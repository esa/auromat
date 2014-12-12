# Copyright European Space Agency, 2013

from __future__ import division, print_function

from six.moves import zip, map
from six import string_types
import warnings
import os
import sys
import gc
import fnmatch
import time
import json
from datetime import datetime
from collections import OrderedDict

if sys.version_info.major == 2:
    try:
        from numap.NuMap import NuMap
    except ImportError as e:
        print('parallel processing not available (NuMap missing)')
        print(repr(e))
else:
    print('parallel processing not available (NuMap not supported on Python 3 yet)')

import numpy as np

import auromat.fits

from auromat.coordinates.geodesic import wgs84A, wgs84B
from auromat.coordinates.intersection import ellipsoidLineIntersects
from auromat.mapping.mapping import BaseMappingProvider, FileImageMixin,\
    sanitize_data, ArrayImageMixin
from auromat.mapping.astrometry import BaseAstrometryMapping,\
    ImageMaskAstrometryMixin
from auromat.util.decorators import lazy_property, inherit_docs
from auromat.coordinates.ephem import EphemerisCalculator

@inherit_docs
class SpacecraftMappingProvider(BaseMappingProvider):
    def __init__(self, imageSequenceFolder, wcsFolder=None, imageFileExtension=None, timeshift=None, 
                 noradId=None, tleFolder=None, spacetrack=None, altitude=110, maxTimeOffset=3,
                 sequenceInParallel=False):
        """        
        :param imageSequenceFolder: folder path or a list of image file paths
        :param wcsFolder: folder path or a list of wcs file paths;
                          optional if imageSequenceFolder is a folder path and contains
                          the wcs files
        """
        BaseMappingProvider.__init__(self, maxTimeOffset=maxTimeOffset)
        if wcsFolder is None:
            assert not isinstance(imageSequenceFolder, list),\
                   'The wcsFolder parameter is required if imageSequenceFolder is a list'
            wcsFolder = imageSequenceFolder
        if isinstance(imageSequenceFolder, list) and isinstance(wcsFolder, list):
            self.imagePaths = imageSequenceFolder
            self.wcsPaths = wcsFolder
            self._imageFileExtension = os.path.splitext(self.imagePaths[0])[1][1:]
            self._checkEachWcsHasOneImage()
            self._sortByDate()
        elif not isinstance(imageSequenceFolder, list) and not isinstance(wcsFolder, list):
            self.imageSequenceFolder = imageSequenceFolder
            self.wcsFolder = wcsFolder
            self._imageFileExtension = imageFileExtension
            self.reload()            
        else:
            raise ValueError('imageSequenceFolder and wcsFolder must be both path lists or folder paths')
                
        self.timeshift = timeshift
        self.noradId = noradId
        self.tleFolder = tleFolder
        self.spacetrack = spacetrack
        self.altitude = altitude
        
        metadataPath = os.path.join(os.path.dirname(self.imagePaths[0]), 'metadata.json')
        if os.path.exists(metadataPath):
            with open(metadataPath, 'r') as fp:
                self.metadata = json.load(fp, object_hook=_parseDates)
        else:
            self.metadata = None
                
        self._sequenceInParallel = sequenceInParallel
    
    def __len__(self):
        return len(self.wcsPaths)
    
    def reload(self):
        """
        Refresh to current disk state if imageSequenceFolder and wcsFolder
        are folders instead of file path lists.
        """
        wcsFilenames = os.listdir(self.wcsFolder)
        wcsPaths = [os.path.join(self.wcsFolder, f) for f in wcsFilenames]
        self.wcsPaths = fnmatch.filter(wcsPaths, '*.wcs')
        
        imageFilenames = os.listdir(self.imageSequenceFolder)
        imagePaths = [os.path.join(self.imageSequenceFolder, f) for f in imageFilenames]

        try:
            self.imagePaths = fnmatch.filter(imagePaths, '*.' + self.imageFileExtension)
        except ValueError:
            self.imagePaths = []
            self.wcsPaths = []
        
        self._checkEachWcsHasOneImage()
        self._sortByDate()
        
    def _checkEachWcsHasOneImage(self):
        wcsFilenames = map(os.path.basename, self.wcsPaths)
        ids = [os.path.splitext(f)[0] for f in wcsFilenames]
        imageFilenames = list(map(os.path.basename, self.imagePaths))
        imageIds = list(filter(lambda id_: id_ + '.' + self.imageFileExtension in imageFilenames, ids))
        assert len(imageIds) == len(ids), 'image ids: ' + str(imageIds) + '; wcs ids: ' + str(ids)
        self.ids = ids

    def _sortByDate(self):
        dates = {auromat.fits.getShiftedPhotoTime(auromat.fits.readHeader(p)): (p, id_)
                 for p, id_ in zip(self.wcsPaths, self.ids)}
        dates = OrderedDict(sorted(dates.items(), key=lambda k_v: k_v[0]))
        self.dates = dates.keys()
        self.wcsPaths = [p for p,_ in dates.values()]
        self.ids = [id_ for _,id_ in dates.values()]
    
    @property
    def imageFileExtension(self):
        """ e.g. 'jpg' """
        if self._imageFileExtension is None:
            # try to find extension ourselves
            imageFilenames = os.listdir(self.imageSequenceFolder)
            wcsFilenames = fnmatch.filter(os.listdir(self.wcsFolder), '*.wcs')
            if self.imageSequenceFolder == self.wcsFolder:
                imageFilenames = set(imageFilenames) - set(wcsFilenames)
            for wcsFilename in wcsFilenames:
                fileBase = os.path.splitext(wcsFilename)[0]
                matches = fnmatch.filter(imageFilenames, fileBase + '.*')
                if len(matches) == 1:
                    self._imageFileExtension = os.path.splitext(matches[0])[1][1:]
                    break
                elif len(matches) > 1:
                    raise ValueError('Image file extension not given but multiple candidates exist: ' + str(matches))
            if self._imageFileExtension is None:
                raise ValueError('Image file extension could not be determined. Make sure that there exists at least ' +
                                 'one .wcs file and a corresponding image with the same filename base.')
        
        return self._imageFileExtension    
    
    @property
    def range(self):
        return self.dates[0], self.dates[-1]
    
    @property
    def unsolvedIds(self):
        imageFilenames = map(os.path.basename, self.imagePaths)
        imageIds = [os.path.splitext(f)[0] for f in imageFilenames]
        unsolvedIds = filter(lambda id_: id_ not in self.ids, imageIds)
        return sorted(unsolvedIds)
    
    def _getIdxWithOffset(self, date):        
        idx = auromat.utils.findNearest(self.dates, date)
        offset = abs(self.dates[idx]-date).total_seconds()
        return idx, offset
            
    def contains(self, date):
        _, offset = self._getIdxWithOffset(date)
        return offset <= self.maxTimeOffset
    
    def get(self, date):
        idx, offset = self._getIdxWithOffset(date)
        if offset > self.maxTimeOffset:
            raise ValueError('No image found')
        
        identifier = self.ids[idx]
        
        imagePath = os.path.join(self.imageSequenceFolder, identifier + '.' + self.imageFileExtension)
        wcsPath = self.wcsPaths[idx]
        
        if self.metadata:
            metadata = dict(list(self.metadata['sequence_metadata'].items()) + 
                            list(self.metadata['image_metadata'][identifier].items()))
        else:
            metadata = None
        
        mapping = getMapping(imagePath, wcsPath, self.timeshift, 
                             self.noradId, self.tleFolder, self.spacetrack, 
                             altitude=self.altitude,
                             metadata=metadata)
        return mapping
    
    def getById(self, identifier):
        matchedIds = filter(lambda id_: identifier in id_, self.ids)
        assert len(matchedIds) == 1, 'Ambiguous identifier: ' + str(matchedIds)
        identifier = matchedIds[0]
        idx = self.ids.index(identifier)
        return self.get(self.dates[idx])
    
    def getSequence(self, dateBegin=None, dateEnd=None):
        assert dateBegin is None and dateEnd is None, 'Date ranges not supported'
        try:
            self.imageFileExtension
        except ValueError as e:
            warnings.warn(str(e) + ' Returning empty sequence.')
            return []
        
        imagePaths = [os.path.join(self.imageSequenceFolder, id_ + '.' + self.imageFileExtension) for id_ in self.ids]
        
        if self.metadata:
            seqmeta = list(self.metadata['sequence_metadata'].items())
            metadatas = [dict(seqmeta + list(self.metadata['image_metadata'][k].items())) for k in self.ids]
        else:
            metadatas = None
        
        return getMappingSequence(imagePaths, self.wcsPaths, metadatas=metadatas,
                                 timeshift=self.timeshift, noradId=self.noradId, 
                                 tleFolder=self.tleFolder, spacetrack=self.spacetrack, 
                                 altitude=self.altitude,
                                 parallel=self._sequenceInParallel)

@inherit_docs  
class SpacecraftMappingPathProvider(BaseMappingProvider):
    def __init__(self, imagePaths, wcsPaths, metadataPath=None, timeshift=None, 
                 noradId=None, tleFolder=None, spacetrack=None, altitude=110, maxTimeOffset=3,
                 sequenceInParallel=False, fastCenterCalculation=False):
        BaseMappingProvider.__init__(self, maxTimeOffset=maxTimeOffset)
        assert len(imagePaths) == len(wcsPaths)
        self.imagePaths, self.wcsPaths = self._sortByDate(imagePaths, wcsPaths)
                
        self.timeshift = timeshift
        self.noradId = noradId
        self.tleFolder = tleFolder
        self.spacetrack = spacetrack
        self.altitude = altitude
        self.sequenceInParallel = sequenceInParallel
        self.fastCenterCalculation = fastCenterCalculation
        
        if metadataPath and os.path.exists(metadataPath):
            with open(metadataPath, 'r') as fp:
                self.metadata = json.load(fp, object_hook=_parseDates)
        else:
            self.metadata = None
             
    def __len__(self):
        return len(self.wcsPaths)

    @staticmethod
    def _sortByDate(imagePaths, wcsPaths):
        def date(wcsPath_imagePath):
            wcsPath = wcsPath_imagePath[0]
            wcsHeader = auromat.fits.readHeader(wcsPath)
            return auromat.fits.getPhotoTime(wcsHeader)
        paths = sorted(zip(wcsPaths, imagePaths), key=date)
        wcsPaths = [wcsPath for wcsPath, _ in paths]
        imagePaths = [imagePath for _, imagePath in paths]
        return imagePaths, wcsPaths
    
    @property
    def imageFileExtension(self):
        return os.path.splitext(self.imagePaths[0])[1][1:]
    
    @property
    def range(self):
        fromDate = getMapping(self.imagePaths[0], self.wcsPaths[0]).photoTime
        toDate = getMapping(self.imagePaths[-1], self.wcsPaths[-1]).photoTime
        return fromDate, toDate
    
    def contains(self, date):
        raise NotImplementedError
        
    def get(self, date):
        # TODO implement provider access by date
        raise NotImplementedError
    
    def getById(self, identifier):
        raise NotImplementedError
    
    def getSequence(self, dateBegin=None, dateEnd=None):
        assert dateBegin is None and dateEnd is None, 'Date ranges not supported'
        
        if self.metadata:
            keys = [os.path.splitext(os.path.basename(p))[0] for p in self.imagePaths]
            seqmeta = list(self.metadata['sequence_metadata'].items())
            metadatas = [dict(seqmeta + list(self.metadata['image_metadata'][k].items())) for k in keys]
        else:
            metadatas = None
        
        return getMappingSequence(self.imagePaths, self.wcsPaths,
                                 timeshift=self.timeshift, noradId=self.noradId, 
                                 tleFolder=self.tleFolder, spacetrack=self.spacetrack, 
                                 altitude=self.altitude,
                                 parallel=self.sequenceInParallel, 
                                 fastCenterCalculation=self.fastCenterCalculation,
                                 metadatas=metadatas)

def _getMappingSequenceArgs(imagePathsOrArrays, wcsPaths, timeshift=None, 
                            noradId=None, tleFolder=None, spacetrack=None, altitude = 110,
                            fastCenterCalculation=False, metadatas=None):   
    if not metadatas:
        metadatas = [{}] * len(wcsPaths)
        
    return (dict(imagePathOrArray=imagePathOrArray, wcsPathOrHeader=wcsPath,
                 timeshift=timeshift, noradId=noradId,
                 tleFolder=tleFolder, spacetrack=spacetrack, 
                 altitude=altitude,
                 fastCenterCalculation=fastCenterCalculation,
                 metadata=metadata)
            for imagePathOrArray, wcsPath, metadata in zip(imagePathsOrArrays, wcsPaths, metadatas))
        
def getMappingSequence(imagePathsOrArrays, wcsPaths, metadatas=None, timeshift=None, 
                       noradId=None, tleFolder=None, spacetrack=None, altitude = 110,
                       parallel=False, fastCenterCalculation=False):
    """
    Returns a generator of SpacecraftMapping objects
    for all images in 'imageSequenceFolder' which have a solution in 'wcsFolder'.
    The order corresponds to the sorted filenames in 'wcsFolder'.
    
    :param iterable imagePathsOrArrays:
    :param list wcsPaths:
    """
    mappingArgsArr = _getMappingSequenceArgs(imagePathsOrArrays, wcsPaths, 
                                             timeshift, noradId, tleFolder, spacetrack, 
                                             altitude, fastCenterCalculation,
                                             metadatas=metadatas)
    
    if parallel:
        return _getMappingsParallel(mappingArgsArr)
    else:
        def mappingFromKw(kw):
            mapping = getMapping(**kw)
            # see _getMappingsParallel
            gc.collect()
            return mapping
        return map(mappingFromKw, mappingArgsArr)

def _getMappingsParallel(mappingArgsArr):
    # Each worker process takes 2-4GiB!
    # We use only one worker as the main process is usually slower in consuming
    # the mappings.
    workerCount = 1
    
    # TODO use iterator class instead of yield to conserve memory 
    # (local variable 'mapping' holds on to reference and is only released on next iteration)
    
    mappings = NuMap(_getCalculatedMappingFromArgs, mappingArgsArr,
                     worker_type='process', worker_num=workerCount, buffer=workerCount)
    try:
        mappings.start()
        for mapping in mappings:
            yield mapping
            
            # Due to some reference cycles there are numpy arrays which don't
            # get freed implicitly. As the arrays we work with are quite huge
            # this adds up quickly. The problem is that currently the garbage collector
            # doesn't know about the real size of the numpy arrays (as they are C extension
            # types and there is no API yet for communicating the real size to the Python
            # interpreter). Therefore the thresholds for triggering a garbage collection
            # are seldomly reached and instead we consume more and more memory and eventually
            # run out of it. To fight against this, we manually run a collection to force
            # freeing up native memory.
            gc.collect()
    finally:
        mappings.stop(ends=[0])
   
def _getCalculatedMappingFromArgs(kwargs):
    """
    A helper function which gets a mapping and forces the
    calculation of its (lazy) properties.
    See getMappingSequence().
    """
    try:
        os.nice(10) # only on UNIX systems
    except:
        pass
    mapping = getMapping(**kwargs)
    # force calculation within worker process
    mapping.boundingBox
    mapping.elevation
    return mapping
            

def getMapping(imagePathOrArray, wcsPathOrHeader, timeshift=None, noradId=None, tleFolder=None, spacetrack=None, 
               altitude=110, fastCenterCalculation=False, metadata=None, nosanitize=False,
               identifier=None):
    """
    If timeshift is None, then the wcs header is first checked for a shifted timestamp
    and corresponding spacecraft position. In case no shifted timestamp exists, the
    wcs header is checked for the original timestamp and spacecraft position. If only the
    timestamp exists (which may be the case for externally produced wcs files), the 
    spacecraft position is calculated from two-line elements. If the latter applies or
    'timeshift' is given, then tleFolder must be given.
    If the tleFolder doesn't contain a %noradid%.tle file, then spacetrack is used to download
    the data (or an error is raised if spacetrack is None). The NORAD ID is determined from
    the noradId parameter, or if that is None from the wcs header. If in the latter case 
    the wcs header doesn't contain the NORAD ID, then the ISS ID (25544) is used as a default and a
    warning is printed. 

    :param imagePathOrArray:
    :param wcsPathOrHeader:
    :param datetime.timedelta timeshift: if set, overrides the shifted timestamp stored in the wcs headers
    :param noradId: if set, overrides the NORAD ID stored in the wcs headers
    :param tleFolder: folder containing TLE files named noradid.tle
    :param spacetrack: a Spacetrack class instance
    :param altitude:
    :rtype: BaseSpacecraftMapping
    """
    wcsHeader, photoTime, originalPhotoTime, cameraPosGCRS = \
        _prepareMappingParams(wcsPathOrHeader, timeshift, noradId, tleFolder, spacetrack)
    
    isImageArray = not isinstance(imagePathOrArray, string_types)
    isWcsHeader = not isinstance(wcsPathOrHeader, string_types)
        
    if identifier is None:
        if not isImageArray:
            identifier = os.path.splitext(os.path.basename(imagePathOrArray))[0]
        elif not isWcsHeader:
            identifier = os.path.splitext(os.path.basename(wcsPathOrHeader))[0]
    
    if isImageArray:
        cls = ArraySpacecraftMapping        
    else:
        cls = FileSpacecraftMappingUnsanitized if nosanitize else FileSpacecraftMapping
        
    mapping = cls(wcsHeader, altitude, imagePathOrArray, cameraPosGCRS, photoTime, 
                  identifier, metadata,
                  originalPhotoTime=originalPhotoTime, 
                  fastCenterCalculation=fastCenterCalculation)            
    return mapping 

def _prepareMappingParams(wcsPathOrHeader, timeshift=None, noradId=None, tleFolder=None, spacetrack=None):
    if noradId is not None:
        noradId = int(noradId)
    
    if isinstance(wcsPathOrHeader, string_types):
        fitsWcsHeader = auromat.fits.readHeader(wcsPathOrHeader)
    else:
        fitsWcsHeader = wcsPathOrHeader
    
    originalPhotoTime = auromat.fits.getPhotoTime(fitsWcsHeader)
    if originalPhotoTime is None:
        raise ValueError('DATE-OBS missing in FITS header')

    if timeshift is not None:
        photoTime = originalPhotoTime + timeshift
        cameraPosGCRS = None
    else:
        cameraPosGCRS, photoTime_, _ = auromat.fits.getShiftedSpacecraftPosition(fitsWcsHeader)
        if cameraPosGCRS is not None:
            photoTime = photoTime_
        else:        
            photoTime = originalPhotoTime
            cameraPosGCRS, _ = auromat.fits.getSpacecraftPosition(fitsWcsHeader)
            if cameraPosGCRS is None:
                warnings.warn('Spacecraft position is missing in FITS header, will recalculate')
    
    if cameraPosGCRS is None:
        if noradId is None:
            noradId = auromat.fits.getNoradId(fitsWcsHeader)
            if noradId is None:
                warnings.warn('NORAD ID is missing in FITS header, assuming ISS (25544)')
                noradId = 25544
        
        if tleFolder is None:
            raise ValueError('You need to specify tleFolder to calculate spacecraft positions')

        tleFilePath = os.path.join(tleFolder, str(noradId) + '.tle')
        if os.path.exists(tleFilePath):
            # the EphemerisCalculator doesn't need to be cached, fast enough (0.007s)
            ephemCalculator = EphemerisCalculator(tleFilePath)
            if not ephemCalculator.contains(photoTime):
                if spacetrack is None:
                    raise ValueError('Please update ' + tleFilePath + ' or ' + 
                                     'supply a spacetrack instance for automatic download')
                spacetrack.updateTLEsFor(noradId, tleFilePath, photoTime)
                ephemCalculator = EphemerisCalculator(tleFilePath)
            
        elif spacetrack is not None:
            spacetrack.updateTLEsFor(noradId, tleFilePath, photoTime)
            ephemCalculator = EphemerisCalculator(tleFilePath)
            
        else:
            raise ValueError('Please put ' + str(noradId) + '.tle inside ' + tleFolder + ' or ' + 
                             'supply a spacetrack instance for automatic download')
            
        cameraPosGCRS = ephemCalculator(photoTime)
    
    return fitsWcsHeader, photoTime, originalPhotoTime, cameraPosGCRS 

class BaseSpacecraftMapping(BaseAstrometryMapping):
    """
    A mapping which is based on having a camera in/on a spacecraft looking both on earth 
    and the stars and where no exact camera pointing is known.
    The stars were then used to derive a WCS definition with which it is possible to
    calculate the direction vector of each pixel.
    """

    def __init__(self, wcsHeader, alti, cameraPosGCRS, photoTime, identifier, metadata=None, 
                 originalPhotoTime=None, fastCenterCalculation=False):
        BaseAstrometryMapping.__init__(self, wcsHeader, alti, cameraPosGCRS, photoTime, 
                                       identifier, metadata, 
                                       fastCenterCalculation=fastCenterCalculation)
        if originalPhotoTime is None:
            originalPhotoTime = photoTime
        self._originalPhotoTime = originalPhotoTime
            
    @property
    def originalPhotoTime(self):
        return self._originalPhotoTime
        
    @lazy_property
    def intersectsEarth(self):
        """
        Returns a boolean array indicating whether a pixel center intersects with the earth.
        """
        direction = self.cameraToPixelCenterDirection
        t0 = time.time()
        intersectsEarth = ellipsoidLineIntersects(wgs84A, wgs84B,
                                                  self.cameraPosGCRS,
                                                  direction.reshape(-1,3))
        print('intersectsEarth:', time.time()-t0, 's')
        intersectsEarth = intersectsEarth.reshape(self.cameraToPixelCenterDirection.shape[0],
                                                  self.cameraToPixelCenterDirection.shape[1])
        return intersectsEarth
    
    def isConsistent(self, starPxCoords=None):
        """
        Checks if the photo timestamp and astrometric solution used for mapping is plausible by analysing
        the mapping result.
        
        Note that in general there are virtually no false solves when using astrometry.net.
        
        :param starPxCoords: array of shape (n,2) containing x,y pixel coordinates of stars which have been
                             used for obtaining an astrometry solution;
                             for astrometry.net, the quad stars can be used for this purpose,
                             see auromat.solving.readQuadMatch()
        :rtype: True if consistent, False if not
        """
        if np.all(self.intersectsEarth):
            # Although we solved the image using stars, every pixel intersects
            # with the modelled earth. Thus, the camera position is such
            # that the camera would look directly at the earth, with no starfield in the image.
            # The timestamp and/or astrometric solution must be wrong. 
            return False
        elif not np.any(self.intersectsEarth):
            # No pixel intersects with the modelled earth. As we assume that the images always contain
            # a part of the earth, this must again be a wrong timestamp and/or astrometric solution.
            return False
        
        if starPxCoords is not None:
            starCoveredByEarth = self.intersectsEarth[starPxCoords[:,1],starPxCoords[:,0]]
            if np.any(starCoveredByEarth):
                # There is at least one star used for the astrometry solution which would
                # be covered by the modelled earth. Therefore, the timestamp and/or astrometric
                # solution must be wrong.
                return False
        
        return True

@inherit_docs
class FileSpacecraftMappingUnsanitized(ImageMaskAstrometryMixin, FileImageMixin, BaseSpacecraftMapping):
    """
    .. warning:: Consider using FileSpacecraftMapping instead of this class.
                Masking is not supported here.
    
    The purpose of this class is to access certain properties in a very efficient way by
    skipping any sanitization.
    See auromat.test.draw_test.testParallelsMeridiansPlotOptimized for an example usage of this
    behaviour.
    """
    
    def __init__(self, wcsHeader, alti, imagePath, cameraPosGCRS, photoTime, identifier, metadata=None,
                 originalPhotoTime=None, fastCenterCalculation=False):
        ImageMaskAstrometryMixin.__init__(self)
        FileImageMixin.__init__(self, imagePath)
        BaseSpacecraftMapping.__init__(self, wcsHeader, alti, cameraPosGCRS, photoTime, identifier, metadata, 
                                       originalPhotoTime=originalPhotoTime,
                                       fastCenterCalculation=fastCenterCalculation)
    
    def createMasked(self, centerMask):
        raise RuntimeError('Masking is not supported for unsanitized mappings, ' +
                           'please use nosanitize=False in getMapping()')

FileSpacecraftMapping = sanitize_data(FileSpacecraftMappingUnsanitized)

@sanitize_data
@inherit_docs
class ArraySpacecraftMapping(ImageMaskAstrometryMixin, ArrayImageMixin, BaseSpacecraftMapping):   
    """
    Like FileSpacecraftMapping but accepts an RGB image array instead of an image file path.
    """
    def __init__(self, wcsHeader, alti, img, cameraPosGCRS, photoTime, identifier, metadata=None,
                 originalPhotoTime=None, fastCenterCalculation=False):
        ImageMaskAstrometryMixin.__init__(self)
        ArrayImageMixin.__init__(self, img)
        BaseSpacecraftMapping.__init__(self, wcsHeader, alti, cameraPosGCRS, photoTime, identifier, metadata, 
                                       originalPhotoTime=originalPhotoTime,
                                       fastCenterCalculation=fastCenterCalculation)
    
isoDateFormat = '%Y-%m-%dT%H:%M:%S.%f'

def _parseDates(dic):
    keys = {'date'} & set(dic.keys())
    for k in keys:
        dic[k] = datetime.strptime(dic[k], isoDateFormat)
    return dic