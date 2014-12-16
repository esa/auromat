# Copyright European Space Agency, 2013

from __future__ import division, absolute_import, print_function

from six.moves import map
from datetime import datetime
import os
from collections import OrderedDict
import json
import warnings
import glob
import numpy as np

from auromat.mapping.mapping import BaseMappingProvider
import auromat.utils
from auromat.util.url import downloadFile
from auromat.mapping.spacecraft import getMapping, getMappingSequence
from auromat.util.os import makedirs
from auromat.util.lensdistortion import getLensfunModifierFromParams
from auromat.util.image import saveImage, croppedImage
from auromat.util.decorators import inherit_docs

try:
    import rawpy
    import rawpy.enhance
    import lensfunpy
except ImportError as e:
    print(repr(e))
    warnings.warn('rawpy or lensfunpy is missing, no RAW support available')

defaultBaseUrl = 'http://arrrgh-tools.cosmos.esa.int/api/georef_seqs/'

isoDateFormat = '%Y-%m-%dT%H:%M:%S.%f'

def _parseDates(dic):
    keys = {'date', 'date_start', 'date_end'} & set(dic.keys())
    for k in keys:
        dic[k] = datetime.strptime(dic[k], isoDateFormat)
    return dic

@inherit_docs
class ISSMappingProvider(BaseMappingProvider):
    """
    Provider for ESA's ISS Auroral Photography Mappings.
    """
    def __init__(self, cacheFolder, id_=None, useRaw=True, altitude=110, 
                 sequenceInParallel=False, fastCenterCalculation=False, maxTimeOffset=3,
                 raw_white_balance=None, raw_gamma=(1,1), raw_bps=16, raw_auto_bright=False,
                 noRawPostprocessCaching=True,
                 baseUrl=defaultBaseUrl, offline=False):
        """
        
        :param cacheFolder: folder where images and WCS files are downloaded to
                            Note that each sequence must be in its own folder!
        :param int id_: sequence id, can be omitted in later calls due to caching
        :param useRaw: 
            If True, download raw images and apply necessary pre-processing
            locally if necessary (rotation, lens distortion correction,
            bad pixel removal).
            This requires rawpy and lensfunpy.
            If the sequence is not available in RAW format, then JPEGs will
            be downloaded instead.
        :param altitude: in km
        :param raw_white_balance: (r,g,b) tuple of multipliers for each color.
            If not given, uses white balance from data set (corresponds to daylight). 
        :param raw_gamma: (inv_gamma,toe_slope) tuple. 
            For visually pleasing images, use (1/2.222,4.5), see recommendation BT.709.
            For linear images (photon count corresponds linearly to color values), use (1,1).
        :param raw_bps: 8 or 16, bits per color sample
        :param raw_auto_bright: 
            If True, automatically brightens the image such that 1% of all pixels are
            fully saturated. Note that this may destroy useful image information.
        :param noRawPostprocessCaching: 
            If True, then postprocessed RAW files are not written to disk as .tiff files.
            This saves disk space but requires re-computation if a mapping is requested
            multiple times. If False, then postprocessed images are cached and must be
            deleted with removePostProcessedImages() if different RAW postprocessing
            settings should be used.
        :param baseUrl: API base url to the mapping sequences
        :param offline: if True, then missing data is not automatically downloaded,
                        instead an exception is raised
        """
        if raw_bps == 16 and not noRawPostprocessCaching:
            noRawPostprocessCaching = True
            print('noRawPostprocessCaching=False can currently not be used together with raw_bps=16,'
                  'the parameter is implicitly set to True')
            # This is because there is data corruption when saving 16bit Tiff images
            # with scikit-image (0.11dev) and Pillow (2.6). The exact cause is not known.
            
        BaseMappingProvider.__init__(self, maxTimeOffset=maxTimeOffset)
        makedirs(cacheFolder)
        self.cacheFolder = cacheFolder
        self.noRawPostprocessCaching = noRawPostprocessCaching
        self.offline = offline
        
        self.apiDataPath = os.path.join(cacheFolder, 'api.json')
        if not os.path.exists(self.apiDataPath) and not offline:
            if not id_:
                raise ValueError('The id_ parameter must be given the first time')
            url = baseUrl + str(id_)
            downloadFile(url, self.apiDataPath)
        with open(self.apiDataPath, 'r') as fp:
            self.apiData = json.load(fp, object_hook=_parseDates)
                    
        self.metadataPath = os.path.join(cacheFolder, 'metadata.json')
        if not os.path.exists(self.metadataPath) and not offline:
            downloadFile(self.apiData['metadata_uri'], self.metadataPath)
        with open(self.metadataPath, 'r') as fp:
            self.metadata = json.load(fp, object_hook=_parseDates)
            
        self.apiImages = OrderedDict(sorted(self.apiData['images'].items(), key=lambda k_v: k_v[1]['date']))
        self.useRaw = useRaw and 'raw_extension' in self.apiData
        self.altitude = altitude
        self.sequenceInParallel = sequenceInParallel
        self.fastCenterCalculation = fastCenterCalculation
        
        self.processedImagePaths = {}
        self.wcsPaths = {}
        if self.useRaw:
            self.raw_white_balance = raw_white_balance
            self.raw_gamma = raw_gamma
            self.raw_bps = raw_bps
            self.raw_no_auto_bright = not raw_auto_bright
            self.rawImagePaths = {}
            self.badPixelsPath = os.path.join(cacheFolder, 'bad_pixels.gz')
            if not os.path.exists(self.badPixelsPath) and not offline:
                downloadFile(self.apiData['raw_bad_pixels_uri'], self.badPixelsPath)
            self.badPixels = np.loadtxt(self.badPixelsPath, int)
    
    @property
    def range(self):
        return self.apiData['date_start'], self.apiData['date_end']
        
    def _getIdxWithOffset(self, date):
        dates = [i['date'] for i in self.apiImages.values()]
        
        idx = auromat.utils.findNearest(dates, date)
        offset = abs(dates[idx]-date).total_seconds()
        return idx, offset
        
    def contains(self, date):
        _, offset = self._getIdxWithOffset(date)
        return offset <= self.maxTimeOffset
    
    def get(self, date):
        idx, offset = self._getIdxWithOffset(date)
        if offset > self.maxTimeOffset:
            raise ValueError('No image found')
        
        key = list(self.apiImages.keys())[idx]
        
        self._downloadFiles(key)
        wcsPath = self.wcsPaths[key]
        imagePathOrArray = self._processedImage(key)
        
        metadata = dict(list(self.metadata['sequence_metadata'].items()) +
                        list(self.metadata['image_metadata'][key].items()))
        
        mapping = getMapping(imagePathOrArray, wcsPath, altitude=self.altitude, 
                             fastCenterCalculation=self.fastCenterCalculation,
                             metadata=metadata, identifier=key)
        return mapping
    
    def getById(self, identifier):
        return self.get(self.apiImages[identifier]['date'])
    
    def getSequence(self, dateBegin=None, dateEnd=None):
        keys = self.download(dateBegin, dateEnd)
        
        wcsPaths = [self.wcsPaths[k] for k in keys]
        imagePathsOrArrays = map(self._processedImage, keys)
        
        metadatas = [dict(list(self.metadata['sequence_metadata'].items()) + 
                          list(self.metadata['image_metadata'][k].items()))
                     for k in keys]
        
        return getMappingSequence(imagePathsOrArrays, wcsPaths, altitude=self.altitude, 
                                  parallel=self.sequenceInParallel, 
                                  fastCenterCalculation=self.fastCenterCalculation,
                                  metadatas=metadatas)
    
    def download(self, dateBegin=None, dateEnd=None):
        """
        Download part of or the whole sequence to the cache folder.
        An error is raised if self.offline is True and data has not
        been downloaded yet.
        """
        if not dateBegin:
            dateBegin = self.range[0]
        if not dateEnd:
            dateEnd = self.range[1]
        keys = list(k for k,_ in filter(lambda k_v: dateBegin <= k_v[1]['date'] <= dateEnd, self.apiImages.items()))    
        for key in keys:
            self._downloadFiles(key)
        return keys
    
    def removePostProcessedImages(self):
        """
        Removes postprocessed RAW images.
        Useful when noRawPostprocessCaching=False and different RAW
        postprocessing settings need to be applied.
        """
        if not self.useRaw:
            raise ValueError('useRaw must be True')
        for p in glob.glob(os.path.join(self.cacheFolder, '*.tiff')):
            os.remove(p)
        self.processedImagePaths = {}
    
    def _processedImage(self, key):
        if key in self.processedImagePaths:
            return self.processedImagePaths[key]
        assert self.useRaw
        rawPath = self.rawImagePaths[key]
       
        raw = rawpy.imread(rawPath)           
        rawpy.enhance.repair_bad_pixels(raw, self.badPixels)
        if self.raw_white_balance:
            if not (raw.color_desc == 'RGBG' and raw.num_colors == 3):
                raise NotImplementedError
            wb = [self.raw_white_balance[0], self.raw_white_balance[1], 
                  self.raw_white_balance[2], self.raw_white_balance[1]]
        else:
            wb = self.apiData['raw_white_balance']
        rgb = raw.postprocess(user_wb=wb, output_bps=self.raw_bps, 
                              no_auto_bright=self.raw_no_auto_bright,
                              gamma=self.raw_gamma, user_flip=False)
        
        if self.apiData['raw_is_upside_down']:
            # rotate 180deg
            rgb = rgb[::-1,::-1]
        
        # correct lens distortion
        dist_corr = self.apiData['distortion_correction']
        if dist_corr:
            # TODO undistCoords could be cached to avoid recomputing them each time
            mod = getLensfunModifierFromParams(dist_corr['model'], dist_corr['params'], 
                                               rgb.shape[1], rgb.shape[0])
            undistCoords = mod.apply_geometry_distortion()
            rgb = lensfunpy.util.remap(rgb, undistCoords)
        
        # WCS solutions from the ISS dataset are based on cropped images,
        # therefore we crop here as well to make it match.
        rgb = croppedImage(rgb, divisible_by=16)
        
        if self.noRawPostprocessCaching:
            return rgb
        else:       
            processedImagePath = os.path.join(self.cacheFolder, key + '.tiff')
            saveImage(processedImagePath, rgb)
            self.processedImagePaths[key] = processedImagePath
            return processedImagePath
        
    def _downloadFiles(self, key):
        apiImage = self.apiImages[key]
        imageUrl = apiImage['raw_uri' if self.useRaw else 'image_uri']
        wcsUrl = apiImage['wcs_uri']
        imageExt = self.apiData['raw_extension' if self.useRaw else 'image_extension']

        imagePath = os.path.join(self.cacheFolder, key + imageExt)
        wcsPath = os.path.join(self.cacheFolder, key + '.wcs')
        
        if not os.path.exists(imagePath):        
            assert not self.offline
            downloadFile(imageUrl, imagePath)
            
        if not os.path.exists(wcsPath):
            assert not self.offline
            downloadFile(wcsUrl, wcsPath)
        
        if self.useRaw:
            self.rawImagePaths[key] = imagePath
        else:
            self.processedImagePaths[key] = imagePath
        self.wcsPaths[key] = wcsPath        
    