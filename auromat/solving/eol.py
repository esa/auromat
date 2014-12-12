# Copyright European Space Agency, 2013

"""
This module allows to easily download images
from `NASA's Earth Observation website <http://eol.jsc.nasa.gov>`_ in JPEG and RAW format.

As there is no API on NASA's end, we have to rely on a certain URL structure,
and, in case of RAW files, HTML structure. As this is not robust, it may fail
if the website is restructured. In that case, the code has to be adapted. 

Notes on JPEGs
--------------
JPEGs from the EOL archive are produced from the RAW camera files.
Different post-processing settings have been applied during the production
of these JPEGs, e.g. colour or exposure correction, or 180 degree rotation.
This is also true for images that belong to a single sequence of images, that is,
colour/exposure might change suddenly. The JPEGs are therefore not
suitable for scientific purposes. On the other hand, they often have hot pixels
removed. A lens distortion correction is typically not applied.

In theory the JPEGs can be used for astrometry while later using the RAW files
for scientic purposes. This is however a risky process as it has to be guaranteed
that the image orientation matches and that the lens distortion was not corrected
in the JPEGs already. To prevent checking these things each time, it is better to
use the RAW files in the first place and prepare them for astrometry ourselves,
that is, removing hot pixels and possibly noise.
"""

from __future__ import division, print_function, absolute_import

from six.moves import range
import six.moves.urllib as urllib
import os
import re
import shutil
import json
import warnings
from time import sleep
from datetime import datetime, timedelta
from collections import namedtuple

from auromat.util import exiftool
from auromat.util.os import makedirs
from auromat.util.url import urlResponseCode, downloadFiles 

try:
    import lensfunpy
    from auromat.util.lensdistortion import CameraNotFoundInDBError, LensNotFoundInDBError,\
        CameraNotFoundInEXIFError, LensNotFoundInEXIFError, getLensfunModifier
    import auromat.util.lensdistortion
except Exception as e:
    print(str(e))
    warnings.warn('lensfunpy not found, reduced functionality in auromat.solving.eol')

metadataFilename = 'meta.json'

class SequenceMetadata(object):
    def __init__(self, mission, roll, fromFrame, toFrame, pattern, frameGaps, lensDistortionCorrected,
                 lensDistortionCorrectionParams=None):
        self.mission = mission
        self.roll = roll
        self.fromFrame = fromFrame
        self.toFrame = toFrame
        self.pattern = pattern
        self.frameGaps = frameGaps
        self.lensDistortionCorrected = lensDistortionCorrected
        self.lensDistortionCorrectionParams = lensDistortionCorrectionParams

class LensDistortionCorrectionParams(object):
    def __init__(self, cameraMaker, cameraModel, cameraVariant, lensMaker, lensModel, 
                 focalLength, aperture):  
        self.cameraMaker = cameraMaker
        self.cameraModel = cameraModel
        self.cameraVariant = cameraVariant
        self.lensMaker = lensMaker
        self.lensModel = lensModel
        self.focalLength = focalLength
        self.aperture = aperture

jpgUrlPattern = 'http://eol.jsc.nasa.gov/sseop/images/ESC/large/{mission}/{mission}-{roll}-{frame}.JPG'
jpgFilePattern = '{mission}-{roll}-{frame}.jpg' # the filename on disk

# 'file' is extracted from the photoPage
photoPageUrlPattern = 'http://eol.jsc.nasa.gov/scripts/sseop/photo.pl?mission={mission}&roll={roll}&frame={frame}'
rawFilePhotoPagePattern = r'<a href=RequestOriginalImage.pl\?mission=[A-Z\d]+&roll=[A-Z\d]+&frame=[\d]+&file=([\w\.]+)>'
rawRequestUrlPattern = 'http://eol.jsc.nasa.gov/scripts/sseop/RequestOriginalImage.pl?mission={mission}&roll={roll}&frame={frame}&file={file}'
rawUrlPattern = 'http://eol.jsc.nasa.gov/sseop/OriginalImagery/{file}'
rawFilePatternNoExt = '{mission}-{roll}-{frame}' # extension not in here, could be .nef or something else

auroraVideosUrl = 'http://eol.jsc.nasa.gov/Videos/CrewEarthObservationsVideos/Videos_Aurora.htm'
auroraVideosPattern = r'<a name="([a-zA-Z\d_]+)">(.+?)</a>.+?' +\
                      '<a href="/scripts/sseop/photo.pl\?mission=([A-Z\d]+)&roll=([A-Z\d]+)&frame=([\d]+)" target="_blank">' +\
                      '<nobr>[A-Z\d-]+</a> to ' +\
                      '<a href="/scripts/sseop/photo.pl\?mission=([A-Z\d]+)&roll=([A-Z\d]+)&frame=([\d]+)" target="_blank">'

def downloadImages(folderPath, ids, format_):
    """
    Download images given by (mission,roll,frame) tuples in the specified
    format.
    
    Note: Use :func:`downloadImageSequence` to download a consecutive sequence
    of images. This function handles frame gaps (gaps in numbering) properly.
    
    :param folderPath: download location
    :param ids: list of tuples (mission,roll,frame)
    :param format_: jpg or raw
    """
    if format_ == 'jpg':
        return downloadImagesJpg(folderPath, ids)
    elif format == 'raw':
        # TODO implement RAW download of single frames
        raise NotImplementedError
    else:
        raise ValueError('Unknown format: ' + format_)

def downloadImagesJpg(folderPath, ids):
    """
    Download JPEG images given by (mission,roll,frame) tuples and return paths if successfull.
    On any error, False is returned.
    Files that are already existing are not downloaded again.
    
    :param folderPath: download location
    :param ids: list of tuples (mission,roll,frame)
    :rtype: list of str | False
    """
    urls = [jpgUrlPattern.format(mission=mission, roll=roll, frame=frame)
            for mission, roll, frame in ids]
    paths = [os.path.join(folderPath, jpgFilePattern.format(mission=mission, 
                                                            roll=roll, 
                                                            frame=frame))
             for mission, roll, frame in ids]
    
    makedirs(folderPath)
    
    if downloadFiles(urls, paths):
        return paths
    else:
        return False

def downloadImageSequence(folderPath, mission, fromFrame, toFrame, format_, roll='E', lensDistortionCorrected=False):
    """
    Download an image sequence in the specified format and return
    a tuple (metadata, []) on success or (False, errors) in case of errors.
    
    :param folderPath: download location
    :param format_: jpg or raw
    :rtype: tuple (SequenceMetadata, failure list)
    """
    if format_ == 'jpg':
        return _downloadImageSequenceJpg(folderPath, mission, fromFrame, toFrame, roll, 
                                         lensDistortionCorrected=lensDistortionCorrected)
    elif format_ == 'raw':
        return _downloadImageSequenceRaw(folderPath, mission, fromFrame, toFrame, roll)
    else:
        raise ValueError('Unsupported format: ' + format_)

def _downloadImageSequenceJpg(folderPath, mission, fromFrame, toFrame, roll='E', lensDistortionCorrected=False):
    # first, download in temp folder, then copy over and remove temp folder if successful
    tempFolderPath = os.path.join(folderPath, 'in_progress')
    
    metadataPath = os.path.join(folderPath, metadataFilename)
    
    fromFrame, toFrame = int(fromFrame), int(toFrame)
    
    # check if already fully downloaded
    firstImage = os.path.join(folderPath, jpgFilePattern.format(mission=mission, 
                                                                roll=roll, 
                                                                frame=fromFrame))
    if os.path.exists(firstImage):
        # as the files are only moved over at the very end, it is enough to
        # check for existance of the first image
        
        # write metadata if not existing yet (for whatever reason..)
        if not os.path.exists(metadataPath):
            frameGaps = []
            for frame in range(fromFrame, toFrame+1):
                imagePath = os.path.join(folderPath, jpgFilePattern.format(mission=mission, 
                                                                           roll=roll, 
                                                                           frame=frame))
                if not os.path.exists(imagePath):
                    frameGaps.append(frame)
            
            meta = SequenceMetadata(mission=mission, roll=roll, fromFrame=fromFrame,
                                    toFrame=toFrame, pattern=jpgFilePattern,
                                    frameGaps=frameGaps, 
                                    lensDistortionCorrected=lensDistortionCorrected)
            storeMetaData(metadataPath, meta)
        else:
            meta = loadMetaData(metadataPath)

        return meta, []
    
    makedirs(folderPath, tempFolderPath)    
    
    frames = range(fromFrame, toFrame+1)
    
    urls = [jpgUrlPattern.format(mission=mission, roll=roll, frame=frame)
            for frame in frames]
    paths = [os.path.join(tempFolderPath, jpgFilePattern.format(mission=mission, 
                                                                roll=roll, 
                                                                frame=frame))
             for frame in frames]
    
    print('downloading sequence frames', fromFrame, 'to', toFrame, 'of', mission + '-' + roll) 
    
    _, errors = downloadFiles(urls, paths, retFailures=True)
    
    # We ignore 404s for frames which are not the start or end.
    # This is because there are sometimes gaps in frame numbers.
    # E.g. for ISS030 the frames 115426 to 115442 don't exist within the
    #      sequence 114986 to 115574
    failures = [] 
    frameGaps = []
    for url, error in errors:
        if isinstance(error, urllib.error.HTTPError):
            i = urls.index(url)
            frame = frames[i]
            if error.code == 404:
                if fromFrame < frame < toFrame:
                    frameGaps.append(frame)
                    continue
                else:
                    raise ValueError('Start/end frame ' + str(frame) + ' not downloadable (404)')
            else:
                failures.append((url, error.code))
        else:
            failures.append((url, error))
        
    if len(failures) > 0:
        return False, failures

    for filename in os.listdir(tempFolderPath):
        shutil.move(os.path.join(tempFolderPath, filename), folderPath)
    
    os.rmdir(tempFolderPath)
    
    meta = SequenceMetadata(mission=mission, roll=roll, fromFrame=fromFrame,
                            toFrame=toFrame, pattern=jpgFilePattern,
                            frameGaps=frameGaps, 
                            lensDistortionCorrected=lensDistortionCorrected)
    storeMetaData(metadataPath, meta)
    
    return meta, []

def _downloadImageSequenceRaw(folderPath, mission, fromFrame, toFrame, roll='E'):
    assert roll == 'E' # only those have RAW files
    
    # first, download in temp folder, then copy over and remove temp folder if successful
    tempFolderPath = os.path.join(folderPath, 'in_progress')
    
    metadataPath = os.path.join(folderPath, metadataFilename)
    
    fromFrame, toFrame = int(fromFrame), int(toFrame)
    
    # check if already fully downloaded
    if os.path.exists(metadataPath):
        return True
    
    makedirs(folderPath, tempFolderPath)

    # first, we determine the RAW filename pattern by looking at a photo page
    firstPhotoPageUrl = photoPageUrlPattern.format(mission=mission, roll=roll, frame=fromFrame)
    photoPageContent = urllib.request.urlopen(firstPhotoPageUrl).read()
    
    match = re.search(rawFilePhotoPagePattern, photoPageContent)
    if match is None:
        raise RuntimeError('Could not find RAW filename on page ' + firstPhotoPageUrl)
    
    rawFilename = match.group(1)
    rawFileBase, rawFileExt = os.path.splitext(rawFilename)
    
    assert mission in rawFileBase or mission.lower() in rawFileBase
    assert roll in rawFileBase or roll.lower() in rawFileBase
    assert str(fromFrame) in rawFileBase
    
    rawFileBasePattern = rawFileBase
    if mission in rawFileBase:
        rawFileBasePattern = rawFileBase.replace(mission, '{mission}')
        missionCased = mission
    elif mission.lower() in rawFileBasePattern:
        rawFileBasePattern = rawFileBasePattern.replace(mission.lower(), '{mission}')
        missionCased = mission.lower()
    else:
        raise RuntimeError('Could not find mission name in ' + rawFileBase)

    if roll in rawFileBasePattern:
        rawFileBasePattern = rawFileBasePattern.replace(roll, '{roll}')
        rollCased = roll
    elif roll.lower() in rawFileBasePattern:
        rawFileBasePattern = rawFileBasePattern.replace(roll.lower(), '{roll}')
        rollCased = roll.lower()
    else:
        raise RuntimeError('Could not find roll name in ' + rawFileBase)
    
    frameZfilled = lambda frame: str(frame).zfill(6)
    if frameZfilled(fromFrame) in rawFileBasePattern:
        rawFileBasePattern = rawFileBasePattern.replace(frameZfilled(fromFrame), '{frame}')
        frameFn = frameZfilled
    elif str(fromFrame) in rawFileBasePattern:
        rawFileBasePattern = rawFileBasePattern.replace(str(fromFrame), '{frame}')
        frameFn = str
    else:
        raise RuntimeError('Could not find frame number in ' + rawFileBase)

    rawFilenamePattern = rawFileBasePattern + rawFileExt
    print('Raw filename pattern: ' + rawFilenamePattern)
    
    frames = range(fromFrame, toFrame+1)
    rawFilenames = [rawFilenamePattern.format(mission=missionCased, roll=rollCased, frame=frameFn(frame))
                    for frame in frames]
    rawRequestUrls = [rawRequestUrlPattern.format(mission=mission, roll=roll, frame=frame, file=rawFilename)
                      for frame, rawFilename in zip(frames, rawFilenames)]
    rawUrls = [rawUrlPattern.format(file=rawFilename) for rawFilename in rawFilenames]
    
    rawFilePatternDisk = rawFilePatternNoExt + rawFileExt.lower()
    rawFilenamesDisk = [rawFilePatternDisk.format(mission=mission, roll=roll, frame=frame) for frame in frames]
    paths = [os.path.join(tempFolderPath, rawFilenameDisk) for rawFilenameDisk in rawFilenamesDisk]
    
    # jpg URLs are used to check if the frame exists (or whether there's a frame gap)
    jpgUrls = [jpgUrlPattern.format(mission=mission, roll=roll, frame=frame)
               for frame in frames]
    
    frameGaps = []
    failures = []
    queue = []
    
    for frame, jpgUrl, rawUrl, rawRequestUrl, path in zip(frames, jpgUrls, rawUrls, rawRequestUrls, paths):
        if os.path.exists(path):
            continue
        try:
            code = urlResponseCode(jpgUrl)
            if code == 200:
                queue.append((rawUrl, rawRequestUrl, path))
                print('Got 200, added frame ' + str(frame) + ' to queue')
            elif code == 404:
                if fromFrame < frame < toFrame:
                    frameGaps.append(frame)
                    print('Got 404, ignoring frame ' + str(frame))
                else:
                    raise ValueError('Start/end frame ' + str(frame) + ' not downloadable (404)')
            else:
                failures.append((rawUrl, code))
                print('Failure: Unexpected response code for jpgUrl: ' + str(code))
        except Exception as e:
            failures.append((rawUrl, e))
            print('Failure: ' + repr(e))
            
    # download RAW files in batches to avoid overloading the server
    batchSize = 30
    batches = [queue[i:i+batchSize] for i in range(0, len(queue), batchSize)]
    for batch in batches:
        batchUrls = []
        batchPaths = []
        for rawUrl, rawRequestUrl, path in batch:
            try:
                code = urlResponseCode(rawRequestUrl)
                if code == 200:
                    print('queried ' + rawRequestUrl)
                    batchUrls.append(rawUrl)
                    batchPaths.append(path)
                else:
                    failures.append((rawUrl, code))
                    print('Failure: Unexpected response code for rawRequestUrl: ' + str(code))
            except Exception as e:
                failures.append((rawUrl, e))
                print('Failure: ' + repr(e))
        
        # now check rawUrls until the files are available for download
        # The request "may take 5 minutes or more to complete" (quote from request page)
        success, failures_ = downloadFiles(batchUrls, batchPaths, retFailures=True)
        failureCount = len(failures_)
        lastFailureCountDecrease = datetime.now()
        while not success and datetime.now() - lastFailureCountDecrease < timedelta(minutes=8):
            sleep(30)
            success, failures_ = downloadFiles(batchUrls, batchPaths, retFailures=True)
            if len(failures_) < failureCount:
                lastFailureCountDecrease = datetime.now()
                failureCount = len(failures_)
        
        failures.extend(failures_)
        
    if len(failures) > 0:
        return False, failures
        
    for filename in os.listdir(tempFolderPath):
        shutil.move(os.path.join(tempFolderPath, filename), folderPath)
    
    os.rmdir(tempFolderPath)
    
    meta = SequenceMetadata(mission=mission, roll=roll, fromFrame=fromFrame,
                            toFrame=toFrame, pattern=rawFilePatternDisk,
                            frameGaps=frameGaps,
                            lensDistortionCorrected=False)
    storeMetaData(metadataPath, meta)
    
    return meta, [] 

Sequence = namedtuple('Sequence', ['mission', 'roll', 'fromFrame', 'toFrame', 'title', 'urlAnchor'])
def extractAuroraSequences():
    """
    Extracts metadata of all sequences found on
    http://eol.jsc.nasa.gov/Videos/CrewEarthObservationsVideos/Videos_Aurora.htm.
    """
    content = urllib.request.urlopen(auroraVideosUrl).read()
    sequences = []
    for match in re.finditer(auroraVideosPattern, content, re.DOTALL):
        urlAnchor, title = match.group(1,2)
        mission, roll, fromFrame = match.group(3,4,5)
        mission_, roll_, toFrame = match.group(6,7,8)
        assert mission == mission_ and roll == roll_
        sequences.append(Sequence(mission, roll, int(fromFrame), int(toFrame), title, urlAnchor))
        
    return sequences

def storeMetaData(jsonPath, meta):
    if os.path.isdir(jsonPath):
        jsonPath = os.path.join(jsonPath, metadataFilename)
    
    metaDict = meta.__dict__
    if meta.lensDistortionCorrectionParams is not None:
        metaDict['lensDistortionCorrectionParams'] = meta.lensDistortionCorrectionParams.__dict__
    
    with open(jsonPath, 'w') as fp:
        json.dump(metaDict, fp, indent=4)

def loadMetaData(jsonPath):
    if os.path.isdir(jsonPath):
        jsonPath = os.path.join(jsonPath, metadataFilename)
    print('loading ' + jsonPath)
    with open(jsonPath) as fp:
        metaDict = json.load(fp)
    meta = SequenceMetadata(**metaDict)
    if meta.lensDistortionCorrectionParams is not None:
        meta.lensDistortionCorrectionParams = LensDistortionCorrectionParams(**meta.lensDistortionCorrectionParams)
    return meta

def filenameOf(frame, meta):
    return _filenameOf(meta.mission, meta.roll, frame, meta.pattern)
    
def _filenameOf(mission, roll, frame, pattern):
    return pattern.format(mission=mission,
                          roll=roll,
                          frame=frame)

def frameIter(meta):
    for frame in range(meta.fromFrame, meta.toFrame+1):
        if frame not in meta.frameGaps:
            yield frame
            
def filenameIter(meta):
    for frame in frameIter(meta):
        yield filenameOf(frame, meta), frame

def correctLensDistortion(folderPath, undistFolderPath, lensfunDbObj=None):
    """
    Corrects the lens distortion of all images in `folderPath` using
    lensfun's distortion profile database.
    
    It is assumed that all images have the same camera and lens.
    Images are skipped whose corrected version already exists in undistFolderPath.
    """
    meta = loadMetaData(folderPath)
    firstImagePath = os.path.join(folderPath, filenameOf(meta.fromFrame, meta))
    mod, cam, lens = getLensfunModifier(firstImagePath, lensfunDbObj=lensfunDbObj)
    
    makedirs(undistFolderPath)
    
    print('starting lens distortion correction for ' + folderPath)
    
    splitted = os.path.splitext(meta.pattern)
    undistPattern = splitted[0] + '_dc' + splitted[1]
       
    with exiftool.ExifTool() as et:
        for filename, frame in filenameIter(meta):
            imagePath = os.path.join(folderPath, filename)
            filenameUndist = _filenameOf(meta.mission, meta.roll, frame, undistPattern)
            undistImagePath = os.path.join(undistFolderPath, filenameUndist)
            if os.path.exists(undistImagePath):
                continue
            auromat.util.lensdistortion.correctLensDistortion(imagePath, undistImagePath, exiftoolObj=et, mod=mod)
    
    dcParams = LensDistortionCorrectionParams(cam.maker, cam.model, cam.variant, lens.maker, lens.model,
                                              mod.focal_length, mod.aperture)
    
    meta.pattern = undistPattern
    meta.lensDistortionCorrected = True
    meta.lensDistortionCorrectionParams = dcParams
    
    storeMetaData(undistFolderPath, meta)
    
if __name__ == '__main__':
    from os.path import expanduser
    home = expanduser("~")
    rootFolderPath = os.path.join(home, 'data/images/aurora_sequences_jpg')
    rootUndistFolderPath = os.path.join(home, 'data/images/aurora_sequences_jpg_dc')
    success, _ = downloadAuroraSequences(rootFolderPath, 'jpg')
    if success:
        print('All aurora sequences were downloaded successfully')
                
        lensfunDb = lensfunpy.Database()
        
        folders = os.listdir(rootFolderPath)
        folderPaths = [os.path.join(rootFolderPath, f) for f in folders]
        folderPaths = filter(os.path.isdir, folderPaths)
        
        for folderPath in folderPaths:
            undistFolderPath = os.path.join(rootUndistFolderPath, os.path.basename(folderPath))
            try:
                correctLensDistortion(folderPath, undistFolderPath, lensfunDb)
            except (CameraNotFoundInEXIFError, LensNotFoundInEXIFError, LensNotFoundInDBError, CameraNotFoundInDBError) as e:
                print('Lens correction not done for ' + folderPath + ': ' + str(e))

    