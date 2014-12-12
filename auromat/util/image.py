# Copyright European Space Agency, 2013

from __future__ import absolute_import, division

import warnings
import os
from datetime import datetime

import numpy as np
import numpy.ma as ma
import skimage.io
import skimage.color
import exifread

from PIL import Image

def loadImage(imagePath):
    """
    Return RGB image in native color range (e.g. [0,255] for uint8).
    Ignores the alpha channel.
    
    Note that this function is not meant for reading RAW files. 
    Use `rawpy <https://pypi.python.org/pypi/rawpy>`_ with desired postprocessing settings instead.
    
    :param imagePath:
    :rtype: rgb array of shape (height,width,3)
    """
    rgb = skimage.io.imread(imagePath)
    
    if rgb.ndim == 2:
        rgb = skimage.color.gray2rgb(rgb)
    
    # ignore alpha if available
    rgb = rgb[:,:,:3]
            
    assert rgb.ndim == 3 and rgb.shape[2] == 3,\
           imagePath + '; wrong shape: ' + str(rgb.shape)
        
    return rgb

def saveImage(imagePath, im, **kw):
    """
    :param imagePath: output path
    :param im: the image
    
    Additional optional keywords for JPEG:
    
    :param quality: 1 to 95
    :param subsampling: '4:4:4', '4:2:2', or '4:1:1'
    """
    if os.path.splitext(imagePath)[1].lower() in ['.jpeg', '.jpg']:
        # for jpg we use Pillow directly, as skimage curently doesn't allow
        # to set the JPEG quality
        im = Image.fromarray(im)
        im.save(imagePath, optimize=True, **kw)
    else:
        skimage.io.imsave(imagePath, im)

def croppedImage(im, divisible_by=16):
    """
    Return image cropped to the next smaller width/height divisable by the given factor.
    Cropping is done on each side and will fail if the amount to crop cannot
    be divided by 2.
    """
    cropped_height = int(im.shape[0]/divisible_by)*divisible_by
    cropped_width = int(im.shape[1]/divisible_by)*divisible_by
    assert (im.shape[0]-cropped_height) % 2 == 0
    assert (im.shape[1]-cropped_width) % 2 == 0
    vcrop = (im.shape[0]-cropped_height)//2
    hcrop = (im.shape[1]-cropped_width)//2
    im = im[vcrop:vcrop+cropped_height,hcrop:hcrop+cropped_width]
    return im

def _normalizeImage(rgb):
    """
    Converts (masked) image to float [0,255] color range.
    If it is a masked image, then it is converted to a non-masked
    image with invalid entries replaced by nan.
    """    
    if rgb.dtype == np.uint8:
        pass
    
    elif rgb.dtype == np.uint16:
        rgb = rgb * (255/65535)
        
    else:
        raise NotImplementedError('Image format ' + str(rgb.dtype) + ' not supported')
    
    rgb = np.require(rgb, np.float)
    
    if ma.isMaskedArray(rgb):
        rgb = rgb.filled(np.nan)
    
    return rgb

def image2mpl(rgb):
    """
    Converts image to [0,1] range suitable for matplotlib.
    """
    rgb = _normalizeImage(rgb)
    return rgb / 255

def image2cv(rgb):
    """
    Converts RGB image to BGR uint8 image suitable for use within OpenCV.
    """
    import cv2 as cv
    rgb = _normalizeImage(rgb)
    rgb = np.require(rgb, np.uint8)
    bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
    return bgr

def readExifTime(imagePath):
    """
    Reads the date/time an image was taken from its EXIF header.
    If available, sub second time is used as the fractional part
    of the seconds.
    
    :type imagePath: string
    :rtype: datetime
    """
    with open(imagePath, 'rb') as p:
        tags = exifread.process_file(p, details=False)
        timeTakenExif = tags['EXIF DateTimeOriginal']
        subsec = tags.get('EXIF SubSecTimeOriginal', '0')
    return convertExifDate(str(timeTakenExif), str(subsec))

def convertExifDate(dateStr, subsecStr = None):
    """
    Converts an EXIF date/time string to a datetime object.
    
    :param dateStr: date of format %Y:%m:%d %H:%M:%S
    :param subsecStr: float part of seconds as string, e.g. "05"
    :rtype: datetime.datetime
    """
    if subsecStr is None:
        subsecStr = '0'
    else:
        # If there are no leading zeros, then exiftool returns an int
        # instead of a string. To support this case, we convert whatever
        # we get to a string.
        subsecStr = str(subsecStr)
    return datetime.strptime(dateStr + '.' + subsecStr, '%Y:%m:%d %H:%M:%S.%f')

def readFocalLength35mm(imagePath):
    """
    Reads the focal length corresponding to 35mm film from the EXIF header of an image.
    
    :type imagePath: string
    :rtype: int|None
    """
    with open(imagePath, 'rb') as p:
        tags = exifread.process_file(p, details=False, stop_tag='FocalLengthIn35mmFilm')
        try:
            # Note: FocalLength is a ratio while FocalLengthIn35mmFilm is an integer
            focalLength35mm = tags['EXIF FocalLengthIn35mmFilm'].values[0]
        except KeyError:
            return None
    return focalLength35mm

def readExposureTime(imagePath):
    """
    Reads the exposure time in seconds from the EXIF header of an image.
    
    :type imagePath: string
    :rtype: float|None
    """
    with open(imagePath, 'rb') as p:
        tags = exifread.process_file(p, details=False, stop_tag='ExposureTime')
        try:
            exposureTimeRatio = tags['EXIF ExposureTime'].values[0]
            exposureTime = exposureTimeRatio.num / exposureTimeRatio.den
        except KeyError:
            return None
    return exposureTime
