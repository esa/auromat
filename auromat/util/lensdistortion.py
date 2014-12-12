# Copyright European Space Agency, 2013

"""
This module provides functions to extract camera and lens information from
EXIF data and correct lens distortion using the 
`lensfunpy <https://pypi.python.org/pypi/lensfunpy>`_ library.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import lensfunpy
import lensfunpy.util

from auromat.util import exiftool
from auromat.util.image import loadImage, saveImage
from auromat.utils import vectorLengths

class LensNotFoundInEXIFError(Exception):
    pass
class CameraNotFoundInEXIFError(Exception):
    pass
class CameraNotFoundInDBError(Exception):
    pass
class LensNotFoundInDBError(Exception):
    pass

distortionTags = ['EXIF:Model', 'EXIF:Make', 'Composite:LensID',
                  # and some additional tags used by getLensfunModifier():
                  'EXIF:FocalLength#','Composite:Aperture#', 'Composite:ImageSize']

def _readExifTags(imagePath, tagNames, exiftoolObj=None):
    if exiftoolObj:
        assert not exiftoolObj.nums
        tags = exiftoolObj.get_tags(tagNames, imagePath)
    else:
        with exiftool.ExifTool(nums=False) as et:
            tags = et.get_tags(tagNames, imagePath)
    return tags

def findCameraAndLensFromExif(tags, lensfunDbObj=None, minAcceptedScore=100, raiseIfNotFoundInDB=True):
    if not tags.get('Composite:LensID'):
        raise LensNotFoundInEXIFError('No LensID in EXIF data')
    
    if not tags.get('EXIF:Make') or not tags.get('EXIF:Model'):
        raise CameraNotFoundInEXIFError('No camera make/model in EXIF data')
 
    if lensfunDbObj is None:
        db = lensfunpy.Database()
    else:
        db = lensfunDbObj
    
    cams = db.find_cameras(tags['EXIF:Make'], tags['EXIF:Model'], loose_search=False)
    if not cams:
        if raiseIfNotFoundInDB:
            raise CameraNotFoundInDBError('Camera "' + tags['EXIF:Make'] + ' - ' + tags['EXIF:Model'] + '" not found in DB!')
        else:
            return None, None
    # FIXME cam score is always 0
#    if cams[0].score < minAcceptedScore:
#        raise CameraNotFoundInDBError('Camera "' + tags['EXIF:Make'] + ' - ' + tags['EXIF:Model'] + '" not found! ' + 
#                                  'Closest was "' + cams[0].maker + ' - ' + cams[0].model + '" with score ' + 
#                                  str(cams[0].score) + ' (<' + str(minAcceptedScore) + ')' )
    cam = cams[0]
     
    lenses = db.find_lenses(cam, None, tags['Composite:LensID'], loose_search=True)
    if not lenses or lenses[0].score < minAcceptedScore:
        if raiseIfNotFoundInDB:
            if not lenses:
                raise LensNotFoundInDBError('Lens "' + tags['Composite:LensID'] + '" not found in DB!')
            if lenses[0].Score < minAcceptedScore:
                raise LensNotFoundInDBError('Lens "' + tags['Composite:LensID'] + '" not found in DB! ' +
                                        'Closest was "' + lenses[0].model + '" with score ' + 
                                        str(lenses[0].score) + ' (<' + str(minAcceptedScore) + ')' )
        else:
            return cam, None
        
    lens = lenses[0]
    if lens.model != tags['Composite:LensID']:
        print('NOTE: Using lensfun lens "' + lens.model + '" with score ' + str(lens.score) + \
              ' (EXIF: "' + tags['Composite:LensID'] + '")')

    return cam, lens

def findCameraAndLens(imagePath, lensfunDbObj=None, exiftoolObj=None, retTags=False, minAcceptedScore=100):
    """
    Finds camera and lens from EXIF data.
    
    Note: The lensfunDbObj reference must be kept as long as the returned camera and lens
    objects shall be used. Once lensfunDbObj gets garbage collected, the camera and lens
    objects get unusable (as their underlying C objects get freed). 
    """
    tags = _readExifTags(imagePath, distortionTags, exiftoolObj)
    cam, lens = findCameraAndLensFromExif(tags, lensfunDbObj, minAcceptedScore)
    
    if retTags:
        return cam, lens, tags
    else:
        return cam, lens

def getLensfunModifierFromExif(tags, width=None, height=None, lensfunDbObj=None, distance=10000):
    """
    WARNING: Not setting width and height may produce surprising results for RAW files.
    If width and height are not set, then Composite:ImageSize is used.
    This tag contains the full RAW size, but many RAW decoders produce slightly
    cropped images. Therefore it may be necessary to first decode the RAW image
    and determine the width and height directly.
             
    :param dict tags: must contain 'EXIF:Model', 'EXIF:Make', 'Composite:LensID',
                      'EXIF:FocalLength', 'Composite:Aperture' 
                      and optionally 'Composite:ImageSize' if width,height is None 
    """
    cam, lens = findCameraAndLensFromExif(tags, lensfunDbObj)
    
    if width is None:
        width, height = tags['Composite:ImageSize'].split('x')
        width, height = int(width), int(height)
    
    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(tags['EXIF:FocalLength'], tags['Composite:Aperture'], distance=distance)

    return mod, cam, lens

def getLensfunModifierFromParams(model, params, width, height):
    """
    
    :param str model: 'ptlens', 'poly3', or 'poly5'
    :param list params: a list of 1, 2 or 3 parameters, depending on the model
    :param width: image width in pixels
    :param height: image height in pixels
    """
    xml = lensfunXML(model, *params)
    db = lensfunpy.Database(xml=xml, load_common=False)
    cam = db.cameras[0]
    lens = db.lenses[0]
    mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    mod.initialize(1, 1)
    return mod

def getLensfunModifier(imagePath, width=None, height=None, lensfunDbObj=None, exiftoolObj=None, distance=10000):
    """
    See :func:`getLensfunModifierFromExif` for a WARNING on not setting width and height.
    """
    tags = _readExifTags(imagePath, distortionTags, exiftoolObj)
    mod, cam, lens = getLensfunModifierFromExif(tags, width, height, lensfunDbObj, distance)
    return mod, cam, lens

def correctLensDistortion(imagePath, undistImagePath, 
                          lensfunDbObj=None, mod=None, 
                          exiftoolObj=None, preserveExif=True,
                          **saveImageKws):
    """
    Correct lens distortion of an image using its EXIF headers and the lensfun library.
    If the camera or lens are not found in the lensfun database, an exception is raised.
    
    :param undistImagePath: path to output image; folders must already exist! 
    :param exiftoolObj: if not None, use the given exiftool object (must have nums=False)
    :param lensfunDbObj: if not None, use the given lensfun.Database object
    :param mod: if not None, use this Modifier instead of calling getLensfunModifier()
                Note: lensfunDbObj is not used in this case.
    :raise ValueError: when lens wasn't found in EXIF data
    :raise CameraNotFoundInDBError: when the camera wasn't found in lensfun database
    :raise LensNotFoundInDBError: when the lens wasn't found in lensfun database  
    """
    im = loadImage(imagePath)
    if mod is None:
        height, width = im.shape[0], im.shape[1]
        mod, _, _ = getLensfunModifier(imagePath, width, height, lensfunDbObj, exiftoolObj)

    undistCoords = mod.apply_geometry_distortion()    
    imUndistorted = lensfunpy.util.remap(im, undistCoords)
    saveImage(undistImagePath, imUndistorted, **saveImageKws)
    
    # TODO set a flag indicating that lens correction has been done
    #      there is no standard flag for that
    #      the closest one is Xmp.digiKam.LensCorrectionSettings
    #      see http://api.kde.org/extragear-api/graphics-apidocs/digikam/html/classDigikam_1_1LensFunFilter.html
    #      adding custom XMP tags requires to change some exiftool config file...
    if preserveExif:
        if exiftoolObj:
            exiftoolObj.copy_tags(imagePath, undistImagePath)
        else:
            with exiftool.ExifTool() as et:
                et.copy_tags(imagePath, undistImagePath)

def lensfunXML(model, *params):
    """
    Return XML in lensfun DB format for the specified distortion profile.
    """
    if model == 'ptlens':
        dist = 'model="ptlens" a="{a}" b="{b}" c="{c}"'.format(a=params[0], b=params[1], c=params[2])
    elif model == 'poly3':
        dist = 'model="poly3" k1="{k1}"'.format(k1=params[0])
    elif model == 'poly5':
        dist = 'model="poly5" k1="{k1}" k2="{k2}"'.format(k1=params[0], k2=params[1])
    else:
        raise ValueError
        
    return """
<lensdatabase>
    <mount>
        <name>Generic</name>
    </mount>
    <camera>
        <maker>CUSTOM</maker>
        <model>CUSTOM</model>
        <mount>Generic</mount>
        <cropfactor>1.0</cropfactor>
    </camera>
    <lens>
        <maker>CUSTOM</maker>
        <model>CUSTOM</model>
        <mount>Generic</mount>
        <cropfactor>1.0</cropfactor>
        <calibration>
            <distortion focal="1" {dist} />
        </calibration>
    </lens>
</lensdatabase>
    """.format(dist=dist)
             
def lensDistortionPixelDistances(imagePath=None, mod=None, lensfunDbObj=None, exiftoolObj=None, retH=False):
    """
    Return the difference between the distances from the image center to the distorted and undistorted
    pixel locations.
    
    See :func:`correctLensDistortion` for parameters.
    
    :rtype: ndarray of shape (h,w)
    """
    if not mod:
        mod,_,_ = getLensfunModifier(imagePath, lensfunDbObj, exiftoolObj)
        
    undistCoordsXY = mod.apply_geometry_distortion()

    height, width = undistCoordsXY.shape[0], undistCoordsXY.shape[1]

    y, x = np.mgrid[0:undistCoordsXY.shape[0], 0:undistCoordsXY.shape[1]]
    coordsXY = np.dstack((x,y))
    
    center = np.array([width/2, height/2])
    
    vectorsDist = (coordsXY - center).reshape(-1, 2)
    vectorsUndist = (undistCoordsXY - center).reshape(-1, 2)
    
    hDist = vectorLengths(vectorsDist).reshape(coordsXY.shape[0], coordsXY.shape[1])
    hUndist = vectorLengths(vectorsUndist).reshape(coordsXY.shape[0], coordsXY.shape[1])
    
    distance = hDist - hUndist
    
    if retH:
        return distance, hDist, hUndist
    else:
        return distance
