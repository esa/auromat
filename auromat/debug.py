# Copyright European Space Agency, 2013

"""
Some methods for easy debugging of parameters on the shell.
"""

from __future__ import division

import os
import sys
import tempfile
import subprocess

import auromat.mapping.spacecraft
import auromat.resample
import auromat.draw
from auromat.util import exiftool
from auromat.util.image import saveImage, loadImage
from auromat.solving.masking import maskStarfield
from auromat.draw import saveFig

def debugHorizon(imagePath, fitsWcsPath, tleFolder, photoTimeDelta=None, altitude=110):
    """
    Visualizes the horizon of the modelled earth and inflated earth (=aurora)
    for the given parameters.
    
    :param datetime.timedelta photoTimeDelta: amount to shift the image time (affects calculated
        camera position and therefore the geodetic coordinates)
    """    
    mapping = auromat.mapping.spacecraft.getMapping(imagePath, fitsWcsPath, photoTimeDelta,
                                                   tleFolder=tleFolder, altitude=altitude) 
    
    tmpPath = tempfile.mktemp(suffix='.jpg')
    saveFig(tmpPath, auromat.draw.drawHorizon(mapping))
    
    _openFile(tmpPath)
    
def debugPlot(imagePath, fitsWcsPath, tleFolder, photoTimeDelta=None, altitude=0, minElevation=10):
    """
    Plot the georeferenced image on a map. This allows to play with the image timestamp
    and mapping altitude.
    
    :param imagePath:
    :param fitsWcsPath:
    :param tleFolder:
    :param datetime.timedelta photoTimeDelta: amount to shift the image time (affects calculated
        camera position and therefore the geodetic coordinates)
    :param altitude: in km
    :param minElevation: in degrees
    """
     
    mapping = auromat.mapping.spacecraft.getMapping(imagePath, fitsWcsPath, photoTimeDelta, 
                                                   tleFolder=tleFolder, altitude=altitude)
    
    mapping = mapping.maskedByElevation(minElevation)
    mapping = auromat.resample.resample(mapping)
        
    tmpPath = tempfile.mktemp(suffix='.png')
    saveFig(tmpPath, auromat.draw.drawPlot(mapping))
    _openFile(tmpPath)    

def maskAllInFolder(folderPath, outputFolderPath, ext='jpg', maskFn=maskStarfield, preserveExif=True):
    """
    Masks the starfield of all images in `folderPath` and stores the masked images
    in `outputFolderPath`.
    
    :param str ext: image extension
    :param function maskFn: the masking function to use, 
        by default the standard algorithm from :mod:`auromat.solving.masking`
    :param bool preserveExif: whether to copy EXIF data (exiftool must be installed)
    """
    if preserveExif:
        et = exiftool.ExifTool()
        et.start()
    
    for imageFilename in os.listdir(folderPath):
        imagePath = os.path.join(folderPath, imageFilename)
        maskedPath = os.path.join(outputFolderPath, os.path.splitext(imageFilename)[0] + '.' + ext)
        mask, _ = maskFn(imagePath)
        im = loadImage(imagePath)
        im[~mask] = 0
        saveImage(maskedPath, im)
        
        if preserveExif:
            et.copy_tags(imagePath, maskedPath)
    
    if preserveExif:
        et.terminate()
    
def _openFile(path):
    if sys.platform.startswith('darwin'):
        subprocess.call(('open', path))
    elif os.name == 'nt':
        subprocess.call(('start', path), shell=True)
    elif os.name == 'posix':
        subprocess.call(('xdg-open', path))