# Copyright European Space Agency, 2013

"""
This module georeferences images taken from a spacecraft (e.g.
a camera onboard the ISS) and looking down at earth 
using the starfield in the images.

The resulting calibrations can be used with the corresponding
:mod:`auromat.mapping.spacecraft` module to compute geodetic coordinates
for the image pixels.
"""

from __future__ import division, absolute_import, print_function

from six.moves import zip
import os

import auromat.fits
from auromat.coordinates.spacetrack import Spacetrack
from auromat.coordinates.ephem import EphemerisCalculator
from auromat.util.image import readExifTime
from auromat.solving.masking import maskStarfield
import auromat.solving.solving
from auromat.util.os import makedirs

supportedImageFormats = ('.jpg', '.jpeg', '.png')

def solve(imagePath, wcsPath, tleFolder, spacetrackUser, spacetrackPassword, noradId, 
          debugOutputFolder=None, overwrite=False, maskingFn=maskStarfield, solveTimeout=60*5, 
          noAstrometryPlots=False, channel=None, sigma=None, 
          oddsToSolve=None, verbose=False):
    """
    Return True if the image could be georeferenced, otherwise False.
    
    :param wcsPath: path of output file containing calibration WCS headers
    :param tleFolder: folder where TLE files are/shall be stored
    :param spacetrackUser, spacetrackPassword: 
        Credentials to space-track.org used to download TLE data which in turn are
        used to calculte the spacecraft position at the time the image was taken.
        These can be None if the `tleFolder` contains relevant data already.
    :param noradId: NORAD ID of the spacecraft
    :param debugOutputFolder: folder to store debug images in from masking and astrometry
    :param overwrite: True overwrites an existing wcs file, False raises an exception
    :param maskingFn: the function to use for masking the starfield in the image
    :param solveTimeout: maximum time in seconds to spend for one astrometry run
                         (note that multiple parameter combinations are tried and each
                          counts as a run)
    :param noAstrometryPlots: whether to let astrometry.net produce star overlay plots
                              (only relevant when `debugOutputFolder` is given)
    :param channel: R,G,B, or None (average of all channel), the channel to use for masking
                    and astrometry
    :param oddsToSolve: default 1e9, see astrometry.net docs
    :return: True if astrometry was successful (in which case a file at `wcsPath` was written),
             otherwise False.
    :rtype: bool
    """
    wcsHeaders = _solveImages([imagePath], tleFolder, spacetrackUser, spacetrackPassword, noradId, 
                              maskingFn, debugOutputFolder, solveTimeout=solveTimeout,
                              noAstrometryPlots=noAstrometryPlots,
                              channel=channel, sigma=sigma, oddsToSolve=oddsToSolve, verbose=verbose)
    wcsHeader = next(wcsHeaders)
    if wcsHeader is None:
        return False
    auromat.fits.writeHeader(wcsPath, wcsHeader, overwrite=overwrite)
    return True

def solveSequence(imageSequenceFolder, wcsFolder, tleFolder, spacetrackUser, spacetrackPassword, noradId, 
                  debugOutputFolder=None, parallel=True, maxWorkers=None, 
                  solveTimeout=60*5, maskingFn=maskStarfield, oddsToSolve=None,
                  noAstrometryPlots=False, channel=None, sigma=None, verbose=False):
    """
    Returns a generator containing (imagePath, wcsPath) tuples of successfully solved images.
    This allows to execute actions directly after each solve.
    Images which are already solved are ignored.
    
    See :func:`solve` for parameters.
    """    
    fileNames = os.listdir(imageSequenceFolder)
    imageFileNames = sorted(filter(lambda f: f.endswith(supportedImageFormats), fileNames))
    imagePaths = map(lambda f: os.path.join(imageSequenceFolder, f), imageFileNames)
    return solveImages(imagePaths, wcsFolder, tleFolder, spacetrackUser, spacetrackPassword, noradId, 
                       debugOutputFolder, parallel=parallel, maxWorkers=maxWorkers,
                       solveTimeout=solveTimeout,
                       maskingFn=maskingFn, noAstrometryPlots=noAstrometryPlots,
                       oddsToSolve=oddsToSolve,
                       channel=channel, sigma=sigma, verbose=verbose)
    
def solveImages(imagePaths, wcsFolder, tleFolder, spacetrackUser, spacetrackPassword, noradId, 
                debugOutputFolder=None, parallel=True, maxWorkers=None, 
                solveTimeout=60*5, maskingFn=maskStarfield,
                returnUnsolved=False, noAstrometryPlots=False, oddsToSolve=None,
                channel=None, sigma=None, verbose=False):
    """
    Returns a generator containing (imagePath, wcsPath) tuples of successfully solved images.
    This allows to execute actions directly after each solve.
    Images which are already solved are ignored.
    
    See :func:`solve` for parameters.
    """
    makedirs(wcsFolder)
    unsolvedImagePaths = []
    unsolvedWcsPaths = []
    for imagePath in imagePaths:
        fileName = os.path.basename(imagePath)
        fileBase = os.path.splitext(fileName)[0]
        wcsPath = os.path.join(wcsFolder, fileBase + '.wcs')
        if not os.path.exists(wcsPath):
            unsolvedImagePaths.append(imagePath)
            unsolvedWcsPaths.append(wcsPath)
        
    wcsHeaders = _solveImages(unsolvedImagePaths, 
                              tleFolder, spacetrackUser, spacetrackPassword, 
                              noradId, maskingFn,
                              debugOutputFolder=debugOutputFolder, 
                              parallel=parallel, maxWorkers=maxWorkers, 
                              solveTimeout=solveTimeout,
                              noAstrometryPlots=noAstrometryPlots, oddsToSolve=oddsToSolve,
                              channel=channel, sigma=sigma, verbose=verbose)
    
    for (imagePath, wcsPath, wcsHeader) in zip(unsolvedImagePaths, unsolvedWcsPaths, wcsHeaders):
        if wcsHeader is not None:
            auromat.fits.writeHeader(wcsPath, wcsHeader)
            yield (imagePath, wcsPath)
        elif returnUnsolved:
            yield (imagePath, None)

def _solveImages(imagePaths, tleFolder, spacetrackUser, spacetrackPassword, noradId, maskingFn, 
                 debugOutputFolder=None, parallel=True, solveTimeout=60*5, maxWorkers=None, 
                 noAstrometryPlots=False, channel=None, sigma=None, oddsToSolve=None, 
                 verbose=False):
    """
    Returns a generator containing the wcsHeader (or None) for each image.
    """    
    noradId = int(noradId)
            
    originalPhotoTimes = list(map(readExifTime, imagePaths))
    latestPhotoTime = max(originalPhotoTimes)
    
    tleFilePath = os.path.join(tleFolder, str(noradId) + '.tle')
    spacetrack = Spacetrack(spacetrackUser, spacetrackPassword)
    spacetrack.updateTLEsFor(noradId, tleFilePath, latestPhotoTime)

    ephemCalculator = EphemerisCalculator(tleFilePath)    
    cameraPositionsGCRS = map(ephemCalculator, originalPhotoTimes)
    
    wcsHeaders = auromat.solving.solving.solveImages(imagePaths, channel=channel,
                                                     maskingFn=maskingFn,
                                                     solveTimeout=solveTimeout, 
                                                     debugOutputFolder=debugOutputFolder, 
                                                     parallel=parallel, maxWorkers=maxWorkers,
                                                     noAstrometryPlots=noAstrometryPlots,
                                                     sigma=sigma, verbose=verbose,
                                                     oddsToSolve=oddsToSolve)
    
    for (wcsHeader, originalPhotoTime, cameraPosGCRS) in zip(wcsHeaders, originalPhotoTimes, cameraPositionsGCRS):
        if wcsHeader is not None:
            auromat.fits.setNoradId(wcsHeader, noradId)
            auromat.fits.setSpacecraftPosition(wcsHeader, cameraPosGCRS, originalPhotoTime)
        yield wcsHeader

    