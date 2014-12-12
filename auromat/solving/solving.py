# Copyright European Space Agency, 2013

from __future__ import division, print_function, absolute_import

from six.moves import map, _thread
import warnings
import time
import os
import errno
import shutil
import tempfile
import psutil
import futures
import subprocess
import multiprocessing
from functools import partial

from astropy import units as u

import auromat.fits
from auromat.solving.masking import maskStarfield
from auromat.util.image import readFocalLength35mm, loadImage, saveImage
from auromat.util.os import makedirs
import sys

from distutils.version import LooseVersion
from threading import Thread
try:
    import pyfits; 'OPTIONAL'
    PYFITS_TOO_OLD = LooseVersion(pyfits.__version__) < '3.1.0'
except:
    # if unavailable
    PYFITS_TOO_OLD = True

"""
A general purpose astrometry solving module. It has functionality which
allows to solve images that not only contain stars but also other elements.
As it was developed for solving images taken from the ISS towards the earth
(meaning that images contain parts of the earth and may contain spacecraft structures)
it may be biased towards it, yet it very likely is suitable for other kinds
of images as well. 
"""

def solveImages(imagePaths, channel=None, maskingFn=maskStarfield, solveTimeout=60*5, parallel=True, 
                debugOutputFolder=None, noAstrometryPlots=False, pixelError=10, oddsToSolve=None, 
                sigma=None, maxWorkers=None,
                astrometryBinPath=None, useModifiedPath=False, verbose=False):
    """
    Solves multiple images in parallel. See :func:`solveImage` for parameter documentation.
    
    Note: This only works because the actual solving is done in an external
          process (astrometry.net), otherwise nothing would have been gained
          due to Pythons GIL.
    """
    if maxWorkers is None:
        maxWorkers = multiprocessing.cpu_count()
    if parallel and len(imagePaths) > 1:
        workerCount = min(multiprocessing.cpu_count(), maxWorkers)
        
        with futures.ThreadPoolExecutor(max_workers=workerCount) as executor:
            for wcsHeader in executor.map(lambda imagePath: solveImage(imagePath, channel=channel,
                                                                       maskingFn=maskingFn, 
                                                                       solveTimeout=solveTimeout, 
                                                                       debugOutputFolder=debugOutputFolder,
                                                                       noAstrometryPlots=noAstrometryPlots,
                                                                       astrometryBinPath=astrometryBinPath,
                                                                       useModifiedPath=useModifiedPath,
                                                                       pixelError=pixelError,
                                                                       oddsToSolve=oddsToSolve,
                                                                       sigma=sigma,
                                                                       verbose=verbose), 
                                          imagePaths):
                yield wcsHeader
    else:
        for wcsHeader in map(lambda imagePath: solveImage(imagePath, channel=channel,
                                                          maskingFn=maskingFn, 
                                                          solveTimeout=solveTimeout, 
                                                          debugOutputFolder=debugOutputFolder,
                                                          noAstrometryPlots=noAstrometryPlots,
                                                          astrometryBinPath=astrometryBinPath,
                                                          useModifiedPath=useModifiedPath,
                                                          pixelError=pixelError,
                                                          oddsToSolve=oddsToSolve,
                                                          sigma=sigma,
                                                          verbose=verbose), 
                             imagePaths):
            yield wcsHeader

def solveImage(imagePath, channel=None, maskingFn=maskStarfield, sigma=None,
               solveTimeout=60*5, debugOutputFolder=None, noAstrometryPlots=False,
               arcsecRange=None, astrometryBinPath=None, useModifiedPath=False,
               parameters=['xy', 'xy2', 'xy4', 's'], pixelError=10, oddsToSolve=None,
               verbose=False):
    """
    Tries different combinations to solve the image using astrometry.net.
    
    :param imagePath:
    :param channel: the channel to use for source extraction
        'R','G','B', or None for combining all channels into a grayscale image
    :param maskingFn: function to use for masking the given image,
                      if None, masking is skipped
    :param sigma: noise level of the image (optional)
    :param solveTimeout: maximum time in seconds after which astrometry.net is killed
    :param debugOutputFolder: if given, the path to which debug files are written
    :param noAstrometryPlots: whether to let astrometry.net generate plots,
                              if True, then debugOutputFolder must be given 
    :param arcsecRange: tuple(low,high), if not given, then it is guessed from
                        the image file if possible
    :param astrometryBinPath: path to the bin/ folder of astrometry.net;
                              if not given, then whatever is in PATH will be used
    :param bool useModifiedPath: invokes astrometry.net with /usr/bin/env PATH=os.environ['PATH']
                                 This may be useful when the PATH was modified after launching Python.
    :param int pixelError: size of pixel positional error, use higher values (e.g. 10)
                           if image contains star trails (ISS images)
    :param oddsToSolve: default 1e9, see astrometry.net docs
    :rtype: dictionary containing FITS WCS header, or None if solving failed
    """
    imageBase = os.path.splitext(os.path.basename(imagePath))[0]

    tmpDir = tempfile.mkdtemp()
    tmpDirAstrometry = os.path.join(tmpDir, 'astrometry')
    
    # TODO we need to return which solving strategy (downsampling, extractor) was used
    #      -> this should probably be added to the .axy header
    #      as astrometry.net repeatedly downsamples in certain cases it
    #      is hard to determine the correct downsampling that was used
    #      -> see also http://trac.astrometry.net/ticket/1117
        
    if maskingFn is None:
        imageMaskedPath = imagePath
        maskedSuffix = ''
        imageSize = None
        if channel is not None:
            raise NotImplementedError
    else:
        maskedSuffix = '_masked'
        imageMaskedPath = os.path.join(tmpDir, imageBase + maskedSuffix + ".png")
        
        # step 1: create starfield-masked image
        t0 = time.time()
        
        if not debugOutputFolder:
            debugPathPrefix = None
        else:
            debugPathPrefix=os.path.join(debugOutputFolder, imageBase) + '_'
        mask, sigma_ = maskingFn(imagePath, 
                                 debugPathPrefix=debugPathPrefix)
        if sigma is None:
            sigma = sigma_
        im = loadImage(imagePath)
        im[~mask] = 0
        if channel is None:
            pass
        elif channel.lower() == 'r':
            im = im[:,:,0]
        elif channel.lower() == 'g':
            im = im[:,:,1]
        elif channel.lower() == 'b':
            im = im[:,:,2]
        else:
            raise ValueError('channel is "{}" but must be R,G,B or None'.format(channel))
        saveImage(imageMaskedPath, im)
        imageSize = (im.shape[0], im.shape[1])
        
        print('masking:', time.time()-t0, 's')       
        
        if debugOutputFolder is not None:
            shutil.copy(imageMaskedPath, debugOutputFolder)
               
    # step 2: invoke astrometry.net
        
    # Note that astrometry.net assumes that images are taken from earth.
    # As the ISS is very near to earth, the error is most likely below the
    # accuracy that astrometry.net provides and also below the common pixel
    # resolution of images and can therefore be ignored.
    keepTempFiles = False if debugOutputFolder is None else True
    
    if debugOutputFolder is None:
        noAstrometryPlots = True
        
    logFilename = imageBase + maskedSuffix + '.log'
    logFilenameOutput = imageBase + '.log'
    logPath = os.path.join(tmpDirAstrometry, logFilename)
    logBaseOutput = imageBase + '_'
    
    objsBase = imageBase + maskedSuffix + '_objs'
    objsBaseOutput = imageBase + '_objs'
    objsPath = os.path.join(tmpDirAstrometry, objsBase + '.png')
    
    indxBase = imageBase + maskedSuffix + '_indx'
    indxBaseOutput = imageBase + '_indx'
    indxFilename = indxBase + '.png'
    indxFilenameOutput = indxBaseOutput + '.png'
    indxPath = os.path.join(tmpDirAstrometry, indxFilename)
    
    matchFilename = imageBase + maskedSuffix + '.match'
    matchFilenameOutput = imageBase + '.match'
    matchPath = os.path.join(tmpDirAstrometry, matchFilename)
    
    indxXyFilename = imageBase + maskedSuffix + '.xyls'
    indxXyFilenameOutput = imageBase + '.xyls'
    indxXyPath = os.path.join(tmpDirAstrometry, indxXyFilename)
    
    axyFilename = imageBase + maskedSuffix + '.axy'
    axyFilenameOutput = imageBase + '.axy'
    axyPath = os.path.join(tmpDirAstrometry, axyFilename)
    axyBaseOutput = imageBase + '_'

    corrFilename = imageBase + maskedSuffix + '.corr'
    corrFilenameOutput = imageBase + '.corr'
    corrPath = os.path.join(tmpDirAstrometry, corrFilename)
    
    # remove old debug files first
    if keepTempFiles:
        for filename in [matchFilenameOutput, indxFilenameOutput,
                         indxXyFilenameOutput, corrFilenameOutput, axyFilenameOutput, logFilenameOutput,
                         objsBaseOutput + '_xy2.jpg', objsBaseOutput + '_xy4.jpg', objsBaseOutput + '_xy.jpg',
                         objsBaseOutput + '_s.jpg',
                         axyBaseOutput + 'xy2.axy', axyBaseOutput + 'xy4.axy', axyBaseOutput + 'xy.axy',
                         axyBaseOutput + 's.axy',
                         logBaseOutput + 'xy2.log', logBaseOutput + 'xy4.log', logBaseOutput + 'xy.log',
                         logBaseOutput + 's.log']:
            path = os.path.join(debugOutputFolder, filename)
            if os.path.exists(path):
                os.remove(path)
    
    t0 = time.time()
    
    def _copyTempFiles(fitsWcsHeader, name):
        if not keepTempFiles:
            return
        if os.path.exists(objsPath):
            p = os.path.join(debugOutputFolder, objsBaseOutput + '_' + name + '.jpg')
            saveImage(p, loadImage(objsPath)) # convert png to jpg
            
        if not fitsWcsHeader and os.path.exists(axyPath):
            p = os.path.join(debugOutputFolder, axyBaseOutput + name + '.axy')
            shutil.copy(axyPath, p)
            
        if not fitsWcsHeader and os.path.exists(logPath):
            p = os.path.join(debugOutputFolder, logBaseOutput + name + '.log')
            shutil.copy(logPath, p)
    
    # first try astrometry.net's own star extraction (simplexy) with different downsampling options
    
    if arcsecRange:
        if len(arcsecRange) != 2:
            raise ValueError('arcsecRange must be a pair (low,high)')
        arcsecLowHigh = arcsecRange
    else:
        arcsecLowHigh = estimateArcSecRange(imagePath, imageSize)
    _solve = partial(_solveStarfield, imageMaskedPath, tmpDirAstrometry,
                     keepTempFiles=keepTempFiles,
                     timeout=solveTimeout,
                     sigma=sigma,
                     plotsBgImagePath=imagePath,
                     noPlots=noAstrometryPlots,
                     arcsecPerPxLowHigh=arcsecLowHigh,
                     pixelError=pixelError,
                     oddsToSolve=oddsToSolve,
                     astrometryBinPath=astrometryBinPath,
                     useModifiedPath=useModifiedPath,
                     verbose=verbose)
    
    # "The downsampling just seems to generally do a better job of source detection on
    #  most of the images we get, at least at about 2-4. I think it has to do with either
    #  the PSF size (default is too narrow) or saturation (downsampling smears out 
    #  saturated pixels, making them not so saturated)." (Dustin Lang)
    # (https://groups.google.com/d/msg/astrometry/Pp_MZD6s4w8/muuH-1T_zpAJ)
    fitsWcsHeader = None
    if 'xy2' in parameters:
        fitsWcsHeader = _solve(useSextractor=False, downsample=2)    
        _copyTempFiles(fitsWcsHeader, 'xy2')
    
    # Astrometry.net might have already tried downsample=4 when downsample=2 was requested
    # in case no stars could be extracted, so it might happen that we repeat this below.
    # We have to try it anyway because with downsample=2 it could have happened that
    # only 1 or 2 stars in the corners were detected after which astrometry didn't repeat 
    # downsampling and just failed.
    # There should be an astrometry flag to disable the repeated downsampling,
    # see also https://groups.google.com/d/msg/astrometry/qNOgWTL1pVA/BNIbqkXEM-gJ
    # Note that with repeated downsampling, astrometry re-uses the sigma (noise level)
    # from the original downsampling, so there may be slight differences if
    # astrometry is manually called with downsample=4 because it recomputes sigma
    # based on downsample=4. 
    # NOTE: this is the behaviour of the -D flag of image2xy, which cannot be set from solve-field! 
    
    if fitsWcsHeader is None and 'xy4' in parameters:
        if keepTempFiles and os.path.exists(tmpDirAstrometry):
            shutil.rmtree(tmpDirAstrometry)
        fitsWcsHeader = _solve(useSextractor=False, downsample=4)          
        _copyTempFiles(fitsWcsHeader, 'xy4')
                    
    # if this didn't work, we try SExtractor for star extraction
    
    if fitsWcsHeader is None and 's' in parameters:
        if keepTempFiles and os.path.exists(tmpDirAstrometry):
            shutil.rmtree(tmpDirAstrometry)
        fitsWcsHeader = _solve(useSextractor=True, downsample=None)  
        _copyTempFiles(fitsWcsHeader, 's')
            
    # in case the input image has a low resolution (which is not the case for ISS images)
    # we also might have luck with downsampling disabled and using simplexy:
    
    if fitsWcsHeader is None and 'xy' in parameters:
        if keepTempFiles and os.path.exists(tmpDirAstrometry):
            shutil.rmtree(tmpDirAstrometry)
        fitsWcsHeader = _solve(useSextractor=False, downsample=None)        
        _copyTempFiles(fitsWcsHeader, 'xy')
        
    print('solving:', time.time()-t0, 's')
    
    if fitsWcsHeader is not None:
        if keepTempFiles and os.path.exists(indxPath):
            p = os.path.join(debugOutputFolder, indxBaseOutput + '.jpg')
            saveImage(p, loadImage(indxPath)) # convert png to jpg
        
        tempPaths = [matchPath, indxXyPath, axyPath, corrPath, logPath]
        outputFilenames = [matchFilenameOutput, indxXyFilenameOutput, axyFilenameOutput, 
                           corrFilenameOutput, logFilenameOutput]
        if keepTempFiles:
            outputPaths = [os.path.join(debugOutputFolder,f) for f in outputFilenames]
            for tempPath, outputPath in zip(tempPaths, outputPaths):    
                if os.path.exists(tempPath):
                    shutil.move(tempPath, outputPath)
                else:
                    print('WARNING!! {} does not exist after successfully solving, but it should!'.format(tempPath), file=sys.stderr)
                        
    shutil.rmtree(tmpDir)
        
    return fitsWcsHeader

def estimateArcSecRange(imagePath, imageSize=None):
    focal35mm = readFocalLength35mm(imagePath)
    if focal35mm is None:
        return None
    focal35mmLow = focal35mm*0.9
    focal35mmHigh = focal35mm*1.1
    if imageSize is None:
        _,w, = loadImage(imagePath).shape
    else:
        _,w = imageSize
    # see http://cbellh47.blogspot.nl/2010/01/astrometry-101-pixel-scale.html
    pixelSizeMm = 35/w
    arcsecLow = (pixelSizeMm/focal35mmHigh * u.rad).to(u.arcsec).value
    arcsecHigh = (pixelSizeMm/focal35mmLow * u.rad).to(u.arcsec).value
    return (arcsecLow, arcsecHigh)

def _solveStarfield(imagePath, tmpDir=None, keepTempFiles=False, timeout=60*1, 
                    useSextractor=True, downsample=2, sigma=None, 
                    searchField=None, arcsecPerPxLowHigh=None, 
                    pixelError=10, oddsToSolve=None,
                    plotsBgImagePath=None, noPlots=False,
                    astrometryBinPath=None, useModifiedPath=False, verbose=False):
    """
    NOTE: The astrometry/bin folder must be in the PATH. 
    
    :param imagePath:
    :param timeout: time in seconds after which the solving process is terminated
                    If solving fails, increase this. astrometry.net tries to solve
                    using bright stars first, and if these are not in the center then
                    the distortion may be too high and only fainter stars might lead
                    to a successful solve, which in turn needs longer processing time.
    :param useSextractor: sextractor sometimes delivers better results than
                          the built-in star extraction of astrometry.net (image2xy)
    :param int|None downsample: Whether and how much astrometry should downsample the image before solving
    :param sigma: noise level override
    :param (ra,dec,radius)|None searchField: search only within 'radius' of the field center 'ra','dec', all in degrees
    :param str plotsBgImagePath: path to .jpg file to use as background for all plots
    :param tuple arcsecPerPxLowHigh: tuple of lower and upper arcsecPerPx to restrict search
    :param int pixelError: size of pixel positional error, use higher values (e.g. 10)
                           if image contains star trails (ISS images)
    :param oddsToSolve: default 1e9, see astrometry.net docs
    :param bool useModifiedPath: invokes astrometry.net with /usr/bin/env PATH=os.environ['PATH']
                                 This may be useful when the PATH was modified after launching Python, Unix only.
    :rtype: wcs header, or None if no solution was found
    """
    # adapt these constants if necessary for newer astrometry.net versions
    solvefieldName = 'solve-field'
    backendName = 'astrometry-engine'
    
    if tmpDir is None and keepTempFiles is True:
        print("solveStarfield: tmpDir is not set but keepTempFiles is true, this doesn't make much sense")
    
    if tmpDir is None:
        tmpDir = tempfile.mkdtemp()
    tmpTmpDir = os.path.join(tmpDir, "tmp")
    makedirs(tmpTmpDir)
        
    imageBase = os.path.splitext(os.path.basename(imagePath))[0]
        
    solvedPath = os.path.join(tmpDir, imageBase + ".solved")
    wcsPath = os.path.join(tmpDir, imageBase + ".wcs")
    matchPath = os.path.join(tmpDir, imageBase + ".match")
    indxXyPath = os.path.join(tmpDir, imageBase + ".xyls")
    corrPath = os.path.join(tmpDir, imageBase + ".corr")
    logPath = os.path.join(tmpDir, imageBase + ".log")
    
    if not astrometryBinPath:
        astrometryBinPath = ''
    
    args = [os.path.join(astrometryBinPath, solvefieldName)]
    
    if useModifiedPath:
        args = ['/usr/bin/env', 'PATH=' + os.environ['PATH']] + args
    
    args += ["--cpulimit", str(timeout)] # see https://github.com/dstndstn/astrometry.net/issues/6
    args += ["--dir", tmpDir, "--temp-dir", tmpTmpDir, "--no-delete-temp"]
    # TODO there are no params for the plot filenames (..-indx.png and ..-objs.png)
    #      and also not for .axy
    args += ["--wcs", wcsPath, "--solved", solvedPath, "--match", matchPath]
    args += ["--index-xyls", indxXyPath, "--corr", corrPath]
    args += ["--crpix-center"]
    args += ["--no-background-subtraction"]
    
    if PYFITS_TOO_OLD:
        args += ["--no-remove-lines"]
        args += ["--no-fits2fits"]

    if arcsecPerPxLowHigh is not None:
        arcsecLow, arcsecHigh = arcsecPerPxLowHigh
        args += ["--scale-low", str(arcsecLow), "--scale-high", str(arcsecHigh), "--scale-units", "arcsecperpix"]
    
    args += ["--no-tweak"] # no SIP polynomial; we correct lens distortion before-hand
    args += ["--pixel-error", str(pixelError)]
    
    if oddsToSolve:
        args += ["--odds-to-solve", str(oddsToSolve)]
    
    if verbose:
        args += ["--verbose"]
    
    if sigma:
        args += ["--sigma", str(sigma)]
    
    if searchField:
        ra,dec,radius = searchField
        args += ["--ra", str(ra), "--dec", str(dec), "--radius", str(radius)]
    
    if downsample:
        args += ["--downsample", str(downsample)]
    
    if useSextractor:
        args += ["--use-sextractor"]

    if plotsBgImagePath:
        args += ["--plot-bg", plotsBgImagePath]
        
    if keepTempFiles:
        print('astrometry files in', tmpDir)
        if noPlots:
            args += ["--no-plots"]
    else:
        args += ["--no-plots"]
        args += ["--new-fits", "none", "--index-xyls", "none", "--rdls", "none", "--corr", "none"]
        
    args += [imagePath]

    print(' '.join(args))
    
    def print_and_store_output(out, logPath):
        # work-around the fact that using sys.stdout in subprocess breaks
        # if sys.stdout got redirected to a non-file object (e.g. StringIO)
        with open(logPath, 'w') as logfile:
            logfile.write(' '.join(args) + '\n')
            for line in iter(out.readline, b''):
                print(line, end='')
                logfile.write(line)
            out.close()
    
    try:
        process = psutil.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        t = Thread(target=print_and_store_output, args=(process.stdout, logPath))
        t.daemon = True # thread dies with the program, should not be needed if no error occurs
        t.start()
        
    except OSError as e:
        if e.errno == errno.ENOENT:
            raise RuntimeError('The "' + solvefieldName + '" program from astrometry.net could not be launched. ' + \
                               'Make sure it is in the PATH!')
        else:
            raise
     
    try:
        process.wait(timeout+10)
    except: # psutil.TimeoutExpired or KeyboardInterrupt (Ctrl+C)
        print('astrometry.net timeout reached, killing processes now')
        
        backend = list(filter(lambda p: p.name() == backendName, process.get_children(True)))
        processes = [process] + backend
        
        for p in processes:
            print('terminating {} (pid {})'.format(p.name(), p.pid))
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
            
        # if astrometry is just writing its solution, let it have some time to do so
        t = 5
        _, alive = psutil.wait_procs(processes, t)
        if alive:
            # This shouldn't happen.
            warnings.warn('solve-field or backend did NOT exit in ' + t + 's after SIGTERM was sent!' +
                          '...killing them now and ignoring results')
            for p in processes:
                print('killing {} (pid {})'.format(p.name(), p.pid))
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            if not keepTempFiles:
                shutil.rmtree(tmpDir)
            return None

    # FIXME astrometry seems to leave files in an inconsistent state 
    #       if it is terminated while persisting results
    #       -> e.g. it seems that the .solved file is written first, and then the rest
    #  -> quick work-around: check for .solved AND .wcs, and catch wcs reading exceptions
    #     may still fail in the end, but the probability is lower
    #  in theory, astrometry should handle SIGTERM accordingly instead of just exiting immediately
       
    try:
        if not os.path.exists(solvedPath) or not os.path.exists(wcsPath):
            return None
        fitsWcsHeader = auromat.fits.readHeader(wcsPath)
    except Exception as e:
        warnings.warn('error reading wcs file ' + wcsPath)
        print(repr(e))
        return None
    finally:
        if not keepTempFiles:
            shutil.rmtree(tmpDir)
        
    return fitsWcsHeader

    