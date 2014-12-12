# Copyright European Space Agency, 2013

"""
The module contains algorithms to automatically mask those image areas
which are most likely containing the starfield.

It does so by using a combination of constant thresholding, adaptive thresholding,
line detection and contour detection. It assumes that the starfield has a more or
less constant background brightness. It was developed and tested with ISS aurora
images, so it may be biased towards those.
"""

from __future__ import division, print_function, absolute_import

import os
import traceback
from collections import namedtuple

from math import pi
import numpy as np
from scipy.signal import convolve2d
import cv2 as cv

try:
    IMWRITE_PNG_COMPRESSION = cv.IMWRITE_PNG_COMPRESSION
    IMWRITE_JPEG_QUALITY = cv.IMWRITE_JPEG_QUALITY
except AttributeError:
    IMWRITE_PNG_COMPRESSION = cv.cv.CV_IMWRITE_PNG_COMPRESSION
    IMWRITE_JPEG_QUALITY = cv.cv.CV_IMWRITE_JPEG_QUALITY

from auromat.draw import drawHistogram, saveFig

from auromat.solving.noiseestimation import _estimateNoiseLevel as _doEstimateNoiseLevel
from auromat.solving.viewasblocks import view_as_blocks
  

# Paradigm: The less false stars, the higher the solving probability and
#           the higher the accuracy.
# When there are less false stars then it's much easier for astrometry.net to tweak
# a solution (=align the stars with other stars of the image except the already matched
# stars from the quad/triangle). 

def maskStarfieldRect(imagePath, topLeft, bottomRight, debugPathPrefix=None):
    """
    Mask an image using the given coordinates representating a rectangle.
    
    :param imagePath:
    :param topLeft: (x,y) pair of top left rectangle pixel
    :param bottomRight: (x,y) pair of bottom right rectangle pixel
    :rtype: tuple (mask, sigma)
    """
    if isinstance(imagePath, np.ndarray):
        im = imagePath
    else:
        im = cv.imread(imagePath)
    h,w = im.shape[0], im.shape[1]
    
    x1,y1 = topLeft
    x2,y2 = bottomRight
    
    mask = np.zeros((h,w), np.bool)
    mask[y1:y2+1,x1:x2+1] = True
    
    sigma = _estimateNoiseLevel(im[y1:y2+1,x1:x2+1,0])
    
    return mask, sigma

def _binarizeStarfieldImage(imgray, fudge=20):
    """
    
    :param imgray: grayscale image as returned by cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    :rtype: binary image where stars are retained
    """
    maxThreshold = 150
    
    # Find the average background brightness of the starfield.
    # In most cases this corresponds to the first spike of the image histogram,
    # assuming that the starfield background is the darkest part of the image.
    # The threshold for binarization is then the average plus an empirical value.
    hist = cv.calcHist([imgray],[0],None,[256],[0,255]).reshape(256)
    # Sometimes there are small bumps before the first main spike.
    # To prevent choosing the wrong spike, the histogram is slightly smoothed.
    hist[1:-1] = (hist[:-2] + hist[1:-1] + hist[2:]) / 3 # smoothing with window=3
    histDiff = hist[1:] - hist[:-1]
    firstSpike = np.argmax(histDiff<0)
    threshold = min(firstSpike + fudge, maxThreshold)
        
    _,binary = cv.threshold(imgray, threshold, 255, cv.THRESH_BINARY)
    return binary, hist, threshold, firstSpike
    
def _findAndCategorizeContours(binary):
    # see auromat.utils.outline_opencv for why we need to pad the image
    imCv = np.zeros((binary.shape[0]+2, binary.shape[1]+2), dtype=np.uint8)
    imCv[1:-1,1:-1] = binary
    contours,_ = cv.findContours(imCv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)    
    contours = np.asarray(contours)
    contours -= 1

    area = np.asarray([cv.contourArea(c) for c in contours])
    rectAxes = np.asarray([cv.minAreaRect(c)[1] for c in contours])

    # isBigContour needs to be big enough to not discard bigger stars!
    # this could leave some spacecraft structures intact which may or may not confuse astrometry    
    # TODO the ratio below should depend on the estimated celestial pixel scale
    #      and the exposure time (longer exposure = longer star trails) 
    bigContourAreaRatio = 0.000013 # 0.0013% of the image area (~160 pixels for 12MP images)
    bigContourArea = bigContourAreaRatio*(binary.shape[0]*binary.shape[1])
    isBigContour = area > int(bigContourArea)
    isSmallContour = ~isBigContour
    
    longRatioThreshold = 5
    with np.errstate(divide='ignore', invalid='ignore'): # division produces nans and infs
        rectRatio = rectAxes[:,0]/rectAxes[:,1] if len(contours) > 0 else np.array([])

    with np.errstate(invalid='ignore'):
        isLongContour = np.logical_and(area > 20, # exclude very tiny long contours (could be stars) and inf ratios
                                       np.logical_or(rectRatio > longRatioThreshold, 
                                                     rectRatio < 1/longRatioThreshold)
                                       )
    isSmallLongContour = np.logical_and(isSmallContour, isLongContour)
    isSmallShortContour = np.logical_and(isSmallContour, ~isLongContour)
    
    return contours, area, isBigContour, isSmallLongContour, isSmallShortContour

def _getBlockShape(im):
    # assumes landscape images
    
    # roughly square blocks
    blocksX = 16
    blocksY = 12
    
    if im.shape[0] % blocksY != 0:
        # if 12 doesn't work, we use 8, assuming that height is divisible by 16
        blocksY = 8
        
    if im.shape[0] % blocksY != 0 or im.shape[1] % blocksX != 0:
        raise NotImplementedError('(width, height) of image must be divisible by (' +
                                  str(blocksX) + ',' + str(blocksY) + ') for block masking, ' +
                                  str(im.shape[1]) + 'x' + str(im.shape[0]))
    
    blockH = im.shape[0]//blocksY
    blockW = im.shape[1]//blocksX    
    return (blockH, blockW)

def _createStarfieldMask(im, contours, area, isBigContour, isSmallLongContour, 
                         blackenLowerPart = True):
    mask = np.ones((im.shape[0], im.shape[1]), np.bool)
    
    blockH, blockW = _getBlockShape(im)
    
    if blackenLowerPart:
        # 1. find biggest contour
        # 2. check if the top end is within the bottom two thirds and the lower end
        #    within the bottom half of the image
        # 3. if 2 is satisfied, paint everything black from the top of the contour to the bottom of the image
        #    Note: the assumption is that such a big chunk is likely be a part of the earth
        # 4. if 2 is not satisfied, paint the lower half of the image black (fallback)
        # This procedure helps in case the image is very dark and faint citylights would be recognized as stars.
        biggestContour = contours[np.argmax(area)]
        _,y,_,h = cv.boundingRect(biggestContour)
        if y > im.shape[0]/3 and y+h > im.shape[0]/2:
            fromy = y
        else:
            fromy = im.shape[0]//2
        
        # cut off below the corresponding block
        fromyBlockBoundary = int(np.ceil(fromy/blockH)*blockH)
        mask[fromyBlockBoundary:] = False


    # devide image into blocks and paint those blocks black which contain big or (small and long)
    # contours, as these are very likely spacecraft structures
    assert isBigContour is not None or isSmallLongContour is not None
    if isSmallLongContour is None:
        isOffendingContour = isBigContour
    else:
        isOffendingContour = np.logical_or(isBigContour, isSmallLongContour)
    
    # draw and fill all offending contours on a blank binary image
    # then for each block check if there are pixels which are black
    imFilledOffenders = np.zeros(mask.shape, np.uint8)
    cv.fillPoly(imFilledOffenders, contours[isOffendingContour], 255)

    blockViewMask = view_as_blocks(mask, (blockH,blockW))
    blockViewOffenders = view_as_blocks(imFilledOffenders, (blockH,blockW))

    isBlockContainingOffenders = (blockViewOffenders==255).any(axis=-1).any(axis=-1)
    blockViewMask[isBlockContainingOffenders] = False

    return mask

def _masked_adaptive_threshold(image,mask,max_value,size,C):
    """
    thresholds only using the unmasked pixels
    Note: becomes slightly inaccurate on very dark areas
    Note: image needs to be black at masked pixels
    
    adapted from http://stackoverflow.com/a/10551103
    """
    mask = mask.astype(np.uint8) * 255
    conv = cv.blur(image, (size, size)).astype(float)
    number_neighbours = cv.blur(mask, (size, size)).astype(float)
    with np.errstate(invalid='ignore'):
        image = image-255*(conv/number_neighbours)    
    binary = np.zeros_like(image, dtype=np.uint8)
    binary[np.logical_and(image > -C, mask)] = max_value
    return binary

def maskStarfield(imagePath, channel=None, blackenLowerPart=True, ignoreVeryDark=True,
                  debugPathPrefix=None, debugJpegQuality=80):
    """
    Automatic masking of the starfield in the given image using
    a combination of image processing and object detection steps.
    
    :param str channel: the channel to use for analysis
        'R','G','B', or None for combining all channels into a grayscale image
    :param bool blackenLowerPart: If the earth is in the lower part of the image
                             then this should be set to True as it will
                             broadly mask parts of the earth not detected otherwise.
    :param bool ignoreVeryDark: If True, then areas (block) which are almost totally black
                           are not considered as starfield. This is sometimes useful
                           to ignore very dark spacecraft structures which would later
                           on be detected as stars.
    :param str debugPathPrefix: if given, the folder in which to store debug images
                                illustrating the different processing stages
    :param int debugJpegQuality: JPEG quality from 0 to 100 used for storing debug images;
                                 only contour images are currently saved as JPEG
    :rtype: tuple (mask, sigma)
    """    
    if debugPathPrefix:
        red = (0,0,255)
        green = (0,255,0)
        orange = (0,106,255)

        debugHistogramPath = debugPathPrefix + 'hist.svg'
        debugThresholdedImagePath = debugPathPrefix + 'thresh.png'
        debugContoursImagePath = debugPathPrefix + 'cont.jpg'
        debugContoursMaskImagePath = debugPathPrefix + 'cont_mask.jpg'
        debugAdaptiveThresholdedImagePath = debugPathPrefix + 'thresh_adapt.png'
        debugCutoffImagePath = debugPathPrefix + 'cutoff.jpg'
    
    if isinstance(imagePath, np.ndarray):
        # assume RGB image array
        im = np.require(imagePath, np.uint8, 'C')
        im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
        imagePath = '[array]'
    else:
        im = cv.imread(imagePath)
    if channel is None:
        imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    elif channel.lower() == 'r':
        imgray = im[:,:,2]
    elif channel.lower() == 'g':
        imgray = im[:,:,1]
    elif channel.lower() == 'b':
        imgray = im[:,:,0]
    else:
        raise ValueError('channel is "{}" but must be R,G,B or None'.format(channel))
    
    # opencv requires a contiguous array..
    imgray = np.require(imgray, np.uint8, 'C')
    
    # Step 1: Find dark areas which might be starfield
    
    fudge = 20
    binary, hist, threshold, firstSpike = _binarizeStarfieldImage(imgray, fudge=fudge)
    contours, area, isBigContour, isSmallLongContour, isSmallShortContour = _findAndCategorizeContours(binary)
    mask = _createStarfieldMask(im, contours, area, isBigContour, None, blackenLowerPart)
    starfieldAreaRatio = np.sum(mask) / (mask.shape[0]*mask.shape[1])
        
    while starfieldAreaRatio < 0.1:
        # A part other then the starfield was probably picked as reference threshold because it was darker.
        # Remember that the first spike in the histogram (=darkest part) is used to determine the threshold.
        # To fix this situation, we just raise the threshold and hope for the best.
        print('Starfield area is only ' + "{0:.2f}".format(starfieldAreaRatio*100) +\
              '% (< 10%). Trying a higher threshold. ' +\
              '(' + os.path.basename(imagePath) + ')')
        fudge += 20
        binary, hist, threshold, firstSpike = _binarizeStarfieldImage(imgray, fudge=fudge)
        contours, area, isBigContour, isSmallLongContour, isSmallShortContour = _findAndCategorizeContours(binary)
        # FIXME use small long contours as well for masking, why did we disable that?
        mask = _createStarfieldMask(im, contours, area, isBigContour, None, blackenLowerPart)
        
        starfieldAreaRatio = np.sum(mask) / (mask.shape[0]*mask.shape[1])
        if starfieldAreaRatio >= 0.1:
            print('Starfield area is now ' + "{0:.2f}".format(starfieldAreaRatio*100) + '%')
        elif fudge > 100:
            print('giving up')
            break
        
    if debugPathPrefix and debugHistogramPath:
        vlines = [(firstSpike, 'red'), (threshold, 'blue')]
        try:
            saveFig(debugHistogramPath, drawHistogram(hist, vlines, 
                                                      xlabel='Intensity', 
                                                      ylabel='Pixel Count',
                                                      linecolor='black'))
        except:
            ex = traceback.format_exc()
            with open(debugPathPrefix + 'matplotlib.EXCEPTION', 'w') as fp:
                fp.write(ex)
        
    if debugPathPrefix and debugThresholdedImagePath:
        cv.imwrite(debugThresholdedImagePath, binary, [IMWRITE_PNG_COMPRESSION, 9])
        
    if debugPathPrefix and debugContoursImagePath:
        imContours = im.copy()
        cv.drawContours(imContours,contours[isBigContour],-1,red,2)
        cv.drawContours(imContours,contours[isSmallLongContour],-1,orange,2)
        cv.drawContours(imContours,contours[isSmallShortContour],-1,green,2)
        cv.imwrite(debugContoursImagePath, imContours, [IMWRITE_JPEG_QUALITY, debugJpegQuality])
        del imContours

    if debugPathPrefix and debugContoursMaskImagePath:
        # draw the borders of the current mask onto the original image
        imContoursMask = im.copy()
        res = _findAndCategorizeContours(mask)
        contours = res[0]
        cv.drawContours(imContoursMask,contours,-1,(255,255,255),3)
        cv.imwrite(debugContoursMaskImagePath, imContoursMask, [IMWRITE_JPEG_QUALITY, debugJpegQuality])
        del imContoursMask

    imgray[~mask] = 0
    
    # Step 2: Filter out dark areas which are probably not starfield
    
    # Step 2a: try to find lines and mask blocks containing them
    binary = _masked_adaptive_threshold(imgray, mask, 255, 89, -1)
    if debugPathPrefix and debugAdaptiveThresholdedImagePath:
        binaryC = cv.cvtColor(binary, cv.COLOR_GRAY2BGR)

    binary = cv.medianBlur(binary, 3)
    lines = cv.HoughLinesP(binary.copy(), 1, pi/180, 200, minLineLength=100, maxLineGap=4)
    
    blockShape = _getBlockShape(im)
    blockH, blockW = blockShape
    blockViewMask = view_as_blocks(mask, blockShape)
    
    if lines is not None:        
        # draw all lines on a blank binary image
        # then for each block check if it contains part of a line
        imFilledOffenders = np.zeros(mask.shape, np.uint8)
        
        for line in lines[0,:]:
            cv.line(imFilledOffenders, (line[0], line[1]), (line[2], line[3]), 255)
            
            if debugPathPrefix and debugAdaptiveThresholdedImagePath:
                cv.line(binaryC, (line[0], line[1]), (line[2], line[3]), red, 5)

        blockViewOffenders = view_as_blocks(imFilledOffenders, blockShape)
    
        isBlockContainingOffenders = (blockViewOffenders==255).any(axis=-1).any(axis=-1)
        blockViewMask[isBlockContainingOffenders] = False
        
    if debugPathPrefix and debugAdaptiveThresholdedImagePath:
        cv.imwrite(debugAdaptiveThresholdedImagePath, binaryC, [IMWRITE_PNG_COMPRESSION, 9])
       
    # Step 2b: make very dark pixels black and mask all-black blocks
    if ignoreVeryDark:
        if debugPathPrefix and debugCutoffImagePath:
            wasStarfieldBlock = blockViewMask.all(axis=-1).all(axis=-1)
        
        imgrayCutoff = imgray.copy()
        # blurring will wash out very tiny artifacts (which could also be faint small stars)
        # this helps here because we require that absolutely all of a block must be black
        imgrayCutoff = cv.blur(imgrayCutoff, (3,3))
        cutoffThreshold = max(30, firstSpike + 20)
        print('cutoffThreshold:', cutoffThreshold, os.path.basename(imagePath))
        imgrayCutoff[imgrayCutoff<cutoffThreshold] = 0
    
        blockViewCutoff = view_as_blocks(imgrayCutoff, blockShape)
        isBlockPureBlack = (blockViewCutoff==0).all(axis=-1).all(axis=-1)
        
        if debugPathPrefix and debugCutoffImagePath:
            imCutoff = im.copy()
            imCutoff[~mask] = 0
            
            blockViewCutoffC = view_as_blocks(imCutoff, (blockH,blockW,3))
            blockGotMasked = np.logical_and(wasStarfieldBlock, isBlockPureBlack)
            # draw rectangle around each block that was masked
            blockViewCutoffC[blockGotMasked,0,:4,:] = red
            blockViewCutoffC[blockGotMasked,0,-4:,:] = red
            blockViewCutoffC[blockGotMasked,0,:,:4] = red
            blockViewCutoffC[blockGotMasked,0,:,-4:] = red
            cv.imwrite(debugCutoffImagePath, imCutoff, [IMWRITE_JPEG_QUALITY, debugJpegQuality])
            del imCutoff
        
        blockViewMask[isBlockPureBlack] = False
        
    # Step 3: Filter out lonely starfield blocks (surrounded by non-starfield blocks)
    isStarfieldBlock = blockViewMask.all(axis=-1).all(axis=-1)
    
    neighborCountKernel = np.ones((3,3), dtype=int)
    neighborCountKernel[1,1] = 0
    neighbors = convolve2d(isStarfieldBlock.astype(int), neighborCountKernel, mode='same')
    isLonelyBlock = np.logical_and(isStarfieldBlock, neighbors == 0)
    blockViewMask[isLonelyBlock] = False
       
    # estimate noise level using the biggest starfield rectangle we got
    (rectY,rectX), (rectH,rectW) = _max_size_rectangle(isStarfieldBlock, value=True)
    rectY, rectH = rectY*blockH, rectH*blockH
    rectX, rectW = rectX*blockW, rectW*blockW
    imgrayBiggestRect = imgray[rectY:rectY+rectH, rectX:rectX+rectW]
    sigma = _estimateNoiseLevel(imgrayBiggestRect)
    print('Sigma:', sigma)
    if debugPathPrefix:
        with open(os.path.join(debugPathPrefix + '.sigma'), 'w') as fp:
            fp.write(str(sigma))

    return mask, sigma

def _estimateNoiseLevel(imgray):
    sigma = _doEstimateNoiseLevel(imgray)
    # sigma seems to be often lower than what astrometry.net calculates
    # NOTE: this is a dirty HACK, sometimes sigma is too high
    sigma = max(0.9, sigma * 2.5)
    return sigma

def _max_size_rectangle(mat, value=True):
    """Return (row,column),(height, width) of the largest rectangle containing 
    all `value`'s.
    """
    # adapted from https://gist.github.com/zed/776423
    
    #The original version was modified such that also the position of the largest
    #rectangle is returned, in addition to its size.
    #
    #Copyright (c) 2014, zed <isidore.john.r@gmail.com>
    #
    #Permission to use, copy, modify, and/or distribute this software for any
    #purpose with or without fee is hereby granted, provided that the above
    #copyright notice and this permission notice appear in all copies.
    #
    #THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
    #WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
    #MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
    #ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
    #WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
    #ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
    #OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
    area = np.product
    it = iter(mat)
    hist = [(el==value) for el in next(it, [])]
    max_size, max_column_index = _max_rectangle_size_hist(hist)
    max_row_index_bottom = 0
    for bottom_row_index, row in enumerate(it):
        hist = [(1+h) if el == value else 0 for h, el in zip(hist, row)]
        row_max_size, row_max_column_index = _max_rectangle_size_hist(hist)
        if area(row_max_size) > area(max_size):
            max_size = row_max_size
            max_row_index_bottom = bottom_row_index + 1
            max_column_index = row_max_column_index
    return (max_row_index_bottom-max_size[0]+1, max_column_index), max_size 

def _max_rectangle_size_hist(histogram):
    """Return size and column of the largest rectangle that fits entirely under
    the histogram.
    """
    # see _max_size_rectangle for origin and license text
    
    # The original version was modified such that also the column of the largest
    # rectangle is returned, in addition to its size.    
    Info = namedtuple('Info', 'start height')
    area = np.product
    stack = []
    top = lambda: stack[-1]
    max_size = (0, 0) # height, width of the largest rectangle
    max_column_index = 0
    pos = 0 # current position in the histogram
    for pos, height in enumerate(histogram):
        start = pos # position where rectangle starts
        while True:
            if not stack or height > top().height:
                stack.append(Info(start, height)) # push
            elif stack and height < top().height:
                current_size = (top().height, (pos - top().start))
                if area(current_size) > area(max_size):
                    max_size = current_size
                    max_column_index = top().start
                start, _ = stack.pop()
                continue
            break # height == top().height goes here
    pos += 1    
    for start, height in stack:
        if area((height, (pos - start))) > area(max_size):
            max_size = (height, (pos - start))
            max_column_index = start
    return max_size, max_column_index
