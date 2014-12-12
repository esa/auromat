# Copyright European Space Agency, 2013

"""
A collection of utility functions which should eventually become part
of the :mod:`auromat.util` package.
"""

from __future__ import division, print_function, absolute_import

import logging
import bisect
from itertools import groupby

import numpy as np
from numpy.core.umath_tests import inner1d # vector-wise dot product
import itertools

from scipy.spatial import Delaunay

import skimage.measure

import matplotlib.path
try:
    import matplotlib.nxutils
except:
    pass
            
def vectorLengths(vectors):
    """ `np.linalg.norm(vectors, axis=1)` for numpy < 1.8. """
    vectors = np.asarray(vectors)
    return np.sqrt((vectors*vectors).sum(axis=1))

def unitVectors(vectors):
    """ Return the unit vectors of an array of vectors. """
    vectors = np.asarray(vectors)
    return vectors / vectorLengths(vectors)[...,None]

def angleBetween(v1, v2):
    """ Return the angles in radians between two unit vector arrays.
        Angles are in [0,pi]. """
    # When vectors are equal their dot product is 1, but
    # due to rounding it can be slightly above 1 which would then
    # result in arccos returning NaN. Therefore we clip the values to [-1,1]
    dot = np.clip(inner1d(v1, v2), -1, 1)
    angle = np.arccos(dot)
    return angle

def signedAngleBetween(v1, v2):
    """ 
    Return the angles in radians between two 2D vector arrays.
    Angles are in [-pi,pi].
    
    :see: http://stackoverflow.com/a/2150475
    """
    angle = np.arctan2(v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0], v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1])
    return angle

def pointsInsidePolygon(points, polygon):
    """
    Return for each point if it lies inside the given polygon.
    
    :param points: shape (n,2)
    :param polygon: unclosed, shape (n,2)
    :rtype: boolean ndarray of shape (n,)
    """
    path = matplotlib.path.Path(polygon)
    try:
        # only in mpl >= 1.2.0        
        isInside = path.contains_points(points)
    except AttributeError:
        # we are on an older version, try nxutils instead
        # nxutils is deprecated in mpl >= 1.2.0
        isInside = matplotlib.nxutils.points_inside_poly(points, polygon)
    return isInside

def _outline_opencv(im):
    import cv2 as cv
    # "Also, the function does not take into account 1-pixel border of the image
    # (it's filled with 0's and used for neighbor analysis in the algorithm),
    # therefore the contours touching the image border will be clipped."
    # (findContours docs)
    # -> as we want the borders, we pad it with 1px on each side
    imCv = np.zeros((im.shape[0]+2, im.shape[1]+2), dtype=np.uint8)
    imCv[1:-1,1:-1] = im
    # cv.CHAIN_APPROX_SIMPLE *must not* be used here, as it would turn concave parts
    # into convex parts!
    contours,_ = cv.findContours(imCv, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    if len(contours) > 1:
        area = list(map(cv.contourArea, contours))
        contour = contours[np.argmax(area)]
        logging.warn('The binary data contains multiple contours! Returning the biggest one.')
    else:
        contour = contours[0]
    return np.asarray(contour).reshape(-1,2) - 1

def _outline_skimage(im):
    imBorder = np.zeros((im.shape[0]+2, im.shape[1]+2), dtype=bool)
    imBorder[1:-1,1:-1] = im
    # by using a level near 1 (=True) we get coordinates that can
    # be directly used as array indices (after rounding)
    level = 0.99
    contours = skimage.measure.find_contours(imBorder, level)
    
    def _fixContour(contour):
        # contrary to the API docs, the returned coordinates are (y,x)
        # instead of (x,y)
        # see https://github.com/scikit-image/scikit-image/issues/1140
        contour = np.fliplr(contour)
        
        # subtract padding and round to int
        contour = np.round(contour-1).astype(int)
        
        # remove consecutive duplicate values
        # TODO why are these here in the first place?
        contour = withoutConsecutiveDuplicates(contour)
    
        # un-close polygon
        contour = contour[:-1]
               
        return contour
        
    if len(contours) > 1:
        contours = list(map(_fixContour, contours))
        # degenerate contours may result from an initial contour such as
        # [[ 186.    320.01]
        #  [ 185.99  320.  ]
        #  [ 186.    319.99]
        #  [ 186.01  320.  ]
        #  [ 186.    320.01]]
        # after _fixContour() the contour is empty
        contours = list(filter(lambda c: len(c) > 2, contours))
        area = list(map(polygonArea, contours))
        contour = contours[np.argmax(area)]
        logging.warn('The binary data contains multiple contours! Returning the biggest one.')
    else:
        contour = _fixContour(contours[0])
    
    return contour

def outline(im):
    """ 
    Finds the outline of a binary image, assuming that the inner structure
    is filled with True's. The returned points are in clockwise order and can
    be used as a polygon.
    This works for concave forms as well.
    
    :param im: shape (h,w)
    :rtype: ndarray of shape (n,2) in x,y order
    """
    return _outline_skimage(im)

def polygonArea(poly, signed=False):
    """
    Return area of an unclosed polygon.
    
    :see: http://www.mathopenref.com/coordpolygonarea.html
    :param poly: (n,2)-array
    """
    # There is no equivalent function in skimage yet.
    # See https://groups.google.com/d/topic/scikit-image/--pXW-fPbGg/discussion
    
    # we need a plain list for the following operations
    if isinstance(poly, np.ndarray):
        poly = poly.tolist()
    segments = zip(poly, poly[1:] + [poly[0]])
    area = 0.5 * sum(x0*y1 - x1*y0
                     for ((x0, y0), (x1, y1)) in segments)
    if not signed:
        area = abs(area)
    return area

def polygonCentroid(poly):
    """
    Return centroid point of an unclosed polygon.
    
    :param poly: (n,2)-array
    :rtype: tuple (x,y)
    """
    # see https://stackoverflow.com/a/14128955
    #Copyright (c) 2013, Stack Overflow user "unutbu"
    #All rights reserved.
    #
    #Redistribution and use in source and binary forms, with or without modification,
    #are permitted provided that the following conditions are met:
    #
    #1. Redistributions of source code must retain the above copyright notice,
    #   this list of conditions and the following disclaimer.
    #
    #2. Redistributions in binary form must reproduce the above copyright notice,
    #   this list of conditions and the following disclaimer in the documentation 
    #   and/or other materials provided with the distribution.
    #
    #3. Neither the name of the copyright holder nor the names of its contributors 
    #   may be used to endorse or promote products derived from this software
    #   without specific prior written permission.
    #
    #THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
    #AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
    #IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
    #ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
    #LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    #DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
    #SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
    #CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, 
    #OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
    #OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    area = polygonArea(poly)
    
    result_x = 0
    result_y = 0
    N = len(poly)
    points = itertools.cycle(poly)
    x1, y1 = next(points)
    for _ in range(N):
        x0, y0 = x1, y1
        x1, y1 = next(points)
        cross = (x0 * y1) - (x1 * y0)
        result_x += (x0 + x1) * cross
        result_y += (y0 + y1) * cross
    result_x /= (area * 6.0)
    result_y /= (area * 6.0)
    return (result_x, result_y)

def _polygonCentroidOpenCV(poly):
    import cv2
    poly = np.require(poly, np.float32, 'C')
    m = cv2.moments(poly.reshape(-1,1,2))
    x = m['m10']/m['m00']
    y = m['m01']/m['m00']
    return x, y

def withoutConsecutiveDuplicates(arr):
    """
    Return a copy of the input array where consecutive duplicates
    (on the first dimension) are removed.
    
    :param arr: list or ndarray
    :rtype: ndarray
    """
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()
    return np.asarray([p[0] for p in groupby(arr)])

def convexHull(points):
    """
    Return the convex hull spanning the given points.
    
    :param points: array of shape (n,2)
    :rtype: array of shape (m,2)
    """
    points = np.asarray(points)
    assert points.ndim == 2 and points.shape[1] == 2

    # TODO using scipy.spatial.ConvexHull lead to inconsistent results
    #      -> possible bug? for now, just use Delaunay
#    if ConvexHull:
#        # scipy.spatial.ConvexHull was introduced in 0.12
#        hull = ConvexHull(points)
#        vertices = points[hull.vertices]
#    else:

    # fall-back to Delaunay to support scipy 0.9 to 0.11
    # the documentation says:
    # "Computing convex hulls via the Delaunay triangulation is
    # inefficient and subject to increased numerical instability."
    hull = Delaunay(points).convex_hull
    vertices = points[np.unique(hull)]

    # sort vertices
    verticesCentered = vertices - vertices.mean(axis=0)
    angles = np.arctan2(verticesCentered[:,0], verticesCentered[:,1])
    vertices = vertices[np.argsort(angles)]
    
    return vertices

def findNearest(a, x):
    """
    Find item in sorted list a that is closest to x and return its index.
    
    If x has the same distance to both neighbors, the index of the left
    one is returned.
    """
    i = bisect.bisect_left(a, x)
    if i == len(a):
        return i-1
    else:
        if a[i] == x or i == 0:
            return i
        d0 = abs(x - a[i-1])
        d1 = abs(x - a[i])
        return i-1 if d0<=d1 else i

def extend(instance, new_class):
    """
    Apply inheritance after object creation.
    
    :see: https://stackoverflow.com/a/8545287
    """
    # note that new_class and instance.__class__ is switched here!
    instance.__class__ = type(
                '%s_extended_with_%s' % (instance.__class__.__name__, new_class.__name__), 
                (new_class, instance.__class__), 
                {}
            )
