# Copyright European Space Agency, 2013

from __future__ import division
import numpy as np
from auromat.utils import vectorLengths

try:
    from numexpr import evaluate as ne
except ImportError:
    ne = None

def sphereLineIntersection(sphereRadius, lineOrigin, lineDirection, directed=True):
    """
    Return the sphere-line intersection points.
            
    :param sphereRadius: radius of sphere with origin [0,0,0]
    :param lineOrigin: point, e.g. [1,2,3]
    :param lineDirection: unit vector or array of unit vectors
    :param bool directed: 
        True, if the line should be directed. In that case, the first intersection
        along the line is returned.
        If False, then the line is infinite in both ends and the intersection which
        is closest to the line origin is returned.
    :rtype: vector or array of vectors
    """
    directionPosDot = np.dot(lineDirection,lineOrigin)
    rootTerm = np.square(directionPosDot)
    rootTerm -= np.dot(lineOrigin,lineOrigin)
    rootTerm += np.square(sphereRadius)
    with np.errstate(invalid='ignore'): # ignore warnings for negative numbers (= no intersection)
        root = np.sqrt(rootTerm)
    directionPosDotNeg = -directionPosDot
    
    if directed:
        isInside = vectorLengths([lineOrigin])[0] < sphereRadius
        if isInside:
            d2 = directionPosDotNeg + root
            dMin = d2
        else:
            d1 = directionPosDotNeg - root
            dMin = d1
        dMin = _filterPointsOutsideDirectedLine(dMin)
    else:
        d1 = directionPosDotNeg - root
        d2 = directionPosDotNeg + root
        dMin = _closestDistance(d1, d2)

    return lineOrigin + dMin[...,None]*lineDirection

def _filterPointsOutsideDirectedLine(d):
    if d.ndim == 0:
        d = np.array(np.nan) if d < 0 else d
    else:
        with np.errstate(invalid='ignore'):
            d[d<0] = np.nan
    return d

def _ellipsoidLineIntersection_np(a, b, lineOrigin, lineDirection, directed=True):
    lineOrigin = np.require(lineOrigin, dtype=np.float64)
    lineDirection = np.require(lineDirection, dtype=np.float64)
    
    # turn into column vectors
    direction = lineDirection.T
    origin = -lineOrigin[:,None]
    
    radius = np.array([[1/a], [1/a], [1/b]])
    directionTimesRadius = direction * radius
    originTimesRadius = origin * radius
    
    # einsum is a bit faster for calculating the element-wise dot product
    # equivalent to np.sum(x*y, axis=0) or inner1d(x.T, y.T)
    directionDotOrigin = np.einsum("ij,ij->j", directionTimesRadius, originTimesRadius)
    directionDotDirection = np.einsum("ij,ij->j", directionTimesRadius, directionTimesRadius)
    originDotOrigin = np.einsum("ij,ij->j", originTimesRadius, originTimesRadius)

    rootTerm = np.square(directionDotOrigin)
    rootTerm -= originDotOrigin*directionDotDirection
    rootTerm += directionDotDirection
    with np.errstate(invalid='ignore'): # ignore warnings for negative numbers (= no intersection)
        np.sqrt(rootTerm, rootTerm)
    root = rootTerm

    if directed:
        if _isInsideEllipsoid(lineOrigin, a, b):
            d2 = directionDotOrigin
            d2 += root
            dMin = d2
        else:
            d1 = directionDotOrigin
            d1 -= root
            dMin = d1
        dMin = _filterPointsOutsideDirectedLine(dMin)
    else:
        d1 = directionDotOrigin - root
        d2 = directionDotOrigin
        d2 += root
        dMin = _closestDistance(d1, d2)
    
    dMin /= directionDotDirection
    
    res = directionTimesRadius
    np.multiply(direction, dMin, res)
    res -= origin
    return res.T

def _ellipsoidLineIntersection_ne(a, b, lineOrigin, lineDirection, directed=True):
    lineOrigin = np.require(lineOrigin, dtype=np.float64)
    lineDirection = np.require(lineDirection, dtype=np.float64)
    
    # turn into column vectors
    direction = lineDirection.T
    origin = -lineOrigin[:,None]
    
    radius = np.array([[1/a], [1/a], [1/b]])
    directionTimesRadius = ne('direction * radius')
    originTimesRadius = ne('origin * radius')
    
    directionDotOrigin = np.einsum("ij,ij->j", directionTimesRadius, originTimesRadius)
    directionDotDirection = np.einsum("ij,ij->j", directionTimesRadius, directionTimesRadius)
    originDotOrigin = np.einsum("ij,ij->j", originTimesRadius, originTimesRadius)
        
    root = ne('sqrt(directionDotOrigin**2 - originDotOrigin*directionDotDirection + directionDotDirection)')
        
    if directed:
        if _isInsideEllipsoid(lineOrigin, a, b):
            d2 = directionDotOrigin
            ne('directionDotOrigin + root', out=d2)
            dMin = d2
        else:
            d1 = directionDotOrigin
            ne('directionDotOrigin - root', out=d1)
            dMin = d1
        dMin = _filterPointsOutsideDirectedLine(dMin)
    else:
        d1 = ne('directionDotOrigin - root')
        d2 = directionDotOrigin
        ne('directionDotOrigin + root', out=d2)
        dMin = _closestDistance(d1, d2)
    
    res = directionTimesRadius
    ne('direction * (dMin / directionDotDirection) - origin', out=res)
    return res.T

def ellipsoidLineIntersection(a, b, lineOrigin, lineDirection, directed=True):
    """
    Return the ellipsoid-line intersection points.

    :note: The ellipsoid is assumed to be at (0,0,0).
    :param a: equatorial axis of the ellipsoid of revolution
    :param b: polar axis of the ellipsoid of revolution
    :param lineOrigin: x,y,z vector
    :param lineDirection: x,y,z array of vectors; not required to be unit vectors
    :param bool directed: 
        True, if the line should be directed. In that case, the first intersection
        along the line is returned.
        If False, then the line is infinite in both ends and the intersection which
        is closest to the line origin is returned.
    :rtype: vector or array of vectors
    """
    if ne:
        return _ellipsoidLineIntersection_ne(a, b, lineOrigin, lineDirection, directed)
    else:
        return _ellipsoidLineIntersection_np(a, b, lineOrigin, lineDirection, directed)

def _ellipsoidLineIntersects_np(a, b, lineOrigin, lineDirection, directed=True):
    lineOrigin = np.require(lineOrigin, dtype=np.float64)
    lineDirection = np.require(lineDirection, dtype=np.float64)
    
    # turn into column vectors
    direction = lineDirection.T
    origin = -lineOrigin[:,None]
    
    radius = np.array([[1/a], [1/a], [1/b]])
        
    directionTimesRadius = direction * radius
    originTimesRadius = origin * radius
    directionDotOrigin = np.einsum("ij,ij->j", directionTimesRadius, originTimesRadius)
    directionDotDirection = np.einsum("ij,ij->j", directionTimesRadius, directionTimesRadius)
    originDotOrigin = np.einsum("ij,ij->j", originTimesRadius, originTimesRadius)

    rootTerm = np.square(directionDotOrigin)
    rootTerm -= originDotOrigin*directionDotDirection
    rootTerm += directionDotDirection
    
    with np.errstate(invalid='ignore'): # ignore warnings for negative numbers (= no intersection)
        if directed:
            np.sqrt(rootTerm, rootTerm)
            root = rootTerm
            if _isInsideEllipsoid(lineOrigin, a, b):
                d2 = directionDotOrigin
                d2 += root
                dMin = d2
            else:
                d1 = directionDotOrigin
                d1 -= root
                dMin = d1
            intersects = dMin >= 0
        else:
            intersects = rootTerm >= 0
    
    return intersects

def _ellipsoidLineIntersects_ne(a, b, lineOrigin, lineDirection, directed=True):
    lineOrigin = np.require(lineOrigin, dtype=np.float64)
    lineDirection = np.require(lineDirection, dtype=np.float64)
    
    # turn into column vectors
    direction = lineDirection.T
    origin = -lineOrigin[:,None]
    
    radius = np.array([[1/a], [1/a], [1/b]])
    directionTimesRadius = ne('direction * radius')
    originTimesRadius = ne('origin * radius')
    
    dirDotOri = np.einsum("ij,ij->j", directionTimesRadius, originTimesRadius)
    dirDotDir = np.einsum("ij,ij->j", directionTimesRadius, directionTimesRadius)
    oriDotOri = np.einsum("ij,ij->j", originTimesRadius, originTimesRadius)

    if directed:
        if _isInsideEllipsoid(lineOrigin, a, b):
            intersects = ne('dirDotOri + sqrt(dirDotOri**2 - oriDotOri*dirDotDir + dirDotDir) >= 0')
        else:
            intersects = ne('dirDotOri - sqrt(dirDotOri**2 - oriDotOri*dirDotDir + dirDotDir) >= 0')
    else:
        intersects = ne('dirDotOri**2 - oriDotOri*dirDotDir + dirDotDir >= 0')
        
    return intersects

def ellipsoidLineIntersects(a, b, lineOrigin, lineDirection, directed=True):
    """
    As :func:`ellipsoidLineIntersection` but returns an array of booleans instead
    of the intersection points.
    """
    if ne:
        return _ellipsoidLineIntersects_ne(a, b, lineOrigin, lineDirection, directed)
    else:
        return _ellipsoidLineIntersects_np(a, b, lineOrigin, lineDirection, directed)

def _isInsideEllipsoid(point, a, b):
    x,y,z = point
    return (x/a)**2 + (y/a)**2 + (z/b)**2 < 1

def _closestDistance(d1, d2):
    """
    Assuming a single line origin, this returns the distances (either d1 or d2)
    whose absolute values are smallest.
    """
    with np.errstate(invalid='ignore'):
        dMin = np.where(np.abs(d1) < np.abs(d2), d1, d2)
    return dMin
