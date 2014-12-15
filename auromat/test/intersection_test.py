# Copyright European Space Agency, 2013

from __future__ import division

import os.path
import unittest
from nose.plugins.attrib import attr
import numpy as np
import numpy.ma as ma
from numpy.testing.utils import assert_array_almost_equal,\
    assert_array_equal, assert_equal
    
from auromat.coordinates.geodesic import wgs84A, wgs84B
from auromat.utils import unitVectors
from auromat.coordinates.transform import geodetic2Ecef
from auromat.coordinates.intersection import sphereLineIntersection,\
    _ellipsoidLineIntersection_np, _ellipsoidLineIntersection_ne,\
    _ellipsoidLineIntersects_np, _ellipsoidLineIntersects_ne
from auromat.mapping import miracle, spacecraft

ellipsoidLineIntersectionFns = [_ellipsoidLineIntersection_np, _ellipsoidLineIntersection_ne]
ellipsoidLineIntersectsFns = [_ellipsoidLineIntersects_np, _ellipsoidLineIntersects_ne]

class Test(unittest.TestCase):

    def testSphereLineIntersection(self):
        sphereRadius = 2
        lineOrigin = [0,3,0]
        lineDirection = [0,-1,0]
        point = sphereLineIntersection(sphereRadius, lineOrigin, lineDirection)
        
        assert_equal(point, [0,2,0])
        
    def testSphereLineIntersectionArray(self):
        sphereRadius = 2
        lineOrigin = [0,3,0]
        lineDirection = unitVectors([[0,-1,0],[-1,-1,0]])
        points = sphereLineIntersection(sphereRadius, lineOrigin, lineDirection)
        
        assert_equal(points, [[0,2,0],[np.nan,np.nan,np.nan]])
        
    def testEllipsoidLineIntersection(self):
        for ellipsoidLineIntersection in ellipsoidLineIntersectionFns:
            p1 = np.array(geodetic2Ecef(np.deg2rad(30),np.deg2rad(60),0))
            p2 = np.array(geodetic2Ecef(np.deg2rad(-30),np.deg2rad(-60),0))
            
            i1 = ellipsoidLineIntersection(wgs84A, wgs84B, p1, [p1-p2], directed=False)
    
            assert_array_almost_equal(i1, [p1])
        
    def testEllipsoidLineIntersectionSphere(self):
        for ellipsoidLineIntersection, ellipsoidLineIntersects in zip(ellipsoidLineIntersectionFns, ellipsoidLineIntersectsFns):
            a = b = 2
            lineOrigin = [0,3,0]
            lineDirection = [[0,-1,0],[0,-1,0],[-1,-1,0]]
            
            points = ellipsoidLineIntersection(a, b, lineOrigin, lineDirection)
            assert_equal(points, [[0,2,0],[0,2,0],[np.nan,np.nan,np.nan]])
            
            intersects = ellipsoidLineIntersects(a, b, lineOrigin, lineDirection)
            assert_equal(intersects, [True, True, False])
        
    def testDirectedIntersection(self):
        for ellipsoidLineIntersection, ellipsoidLineIntersects in zip(ellipsoidLineIntersectionFns, ellipsoidLineIntersectsFns):
            r = 1
            origin = [2,0,0]
            direction = [[1,0,0]]
            intersection = [[1,0,0]]
            noIntersection = [[np.nan, np.nan, np.nan]]
            
            res = sphereLineIntersection(r, origin, direction, directed=False)
            assert_array_equal(res, intersection)
            
            res = sphereLineIntersection(r, origin, direction, directed=True)
            assert_array_equal(res, noIntersection)
                    
            res = ellipsoidLineIntersection(r, r, origin, direction, directed=False)
            assert_array_equal(res, intersection)
            
            intersects = ellipsoidLineIntersects(r, r, origin, direction, directed=False)
            assert_equal(intersects, [True])        
            
            res = ellipsoidLineIntersection(r, r, origin, direction, directed=True)
            assert_array_equal(res, noIntersection)
            
            intersects = ellipsoidLineIntersects(r, r, origin, direction, directed=True)
            assert_equal(intersects, [False])
            
            origin2 = [-2,0,0]
            intersection2 = [[-1,0,0]]
            res = sphereLineIntersection(r, origin2, direction, directed=False)
            assert_array_equal(res, intersection2)
            
            res = sphereLineIntersection(r, origin2, direction, directed=True)
            assert_array_equal(res, intersection2)
            
            direction2 = [[-1,0,0]]
            res = sphereLineIntersection(r, origin2, direction2, directed=False)
            assert_array_equal(res, intersection2)
            
            res = sphereLineIntersection(r, origin2, direction2, directed=True)
            assert_array_equal(res, noIntersection)
            
    def testDirectedIntersectionFromInside(self):
        # line origin is inside sphere/ellipsoid
        
        for ellipsoidLineIntersection, ellipsoidLineIntersects in zip(ellipsoidLineIntersectionFns, ellipsoidLineIntersectsFns):
            r = 2
            origin = [1,0,0]
            direction = [[1,0,0]]
            intersection = [[2,0,0]]
            
            res = sphereLineIntersection(r, origin, direction, directed=False)
            assert_array_equal(res, intersection)
            
            res = sphereLineIntersection(r, origin, direction, directed=True)
            assert_array_equal(res, intersection)
                    
            res = ellipsoidLineIntersection(r, r, origin, direction, directed=False)
            assert_array_equal(res, intersection)
            
            intersects = ellipsoidLineIntersects(r, r, origin, direction, directed=False)
            assert_equal(intersects, [True])        
            
            res = ellipsoidLineIntersection(r, r, origin, direction, directed=True)
            assert_array_equal(res, intersection)
            
            intersects = ellipsoidLineIntersects(r, r, origin, direction, directed=True)
            assert_equal(intersects, [True])
            
            intersection2 = [[-2,0,0]]           
            direction2 = [[-1,0,0]]
            res = sphereLineIntersection(r, origin, direction2, directed=False)
            assert_array_equal(res, intersection)
            
            res = sphereLineIntersection(r, origin, direction2, directed=True)
            assert_array_equal(res, intersection2)
        
    @attr('slow')       
    def testIntersectionBug(self):
        """
        An optimized code path led to choosing the intersection points on the
        other side of the earth for a particular mapping. This mapping
        has (wrongly appearing) intersection points at the top of the image, 
        and these errorneous points led the optimized code path to choose
        the wrong side of the earth, at least from the point of view of
        the correct intersection points at the lower half of the image.
        Thus, this issue will disappear once the actual bug is solved.
        """
        m = _getISSMapping()
        dists = m.distance
        assert dists[-1,2000] < dists[-100,2000]
    
    @attr('slow') 
    def testIntersectionBug2(self):
        """
        For the sequence beginning with ISS029-E-8492 there are intersection
        areas at the top of the image which should not be there.
        These intersections occur because the other ends of the rays actually intersect
        with the earth "behind" the camera. This happens because we intersect an
        infinite line with the earth, instead of a line beginning at the camera position
        towards the earth.
        """
        m = _getISSMapping()
        # check that there is intersection at the bottom
        assert m.lats[-1,2000] is not ma.masked
        
        # check that there is NO intersection at the top
        assert m.lats[0,2000] is ma.masked

def getResourcePath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/" + name)

def _getISSMapping():
    imagePath = getResourcePath('ISS029-E-8492.jpg')
    wcsPath = getResourcePath('ISS029-E-8492.wcs')
    mapping = spacecraft.getMapping(imagePath, wcsPath, fastCenterCalculation=True)
    return mapping

def _getMiracleMapping():
    imagePath = getResourcePath('SOD120304_171900_557_1000.jpg')
    mapping = miracle.getMapping(imagePath)
    return mapping
