# Copyright European Space Agency, 2013

from __future__ import division, absolute_import

import unittest
from auromat.mapping.mapping import BoundingBox
from numpy.testing.utils import assert_array_almost_equal, assert_array_equal
    

class Test(unittest.TestCase):
    
    def testBB(self):
        bb = BoundingBox(latSouth=-60, lonWest=80, 
                         latNorth=-30, lonEast=85)        
        assert_array_almost_equal(bb.center, [-45.03119418083877, 82.5])
        assert_array_almost_equal(bb.size, [482.39311013217343, 3336.5953086140203])

    def testBBDiscontinuity(self):
        bb = BoundingBox(latSouth=-60.646114098, lonWest=82.7852215499, 
                         latNorth=-38.7515567117, lonEast=-178.546517062)
        assert_array_almost_equal(bb.center, [-54.33647117488648, 132.11935224395])
        assert_array_almost_equal(bb.size, [8084.704893634039, 3464.8889697347718])

    def testBBPole(self):
        # north pole
        bb = BoundingBox(latSouth=60, lonWest=-180, latNorth=90, lonEast=180)
        assert_array_almost_equal(bb.center, [90, 0])
        assert_array_almost_equal(bb.size, [6695.78581964, 6695.78581964])
        
        # south pole
        bb = BoundingBox(latSouth=-90, lonWest=-180, latNorth=-60, lonEast=180)
        assert_array_almost_equal(bb.center, [-90, 0])
        assert_array_almost_equal(bb.size, [6695.78581964, 6695.78581964])
    
    def testBBPoint(self):
        bb = BoundingBox(latSouth=50, lonWest=80, 
                         latNorth=50, lonEast=80)
        assert_array_equal(bb.center, [50,80])
        assert_array_equal(bb.size, 0)
        
    def testBBMerge(self):
        bb1 = BoundingBox(latSouth=-55, lonWest=95, latNorth=-45, lonEast=109)
        bb2 = BoundingBox(latSouth=44, lonWest=-164, latNorth=74, lonEast=-35)
        bb = BoundingBox.mergedBoundingBoxes([bb1,bb2])
        assert_array_equal([bb.latSouth,bb.latNorth,bb.lonWest,bb.lonEast],
                           [bb1.latSouth,bb2.latNorth,bb1.lonWest,bb2.lonEast])
        assert_array_almost_equal(bb.center, [21.136113246, -150])
        # size cannot be checked as the bounding box spans more than 180deg longitude
        # which is not supported currently
