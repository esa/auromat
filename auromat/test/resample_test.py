# Copyright European Space Agency, 2013

from __future__ import division

import numpy as np
import numpy.ma as ma
from astropy.coordinates import Angle
import astropy.units as u

import unittest
from nose.plugins.attrib import attr

from auromat.resample import resample, resampleMLatMLT
from auromat.mapping.mapping import GenericMapping, checkPlateCarree
import os
from auromat.mapping.spacecraft import getMapping
from auromat.coordinates.transform import mltToSmLon, rotatePole
import datetime
from numpy.ma.testutils import assert_array_approx_equal

def _testCoords(offset):
    n = 10
    sp, step = np.linspace(offset,offset+10, num=n, retstep=True)
    coord = np.tile(sp, n).reshape(n,n).astype(np.float32)

    r = n*0.4
    y,x = np.ogrid[-r: r+1, -r: r+1]
    mask = x**2+y**2 <= r**2

    coord[mask] = -coord[mask]
    coord[coord>0] = np.nan
    coord[mask] = -coord[mask]
    
    coordCenter = coord[:-1,:-1] + step/2
    
    return coord, coordCenter
    
def testCoords():
    # no discontinuity or pole
    lats, latsCenter = _testCoords(70)
    lats = lats.T
    latsCenter = latsCenter.T
    
    lons, lonsCenter = _testCoords(160)
    
    return lats, lons, latsCenter, lonsCenter
    
def testCoordsDiscontinuity():
    lats, latsCenter = _testCoords(70)
    lats = lats.T
    latsCenter = latsCenter.T
    
    lons, lonsCenter = _testCoords(160)
    
    lons = Angle((lons + 15) * u.deg).wrap_at(180 * u.deg).degree
    lonsCenter = Angle((lonsCenter + 15) * u.deg).wrap_at(180 * u.deg).degree
    return lats, lons, latsCenter, lonsCenter

def testCoordsPole():
    lons, lonsCenter = _testCoords(1)
    lons = Angle((lons - 5) * u.deg).wrap_at(180 * u.deg).degree
    lonsCenter = Angle((lonsCenter - 5) * u.deg).wrap_at(180 * u.deg).degree
    lats = lons.T
    latsCenter = lonsCenter.T
    latsRot, lonsRot = rotatePole(np.deg2rad(lats.flat), np.deg2rad(lons.flat), 0, angle=90, axis=[0,1,0])
    latsCenterRot, lonsCenterRot = rotatePole(np.deg2rad(latsCenter.flat), np.deg2rad(lonsCenter.flat), 0, angle=90, axis=[0,1,0])
    return np.rad2deg(latsRot.reshape(lats.shape)), np.rad2deg(lonsRot.reshape(lons.shape)),\
           np.rad2deg(latsCenterRot.reshape(latsCenter.shape)), np.rad2deg(lonsCenterRot.reshape(lonsCenter.shape))

class Test(unittest.TestCase):
    def test(self):
        np.set_printoptions(precision=1)
        lats, lons, latsCenter, lonsCenter = testCoordsDiscontinuity()

        altitude = 110
        rgb = ma.masked_array((np.random.rand(lats.shape[0]-1,lats.shape[1]-1,3) * 255).astype(np.uint8))
        elevation = np.zeros((rgb.shape[0], rgb.shape[1]))
                
        mapping = GenericMapping(lats, lons, latsCenter, lonsCenter, elevation, altitude, rgb, 
                                 cameraPosGCRS=np.array([0,0,0]), photoTime=datetime.datetime.now(), identifier=None)
        m = resample(mapping, pxPerDeg=1, method='mean')
        m.checkPlateCarree()
        
        mapping = resampleMLatMLT(mapping, arcsecPerPx=100, method='nearest')
        assert not mapping.isPlateCarree
        mlat, mlt = mapping.mLatMlt
        smlon = mltToSmLon(mlt)
        checkPlateCarree(mlat, smlon)
    
    @attr('slow')          
    def testReal(self):
        m1 = _getMapping()
        m2 = resample(m1, pxPerDeg=15, method='mean')
        m2.checkPlateCarree()
        m3 = resample(m2, arcsecPerPx=100, method='nearest')
        m3.checkPlateCarree()
        
        assert_array_approx_equal(_bbToArray(m2.boundingBox), _bbToArray(m1.boundingBox), 1)
        assert_array_approx_equal(_bbToArray(m3.boundingBox), _bbToArray(m1.boundingBox), 1)
    
    @attr('slow')    
    def testPoleBug(self):
        m1 = _getMapping()
        m2 = resample(m1, arcsecPerPx=100, method='mean')
        assert_array_approx_equal(_bbToArray(m2.boundingBox), _bbToArray(m1.boundingBox), 1)
        
def getResourcePath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/" + name)

def _getMapping():
    imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
    wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
    mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
    return mapping

def _bbToArray(bb):
    return [bb.latNorth, bb.latSouth, bb.lonWest, bb.lonEast]
