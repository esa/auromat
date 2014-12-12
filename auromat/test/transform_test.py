# Copyright European Space Agency, 2013

from __future__ import division

import unittest
import numpy as np
from numpy.testing.utils import assert_array_almost_equal, assert_array_equal
import datetime

from auromat.coordinates.transform import geodetic2EcefZero,\
    ecef2Geodetic, geo_to_sm, j2000_to_geo, j2000_to_sm, gei_to_gse, gse_to_gsm, gsm_to_sm,\
    gei_to_geo, geo_to_gei, cartesian_to_spherical, spherical_to_cartesian,\
    _cartesian_to_spherical_np, _cartesian_to_spherical_ne,\
    _spherical_to_cartesian_np, _spherical_to_cartesian_ne

class Test(unittest.TestCase):

    def testCartesianSpherical(self):
        h, w = 20, 10
        x = np.random.rand(h, w)
        y = np.random.rand(h, w)
        z = np.random.rand(h, w)
        r, lat, lon = cartesian_to_spherical(x, y, z)
        xr, yr, zr = spherical_to_cartesian(r, lat, lon)
        assert_array_almost_equal(xr, x)
        assert_array_almost_equal(yr, y)
        assert_array_almost_equal(zr, z)
        
        xyz = spherical_to_cartesian(r, lat, lon, astuple=False)
        assert_array_almost_equal(xyz, np.dstack((x,y,z)))
        
    def testCartesianSphericalShape(self):
        h, w = 20, 10
        x = np.random.rand(h, w)
        y = np.random.rand(h, w)
        z = np.random.rand(h, w)
        
        r, lat, lon = _cartesian_to_spherical_np(x, y, z)
        assert_array_equal(r.shape, x.shape)
        assert_array_equal(lat.shape, r.shape)
        assert_array_equal(lon.shape, r.shape)
        
        r, lat, lon = _cartesian_to_spherical_ne(x, y, z)
        assert_array_equal(r.shape, x.shape)
        assert_array_equal(lat.shape, r.shape)
        assert_array_equal(lon.shape, r.shape)
        
        xr, yr, zr = _spherical_to_cartesian_np(r, lat, lon)
        assert_array_equal(xr.shape, r.shape)
        assert_array_equal(yr.shape, xr.shape)
        assert_array_equal(zr.shape, xr.shape)
        
        xr, yr, zr = _spherical_to_cartesian_ne(r, lat, lon)
        assert_array_equal(xr.shape, r.shape)
        assert_array_equal(yr.shape, xr.shape)
        assert_array_equal(zr.shape, xr.shape)
        
        xyz = _spherical_to_cartesian_np(r, lat, lon, astuple=False)
        assert_array_equal(xyz.shape, r.shape + (3,))
        
        xyz = _spherical_to_cartesian_ne(r, lat, lon, astuple=False)
        assert_array_equal(xyz.shape, r.shape + (3,))
        
        xyz = _spherical_to_cartesian_np(r.ravel(), lat.ravel(), lon.ravel(), astuple=False)
        assert_array_equal(xyz.shape, r.ravel().shape + (3,))
        
        xyz = _spherical_to_cartesian_ne(r.ravel(), lat.ravel(), lon.ravel(), astuple=False)
        assert_array_equal(xyz.shape, r.ravel().shape + (3,))

    def testGeodeticConversions(self):
        for lat in np.linspace(-89.9,89.9):
            for lon in np.linspace(-179.9,179.9):
                x, y, z = geodetic2EcefZero(np.deg2rad(lat), np.deg2rad(lon))
                r = np.rad2deg(ecef2Geodetic(x, y, z))
                assert_array_almost_equal(r,[lat,lon], 11)
        
    def testGeodeticConversionsArray(self):
        lat, lon = np.mgrid[-89:89:5,-179:179:5]
        x, y, z = geodetic2EcefZero(np.deg2rad(lat), np.deg2rad(lon))
    
        r = ecef2Geodetic(x, y, z)
        #print np.rad2deg(r)
        assert_array_almost_equal(np.rad2deg(r),[lat,lon], 11)
    
    # the following values were calculated using SSCWeb
    # with input: lat=50, lon=-100, r=1, date as below
    # http://sscweb.gsfc.nasa.gov/cgi-bin/CoordCalculator.cgi
    date = datetime.datetime(2012,1,25,9,26,55)
    geo = [[-0.11, -0.63, 0.77]]
    j2000 = [[-0.62, 0.16, 0.77]]
    gei = [[-0.62, 0.16, 0.77]]
    gse = [[-0.72, -0.26, 0.64]]
    gsm = [[-0.72, -0.30, 0.62]]
    sm = [[-0.43, -0.30, 0.85]]
    
    # the following 4 tests use each a single rotation matrix
    def testGeiToGeo(self):
        geo = gei_to_geo(self.date, self.gei)
        assert_array_almost_equal(geo, self.geo, 2)
        
    def testGeiToGse(self):
        gse = gei_to_gse(self.date, self.gei)
        assert_array_almost_equal(gse, self.gse, 2)
        
    def testGseToGsm(self):
        gsm = gse_to_gsm(self.date, self.gse)
        assert_array_almost_equal(gsm, self.gsm, 2)
        
    def testGsmToSm(self):
        sm = gsm_to_sm(self.date, self.gsm)
        assert_array_almost_equal(sm, self.sm, 2)
        
    # the following test uses a reverse transformation
    def testGeoToGei(self):
        gei = geo_to_gei(self.date, self.geo)
        assert_array_almost_equal(gei, self.gei, 2)

    # the following 3 tests use several multiplied rotation matrices
    def testJ2000ToGeo(self):
        geo = j2000_to_geo(self.date, self.j2000)
        assert_array_almost_equal(geo, self.geo, 2)
        
    def testJ2000ToSm(self):
        sm = j2000_to_sm(self.date, self.j2000)
        assert_array_almost_equal(sm, self.sm, 2)
    
    def testGeoToSm(self):
        sm = geo_to_sm(self.date, self.geo)
        assert_array_almost_equal(sm, self.sm, 2)
