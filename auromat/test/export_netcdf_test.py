# Copyright European Space Agency, 2013

from __future__ import division, absolute_import

import unittest
from nose.plugins.attrib import attr
from numpy.ma.testutils import assert_array_almost_equal,\
    assert_array_equal, assert_equal
import os
import tempfile
from netCDF4 import Dataset

from auromat.mapping.spacecraft import getMapping
from auromat.resample import resample
import auromat.export.netcdf
from auromat.mapping.netcdf import NetCDFMapping

@attr('slow')
class Test(unittest.TestCase):

    def testRawNetCDFExport(self):
        imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
        wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
        mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
        mapping.checkGuarantees()
        path = tempfile.mktemp(suffix='.nc')
        try:            
            auromat.export.netcdf.write(path, mapping)
            
            with Dataset(path, 'r') as root:
                var = root.variables
                assert set(var.keys()) == set(['time', 'lat', 'lon', 'altitude', 
                                        'lat_bounds', 'lon_bounds', 'mlat', 
                                        'mlt', 'mlat_bounds', 'mlt_bounds', 
                                        'mcrs', 'img_red', 'img_green', 
                                        'img_blue', 'zenith_angle', 
                                        'camera_pos', 'crs'])
            
            cdfmapping = NetCDFMapping(path)
            cdfmapping.checkGuarantees()
            check_equal(cdfmapping, mapping)
                
        finally:
            if os.path.exists(path):
                os.remove(path)

    def testPlateCarreeNetCDFExport(self):
        imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
        wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
        mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
        mapping = resample(mapping, arcsecPerPx=200)
        path = tempfile.mktemp(suffix='.nc')
        try:
            auromat.export.netcdf.write(path, mapping)
            
            cdfmapping = NetCDFMapping(path)
            cdfmapping.checkGuarantees()
            cdfmapping.checkPlateCarree()
            
            check_equal(cdfmapping, mapping)
        finally:
            if os.path.exists(path):
                os.remove(path)
                
def check_equal(cdfmapping, mapping):
    assert_array_equal(cdfmapping.img.shape, mapping.img.shape)
    assert_array_equal(cdfmapping.lats.shape, mapping.lats.shape)

    assert_array_equal(cdfmapping.img, mapping.img)
    assert_array_equal(cdfmapping.lats, mapping.lats)
    assert_array_equal(cdfmapping.lons, mapping.lons)
    assert_array_equal(cdfmapping.latsCenter, mapping.latsCenter)
    assert_array_equal(cdfmapping.lonsCenter, mapping.lonsCenter)
    
    assert_equal(cdfmapping.boundingBox, mapping.boundingBox)
    
    # elevation is stored as float32, so there is some loss of accuracy
    assert_array_almost_equal(cdfmapping.elevation, mapping.elevation, decimal=5)
    
    assert_equal(cdfmapping.photoTime, mapping.photoTime)
    assert_equal(cdfmapping.cameraPosGCRS, mapping.cameraPosGCRS)
              
                               
def getResourcePath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/" + name)
