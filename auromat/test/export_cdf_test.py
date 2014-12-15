# Copyright European Space Agency, 2013

from __future__ import division

import unittest
from nose.plugins.attrib import attr
import os
import tempfile

from auromat.mapping.spacecraft import getMapping
import auromat.export.cdf
from spacepy import pycdf
from auromat.mapping.cdf import CDFMapping
from auromat.test.export_netcdf_test import check_equal

@attr('slow')
class Test(unittest.TestCase):

    def testCDFExport(self):
        imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
        wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
        mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
        path = tempfile.mktemp(suffix='.cdf')
        try:            
            auromat.export.cdf.write(path, mapping)
            
            with pycdf.CDF(path) as root:
                assert set(root.keys()) == set(['Epoch', 'lat', 'lon', 'altitude',
                                                'lat_bounds', 'lon_bounds', 'mlat',
                                                'mlt', 'mlat_bounds', 'mlt_bounds',
                                                'mcrs', 'img_red', 'img_green',
                                                'img_blue', 'zenith_angle', 'camera_pos',
                                                'crs'])
            
            cdfmapping = CDFMapping(path)
            cdfmapping.checkGuarantees()
            check_equal(cdfmapping, mapping)
            
        finally:
            if os.path.exists(path):
                os.remove(path)
                        
            
def getResourcePath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/" + name)
