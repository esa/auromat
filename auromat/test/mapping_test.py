# Copyright European Space Agency, 2013

from __future__ import print_function

import unittest
import os
import numpy as np
from auromat.mapping.spacecraft import getMapping
from auromat.resample import resample
from mpl_toolkits.basemap import Basemap
from auromat.mapping import miracle
from auromat.draw_helpers import generatePolygonsFromMapping

class Test(unittest.TestCase):

    def testSpacecraftMappingNorth2(self):
        m = _getMappingNorth()
        foo = generatePolygonsFromMapping(m)
        print(foo)

    def _testSpacecraftMappingNorth(self):
        m = _getMappingNorth()
        m.checkGuarantees()
        m2 = m.maskedByElevation(10)
        m2.checkGuarantees()
        
        assert np.any(~(m.latsCenter.mask == m2.latsCenter.mask))
        assert np.any(~(m.lats.mask == m2.lats.mask))
        
        m3 = resample(m, arcsecPerPx=100, method='mean')
        m3.checkGuarantees()
        
    def _testSpacecraftMappingSouth(self):
        m = _getMappingSouth()
        m.checkGuarantees()
        m = m.maskedByElevation(10)
        m.checkGuarantees()
        m = resample(m, arcsecPerPx=100, method='mean')
        m.checkGuarantees()
        
    def _testMiracleMapping(self):
        m = _getMiracleMapping()
        m.checkGuarantees()
        m = m.maskedByElevation(10)
        m.checkGuarantees()
        m = resample(m, arcsecPerPx=100, method='mean')
        m.checkGuarantees()

def getResourcePath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/" + name)

def _getMappingNorth():
    imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
    wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
    mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
    return mapping

def _getMappingSouth():
    imagePath = getResourcePath('ISS029-E-8492.jpg')
    wcsPath = getResourcePath('ISS029-E-8492.wcs')
    mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
    return mapping

def _getMiracleMapping():
    imagePath = getResourcePath('SOD120304_171900_557_1000.jpg')
    mapping = miracle.getMapping(imagePath)
    return mapping
