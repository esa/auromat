# Copyright European Space Agency, 2013

from __future__ import division, print_function

import numpy as np

import unittest
from nose.plugins.attrib import attr

import os
from auromat.mapping.spacecraft import getMapping

@attr('slow')
class Test(unittest.TestCase):            
    def testReal(self):
        m = _getMapping1()
        print(np.min(m.elevation), np.max(m.elevation))
        assert 0 <= np.min(m.elevation) <= np.max(m.elevation) <= 90
        
        m = _getMapping2()
        print(np.min(m.elevation), np.max(m.elevation))
        assert 0 <= np.min(m.elevation) <= np.max(m.elevation) <= 90

def getResourcePath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/" + name)

def _getMapping1():
    imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
    wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
    mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
    return mapping

def _getMapping2():
    imagePath = getResourcePath('ISS029-E-8492.jpg')
    wcsPath = getResourcePath('ISS029-E-8492.wcs')
    mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
    return mapping

