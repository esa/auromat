# Copyright European Space Agency, 2013

from __future__ import division

import unittest
import os

import auromat.util.lensdistortion

class Test(unittest.TestCase):
    def test(self):
        imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
        auromat.util.lensdistortion.correctLensDistortion(imagePath, 'test_undist.jpg', preserveExif=False)
        
def getResourcePath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/" + name)
