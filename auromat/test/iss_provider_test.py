# Copyright European Space Agency, 2013

from __future__ import absolute_import, print_function

import unittest
from nose.plugins.attrib import attr
import os.path
from datetime import datetime

from auromat.mapping.iss import ISSMappingProvider
from auromat.resample import resample
try:
    from auromat.draw import drawStereographicMLatMLT, drawScanLinesMLatMLTCo,\
        saveFig
except ImportError as e:
    print(repr(e))
    # import is optional so that the test module import doesn't fail if
    # the tests are not actually run and matplotlib is not installed
    
from auromat.util.coroutine import broadcast

url = 'http://127.0.0.1:5001/api/georef_seqs/12'
cacheFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iss_cache')

@attr('slow')
class Test(unittest.TestCase):
    def _testSingle(self):
        provider = ISSMappingProvider(url, cacheFolder, altitude=110)
        
        mapping = provider.get(datetime(2012,1,25,9,26,57))
        mapping = resample(mapping, arcsecPerPx=100)
        saveFig('test.png', drawStereographicMLatMLT(mapping))
        
    def testSequence(self):
        provider = ISSMappingProvider(url, cacheFolder, altitude=110)
        seq = provider.getSequence()
        broadcast(seq, drawScanLinesMLatMLTCo('test_scanlines.png'))
        