# Copyright European Space Agency, 2013

import unittest
import os
import numpy as np
import tempfile
from nose.plugins.attrib import attr

from auromat.util.movie import createMovie
from auromat.util.image import saveImage


class Test(unittest.TestCase):
    def setUp(self):
        # create test frame for movie creation
        self.framePath = tempfile.mktemp(suffix='.jpg')
        im = np.zeros((2832,4256,3), np.uint8)
        saveImage(self.framePath, im)
        
    def tearDown(self):
        os.remove(self.framePath)

    def testMp4Movie(self):
        imagePaths = [self.framePath]*30
        moviePath = tempfile.mktemp(suffix='.mp4')
        _createTempMovie(moviePath, lambda: createMovie(moviePath, imagePaths, width=1280))
    
    @attr('libvpx')
    def testWebMMovie(self):
        imagePaths = [self.framePath]*30
        moviePath = tempfile.mktemp(suffix='.webm')
        _createTempMovie(moviePath, lambda: createMovie(moviePath, imagePaths, width=1280))
            
def _createTempMovie(moviePath, fn):
    try:
        fn()
    except:
        # if there was an exception we want to see it, and
        # ignore the one we might get from removing a nonexisting file
        if os.path.exists(moviePath):
            os.remove(moviePath)
        raise
    else:
        os.remove(moviePath)
    