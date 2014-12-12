# Copyright European Space Agency, 2013

import unittest
from auromat.util.movie import createMovie
import os
import tempfile


class Test(unittest.TestCase):

    def testMp4Movie(self):
        imagePaths = [getResourcePath('ISS030-E-102170_dc.jpg')]*30
        moviePath = tempfile.mktemp(suffix='.mp4')
        _createTempMovie(moviePath, lambda: createMovie(moviePath, imagePaths, width=1280))
            
    def testWebMMovie(self):
        imagePaths = [getResourcePath('ISS030-E-102170_dc.jpg')]*30
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
    
def getResourcePath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/" + name)