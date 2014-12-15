# Copyright European Space Agency, 2013

from __future__ import division, absolute_import, print_function

import numpy as np
import matplotlib.pyplot as plt

import unittest
from nose.plugins.attrib import attr

from auromat.utils import _outline_opencv, _outline_skimage, polygonArea,\
    polygonCentroid, _polygonCentroidOpenCV
from numpy.testing.utils import assert_almost_equal
from auromat.mapping.spacecraft import getMapping
import os.path

try:
    import cv2 as cv
except ImportError:
    cv = None

def _testIm(n=10):
    coord = np.ones((n,n))

    r = n*0.4
    y,x = np.ogrid[-r: r+1, -r: r+1]
    mask = x**2+y**2 <= r**2

    coord[mask] = -coord[mask]
    coord[coord>0] = np.nan
    # make it non-symmetric
    coord[4,0] = np.nan
    
    return ~np.isnan(coord)

if __name__ == '__main__':
    np.set_printoptions(precision=1)
    
    
#    def testImage():
#        n = 10
#        coord = np.ones((n,n))
#    
#        r = n*0.4
#        y,x = np.ogrid[-r: r+1, -r: r+1]
#        mask = x**2+y**2 <= r**2
#    
#        coord[mask] = -coord[mask]
#        coord[coord>0] = np.nan
#        # make it non-symmetric
#        coord[4,0] = np.nan
#        
#        r = ~np.isnan(coord)
#        return np.hstack((r,r))
#    
#    import skimage.measure
#    contours = skimage.measure.find_contours(testImage(), 0.99)
#    print contours
#    raise
    
    
    
    im = _testIm(3000)
    print(im)
    
    outl = _outline_opencv(im)
    outlsk = _outline_skimage(im)
    
    print(len(outl) == len(outlsk))
    print('opencv:', outl)
    print('skimage:', outlsk)
    
    
    # TODO check that both outlines are the same modulo starting point
    
    for title, p in [('skimage', outlsk), ('opencv', outl)]:
        fig, ax = plt.subplots()
        ax.imshow(im, interpolation='nearest', cmap=plt.cm.gray)
        ax.plot(p[:, 0], p[:, 1], linewidth=2)
        ax.set_title(title)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
    
    
    # TODO test that biggest contour is returned
    print('TWO CONTOURS')
    im2 = np.hstack((im, im))
    print(im2)
    
    outl = _outline_opencv(im2)
    outlsk = _outline_skimage(im2)
    print(len(outl) == len(outlsk))
    print('opencv:', outl)
    print('skimage:', outlsk)
    
    # TODO check that both outlines are the same modulo starting point
    for title, p in [('skimage-2', outlsk), ('opencv-2', outl)]:
        fig, ax = plt.subplots()
        ax.imshow(im2, interpolation='nearest', cmap=plt.cm.gray)
#        ax.plot(p[:, 0], p[:, 1], linewidth=2)
#        ax.set_title(title)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()
    
class Test(unittest.TestCase):
    def testPolygonArea(self):
        poly = [[4, 8],
               [3, 7],
               [2, 7],
               [1, 6],
               [1, 5],
               [1, 4],
               [1, 3],
               [1, 2],
               [2, 1],
               [3, 1],
               [4, 0],
               [5, 1],
               [6, 1],
               [7, 2],
               [7, 3],
               [8, 4],
               [7, 5],
               [7, 6],
               [6, 7],
               [5, 7]]
    
        assert polygonArea(poly) == 37.0
        
    def testPolygonCentroid(self):
        
        poly = [
            (30,50),
            (200,10),
            (250,50),
            (350,100),
            (200,180),
            (100,140),
            (10,200)
            ]
        
        assert_almost_equal(polygonCentroid(poly), (159.2903828197946, 98.88888888888))
        
        if cv:
            assert_almost_equal(polygonCentroid(poly), _polygonCentroidOpenCV(poly))
    
    @attr('slow')
    def testMappingCentroid(self):
        m = _getMapping()
        # for many polygon points like here, the two solutions (opencv using moments and
        # plain python) differ slightly after the 7th decimal
        # for our purposes this should still be accurate enough
        assert_almost_equal(m.centroid, 
                            [55.00295889563608, -99.21825084682715], # using cv.moments 
                            decimal=6)
        
def _getMapping():
    imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
    wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
    mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
    return mapping

def getResourcePath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/" + name)
    