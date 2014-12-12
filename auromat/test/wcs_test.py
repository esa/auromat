# Copyright European Space Agency, 2013

from __future__ import absolute_import, print_function

import os
import time

import numpy as np
from numpy.testing.utils import assert_almost_equal
from astropy.wcs.wcs import WCS
from auromat.fits import readHeader
from auromat.coordinates.wcs import tan_pix2world
    
def wcsTest():
    wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
    header = readHeader(wcsPath)
    
    # the pixel coordinates of which we want the world coordinates
    x, y = np.meshgrid(np.arange(0,4000), np.arange(0,2000))
    x = x.ravel()
    y = y.ravel()
    
    # first, calculate using astropy as reference
    wcs = WCS(header)
    t0 = time.time()
    ra_astropy, dec_astropy = wcs.wcs_pix2world(x, y, 0)
    print('astropy wcs:', time.time()-t0, 's')
    
    # now, check against our own implementation
    #tan_pix2world(header, x, y, 0) # warmup
    t0 = time.time()
    ra, dec = tan_pix2world(header, x, y, 0)
    print('own wcs:', time.time()-t0, 's')
    
    assert_almost_equal([ra, dec], [ra_astropy, dec_astropy])

def getResourcePath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/" + name)

if __name__ == '__main__':
    wcsTest()
    