# Copyright European Space Agency, 2013

from __future__ import absolute_import, print_function

import numpy as np
from numpy.core.umath_tests import matrix_multiply
from astropy.wcs.wcs import WCS
from auromat.utils import vectorLengths
from auromat.coordinates.transformations import euler_matrix
from auromat.coordinates.transform import cartesian_to_spherical,\
    spherical_to_cartesian
    
try:
    from numexpr import evaluate as ne
except ImportError:
    ne = None

def pix2world(wcsHeader, width, height, startX=0, startY=0, corner=True, ascartesian=False):
    """
    Calculate RA, Dec coordinates of a given pixel coordinate rectangle.
           
    ra, dec = pix2world(..)
    ra and dec are indexed as [y,x]
    
    Each array element contains the RA,Dec coords of the top left corner of the
    given pixel if corner==True, otherwise the coords of the pixel center. 
    If corner==True, an additional row and column exists at the bottom and right so that
    it is possible to get the bottom and right corner values for those pixels.
    
    :param dictionary wcsHeader: WCS header
    :param width: width of rectangle
    :param height: height of rectangle
    :param startX: x coordinate of rectangle, can be negative
    :param startY: y coordinate of rectangle, can be negative
    
    If ascartesian=False:
    
    :rtype: tuple(ra, dec) with arrays of shape (height+1,width+1) if corner==True, else (height,width)
    
    If ascartesian=True:
    
    :rtype: array of shape (height[+1],width[+1],3) with x,y,z order 
    """
    if corner:
        startX -= 0.5 # top left corner instead of pixel center
        startY -= 0.5
    x, y = np.meshgrid(np.arange(startX,startX+width+corner), np.arange(startY,startY+height+corner))
    
    # check if TAN projection and use our fast version, otherwise fall-back to astropy
    if wcsHeader['CTYPE1'] == 'RA---TAN' and wcsHeader['CTYPE2'] == 'DEC--TAN' and \
       wcsHeader['LATPOLE'] == 0.0:
        res = tan_pix2world(wcsHeader, x, y, 0, ascartesian=ascartesian)
        
    else:
        wcs = WCS(wcsHeader)
        ra, dec = wcs.all_pix2world(x, y, 0, ra_dec_order=True)
        if ascartesian:
            np.deg2rad(ra, ra)
            np.deg2rad(dec, dec)
            res = spherical_to_cartesian(None, dec, ra, astuple=False)
        else:
            res = ra, dec
    
    return res

def tan_pix2world(header, px, py, origin, ascartesian=False):
    """
    Fast reimplementation of astropy.wcs.wcs.wcs_pix2world with support for
    only the TAN projection. Speedup is about 2x.
    
    :rtype: tuple (ra,dec) in degrees, or cartesian coordinates in one array (h,w,3) 
            if ascartesian=True
    """
    assert origin in [0,1]
    assert px.shape == py.shape
    shape = px.shape
    px = px.ravel()
    py = py.ravel()
    
    assert header['CTYPE1'] == 'RA---TAN'
    assert header['CTYPE2'] == 'DEC--TAN'
    assert header['LATPOLE'] == 0.0
    lonpole = header['LONPOLE']
    
    ra_ref = header['CRVAL1'] # degrees
    dec_ref = header['CRVAL2']
    px_ref = header['CRPIX1']
    py_ref = header['CRPIX2']
    cd = np.array([[header['CD1_1'], header['CD1_2']],
                   [header['CD2_1'], header['CD2_2']]])
                
    # make pixel coordinates relative to reference point and put them in one array
    pxy = np.empty((len(px),2), float)
    pxy[:,0] = px
    pxy[:,0] -= px_ref
    pxy[:,1] = py
    pxy[:,1] -= py_ref
    if origin == 0:
        pxy += 1
    
    # projection plane coordinates
    xy = matrix_multiply(cd, pxy[...,np.newaxis]).reshape(pxy.shape) # [18% of execution time]
    
    del pxy
    
    # native spherical coordinates
    # spherical projection
    if ne:
        x = xy[:,0]
        y = xy[:,1]
        r = ne('sqrt(x*x+y*y)')
    else:
        r = vectorLengths(xy)
    
    if ne:
        lon = ne('arctan2(x, -y)') # [6% of execution time]
    else:
        lon = np.arctan2(xy[:,0], -xy[:,1])
    del xy
        
    #lat = np.arctan(180/(np.pi*r))
    
    # optimized:
    with np.errstate(divide='ignore'):
        np.reciprocal(r, r)
    np.multiply(180/np.pi, r, r)
    if ne:
        ne('arctan(r)', out=r)
    else:
        np.arctan(r, r)
    lat = r
    
    # celestial spherical coordinates
    # spherical rotation
    euler_z = ra_ref+90
    euler_x = 90-dec_ref
    #euler_z2 = lonpole-90
    euler_z2 = -(lonpole-90) # for some reason, this needs to be negative, contrary to paper
    rotmat = euler_matrix(np.deg2rad(euler_z), np.deg2rad(euler_x), np.deg2rad(euler_z2), 'rzxz')[:3,:3]
    
    lmn = spherical_to_cartesian(None, lat, lon, astuple=False) # [12% of execution time]
    lmnrot = matrix_multiply(rotmat, lmn[...,np.newaxis]).reshape(lmn.shape) # [17% of execution time]
    if ascartesian:
        return lmnrot.reshape(shape + (3,))
    
    dec, ra = cartesian_to_spherical(lmnrot[:,0], lmnrot[:,1], lmnrot[:,2], with_radius=False) # [15% of execution time]
    
    np.rad2deg(dec, dec)    
    np.rad2deg(ra, ra)
    # wrap at 360deg so that values are in [0,360]
    ra -= 360
    np.mod(ra, 360, ra) # [12% of execution time]
    
    ra = ra.reshape(shape)
    dec = dec.reshape(shape)
    
    return ra, dec
