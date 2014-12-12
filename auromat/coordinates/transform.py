# Copyright European Space Agency, 2013

from __future__ import division, print_function

from auromat.coordinates.igrf import IGRF_DEFINED_UNTIL_YEAR

__doc__ = """
This module contains fast and memory-efficient algorithms to convert
coordinates between different reference frames or representations.

Note that some functions depend on the IGRF model whose parameters are
defined in the :mod:`igrf` module. In this version, parameters are defined
until the year {}.
""".format(IGRF_DEFINED_UNTIL_YEAR)

import time

from math import sin, atan2, cos, sqrt, pi, fmod, atan
import numpy as np
from numpy.core.umath_tests import matrix_multiply

try:
    from numexpr import evaluate as ne
except ImportError:
    ne = None

from astropy.time import Time
from astropy.coordinates import Angle
import astropy.units as u

from auromat.coordinates.geodesic import wgs84A, wgs84B, Location
from auromat.coordinates.igrf import calcG11, calcH11, calcG01
    
import auromat.coordinates.transformations

rotation_matrix = auromat.coordinates.transformations.rotation_matrix

def _spherical_to_cartesian_np(r, lat, lon, astuple=True):
    """
    As astropy.coordinates.distances.spherical_to_cartesian but more
    optimized. lat and lon must be arrays.
    """
    # using (3,..) instead of (...,3) as shape has better memory access performance
    # x, y, and z are then each a contiguous block of memory
    res = np.empty((3,) + lat.shape, lat.dtype)
    x = res[0]
    y = res[1]
    z = res[2]
    np.cos(lat, x)
    if r is not None:
        x *= r
    y[:] = x
    x *= np.cos(lon)
    y *= np.sin(lon)
    np.sin(lat, z)
    if r is not None:
        z *= r
        
    if astuple:
        return x,y,z
    else:
        res = np.rollaxis(res, 0, res.ndim)
        return res

def _spherical_to_cartesian_ne(r, lat, lon, astuple=True):
    # using (3,..) instead of (...,3) as shape has better memory access performance
    # x, y, and z are then each a contiguous block of memory
    res = np.empty((3,) + lat.shape, lat.dtype)
    x = res[0]
    y = res[1]
    z = res[2]
    if r is None:
        ne('cos(lat)', out=x)
    else:
        ne('r * cos(lat)', out=x)
    y[:] = x
    ne('x * cos(lon)', out=x)
    ne('y * sin(lon)', out=y)
    if r is None:
        ne('sin(lat)', out=z)
    else:
        ne('r * sin(lat)', out=z)
    if astuple:
        return x,y,z
    else:
        res = np.rollaxis(res, 0, res.ndim)
        return res

def spherical_to_cartesian(r, lat, lon, astuple=True):
    """
    Convert spherical to cartesian coordinates. Inputs must be arrays.
    
    Equivalent to `astropy.coordinates.distances.spherical_to_cartesian`
    for array inputs but uses less memory and is faster.
    
    :type r: ndarray or None (=1)
    :rtype: tuple (x,y,z) of ndarray's with shape as input
    """
    if ne:
        return _spherical_to_cartesian_ne(r, lat, lon, astuple)
    else:
        return _spherical_to_cartesian_np(r, lat, lon, astuple)

def _cartesian_to_spherical_np(x, y, z, with_radius=True):
    xsq = x*x
    ysq = y*y
    zsq = z*z

    if with_radius:
        r = xsq + ysq
        r += zsq
        np.sqrt(r, r)

    s = xsq
    s += ysq
    np.sqrt(s, s)

    lon = ysq
    np.arctan2(y, x, lon)
    
    lat = zsq
    np.arctan2(z, s, lat)

    if with_radius:
        return r, lat, lon
    else:
        return lat, lon

def _cartesian_to_spherical_ne(x, y, z, with_radius=True):
    if with_radius:
        xy = ne('x*x + y*y')
        r = ne('sqrt(xy + z*z)')
        lat = ne('arctan2(z, sqrt(xy))')
        lon = xy
        ne('arctan2(y, x)', out=lon)
        return r, lat, lon
    else:
        lat = ne('arctan2(z, sqrt(x*x + y*y))')
        lon = ne('arctan2(y, x)')
        return lat, lon

def cartesian_to_spherical(x, y, z, with_radius=True):
    """
    Convert cartesian to spherical coordinates. Inputs must be arrays.
    
    Equivalent to `astropy.coordinates.distances.cartesian_to_spherical`
    for array inputs but uses less memory and is faster.
    
    :rtype: tuple (r,lat,lon) or (lat,lon) of ndarray's with shape as input
    """
    if ne:
        return _cartesian_to_spherical_ne(x, y, z, with_radius)
    else:
        return _cartesian_to_spherical_np(x, y, z, with_radius)

def geodetic2Ecef(lat, lon, h, a=wgs84A, b=wgs84B):
    """
    Converts geodetic to Earth Centered, Earth Fixed coordinates.
    
    Parameters h, a, and b must be given in the same unit. 
    The values of the return tuple then also have this unit.
    
    :param lat: latitude(s) in radians
    :param lon: longitude(s) in radians
    :param h: height(s)
    :param a: equatorial axis of the ellipsoid of revolution
    :param b: polar axis of the ellipsoid of revolution
    :rtype: tuple (x,y,z)
    """
    lat, lon, h = np.asarray(lat), np.asarray(lon), np.asarray(h)
    e2 = (a*a - b*b) / (a*a) # first eccentricity squared
    n = a / np.sqrt(1 - e2*np.sin(lat)**2)
    latCos = np.cos(lat)
    nh = n+h
    x = nh*latCos*np.cos(lon)
    y = nh*latCos*np.sin(lon)
    z = (n*(1-e2)+h)*np.sin(lat)    
    return x,y,z

def geodetic2EcefZero(lat, lon, a=wgs84A, b=wgs84B):
    """
    Fast version of :func:`geodetic2Ecef` for `h=0`.
    
    :param lat: latitude(s) in radians
    :param lon: longitude(s) in radians
    :param a: equatorial axis of the ellipsoid of revolution
    :param b: polar axis of the ellipsoid of revolution
    :rtype: tuple (x,y,z)
    """
    lat, lon = np.asarray(lat), np.asarray(lon)
    e2 = (a*a - b*b) / (a*a) # first eccentricity squared
    n = a / np.sqrt(1 - e2*np.sin(lat)**2)
    latn = n*np.cos(lat)
    x = latn*np.cos(lon)
    y = latn*np.sin(lon)
    z = n*(1-e2)*np.sin(lat)    
    return x,y,z

def ecef2Geodetic(x, y, z, a=wgs84A, b=wgs84B):
    """
    Convert ECEF to geodetic coordinates.
    
    This function uses the Bowring algorithm from 1985.
    
    The accuracy is at least 11 decimals (in degrees).
    
    :param x,y,z: ECEF coordinates
    :param a: equatorial axis of the ellipsoid of revolution
    :param b: polar axis of the ellipsoid of revolution
    :rtype: tuple (lat,lon) in radians
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    if x.ndim > 0:
        return _ecef2GeodeticOptimized(x, y, z, a, b)
    e2 = (a*a - b*b) / (a*a) # first eccentricity squared
    d = (a*a - b*b) / b
    
    p2 = np.square(x) + np.square(y)
    p = np.sqrt(p2)
    r = np.sqrt(p2 + z*z)
    tu = b*z*(1 + d/r)/(a*p)
    tu2 = tu*tu
    cu3 = (1/np.sqrt(1 + tu2))**3
    su3 = cu3*tu2*tu
    tp = (z + d*su3)/(p - e2*a*cu3)
    lat = np.arctan(tp)
    
    lon = np.arctan2(y,x)
    
    return lat, lon

def _ecef2GeodeticOptimized_ne(x, y, z, a=wgs84A, b=wgs84B):
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

    e2 = (a*a - b*b) / (a*a) # first eccentricity squared 
    d = (a*a - b*b) / b  
    
    p2 = ne('x*x + y*y')
    p = ne('sqrt(p2)')
    tu = p2
    ne('b*z*(1 + d/sqrt(p2 + z*z))/(a*p)', out=tu)
    tu2 = ne('tu*tu')
    cu3 = ne('(1/sqrt(1 + tu2))**3')
    lat = p2
    ne('arctan((z + d*cu3*tu2*tu)/(p - e2*a*cu3))', out=lat)
    
    lon = p
    ne('arctan2(y,x)', out=lon)
    
    return lat, lon

def _ecef2GeodeticOptimized_np(x, y, z, a=wgs84A, b=wgs84B):
    """ array-only, memory-efficient version of ecef2Geodetic, around 10-30% faster """
    e2 = (a*a - b*b) / (a*a)
    d = (a*a - b*b) / b
    
    p2 = np.square(x)
    y2 = np.square(y)
    p2 += y2
    p = y2
    np.sqrt(p2, p)
    r = p2
    z2 = z*z
    r += z2
    np.sqrt(r, r)
    tu = z2
    np.divide(d, r, tu)
    tu += 1
    tu *= b
    tu *= z
    ap = a*p
    tu /= ap
    np.square(tu, r)
    tu2 = r
    cu3 = ap
    np.add(1, tu2, cu3)
    np.sqrt(cu3, cu3)
    np.reciprocal(cu3, cu3)
    cu3 **= 3 # don't replace by the faster cu3*cu3*cu3, it's less accurate!
    su3 = tu
    su3 *= cu3
    su3 *= tu2
    tp = tu2
    np.multiply(d, su3, tp)
    tp += z
    pm = p
    cu3m = cu3
    cu3m *= e2*a
    pm -= cu3m
    tp /= pm
    np.arctan(tp, tp)
    lat = tp
    
    lon = cu3
    np.arctan2(y,x, lon)
    
    return lat, lon

_ecef2GeodeticOptimized = _ecef2GeodeticOptimized_ne if ne else _ecef2GeodeticOptimized_np

def rotatePole(lats, lons, altitude, angle=90, axis=[1,0,0]):
    """
    Rotates the given geodetic lat/lon coordinates around the origin.
    
    :param lats, lons: shape (n,) in radians
    :param altitude: in km
    :param angle: degrees
    :param axis: [1, 0, 0], [0, 1, 0], or [0, 0, 1] for x y z axis
    :rtype: tuple (lats, lons) in radians
    """
    assert lats.ndim == 1 and lons.ndim == 1
    assert len(axis) == 3    

    x,y,z = geodetic2Ecef(lats, lons, altitude, wgs84A, wgs84B)
    xyz = np.asarray([x,y,z]).T
    
    alpha = np.deg2rad(angle)
    rot = rotation_matrix(alpha, axis)[:3,:3]
    
    xyzRot = matrix_multiply(rot,xyz[...,np.newaxis]).reshape(xyz.shape)
    lats, lons = ecef2Geodetic(xyzRot[:,0], xyzRot[:,1], xyzRot[:,2], wgs84A, wgs84B)
    return lats, lons

def j2000ToLatLon(j2000Vecs, time_):
    """
    Convert cartesian J2000 coordinates to geodetic coordinates.
    
    :param j2000Vecs: shape (n,3)
    :param datetime time_:
    :rtype: tuple (latitudes, longitudes) in degrees
    """
    t0 = time.time()
    j2000Vecs = np.asarray(j2000Vecs)
    geoX, geoY, geoZ = j2000_to_geo(time_, j2000Vecs).T
    print('convert J2000-GEO:', time.time()-t0, 's')
        
    t0 = time.time()
    lat, lon = ecef2Geodetic(geoX, geoY, geoZ, wgs84A, wgs84B)
    np.rad2deg(lat, lat)
    np.rad2deg(lon, lon)
    print('convert GEO-LatLon:', time.time()-t0, 's')
    
    return lat, lon

def latLonToJ2000(lat, lon, h, time_):
    """
    Convert geodetic coordinates to cartesian J2000 coordinates.
    
    :param lat: scalar or 1d-array, degrees
    :param lon: scalar or 1d-array, degrees
    :param h: scalar or 1d-array
    :param datetime time_:
    :rtype: tuple (latitudes, longitudes) in degrees
    """
    t0 = time.time()
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    geoX, geoY, geoZ = geodetic2Ecef(lat, lon, h)
    geoVecs = np.asarray([geoX, geoY, geoZ]).T
    isScalar = geoVecs.ndim == 1
    if isScalar:
        geoVecs = np.array([geoVecs])
    print('convert LatLon-GEO:', time.time()-t0, 's')
        
    t0 = time.time()
    j2000 = geo_to_j2000(time_, geoVecs).T
    if isScalar:
        j2000 = j2000[:,0]
    print('convert GEO-J2000:', time.time()-t0, 's')
    
    return j2000

def smLonToMLT(smlons, out=None):
    """
    Convert solar magnetic longitudes to magnetic local time.
    
    :param smlons: in degrees [-180,180]
    :return: magnetic local time [0,24]
    """
    if out is not None:
        np.multiply(smlons, 24/360, out)
        mlt = out
    else:
        mlt = smlons*(24/360)
    mlt += 12
    return mlt

def mltToSmLon(mlt, out=None):
    """
    Convert magnetic local time to solar magnetic longitudes.
    
    :param mlt: in hours [0,24]
    :return: solar magnetic longitudes in degrees [-180,180]
    """
    if out is not None:
        np.subtract(mlt, 12, out)
        smlon = out
    else:
        smlon = mlt-12
    smlon /= (24/360)
    return smlon

def j2000ToMLatMLT(j2000Vecs, time_):
    """
    Convert cartesian J2000 coordinates to MLat/MLT coordinates using the
    IGRF model.
    
    MLat = geomagnetic latitude
    
    MLT = magnetic local time
    
    :param j2000Vecs: shape (n,3)
    :param datetime time_:
    :rtype: tuple (mlat, mlt) in (degrees,hours)
    """
    t0 = time.time()
    j2000Vecs = np.asarray(j2000Vecs)
    smX, smY, smZ = j2000_to_sm(time_, j2000Vecs).T
    print('convert J2000-SM:', time.time()-t0, 's')
    
    t0 = time.time()
    _, smlat, smlon = cartesian_to_spherical(smX, smY, smZ)
    np.rad2deg(smlat, smlat)
    np.rad2deg(smlon, smlon)
    mlat = smlat
    smLonToMLT(smlon, smlon)
    mlt = smlon
    print('convert SM-MLat/MLT:', time.time()-t0, 's')
    
    return mlat, mlt

def geoToMLatMLT(geoVecs, time_):
    """
    Convert ECEF coordinates to MLat/MLT coordinates using the
    IGRF model.
    
    MLat = geomagnetic latitude
    
    MLT = magnetic local time
    
    :param geoVecs: shape (n,3)
    :param datetime time_:
    :rtype: tuple (mlat, mlt) in (degrees,hours)
    """    
    t0 = time.time()
    geoVecs = np.asarray(geoVecs)
    smX, smY, smZ = geo_to_sm(time_, geoVecs).T
    print('convert GEO-SM:', time.time()-t0, 's')
    
    t0 = time.time()
    _, smlat, smlon = cartesian_to_spherical(smX, smY, smZ)
    np.rad2deg(smlat, smlat)
    np.rad2deg(smlon, smlon)
    mlat = smlat
    smLonToMLT(smlon, smlon)
    mlt = smlon
    print('convert SM-MLat/MLT:', time.time()-t0, 's')
    
    return mlat, mlt
    
def smToLatLon(smlats, smlons, time_):
    """
    Convert solar magnetic to geodetic coordinates using the
    IGRF model.
    
    :param smlats: in degrees [-90,90]
    :param smlons: in degrees [-180,180]
    :param time_:
    :rtype: tuple (latitudes, longitudes) in degrees
    """
    t0 = time.time()
    smlats, smlons = np.deg2rad(smlats), np.deg2rad(smlons)
    smX, smY, smZ = spherical_to_cartesian(1, smlats.ravel(), smlons.ravel())
    smVecs = np.array([smX,smY,smZ]).T
    geoX, geoY, geoZ = sm_to_geo(time_, smVecs).T
    print('convert SM-GEO:', time.time()-t0, 's')
    
    t0 = time.time()
    lats, lons = ecef2Geodetic(geoX, geoY, geoZ)
    np.rad2deg(lats, lats)
    np.rad2deg(lons, lons)
    lats = lats.reshape(smlats.shape)
    lons = lons.reshape(smlons.shape)
    print('convert GEO-LatLon:', time.time()-t0, 's')
    return lats, lons

# The following functions are adapted from cxform_manual.c of NASA's cxform library.
# The below versions are much faster as they pre-calculate and multiply multiple rotation matrices
# before converting coordinates.

# the following axis directions for gohlke's rotation_matrix match it with hapgood_matrix defined in cxform
X = [-1,0,0]
Y = [0,1,0]
Z = [0,0,-1]


def mag_lon(et):
    """
    Longitude of Earth's magnetic pole in radians
    """
    fracYearIndex = (et+3155803200.0)/157788000.0
    fracYear = fmod(fracYearIndex, 1.0)

    g11 = calcG11(fracYearIndex, fracYear)
    h11 = calcH11(fracYearIndex, fracYear)

    lambda0 = atan2(h11, g11) + pi
    return lambda0

def mag_lat(et):
    """
    Latitude of Earth's magnetic pole in radians
    """
    fracYearIndex = (et+3155803200.0)/157788000.0
    fracYear = fmod(fracYearIndex, 1.0)

    g01 = calcG01(fracYearIndex, fracYear)
    g11 = calcG11(fracYearIndex, fracYear)
    h11 = calcH11(fracYearIndex, fracYear)
    lambda0 = mag_lon(et)
  
    phi0 = pi/2 - atan((g11*cos(lambda0) + h11*sin(lambda0))/g01)
    return phi0

def date2es(date):
    """
    Converts UTC to ephemeris seconds.
    """
    jd = Time(date, scale='utc').jd
    return (jd - 2451545) * 86400
    # cxform used the following which limits precision unnecessarily to 1sec:
#    return long(round((jd - 2451545) * 86400))

def T0(et):
    """
    Julian Centuries from a certain time to 1 Jan 2000 12:00
    """
    return (et / 86400.0)/36525.0

def H(et):
    """
    time, in hours, since preceding UT midnight
    """
    jd    = (et / 86400.0) - 0.5
    dfrac = jd - int(jd)
    hh    = dfrac * 24.0

    if (hh < 0.0):
        hh += 24.0

    return hh

def lambda0(et):
    """
    Sun's ecliptic longitude in degrees
    """
    M = 357.528 + 35999.050 * T0(et)
    lambd = 280.460 + 36000.772 * T0(et)
    lambda0 = lambd + (1.915 - 0.0048 * T0(et)) * sin(np.deg2rad(M)) + 0.020 * sin(np.deg2rad(2 * M))
    return lambda0

def epsilon(et):
    """
    The obliquity of the ecliptic in degrees
    """
    return 23.439 - 0.013 * T0(et)

def mat_P(et):
    """
    J2000 to GEI matrix
    """
    t0 = T0(et)
    
    mat = rotation_matrix(np.deg2rad(-1.0*(0.64062 * t0  +  0.00030 * t0*t0)), Z)

    mat_tmp = rotation_matrix(np.deg2rad(0.55675 * t0  -  0.00012 * t0*t0), Y)
    mat = np.dot(mat, mat_tmp)
        
    mat_tmp = rotation_matrix(np.deg2rad(-1.0*(0.64062 * t0  +  0.00008 * t0*t0)), Z)
    mat = np.dot(mat, mat_tmp)
    return mat[:3,:3]

def mat_T1(et):
    """
    GEI to GEO matrix
    """
    theta = 100.461 + 36000.770 * T0(et) + 360.0*(H(et)/24.0)

    mat = rotation_matrix(np.deg2rad(theta), Z)
    return mat[:3,:3]

def mat_T2(et):
    """
    GEI to GSE matrix
    """
    mat = rotation_matrix(np.deg2rad(lambda0(et)), Z)
    mat_tmp = rotation_matrix(np.deg2rad(epsilon(et)), X)
    mat = np.dot(mat, mat_tmp)
    return mat[:3,:3]

def vec_Qe(et):
    """
    don't ask. [sic]
    """
    lat = mag_lat(et)
    lon = mag_lon(et)
    
    cos_lat = cos(lat)
    sin_lat = sin(lat)
    cos_lon = cos(lon)
    sin_lon = sin(lon)

    Qg = [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]

    mat = mat_T2(et)
    mat_tmp = mat_T1(et).T

    mat = np.dot(mat, mat_tmp)
    Qe = np.dot(mat, Qg)
    return Qe

def mat_T3(et):
    """
    GSE to GSM matrix
    """
    Qe = vec_Qe(et)
    psi = atan2(np.deg2rad(Qe[1]), np.deg2rad(Qe[2]))
    mat = rotation_matrix(-psi, X)
    return mat[:3,:3]

def mat_T4(et):
    """
    GSM to SM matrix
    """
    Qe = vec_Qe(et)
    mu = atan2(np.deg2rad(Qe[0]), np.deg2rad(sqrt(Qe[1]*Qe[1] + Qe[2]*Qe[2])))
    mat = rotation_matrix(-mu, Y)
    return mat[:3,:3]

def mat_T5(et):
    """
    GEO to MAG matrix
    """
    mat = rotation_matrix(mag_lat(et) - np.deg2rad(90.0), Y)
    mat_tmp = rotation_matrix(mag_lon(et), Z)
    mat = np.dot(mat, mat_tmp)
    return mat[:3,:3]

# def hapgood_matrix(theta, axis):
#     if axis==[1,0,0]:
#         axis = 0
#     elif axis==[0,1,0]:
#         axis = 1
#     elif axis==[0,0,1]:
#         axis = 2
# 
#     sin_theta = np.sin(theta)
#     cos_theta = np.cos(theta)
# 
#     t1 = (axis+1) % 3
#     t2 = (axis+2) % 3
#     if (t1 > t2):
#         tmp = t1
#         t1  = t2
#         t2  = tmp
#     
#     mat = np.zeros((3,3))
# 
#     mat[axis,axis] = 1.0
#     mat[t1,t1]     = cos_theta
#     mat[t2,t2]     = cos_theta
# 
#     mat[t1,t2]     =  sin_theta
#     mat[t2,t1]     = -sin_theta
#     return mat

# Note that the cxform functions of this module (including rotation_matrix) were 
# as a test fully implemented using sympy (arbitrary precision floats) to check 
# whether the repeated matrix multiplications (see below) had a significant effect
# on accuracy due to float rounding errors. The result was that the maximum error
# of any element in the rotation matrices was in the order of e-14.

def mat_j2000_to_geo(et):
    # j2000_twixt_gei -> gei_twixt_geo
    mat = np.dot(mat_T1(et), mat_P(et))
    return mat

def mat_j2000_to_sm(et):
    # j2000_twixt_gei -> gei_twixt_gse -> gse_twixt_gsm -> gsm_twixt_sm
    mat = mat_T4(et).dot(mat_T3(et)).dot(mat_T2(et)).dot(mat_P(et))
    return mat

def mat_geo_to_sm(et):
    # gei_twixt_geo (reverse) -> gei_twixt_gse -> gse_twixt_gsm -> gsm_twixt_sm
    mat = mat_T4(et).dot(mat_T3(et)).dot(mat_T2(et)).dot(mat_T1(et).T)
    return mat

def j2000_to_geo(date, vecsJ2000):
    return x_to_y(mat_j2000_to_geo, date, vecsJ2000)

def geo_to_j2000(date, vecsGeo):
    return x_to_y(mat_j2000_to_geo, date, vecsGeo, reverse=True)

def j2000_to_sm(date, vecsJ2000):
    return x_to_y(mat_j2000_to_sm, date, vecsJ2000)

def geo_to_sm(date, vecsGEO):
    return x_to_y(mat_geo_to_sm, date, vecsGEO)

def sm_to_geo(date, vecsSM):
    return x_to_y(mat_geo_to_sm, date, vecsSM, reverse=True)

def gei_to_geo(date, vecsGEI):
    return x_to_y(mat_T1, date, vecsGEI)

def geo_to_gei(date, vecsGEO):
    return x_to_y(mat_T1, date, vecsGEO, reverse=True)

def gei_to_gse(date, vecsGEI):
    return x_to_y(mat_T2, date, vecsGEI)

def gse_to_gsm(date, vecsGSE):
    return x_to_y(mat_T3, date, vecsGSE)

def gsm_to_sm(date, vecsGSM):
    return x_to_y(mat_T4, date, vecsGSM)

def x_to_y(matFn, date, vecs, reverse=False):
    vecs = np.asarray(vecs)
    assert vecs.ndim == 2 
    assert vecs.shape[1] == 3
        
    et = date2es(date)
    mat = matFn(et)
    if reverse:
        mat = mat.T
    vecsOut = matrix_multiply(mat, vecs[...,np.newaxis]).reshape(vecs.shape)
    return vecsOut

def northGeomagneticPoleLocation(date):
    """
    Calculate the approximate position of the north geomagnetic pole for the
    given date using the IGRF model.
    
    :param datetime date:
    :rtype: named (latitude, longitude) tuple, in degrees
    """
    # TODO are these geocentric or geodetic coordinates??
    et = date2es(date)
    lat, lon = mag_lat(et), mag_lon(et)
    lat = np.rad2deg(lat)
    lon = Angle(lon * u.rad).wrap_at(180 * u.deg).degree
    return Location(lat, lon)

__all__ = map(lambda f: f.__name__,
              [spherical_to_cartesian, cartesian_to_spherical,
               geodetic2Ecef, geodetic2EcefZero, ecef2Geodetic,
               j2000ToLatLon, latLonToJ2000,
               smLonToMLT, mltToSmLon, j2000ToMLatMLT, geoToMLatMLT, smToLatLon,
               northGeomagneticPoleLocation])
    
if __name__ == '__main__':
    from datetime import datetime
    d = datetime(2010,1,1)
    print('Geomagnetic pole at ' + str(d) + ':')
    print(northGeomagneticPoleLocation(d))
    