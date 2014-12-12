# Copyright European Space Agency, 2013

"""
This module resamples mappings in a given resolution in the plate carree
projection, relative to either geodetic or MLat/MLT coordinates.
"""

from __future__ import division, print_function, absolute_import

from six.moves import map

import time

import numpy as np
import numpy.ma as ma

from distutils.version import LooseVersion
import astropy
import copy
from functools import partial
if LooseVersion(astropy.__version__) < '0.4':
    raise RuntimeError('astropy<0.4 is unsupported due to bugs in handling non-contiguous arrays')
from astropy.coordinates import Angle
import astropy.units as u

import scipy.interpolate

from auromat.utils import pointsInsidePolygon, extend
from auromat.mapping.mapping import BaseMapping, MappingCollection,\
    convertMappingToSM, convertSMMappingToGeo
from auromat.coordinates.transform import rotatePole      
from auromat.util.histogram import histogram2d
from auromat.coordinates import geodesic
from auromat.coordinates.geodesic import Location

def plateCarreeResolution(boundingBox, arcsecPerPx):
    """
    Approximates the latitude and longitude resolution of a plate carree
    projection from the given spherical resolution for the area given
    by the bounding box. The approximation is calculated for the bounding box center.
    
    :type boundingBox: auromat.mapping.mapping.BoundingBox
    :param arcsecPerPx: spherical resolution
    :rtype: tuple (latPxPerDeg, lonPxPerDeg)
    """
    degPerPx = (arcsecPerPx * u.arcsec).to(u.degree).value
    latPxPerDeg = 1/degPerPx
    
    latMiddle = (boundingBox.latNorth + boundingBox.latSouth)/2
    middleLeft = Location(latMiddle, boundingBox.lonWest)
    middleRight = Location(latMiddle, boundingBox.lonEast)
    lonMiddleDistance = geodesic.angularDistance(middleLeft, middleRight)
    px = lonMiddleDistance/degPerPx
    lonEast = boundingBox.lonEast
    if boundingBox.lonWest > lonEast:
        lons = lonEast+360 - boundingBox.lonWest
    else:
        lons = lonEast - boundingBox.lonWest
    lonMiddlePxPerDeg = px/lons
    
    return latPxPerDeg, lonMiddlePxPerDeg    

def resampleMLatMLT(mapping, **kw):
    """ Resamples a mapping such that MLat/MLT become regular grids.
    
    See :func:`resample` for parameters.    
    """
    sm = convertMappingToSM(mapping)
    smResampled = resample(sm, **kw)
    geo = convertSMMappingToGeo(smResampled)
    return geo

def resample(mappingOrCollection, pxPerDeg=25, arcsecPerPx=None, containsPole=None, method='mean'):
    """
    Returns a new mapping (or collection) where the colors and elevation
    are resampled into a regular latitude/longitude grid (plate carree projection)
    with y=latitude and x=longitude.
    
    If 'mean' binning is used as resampling method then take into account that
    this will lead to holes for low elevation angles if a high resampling resolution
    is used. This is because binning does not interpolate when there are zero data
    points in a given bin. Mask the mapping by elevation (e.g. 10deg) to get rid
    of the areas with holes.
    
    :param mappingOrCollection:
    :param None|number|tuple pxPerDeg: tuple (latPxPerDeg, lonPxPerDeg) or a number if both are the same
    :param None|number arcsecPerPx: spherical resolution, used to approximate pxPerDeg,
                                    has precedence over pxPerDeg
    :param None|bool containsPole: specify True|False to skip pole checking algorithm
    :param method: binning: 'mean'; interpolation: 'nearest', 'linear', 'cubic';
                   Note that linear and cubic take considerably longer and use much more memory while
                   they don't bring any benefit over 'nearest' if the goal is downsampling.
    :rtype: a subclass of BaseMapping or MappingCollection
    """
    def doResample(mapping, pxPerDeg, arcsecPerPx, containsPole):
        # trigger calculation of properties so that they are not included in the timing measurements
        mapping.lats
        mapping.latsCenter
        mapping.elevation
        mapping.img
        t0 = time.time()
        
        if containsPole is None:
            containsPole = mapping.containsPole
            
        if arcsecPerPx:
            pxPerDeg = plateCarreeResolution(mapping.boundingBox, arcsecPerPx)
        else:
            try:
                _, _ = pxPerDeg
            except TypeError:
                assert pxPerDeg is not None
                pxPerDeg = (pxPerDeg, pxPerDeg)
        print('pxPerDeg: ' + str(pxPerDeg))
        
        imgIsInt = mapping.img.dtype in [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64]
        
        # merge elevation with rgb array and extract channels afterwards
        merged = np.dstack((mapping.img.astype(np.float64).filled(np.nan),
                            mapping.elevation.filled(np.nan)))
        lats, lons, latsCenter, lonsCenter, merged = \
            _resample(mapping.latsCenter.filled(np.nan), mapping.lonsCenter.filled(np.nan), mapping.altitude,
                      merged,
                      lambda: mapping.outline, mapping.boundingBox,
                      pxPerDeg, mapping.containsDiscontinuity, containsPole,
                      method=method)
                
        img, elevation = np.dsplit(merged, [-1])
        if imgIsInt:
            with np.errstate(invalid='ignore'):
                img = np.round(img)
        img = np.require(ma.masked_invalid(img, copy=False), mapping.img.dtype)
        if mapping.img.ndim == 2:
            img = img.reshape(img.shape[0], img.shape[1])            
        elevation = elevation.reshape(elevation.shape[0],elevation.shape[1])
        elevation = ma.masked_invalid(elevation, copy=False)
    
        resampledMapping = mapping.createResampled(lats, lons, latsCenter, lonsCenter, elevation, img)
        
        print('resampling:', time.time()-t0, 's')
        
        return resampledMapping
    
    if isinstance(mappingOrCollection, BaseMapping):
        resampled = doResample(mappingOrCollection, pxPerDeg, arcsecPerPx, containsPole)
        
    elif isinstance(mappingOrCollection, MappingCollection):
        mappings = []
        for mapping in mappingOrCollection.mappings:
            mappings.append(doResample(mapping, pxPerDeg, arcsecPerPx, containsPole))
        resampled = MappingCollection(mappings, mayOverlap=mappingOrCollection.mayOverlap)
    
    else:
        raise ValueError('First argument must be a mapping or a mapping collection, but is: {}'.
                         format(type(mappingOrCollection)))
    
    return resampled   

def _resample(latsCenter, lonsCenter, altitude, data, outlineLatLonFn, boundingBox, pxPerDeg, 
              containsDiscontinuity=False, containsPole=False, method='mean'):
    """
    
    Note: Each channel is resampled on its own.

    :param latsCenter: (h,w)
    :param lonsCenter: (h,w)
    :param data: float data for each pixel center, (h,w,n) with n>0, or (h,w)
    :param pxPerDeg: tuple (latPxPerDeg, lonPxPerDeg)
    :rtype: tuple (lat, lon, latCenter, lonCenter, data)
    """   
    latMin = boundingBox.latSouth
    latMax = boundingBox.latNorth
    lonMin = boundingBox.lonWest
    lonMax = boundingBox.lonEast
        
    if containsPole:
        print('contains pole')
        
        outlineLatLon = outlineLatLonFn()
        outlineLats = outlineLatLon[:,0]
        outlineLons = outlineLatLon[:,1]
        
        # rotation of latitude/poles needs to happen in cartesian space based on earth as sphere
        # only a very small error will be introduced here as the outline form is not a segment of a sphere but an ellipsoid
        # -> as the outline is very small in size, this won't be a problem
        angle = 90
        axis = [1,0,0]
        outlineLats, outlineLons = rotatePole(np.deg2rad(outlineLats), np.deg2rad(outlineLons), 
                                              altitude, angle=angle, axis=axis)
        outlineLats, outlineLons = np.rad2deg(outlineLats), np.rad2deg(outlineLons)
        
        outlineLatLon[:,0] = outlineLats
        outlineLatLon[:,1] = outlineLons
        
        latMin, latMax = np.min(outlineLats), np.max(outlineLats)
        lonMin, lonMax = np.min(outlineLons), np.max(outlineLons)
                
        latsCenter_, lonsCenter_ = rotatePole(np.deg2rad(np.ravel(latsCenter)), np.deg2rad(np.ravel(lonsCenter)), 
                                              altitude, angle=angle, axis=axis)
        latsCenter = np.rad2deg(latsCenter_.reshape(latsCenter.shape))
        lonsCenter = np.rad2deg(lonsCenter_.reshape(lonsCenter.shape))
        
    elif containsDiscontinuity:
        print('contains discontinuity')
        
        outlineLatLon = outlineLatLonFn()
        outlineLats = outlineLatLon[:,0]
        outlineLons = outlineLatLon[:,1]

        # rotate longitudes out of 180° discontinuity; poles stay where they are
        # this introduces no additional error (e.g. due to ellipsoidal form)
        angle = 180
        outlineLons = Angle((outlineLons + angle) * u.deg).wrap_at(angle * u.deg).degree
        outlineLatLon[:,1] = outlineLons
            
        lonMin, lonMax = np.min(outlineLons), np.max(outlineLons)
        
        lonsCenter = Angle((lonsCenter + angle) * u.deg).wrap_at(angle * u.deg).degree

    # create regular plate carree grid within bounding box where y=lat and x=lon
    # Note: For a given pxPerDeg, all resamplings are aligned to the same global grid.
    latPxPerDeg, lonPxPerDeg = pxPerDeg
    assert latPxPerDeg > 0 and lonPxPerDeg > 0
    nLat, nLon, latMinInGrid, latMaxInGrid, lonMinInGrid, lonMaxInGrid =\
        fixedGrid(pxPerDeg, latMin, latMax, lonMin, lonMax)
    assert nLat > 1, 'nlat={}, latMax={}, latMin={}, pxperdeg={}'.format(nLat, latMaxInGrid, latMinInGrid, pxPerDeg)
    assert nLon > 1, 'nlon={}, lonMax={}, lonMin={}, pxperdeg={}'.format(nLon, lonMaxInGrid, lonMinInGrid, pxPerDeg)
    # the center coordinates are the ones which lie on the grid, corners are calculated
    latSpaceCenter, latStep = np.linspace(latMaxInGrid, latMinInGrid, num=nLat, retstep=True)
    lonSpaceCenter, lonStep = np.linspace(lonMinInGrid, lonMaxInGrid, num=nLon, retstep=True)
    
    # skip first and last coordinate center coordinate, otherwise we would have to calculate corner
    # coordinates outside the determined range, which could trigger certain edge cases
    latSpace = latSpaceCenter[:-1] + latStep/2
    lonSpace = lonSpaceCenter[:-1] + lonStep/2
    latSpaceCenter = latSpaceCenter[1:-1]
    lonSpaceCenter = lonSpaceCenter[1:-1]

    #latGrid, lonGrid = np.meshgrid(latSpace, lonSpace, indexing='ij') # 'indexing' not supported in np 1.6
    latGrid, lonGrid = np.dstack(np.meshgrid(latSpace, lonSpace)).T
    latGridCenter, lonGridCenter = np.dstack(np.meshgrid(latSpaceCenter, lonSpaceCenter)).T
    
    # do the actual resampling
    dataResampled = _resampleCenterData(latsCenter, lonsCenter, 
                                        data, latSpaceCenter, lonSpaceCenter, latStep, lonStep,
                                        method)

    # mask grid points which are outside the outline
    # This is needed as 'linear' and 'cubic' only mask points outside the *convex hull*,
    # which is not enough as we have concave forms. In those corner cases the data is interpolated.
    # With 'nearest', nothing is masked.
    # With 'mean', there is no inter/extrapolation, so we can skip the additional masking.
    if method != 'mean':
        # Based on the masked grid points the data is masked if any of its 4 corner points is masked.
        outlineLatLon = outlineLatLonFn()
        latLonGridFlat = np.asarray([np.ravel(latGrid), np.ravel(lonGrid)]).T
        isOutside = ~pointsInsidePolygon(latLonGridFlat, outlineLatLon).reshape(latGrid.shape)      
        mask = np.logical_or.reduce((isOutside[:-1,:-1], isOutside[1:,:-1], isOutside[:-1,1:], isOutside[1:,1:]))
        dataResampled[mask] = np.nan
        
    # rotate back coordinates if previously rotated
    if containsPole:
        angle = -90
        axis = [1,0,0]
        latGridFlat, lonGridFlat = rotatePole(np.deg2rad(latGrid.ravel()), np.deg2rad(lonGrid.ravel()), 
                                              altitude, angle=angle, axis=axis)
        latGrid = np.rad2deg(latGridFlat.reshape(latGrid.shape))
        lonGrid = np.rad2deg(lonGridFlat.reshape(latGrid.shape))
        
        latGridCenterFlat, lonGridCenterFlat = rotatePole(np.deg2rad(latGridCenter.ravel()), np.deg2rad(lonGridCenter.ravel()), 
                                                          altitude, angle=angle, axis=axis)
        latGridCenter = np.rad2deg(latGridCenterFlat.reshape(latGridCenter.shape))
        lonGridCenter = np.rad2deg(lonGridCenterFlat.reshape(latGridCenter.shape))
    elif containsDiscontinuity:
        angle = 180
        lonGrid = Angle((lonGrid + angle) * u.deg).wrap_at(angle * u.deg).degree
        lonGridCenter = Angle((lonGridCenter + angle) * u.deg).wrap_at(angle * u.deg).degree
        
    return latGrid, lonGrid, latGridCenter, lonGridCenter, dataResampled

def fixedGrid(pxPerDeg, latMin, latMax, lonMin, lonMax):
    """
    Aligns the given bounding box to a fixed plate carree grid as defined
    by `pxPerDeg`.
    
    :param lonMin,lonMax: must NOT contain the discontinuity
    """
    latPxPerDeg, lonPxPerDeg = pxPerDeg
    nLatAll = latPxPerDeg*180 + 1
    nLonAll = lonPxPerDeg*360 + 1
    latSpaceAll = np.linspace(-90, 90, int(round(nLatAll)))
    lonSpaceAll = np.linspace(-180, 180, int(round(nLonAll)))
    latMinInGrid = latSpaceAll[np.argmax(latSpaceAll > latMin) - 1]
    latMaxInGrid = latSpaceAll[np.argmax(latSpaceAll >= latMax)]
    lonMinInGrid = lonSpaceAll[np.argmax(lonSpaceAll > lonMin) - 1]
    lonMaxInGrid = lonSpaceAll[np.argmax(lonSpaceAll >= lonMax)]
    nLat = int(round(latPxPerDeg*(latMaxInGrid-latMinInGrid) + 1))
    nLon = int(round(lonPxPerDeg*(lonMaxInGrid-lonMinInGrid) + 1))
    return nLat, nLon, latMinInGrid, latMaxInGrid, lonMinInGrid, lonMaxInGrid

def _resampleCenterData(latsCenter, lonsCenter, centerData, latSpaceCenter, lonSpaceCenter, latStep, lonStep,
                        method):
    """
    :param method: binning: 'mean'; interpolation: 'nearest', 'linear', 'cubic';
                   Note that linear and cubic take considerably longer and use much more memory while
                   they don't bring any benefit over 'nearest' if the goal is downsampling.
    """
    if centerData.ndim == 2:
        scalarData = True
        centerData = centerData[...,None]
    else:
        scalarData = False
    
    # interpolate center data at grid point centers
    centerInLatsFlat, centerInLonsFlat = np.ravel(latsCenter), np.ravel(lonsCenter)
    centerNonNans = ~np.isnan(centerInLatsFlat)
    centerInLatsFlatFiltered = centerInLatsFlat[centerNonNans]
    centerInLonsFlatFiltered = centerInLonsFlat[centerNonNans]
    
    centerFlat = centerData.reshape(-1,centerData.shape[2])
    centerFlatFiltered = centerFlat[centerNonNans]
    
    if method in ['nearest', 'linear', 'cubic']:
        centerResampled = scipy.interpolate.griddata(
                              (centerInLatsFlatFiltered,centerInLonsFlatFiltered), centerFlatFiltered,
                              (latSpaceCenter[:,None], lonSpaceCenter[None,:]), method=method)
    
    elif method == 'mean':
        # this is about 20-50% slower than griddata's 'nearest'
        bins = (len(lonSpaceCenter), len(latSpaceCenter))
        # the bin egdes must be monotonically increasing, therefore we do that and flip it afterwards
        # so that latitudes are decreasing
        range_ = [[lonSpaceCenter[0]-lonStep/2, lonSpaceCenter[-1]+lonStep/2],
                  [latSpaceCenter[-1]+latStep/2, latSpaceCenter[0]-latStep/2]]    
        data = [centerFlatFiltered[:,d] for d in range(centerData.shape[2])]
                
        countAndData,_,_ = histogram2d(centerInLonsFlatFiltered, centerInLatsFlatFiltered, bins=bins, range=range_,
                                       weights=[None]+data)
        count = countAndData[0].T
        
        resampled = []
        for d in range(centerData.shape[2]):
            data = centerFlatFiltered[:,d]
            dataOut = countAndData[d+1].T
            dataOut[count==0.0] = np.nan
            with np.errstate(invalid='ignore'):
                dataOut /= count
            # flip so that latitudes are decreasing
            dataOut = np.flipud(dataOut)
            resampled.append(dataOut)
        centerResampled = np.dstack(resampled)
        
    elif method == 'median':
        raise NotImplementedError
        # https://stackoverflow.com/a/15488537
        # https://stackoverflow.com/a/10324083
        # http://fspaolo.net/code/bin-data.html       
        
    else:
        raise NotImplementedError
    
    assert centerResampled.shape == (len(latSpaceCenter), len(lonSpaceCenter), centerData.shape[2]),\
           str(centerResampled.shape) + ' != ' + str((len(latSpaceCenter), len(lonSpaceCenter), centerData.shape[2]))
    
    if scalarData:
        centerResampled = centerResampled.reshape(centerResampled.shape[0], centerResampled.shape[1])
        
    return centerResampled

def ResampleProvider(provider, **kw):
    """
    Wrap the given mapping provider by resampling every returned mapping.    
    
    :param provider: the provider to wrap
    
    See :func:`resample` for masking parameters.
    """
    resampleFn = partial(resample, **kw)
    class ResamplingProvider(object):          
        def get(self, *args, **kw):
            m = super(ResamplingProvider, self).get(*args, **kw)
            return resampleFn(m)
        
        def getById(self, *args, **kw):
            m = super(ResamplingProvider, self).getById(*args, **kw)
            return resampleFn(m)
            
        def getSequence(self, *args, **kw):
            m = super(ResamplingProvider, self).getSequence(*args, **kw)
            return map(resampleFn, m)
    
    provider = copy.copy(provider)  
    extend(provider, ResamplingProvider)
    return provider
