# Copyright European Space Agency, 2013

"""
This module provides core classes to handle georeferenced images.
Most classes are base classes which are inherited from in other modules
of this package.
"""

from __future__ import division, print_function, absolute_import

from six.moves import range, map
from six import add_metaclass

import logging
import time
from abc import ABCMeta, abstractproperty, abstractmethod
from collections import namedtuple
import copy

import numpy as np
import numpy.ma as ma

import astropy.units as u
from astropy import constants as const
from astropy.coordinates.angles import Angle

from auromat.coordinates.transform import j2000ToLatLon, rotatePole, geoToMLatMLT,\
    geodetic2Ecef, smToLatLon, mltToSmLon, j2000ToMLatMLT
from auromat.coordinates.geodesic import wgs84A, wgs84B, containsOrCrossesPole
import auromat.utils
from auromat.coordinates import geodesic
from auromat.coordinates.geodesic import Location
from auromat.coordinates.intersection import sphereLineIntersection,\
    ellipsoidLineIntersection
from auromat.util.image import loadImage
from auromat.utils import outline, convexHull, polygonCentroid, extend
from auromat.util.decorators import lazy_property, inherit_docs
import warnings

Size = namedtuple('Size', ['width','height'])
PixelScales = namedtuple('PixelScales', ['width', 'height', 'diagonal'])
PixelScale = namedtuple('PixelScale', ['mean', 'median', 'min', 'max'])

class BoundingBox(object):
    """
    Describes a geographical bounding box that can span across
    the discontinuity.
    """
    def __init__(self, latSouth, lonWest, latNorth, lonEast):
        """
        
        :param number latSouth: in degrees [-90,90]
        :param number lonWest: in degrees [-180,180]
        :param number latNorth: in degrees [-90,90]
        :param number lonEast: in degrees [-180,180]
        """
        assert -180 <= lonWest <= 180, 'Longitude: ' + str(lonWest)
        assert -180 <= lonEast <= 180, 'Longitude: ' + str(lonEast)
        assert -90 <= latSouth <= 90, 'Latitude: ' + str(latSouth)
        assert -90 <= latNorth <= 90, 'Latitude: ' + str(latNorth)
        
        self._latSouth = latSouth
        self._lonWest = lonWest
        self._latNorth = latNorth
        self._lonEast = lonEast
        
    @property
    def latSouth(self):
        return self._latSouth
    
    @property
    def lonWest(self):
        return self._lonWest
    
    @property
    def latNorth(self):
        return self._latNorth
    
    @property
    def lonEast(self):
        return self._lonEast
    
    @property
    def topLeft(self):
        """
        North west corner of the bounding box.
        
        :rtype: auromat.coordinates.geodesic.Location
        """
        return Location(self.latNorth, self.lonWest)
    
    @property
    def bottomLeft(self):
        """
        South west corner of the bounding box.
        
        :rtype: auromat.coordinates.geodesic.Location
        """
        return Location(self.latSouth, self.lonWest)
    
    @property
    def topRight(self):
        """
        North east corner of the bounding box.
        
        :rtype: auromat.coordinates.geodesic.Location
        """
        return Location(self.latNorth, self.lonEast)
    
    @property
    def bottomRight(self):
        """
        South east corner of the bounding box.
        
        :rtype: auromat.coordinates.geodesic.Location
        """
        return Location(self.latSouth, self.lonEast)
    
    @lazy_property
    def _minSphericalRectangle(self):
        """
        based on https://stackoverflow.com/a/13503064
        
        useful as stereographic projection parameters for drawing
        
        Note that this currently only works for bounding boxes spanning
        less than 180 degrees in longitude. For greater longitude ranges
        the returned center is still correct but the size is wrong.
        
        :rtype: tuple (center, size) with size a named tuple (width, height) with values in km
                and center a named Location tuple
        """
        if self.containsPole:
            if self.latNorth == 90: # north pole
                center = Location(90, 0)
                width = geodesic.distance(center, Location(self.latSouth, 0))*2
            else: # south pole
                center = Location(-90, 0)
                width = geodesic.distance(center, Location(self.latNorth, 0))*2              
            height = width            
        else:
            lonWest = self.lonWest
            lonEast = self.lonEast
            if lonWest > lonEast:
                lonEast += 360
            lonc = (lonWest+lonEast)/2
            lonc = Angle(lonc * u.deg).wrap_at(180 * u.deg).degree
            
            if lonEast - lonWest > 180:
                warnings.warn('The bounding box spans more than 180deg in longitude. '
                              'The returned size of the minimum spherical rectangle will be incorrect.')
            
            # TODO only works when longitude range is less than 180 deg
            # -> this is because the geodesic distance is not oriented and always
            #    returns the shortest distance
            width = geodesic.distance(self.bottomLeft, self.bottomRight)
            width2 =  geodesic.distance(self.topLeft, self.topRight)
            if width2 > width:
                # southern hemisphere
                width = width2
                bottomCenter = geodesic.intermediate(self.bottomLeft, self.bottomRight, 0.5)
                topDataCenter = Location(self.latNorth, lonc)
                height = geodesic.distance(topDataCenter, bottomCenter)
                center = geodesic.intermediate(topDataCenter, bottomCenter, 0.5)            
            else:
                # northern hemisphere
                topCenter = geodesic.intermediate(self.topLeft, self.topRight, 0.5)
                bottomDataCenter = Location(self.latSouth, lonc)
                height = geodesic.distance(bottomDataCenter, topCenter)
                center = geodesic.intermediate(bottomDataCenter, topCenter, 0.5)
        
        return center, Size(width/1000, height/1000)
    
    @lazy_property
    def center(self):
        """
        Center of the minimum spherical rectangle that fits the
        bounding box.
        
        :rtype: auromat.coordinates.geodesic.Location
        """
        center, _ = self._minSphericalRectangle
        return center
    
    @lazy_property
    def size(self):
        """
        Width and height in km of the minimum spherical rectangle that fits the
        bounding box.
        
        Note that the size is only correct if the bounding box spans
        less than 180 degrees in longitude.
        
        :rtype: Size
        """
        _, size = self._minSphericalRectangle
        return size
    
    @property
    def containsDiscontinuity(self):
        """
        Return whether the bounding box contains the 180 degree discontinuity.
        
        :rtype: bool
        """
        return self.lonWest > self.lonEast or self.containsPole
    
    @property
    def containsPole(self):
        """
        Return whether the bounding box contains the north and/or south pole.
        
        :rtype: bool
        """
        return self.lonWest == -180 and self.lonEast == 180 and \
               (self.latNorth == 90 or self.latSouth == -90)
    
    @staticmethod
    def minimumBoundingBox(latLons):
        """
        Return the smallest bounding box that contains all given
        coordinates.
        
        :param latLons: iterable of (lat,lon) pairs
        :rtype: BoundingBox
        """
        bbs = [BoundingBox(lat, lon, lat, lon) for [lat,lon] in latLons]        
        minBB = BoundingBox.mergedBoundingBoxes(bbs)               
        return minBB
    
    @staticmethod
    def mergedBoundingBoxes(boundingBoxes):
        """
        Return the smallest bounding box that contains all given
        bounding boxes.
        
        :param boundingBoxes: iterable of BoundingBox instances
        :rtype: BoundingBox
        """
        boundingBoxes = list(boundingBoxes)
        lats = [bb.latSouth for bb in boundingBoxes] + [bb.latNorth for bb in boundingBoxes]       
        lons = [(bb.lonWest,bb.lonEast) for bb in boundingBoxes]
        
        latSouth = np.min(lats)
        latNorth = np.max(lats)
        lonWest, lonEast = BoundingBox._minimumBoundingBoxLons(lons)
        
        return BoundingBox(latSouth, lonWest, latNorth, lonEast)
    
    @staticmethod
    def _minimumBoundingBoxLons(lons):       
        lons = np.asarray(lons)
        xs = np.sort(lons.ravel())
        assert len(xs) % 2 == 0
        xs = np.concatenate((xs,[xs[0]+360]))
        
        lonsUnwrapped = np.rad2deg(np.unwrap(np.deg2rad(lons)))
    
        coversInterval = np.zeros(len(xs)-1, dtype=bool)
        for i in range(1, len(xs)):
            for bb in lonsUnwrapped:
                if bb[0] <= xs[i-1] and bb[1] >= xs[i]:
                    coversInterval[i-1] = True
                    break
        
        intervalLengths = xs[1:] - xs[:-1]
        gapLengths = ma.masked_array(intervalLengths, coversInterval)
        biggestGapIdx = np.argmax(gapLengths) # ignoring other possible maximums
        
        lonWest, lonEast = xs[biggestGapIdx+1], xs[biggestGapIdx]
        lonWest = Angle(lonWest * u.deg).wrap_at(180 * u.deg).degree
        lonEast = Angle(lonEast * u.deg).wrap_at(180 * u.deg).degree
                
        return lonWest, lonEast
    
    def __eq__(self, obj):
        return isinstance(obj, BoundingBox) and \
           self.latNorth == obj.latNorth and self.latSouth == obj.latSouth and \
           self.lonWest == obj.lonWest and self.lonEast == obj.lonEast
           
    def _ne__(self, obj):
        return not self == obj
    
    def __repr__(self):
        return 'BoundingBox(latSouth={0}, lonWest={1}, latNorth={2}, lonEast={3})'.format(
                       self.latSouth, self.lonWest, self.latNorth, self.lonEast)

MappingProperties = namedtuple('MappingProperties', 
                               'altitude cameraPosGCRS boundingBox photoTime '
                               'centroid cameraFootpoint identifier')

@add_metaclass(ABCMeta)
class BaseMapping(object):    
    """
    Base class for all mapping objects.
    Describes a georeferenced image for a given altitude.
    
    The following assertions always hold:
    If a pixel in img is defined, then it is guaranteed that the corresponding
    center and corner coordinates are defined, as well as the elevation.
    If a pixel in img is NOT defined, then the corresponding center coordinate and
    elevation are also not defined; one of its corner coordinates may be defined
    in case a neighbouring pixel is defined, otherwise it is not defined.
    
    More formally:
    
    - If lats[y,x] is not masked, then lons[y,x] is not masked, and vice versa.
    - If latsCenter[y,x] is not masked, then lonsCenter[y,x] is not masked, and vice versa.
    - If lats[y,x] is not masked, then at least one of latsCenter[y,x], latsCenter[y,x-1],
      latsCenter[y-1,x-1] or latsCenter[y-1,x] is not masked.
    - If latsCenter[y,x] is not masked, then lats[y,x], lats[y+1,x],
      lats[y+1,x+1] and lats[y,x+1] are not masked.
    - If img[y,x] is not masked, then latsCenter[y,x] is not masked, and vice versa.
    - If elevation[y,x] is not masked, then latsCenter[y,x] is not masked, and vice versa.
    """
    
    def __init__(self, altitude, cameraPosGCRS, photoTime, identifier, metadata=None):
        """
            
        :param number altitude: the altitude in km onto which the image is mapped                    
        :param array-like cameraPosGCRS: [x,y,z] camera position in GCRS coordinates, in km
        :param datetime.datetime photoTime: date the photo was taken
        :param str identifier: a string which uniquely names this mapping (e.g. ISS030-E-84614);
                               it must be usable for parts of a filename
        :param dict metadata: a flat dictionary of informational metadata that is used when
                              exporting a mapping
        """
        assert altitude >= 0
        cameraPosGCRS = np.asarray(cameraPosGCRS)
        assert cameraPosGCRS.shape == (3,)

        self._altitude = altitude
        self._cameraPosGCRS = cameraPosGCRS
        self._photoTime = photoTime
        self._identifier = identifier
        self._metadata = metadata
        
        # cached values of derived attributes
        self._centroid = None
        self._outlines = None
        self._boundingBox = None
        self._pixelScales = None
        self._mlatmlt = None
        self._mlatmltCenter = None
        
    @property
    def properties(self):
        """
        Return information about the mapping as a named tuple.
        
        :rtype: MappingProperties
        """
        return MappingProperties(identifier=self.identifier,
                                 altitude=self.altitude,
                                 cameraPosGCRS=self.cameraPosGCRS,
                                 boundingBox=self.boundingBox,
                                 photoTime=self.photoTime,
                                 centroid=self.centroid,
                                 cameraFootpoint=self.cameraFootpoint)
    
    def checkGuarantees(self):
        """
        Checks if the guarantees defined for coordinate and data arrays are fulfilled.
        See the docstring of this class.
        
        Note: This method is meant only for testing purposes.
        """
        lats, lons = self.lats, self.lons
        latsCenter, lonsCenter = self.latsCenter, self.lonsCenter
        mlat, mlt = self.mLatMlt
        mlatCenter, mltCenter = self.mLatMltCenter
        img = self.img
        elevation = self.elevation
        
        # we use masked arrays, so there must be no NaN's in the unmasked parts
        assert not np.any(np.isnan(lats))
        assert not np.any(np.isnan(latsCenter))
        assert not np.any(np.isnan(mlat))
        assert not np.any(np.isnan(elevation))     
        
        # If lats[y,x] is not nan, then lons[y,x] is not nan, and vice versa.
        assert np.all(np.logical_xor(ma.getmaskarray(lats), ~ma.getmaskarray(lons)))
        
        # If latsCenter[y,x] is not nan, then lonsCenter[y,x] is not nan, and vice versa.
        assert np.all(np.logical_xor(ma.getmaskarray(latsCenter), ~ma.getmaskarray(lonsCenter)))
        
        # If lats[y,x] is not nan, then at least one of latsCenter[y,x], latsCenter[y,x-1],
        # latsCenter[y-1,x-1] or latsCenter[y-1,x] is not nan.
        latsCenterNotNanPadded = np.zeros((latsCenter.shape[0]+2, latsCenter.shape[1]+2), bool)
        latsCenterNotNanPadded[1:-1,1:-1] = ~ma.getmaskarray(latsCenter)
        assert np.all(np.logical_or.reduce((ma.getmaskarray(lats),
                                            latsCenterNotNanPadded[1:,1:],
                                            latsCenterNotNanPadded[1:,:-1],
                                            latsCenterNotNanPadded[:-1,:-1],
                                            latsCenterNotNanPadded[:-1,1:]
                                            )))
        
        # If latsCenter[y,x] is not nan, then lats[y,x], lats[y+1,x],
        # lats[y+1,x+1] and lats[y,x+1] are not nan.
        latsNotNan = ~ma.getmaskarray(lats)
        assert np.all(np.logical_or(ma.getmaskarray(latsCenter),
                                    np.logical_and.reduce((latsNotNan[:-1,:-1],
                                                           latsNotNan[1:,:-1],
                                                           latsNotNan[1:,1:],
                                                           latsNotNan[:-1,1:]
                                                           ))))
        
        # If img[y,x] is not nan, then latsCenter[y,x] is not nan, and vice versa.
        latsCenterNotNan = ~ma.getmaskarray(latsCenter)
        imgNan = ma.getmaskarray(img)
        for d in range(img.shape[2]):
            assert np.all(np.logical_xor(imgNan[:,:,d], latsCenterNotNan))
                
        # If elevation[y,x] is not nan, then latsCenter[y,x] is not nan, and vice versa.          
        assert np.all(np.logical_xor(ma.getmaskarray(elevation), latsCenterNotNan))
        
        # If mlatCenter[y,x] is not nan, then latsCenter[y,x] is not nan, and vice versa.
        assert np.all(np.logical_xor(ma.getmaskarray(mlatCenter), latsCenterNotNan))
        
        # If mltCenter[y,x] is not nan, then latsCenter[y,x] is not nan, and vice versa.
        assert np.all(np.logical_xor(ma.getmaskarray(mltCenter), latsCenterNotNan))
        
        # If mlat[y,x] is not nan, then lats[y,x] is not nan, and vice versa.
        assert np.all(np.logical_xor(ma.getmaskarray(mlat), latsNotNan))
        
        # If mlt[y,x] is not nan, then lats[y,x] is not nan, and vice versa.
        assert np.all(np.logical_xor(ma.getmaskarray(mlt), latsNotNan))
        
    @property
    def altitude(self):
        """
        Mapping altitude in km.
        """
        return self._altitude
    
    @property
    def cameraPosGCRS(self):
        return self._cameraPosGCRS
    
    @lazy_property
    def cameraFootpoint(self):
        """
        The camera footpoint in geodetic coordinates.
        
        :rtype: auromat.coordinates.geodesic.Location
        """
        camFootLat, camFootLon = j2000ToLatLon([self.cameraPosGCRS], self.photoTime)
        return Location(camFootLat[0], camFootLon[0])
    
    @property
    def photoTime(self):
        """
        :rtype: datetime.datetime
        """
        return self._photoTime
    
    @property
    def identifier(self):
        """
        :rtype: str
        """
        return self._identifier
    
    @property
    def metadata(self):
        """
        :rtype: dict
        """
        if self._metadata is None:
            return {}
        else:
            return self._metadata
        
    @abstractproperty
    def lats(self):
        """
        A masked array of shape (img.shape[0]+1, img.shape[1]+1)
        containing the latitude of every pixel corner.
        """
    
    @abstractproperty
    def lons(self):
        """
        A masked array of shape shape (img.shape[0]+1, img.shape[1]+1)
        containing the longitude of every pixel corner.
        """
    
    @abstractproperty
    def latsCenter(self):
        """
        A masked array of shape shape (img.shape[0], img.shape[1])
        containing the latitude of every pixel center.
        """

    @abstractproperty
    def lonsCenter(self):
        """
        A masked array of shape (img.shape[0], img.shape[1])
        containing the longitude of every pixel center.
        """
        
    @property
    def isPlateCarree(self):
        """
        Return whether the latitude/longitude arrays describe a plate carree
        projection (regular grid).
        """
        return isPlateCarree(self.lats, self.lons)
    
    def checkPlateCarree(self):
        """
        Raises an error when the latitude/longitude arrays do not describe a plate carree
        projection (regular grid).
        """
        return checkPlateCarree(self.lats, self.lons)
        
    @property
    def mLatMlt(self):
        """
        Tuple of masked arrays of the geomagnetic latitude/magnetic local time of every pixel corner.
        
        :rtype: tuple(mlat, mlt) with array shape (img.shape[0]+1, img.shape[1]+1).
        """
        if self._mlatmlt is None:
            self._mlatmlt = self._mLatMlt(self.lats, self.lons)
        return self._mlatmlt
    
    @property
    def mLatMltCenter(self):
        """
        Tuple of masked arrays of the geomagnetic latitude/magnetic local time of every pixel center.
        
        :rtype: tuple(mlat, mlt) with shape (img.shape[0], img.shape[1]).
        """
        if self._mlatmltCenter is None:
            self._mlatmltCenter = self._mLatMlt(self.latsCenter, self.lonsCenter)
        return self._mlatmltCenter
    
    def _mLatMlt(self, lats, lons):
        mask = ma.getmaskarray(lats)
        lats, lons = np.deg2rad(lats.data), np.deg2rad(lons.data)
        geoX, geoY, geoZ = geodetic2Ecef(lats, lons, self.altitude, wgs84A, wgs84B)
        geo = np.dstack((geoX,geoY,geoZ))
        mlat, mlt = geoToMLatMLT(geo.reshape(-1,3), self.photoTime)
        mlat = mlat.reshape(geo.shape[0], geo.shape[1])
        mlt = mlt.reshape(geo.shape[0], geo.shape[1])
        mlat = ma.masked_array(mlat, mask)
        mlt = ma.masked_array(mlt, mask)
        return mlat, mlt
    
    @abstractproperty
    def img(self):
        """
        Masked array of shape (h,w,n) and type (u)int.
        """
        
    @abstractproperty
    def img_unmasked(self):
        """
        Like img but as a normal numpy array (not a masked array).
        
        The purpose of this property (compared to img.data) is to implement
        more efficient ways to access the image data, in the case that
        the coordinates (lat, lon, etc.) are not needed. These would be
        calculated to properly mask the image.
        """
    
    @abstractproperty
    def elevation(self):
        """
        Elevation in degrees for each pixel center, that is,
        a masked array of shape (img.shape[0], img.shape[1]).
        """
    
    @abstractproperty
    def rgb(self):
        """
        A representation of img as a masked (h,w,3) uint8 RGB image with a range [0,255].
        
        This is a convenience property for easy display of the image data.
        
        This array shall never be modified directly, only through
        the underlying img.
        
        Note to implementers: If this property is cached then care must be
        taken when a mapping gets masked. For example, when implementing
        :meth:`createMasked` then the cache may be invalidated.
        """
    
    @abstractproperty
    def rgb_unmasked(self):
        """
        Like rgb but as a normal numpy array (not a masked array).
        
        The purpose of this property (compared to rgb.data) is to implement
        more efficient ways to access the image data, in the case that
        the coordinates (lat, lon, etc.) are not needed. These would be
        calculated to properly mask the image.
        """
        
    @abstractmethod
    def createResampled(self, lats, lons, latsCenter, lonsCenter, elevation, img):
        """
        Returns a new mapping object with the given values.
        Each subclass can decide what the most appropriate class for
        the resampled data is. E.g. ThemisMapping uses itself, while
        AstrometryMapping uses GenericMapping. This is important as
        only the original class knows how to interpret the img data
        (if it's not standard RGB in the case of THEMIS).
        See :func:`auromat.resample.resample` for an application of this method.
        
        :param elevation: can be None
        """

    @abstractmethod    
    def createMasked(self, centerMask):
        """
        Return a copy of this mapping with the given mask applied
        to img, latsCenter, lonsCenter, and elevation.
        Implementing classes must override this method and handle
        the lats and lons attributes and possible others.
        See :meth:`maskedByElevation` for an application of this method.
        
        :param centerMask: the mask is not copied and should not be
                           used after anymore after calling this method
        """
        # mask is copied, data is not      
        _latsCenter = ma.masked_array(self.latsCenter.data, centerMask)
        _lonsCenter = ma.masked_array(self.lonsCenter.data, centerMask)
        _img = ma.masked_array(self.img.data, centerMask)
        _elevation = ma.masked_array(self.elevation.data, centerMask)

        class MaskedMapping(object):                
            @property
            def latsCenter(self):
                return _latsCenter
            
            @property
            def lonsCenter(self):
                return _lonsCenter
            
            @property
            def img(self):
                return _img
            
            @property
            def elevation(self):
                return _elevation
            
        m = copy.copy(self)
        extend(m, MaskedMapping)
        return m
            
    @property
    def outline(self):
        """
        The complete outline of this mapping. 
        Note that the outline can be concave.
        """
        full, _ = self._fullAndConvexOutlines
        return full
    
    @property
    def outlineConvexHull(self):
        """
        The convex hull of the regular outline.
        """
        _, convex = self._fullAndConvexOutlines
        return convex
    
    @property
    def _fullAndConvexOutlines(self):
        if self._outlines is None:
            mask = ~ma.getmaskarray(self.lats)            
            outl = outline(mask)
            
            # full outline
            outlLats = self.lats[outl[:,1], outl[:,0]]
            outlLons = self.lons[outl[:,1], outl[:,0]]
            outlLatLon = np.transpose([outlLats, outlLons])
            
            # convex hull
            outlConvex = convexHull(outl)
            outlConvexLats = self.lats[outlConvex[:,1], outlConvex[:,0]]
            outlConvexLons = self.lons[outlConvex[:,1], outlConvex[:,0]]
            outlConvexLatLon = np.transpose([outlConvexLats, outlConvexLons])
            
            self._outlines = outlLatLon, outlConvexLatLon    
            
        return self._outlines        
    
    @property   
    def boundingBox(self):
        """
        In case containsPole is True, this bounding box is degenerate!
        It will span the full longitude range in this case.
        """
        if self._boundingBox is None:
            outlLats = self.outline[:,0]
            outlLons = self.outline[:,1]
            
            latMin, latMax = np.min(outlLats), np.max(outlLats)
            lonMin, lonMax = np.min(outlLons), np.max(outlLons)
            
            # To make pole checking faster we reduce the outline.
            # Removing points from the full outline could produce a 
            # self-intersecting polygon, therefore the convex hull of
            # the full outline is used such that this cannot happen.
            pointCount = len(self.outlineConvexHull)
            sampleCount = min(pointCount, 50)
            indices = np.round(np.linspace(0, pointCount-1, sampleCount)).astype(np.int)           
            reducedOutline = self.outlineConvexHull[indices]
            
            if containsOrCrossesPole(reducedOutline):
                lonWest = -180
                lonEast = 180
                # find out which pole!
                if latMax < 0: # south pole
                    latSouth = -90
                    latNorth = latMax
                else: # north pole
                    latNorth = 90
                    latSouth = latMin
            else:
                # we assume that mappings are smaller than 180deg longitudes
                # and use this assumption to determine if the mapping contains
                # the discontinuity
                if lonMax-lonMin > 180:
                    # contains discontinuity
                    westLons = outlLons>0
                    eastLons = ~westLons
                    lonWest = np.min(outlLons[westLons])
                    lonEast = np.max(outlLons[eastLons])
                else:
                    lonWest = lonMin
                    lonEast = lonMax
                latNorth, latSouth = latMax, latMin
            
            bb = BoundingBox(latSouth, lonWest, latNorth, lonEast)
            self._boundingBox = bb
        
        return self._boundingBox
    
    @property
    def containsDiscontinuity(self):
        """
        Return whether the bounding box contains the discontinuity.
        """
        return self.boundingBox.containsDiscontinuity
        
    @property
    def containsPole(self):
        """
        Return whether the bounding box contains a pole.
        """
        return self.boundingBox.containsPole
    
    @property
    def centroid(self):
        """The centroid of the mapping based on the plate-carree projection.
        
        :rtype: auromat.coordinates.geodesic.Location
        """        

        if self._centroid is None:
            if self.containsPole:
                # TODO rotate away from pole, see resample module
                raise NotImplementedError
            elif self.containsDiscontinuity:
                lats = self.outline[:,0]
                lons = self.outline[:,1]
                lons = Angle((lons + 180) * u.deg).wrap_at(180 * u.deg).degree
                outline = np.transpose([lats, lons])
                
                lat,lon = polygonCentroid(outline)
                
                centroidLon = Angle((lon + 180) * u.deg).wrap_at(180 * u.deg).degree
                self._centroid = Location(lat, centroidLon)
            else:
                lat,lon = polygonCentroid(self.outline)
                self._centroid = Location(lat, lon)
        
        return self._centroid
    
    @property
    def arcSecPerPx(self):
        """
        Min, max, median, and mean angular sizes of pixels/polygons determined for the width, height, and
        diagonal of 1000 polygons.
        
        :rtype: PixelScales
        """
        if self._pixelScales is None:            
            # create polygons without colors 
            # (copied from auromat.draw.createPolygonsAndColors)
            latLonDeg = ma.dstack((self.lats, self.lons))
            verts = ma.concatenate((
                        latLonDeg[0:-1, 0:-1],
                        latLonDeg[0:-1, 1:  ],
                        latLonDeg[1:  , 1:  ],
                        latLonDeg[1:  , 0:-1],
                        ), axis=2)
            verts = verts.reshape((latLonDeg.shape[0]-1)*(latLonDeg.shape[1]-1), 4, 2)
            hasNans = ma.getmaskarray(verts).any(axis=-1).any(axis=-1)
            verts = verts[~hasNans]
            
            # calculate angular sizes for 1000 polygons using the polygon width
            # (just 1000 because geographiclib is quite slow due to no array support)
            polyCount = verts.shape[0]
            sampleCount = min(polyCount, 1000)
            indices = np.round(np.linspace(0, polyCount-1, sampleCount)).astype(np.int)
            
            widthSizesDeg = [geodesic.angularDistance(Location(poly[0][0], poly[0][1]), 
                                                      Location(poly[1][0], poly[1][1]))
                             for poly in verts[indices]]

            heightSizesDeg = [geodesic.angularDistance(Location(poly[1][0], poly[1][1]), 
                                                       Location(poly[2][0], poly[2][1]))
                              for poly in verts[indices]]
            
            diagonalSizesDeg = [geodesic.angularDistance(Location(poly[0][0], poly[0][1]), 
                                                         Location(poly[2][0], poly[2][1]))
                                for poly in verts[indices]]
            
            # reduce to some useful values
            scales = []
            for sizesDeg in [widthSizesDeg, heightSizesDeg, diagonalSizesDeg]:
                meanDegPerPx = np.mean(sizesDeg)
                medianDegPerPx = np.median(sizesDeg)
                minDegPerPx = min(sizesDeg)
                maxDegPerPx = max(sizesDeg)
                
                meanArcSecPerPx = (meanDegPerPx * u.deg).to(u.arcsec).value
                medianArcSecPerPx = (medianDegPerPx * u.deg).to(u.arcsec).value
                minArcSecPerPx = (minDegPerPx * u.deg).to(u.arcsec).value
                maxArcSecPerPx = (maxDegPerPx * u.deg).to(u.arcsec).value
                
                scales.append(PixelScale(meanArcSecPerPx, medianArcSecPerPx, minArcSecPerPx, maxArcSecPerPx))
            
            self._pixelScales = PixelScales(width=scales[0], height=scales[1], diagonal=scales[2])
            
        return self._pixelScales
        
    def maskedByElevation(self, minElevation=10):
        """
        Return a new mapping with data masked below the given minimum elevation.
        A previously applied mask is ignored.
        
        Note that the new mapping reuses the existing arrays and just exchanges
        the masks.
        
        :param int minElevation: 0 to 90, in degrees
        :rtype: BaseMapping
        """
        assert self.elevation is not None
        centerMask = (self.elevation < minElevation).filled(True)
        if np.all(centerMask):
            raise ValueError('minElevation=' + str(minElevation) + ' would mask all pixels!')
        
        newMapping = self.createMasked(centerMask)
        newMapping.setDirty()
                
        return newMapping
        
    def maskedByPolygon(self, polygon):
        """
        Returns a copy of this mapping where the image is masked using the
        given polygon. Only those pixels are retained where all of its corners
        are inside the polygon. A previously applied mask is ignored.
        
        Note that the new mapping reuses the existing arrays and just exchanges
        the masks.
        
        .. warning:: If the mapping or the polygon contains the discontinuity and/or poles then
                 this method tries to handle it in a best-effort approach.
                 It may happen that this approach fails where the resulting mapping could
                 contain invalid data.
        
        :param array-like polygon: ordered points of an unclosed polygon in [lat,lon] order
        :rtype: BaseMapping
        """
        polygon = np.asarray(polygon)
        latLonGrid = np.dstack((self.lats.data, self.lons.data))
        latLonGridFlat = latLonGrid.reshape(-1, 2)

        polyBoundingBox = BoundingBox.minimumBoundingBox(polygon)
        polyContainsPole = containsOrCrossesPole(polygon)   
        if self.containsDiscontinuity or polyBoundingBox.containsDiscontinuity:
            angle = 180
            polygon = polygon.copy()
            for arr in [latLonGridFlat, polygon]:
                arr[:,1] = Angle((arr[:,1] + angle) * u.deg).wrap_at(angle * u.deg).degree
                
        elif self.containsPole or polyContainsPole:
            angle = 90
            xaxis = [1,0,0]
            for arr in [latLonGridFlat, polygon]:
                arr[:,0], arr[:,1] = rotatePole(np.deg2rad(arr[:,0]), 
                                                np.deg2rad(arr[:,1]),
                                                self.altitude, 
                                                angle=angle, axis=xaxis)
              
        isInside = auromat.utils.pointsInsidePolygon(latLonGridFlat, polygon)
        isInside = isInside.reshape(self.lats.shape)
        mask = ~isInside | self.lats.mask

        if np.all(mask):
            raise ValueError('The given mask would mask all pixels!')
        
        # we mask every pixel which misses at least one of its four corner coordinates        
        centerMask = np.logical_or.reduce((mask[:-1,:-1], mask[1:,:-1], mask[:-1,1:], mask[1:,1:]))
        
        newMapping = self.createMasked(centerMask)
        newMapping.setDirty()
                
        return newMapping
    
    def setDirty(self):
        """
        Sets all values of cached properties to None so that they are
        recalculated the next time they are accessed.
        """
        self._outlines = None
        self._boundingBox = None
        self._centroid = None
        self._pixelScales = None
        self._mlatmlt = None
        self._mlatmltCenter = None

def checkPlateCarree(lats, lons):
    """
    Checks whether the given 2D coordinate arrays describe a
    plate carree projection, that is, if the latitudes (longitudes)
    are evenly spaced and monotonically decreasing (increasing).
    
    :param ndarray lats: 2D latitude array
    :param ndarray lons: 2D longitude array
    :raise ValueError: when the projection is not plate carree 
    """
    if ma.isMaskedArray(lats):
        lats, lons = lats.data, lons.data
    if np.any(np.isnan(lats)):
        raise ValueError('coordinates contains NaNs')
    lons = np.unwrap(np.deg2rad(lons))
    # x=lon must be monotonically increasing
    # y=lat must be monotonically decreasing
    if lons[0,-1]-lons[0,0] <= 0:
        raise ValueError('longitudes are not monotonically increasing')
    if lats[0,0]-lats[-1,0] <= 0:
        raise ValueError('latitudes are not monotonically decreasing')
    # lat and lon must be evenly spaced
    eps = 1e-4
    deltaLon = lons[0,1:] - lons[0,:-1]
    isLonRegular = np.max(deltaLon) - np.min(deltaLon) < eps
    if not isLonRegular:
        raise ValueError('longitudes are not evenly spaced')
    deltaLat = lats[:-1,0] - lats[1:,0]
    isLatRegular = np.max(deltaLat) - np.min(deltaLat) < eps
    if not isLatRegular:
        raise ValueError('latitudes are not evenly spaced')

def isPlateCarree(lats, lons):
    """
    Return whether the given 2D coordinate arrays describe a
    plate carree projection, that is, if the latitudes (longitudes)
    are evenly spaced and monotonically decreasing (increasing).
    
    :param ndarray lats: 2D latitude array
    :param ndarray lons: 2D longitude array
    :rtype: bool
    """
    try:
        checkPlateCarree(lats, lons)
    except:
        return False
    else:
        return True
      
class DefaultRGBMixin(object):
    """
    Mixin for use with :class:`BaseMapping`. 
    Interprets an 8 bit (uint8) or 16 bit (uint16) 3-channel image as RGB image
    with the channels in RGB order. 
    """
     
    @property
    def rgb(self):
        return ma.masked_array(self.rgb_unmasked, mask=self.img.mask)
        
    @property
    def rgb_unmasked(self):
        if self.img_unmasked.dtype == np.uint8:
            img = self.img_unmasked
        elif self.img_unmasked.dtype == np.uint16:
            img = self.img_unmasked.copy()
            img *= 255/65535
            img = img.astype(np.uint8)
        else:
            raise NotImplementedError
        
        if img.shape[2] == 3:
            return img
        elif img.shape[2] == 1:
            return np.repeat(img, 3, 2)
        else:
            raise NotImplementedError('Unknown img format')

class ArrayImageMixin(DefaultRGBMixin):
    """
    Mixin for use with :class:`BaseMapping`. 
    Interprets an 8 bit (uint8) or 16 bit (uint16) 3-channel array as RGB image
    with the channels in RGB order. 
    """
    
    def __init__(self, img):
        assert img.ndim == 3
        assert img.dtype in [np.uint8, np.uint16]
        if not ma.isMaskedArray(img):
            img = ma.masked_array(img)
        self._img = img
    
    @property
    def img(self):
        return self._img
    
    @property
    def img_unmasked(self):
        return self._img.data
            
class FileImageMixin(DefaultRGBMixin):
    """
    Mixin for use with :class:`BaseMapping`. 
    Loads an 8 or 16 bit image file and converts it to an RGB image
    in case it is grayscale.
    """
    
    # TODO don't convert grayscale to RGB, only for rgb property later on
    #      -> relevant for MIRACLE
    
    def __init__(self, imagePath):
        self._imagePath = imagePath
        self._img = None
        self._img_unmasked = None
    
    @property
    def img(self):
        if self._img is None:
            self._img = ma.masked_array(self.img_unmasked)
        return self._img
    
    @property
    def img_unmasked(self):
        if self._img_unmasked is None:
            img = loadImage(self._imagePath)
            self._img_unmasked = img
        return self._img_unmasked
    
    @property
    def imagePath(self):
        return self._imagePath

def _doSanitize(lats, lons, latsCenter, lonsCenter, img, elevation, afterMasking=False):
    """
    Sanitizes the given masked data arrays such that _checkGuarantees() returns True.
    This method works directly on the masks of the given arrays.
    
    :param afterMasking: If True, skips steps which are not needed when sanitizing
                         an already sanitized mapping that has been masked.
    """
    t0 = time.time()
    
    centerMaskShared = isinstance(latsCenter.mask, np.ndarray) and \
                       latsCenter.mask is lonsCenter.mask is elevation.mask
    cornerMaskShared = isinstance(lats.mask, np.ndarray) and lats.mask is lons.mask
    
    # first, apply img mask to center coordinates
    imgMask = ma.getmaskarray(img)[:,:,0]
    latsCenter[imgMask] = ma.masked
    if not centerMaskShared:
        lonsCenter[imgMask] = ma.masked
    
    # now, make sure lats,lons,latsCenter,lonsCenter are consistent
    latsCenterNanPadded = np.ones((latsCenter.shape[0]+2, latsCenter.shape[1]+2), bool)
    latsCenterNanPadded[1:-1,1:-1] = ma.getmaskarray(latsCenter)

    allNeighboursMissing = np.logical_and.reduce((latsCenterNanPadded[1:,1:],
                                                  latsCenterNanPadded[1:,:-1],
                                                  latsCenterNanPadded[:-1,:-1],
                                                  latsCenterNanPadded[:-1,1:]))

    lats[allNeighboursMissing] = ma.masked
    if not cornerMaskShared:
        lons[allNeighboursMissing] = ma.masked
    
    if not afterMasking:
        latsNan = ma.getmaskarray(lats)
        anyCornerMissing = np.logical_or.reduce((latsNan[:-1,:-1],
                                                 latsNan[1:,:-1],
                                                 latsNan[1:,1:],
                                                 latsNan[:-1,1:]))
    
        latsCenter[anyCornerMissing] = ma.masked
        if not centerMaskShared:
            lonsCenter[anyCornerMissing] = ma.masked
        
        # and again in case there are new corners without neighbors
        latsCenterNanPadded = np.ones((latsCenter.shape[0]+2, latsCenter.shape[1]+2), bool)
        latsCenterNanPadded[1:-1,1:-1] = ma.getmaskarray(latsCenter)
    
        allNeighboursMissing = np.logical_and.reduce((latsCenterNanPadded[1:,1:],
                                                      latsCenterNanPadded[1:,:-1],
                                                      latsCenterNanPadded[:-1,:-1],
                                                      latsCenterNanPadded[:-1,1:]))
    
        lats[allNeighboursMissing] = ma.masked
        if not cornerMaskShared:
            lons[allNeighboursMissing] = ma.masked
    
    # finally, apply coords mask back to img and elevation
    img[ma.getmaskarray(latsCenter)] = ma.masked
    if not centerMaskShared:        
        elevation[ma.getmaskarray(img)[:,:,0]] = ma.masked
    
    print('sanitize:', time.time()-t0, 's')

def sanitize_data(cls):
    """
    A decorator for classes inheriting from :class:`BaseMapping`. It lazily sanitizes data such that
    :meth:`BaseMapping.checkGuarantees` is satisfied. This decorator assumes the most general case. 
    If a more optimized sanitization technique can be implemented
    for a specific mapping class, then this should be done and this decorator not be used.
    """
    @inherit_docs
    class SanitizedMapping(cls):
                
        @property
        def lats(self):
            return self._sanitized(lambda s: s.lats)
        
        @property
        def lons(self):
            return self._sanitized(lambda s: s.lons)
        
        @property
        def latsCenter(self):
            return self._sanitized(lambda s: s.latsCenter)
        
        @property
        def lonsCenter(self):
            return self._sanitized(lambda s: s.lonsCenter)
        
        @property
        def img(self):
            return self._sanitized(lambda s: s.img)
        
        @property
        def elevation(self):
            return self._sanitized(lambda s: s.elevation)
            
        def createMasked(self, centerMask):
            # mask is copied, data is not
            latslonsmask = ma.getmaskarray(self.lats).copy()
            _lats = ma.masked_array(self.lats.data, latslonsmask)
            _lons = ma.masked_array(self.lons.data, latslonsmask)
            _latsCenter = ma.masked_array(self.latsCenter.data, centerMask)
            _lonsCenter = ma.masked_array(self.lonsCenter.data, centerMask)
            imgMask = np.repeat(centerMask[:,:,None], self.img.shape[2], 2)
            _img = ma.masked_array(self.img.data, imgMask)
            _elevation = ma.masked_array(self.elevation.data, centerMask)
            
            class attrs(object):
                lats=_lats
                lons=_lons
                latsCenter=_latsCenter
                lonsCenter=_lonsCenter
                img=_img
                elevation=_elevation
                        
            class MaskedMapping(object):                    
                @property
                def lats(self):
                    return self._sanitized(lambda s: s.lats)
                
                @property
                def lons(self):
                    return self._sanitized(lambda s: s.lons)
                
                @property
                def latsCenter(self):
                    return self._sanitized(lambda s: s.latsCenter)
                
                @property
                def lonsCenter(self):
                    return self._sanitized(lambda s: s.lonsCenter)
                
                @property
                def img(self):
                    return self._sanitized(lambda s: s.img)
                
                @property
                def elevation(self):
                    return self._sanitized(lambda s: s.elevation)
                
                def _sanitized(self, p):
                    self._sanitize(attrs)
                    return p(attrs)
                
            m = copy.copy(self)
            extend(m, MaskedMapping)
            m.isSanitized = False
            m.afterMasking = True
            return m
            
        def _sanitized(self, p):
            self._sanitize(super(SanitizedMapping, self))
            return p(super(SanitizedMapping, self))
            
        def _sanitize(self, attrs):
            if getattr(self, 'isSanitized', False):
                return
            afterMasking = getattr(self, 'afterMasking', False)
            
            _doSanitize(attrs.lats, attrs.lons, attrs.latsCenter, 
                        attrs.lonsCenter, attrs.img, attrs.elevation,
                        afterMasking=afterMasking)
            self.isSanitized = True
    
    SanitizedMapping.__doc__ = cls.__doc__
    
    return SanitizedMapping

@sanitize_data
@inherit_docs
class GenericMapping(ArrayImageMixin, BaseMapping):
    """
    A mapping consisting of precalculated latitudes/longitudes/elevation values.
    """
    
    def __init__(self, lats, lons, latsCenter, lonsCenter, elev, alti, img, cameraPosGCRS, photoTime, 
                 identifier, metadata=None):
        """
        
        :param ndarray lats: (h+1,w+1) in degrees
        :param ndarray lons: (h+1,w+1) in degrees
        :param ndarray latsCenter: (h,w) in degrees
        :param ndarray lonsCenter: (h,w) in degrees
        :param ndarray elev: (h,w) in degrees; can also be None         
        :param number alti: the altitude in km onto which the image was mapped (e.g. 110)
        :param ndarray img: uint8 or uint16 array of shape (h,w) for grayscale or (h,w,3) for RGB
        :param array-like cameraPosGCRS: [x,y,z] in km
        :param datetime.datetime photoTime:
        """
        h, w = img.shape[0], img.shape[1]
        assert lats.shape == lons.shape == (h+1, w+1)
        assert latsCenter.shape == lonsCenter.shape == (h, w)
        assert elev is None or elev.shape == (h, w)
        
        ArrayImageMixin.__init__(self, img)
        BaseMapping.__init__(self, alti, cameraPosGCRS, photoTime, identifier, metadata)
        self._lats = ma.masked_invalid(lats, copy=False) if not ma.isMA(lats) else lats
        self._lons = ma.masked_invalid(lons, copy=False) if not ma.isMA(lons) else lons
        self._latsCenter = ma.masked_invalid(latsCenter, copy=False) if not ma.isMA(latsCenter) else latsCenter
        self._lonsCenter = ma.masked_invalid(lonsCenter, copy=False) if not ma.isMA(lonsCenter) else lonsCenter
        self._elevation = ma.masked_invalid(elev, copy=False) if not ma.isMA(elev) else elev
        
    @property
    def lats(self):
        return self._lats
    
    @property
    def lons(self):
        return self._lons
    
    @property
    def latsCenter(self):
        return self._latsCenter
    
    @property
    def lonsCenter(self):
        return self._lonsCenter
    
    @property
    def elevation(self):
        return self._elevation
    
    def createResampled(self, lats, lons, latsCenter, lonsCenter, elevation, img):
        mapping = GenericMapping(lats, lons, latsCenter, lonsCenter, elevation, self.altitude, img, 
                                 self.cameraPosGCRS, self.photoTime, self.identifier,
                                 metadata=self.metadata)
        return mapping
    
    @staticmethod
    def fromMapping(mapping):
        """
        Create a :class:`GenericMapping` from the given mapping.
        
        Useful for removing additional data of a more specific mapping class.
        Note that this method can only be used if the image data is compatible
        to :class:`GenericMapping` (standard uint8 or uint16 grayscale/RGB arrays).        
        """
        m = GenericMapping(mapping.lats, mapping.lons, mapping.latsCenter, mapping.lonsCenter, 
                           mapping.elevation, mapping.altitude, mapping.img, 
                           mapping.cameraPosGCRS, mapping.photoTime, mapping.identifier, mapping.metadata)
        try:
            m.isSanitized = mapping.isSanitized
            m.afterMasking = mapping.afterMasking
        except:
            pass
        return m
# for sphinx: this avoids the class being displayed as an alias of SanitizedMapping
# The bases are displayed wrongly though.
GenericMapping.__name__ = 'GenericMapping'

class MappingCollection(object):
    def __init__(self, mappings, identifier, mayOverlap=True):
        """
        A collection of mappings for the same photo time (+- a few seconds).
        
        :type mappings: list of :class:`BaseMapping` objects
        :param string identifier: e.g. THEMIS-2012.01.01.12.30.59 
        :param mayOverlap: True if some mappings may overlap each other
                           This information is used for drawing polygons. When True,
                           then the pixels of all mappings are joined together, sorted
                           by elevation, and drawn in the same order (such that low-elevation
                           polygons of one mapping are overdrawn by higher-elevation polygons
                           of another overlapping mapping).
        """
        self._mappings = mappings
        self._identifier = identifier
        self._mayOverlap = mayOverlap
    
    @property
    def identifier(self):
        return self._identifier
    
    @property
    def empty(self):
        """
        Whether this collection contains no mappings.
        """
        return len(self.mappings) == 0
    
    @property
    def mappings(self):
        return self._mappings
    
    @property
    def mayOverlap(self):
        return self._mayOverlap
    
    def maskedByElevation(self, minElevation=10):
        mappings = [m.maskedByElevation(minElevation) for m in self.mappings]
        return MappingCollection(mappings, self.identifier, self.mayOverlap)
            
    @property
    def boundingBox(self):
        """
        The smallest bounding box containing the bounding boxes of all mappings.
        """
        boundingBoxes = [mapping.boundingBox for mapping in self.mappings]
        return BoundingBox.mergedBoundingBoxes(boundingBoxes)
    
    @property
    def photoTime(self):
        """
        The median of the photo times of all mappings.
        """
        times = sorted(m.photoTime for m in self.mappings)
        return times[len(times)//2]
    
    def __len__(self):
        return len(self._mappings)
               
@add_metaclass(ABCMeta)
class BaseMappingProvider(object):
    """
    Base class for all mapping providers.
    """
    
    def __init__(self, maxTimeOffset):
        """
        :param maxTimeOffset: in seconds
        """
        self.maxTimeOffset = maxTimeOffset
        
    @abstractproperty
    def range(self):
        """
        The dates of the first and last available mappings. 
        
        :rtype: datetime tuple (from, to)
        """
    
    @abstractmethod
    def contains(self, date):
        """
        Return True if there is a mapping for the given date 
        within +-maxTimeOffset.
        
        :param datetime date:
        :rtype: bool
        """
        
    def containsAny(self, dates):
        """
        Return True if there is a mapping for at least one of the given dates
        within +-maxTimeOffset.
        
        :param dates: list of datetime objects
        :rtype: bool
        """
        return any(self.contains(date) for date in dates)
        
    @abstractmethod
    def get(self, date):
        """
        Returns the mapping which is closest to the given date
        within +-maxTimeOffset.
        
        :param datetime date:
        :rtype: BaseMapping or MappingCollection
        :raise ValueError: when no mapping exists for the given date
        """
    
    @abstractmethod
    def getById(self, identifier):
        """
        Returns the mapping with the given identifier.
        
        :param string identifier:
        :rtype: BaseMapping or MappingCollection
        :raise ValueError: when no mapping with the given identifier exists
        """
    
    @abstractmethod
    def getSequence(self, dateBegin=None, dateEnd=None):
        """
        Returns a generator of mappings ordered by date for the given date range.
        If dateBegin and dateEnd are None, then all available mappings are returned.
        
        :param datetime dateBegin: 
        :param datetime dateEnd: inclusive
        :rtype: list of BaseMapping or MappingCollection objects
        """

def MaskByElevationProvider(provider, *args, **kw):
    """
    Wrap the given mapping provider by masking every returned mapping by elevation.    
    
    :param provider: the provider to wrap
    
    See :func:`BaseMapping.maskedByElevation` for masking parameters.
    """
    def mask(m):
        return m.maskedByElevation(*args, **kw)
    class MaskingProvider(object):          
        def get(self, *args, **kw):
            m = super(MaskingProvider, self).get(*args, **kw)
            return mask(m)
        
        def getById(self, *args, **kw):
            m = super(MaskingProvider, self).getById(*args, **kw)
            return mask(m)
            
        def getSequence(self, *args, **kw):
            m = super(MaskingProvider, self).getSequence(*args, **kw)
            return map(mask, m)
    
    provider = copy.copy(provider)  
    extend(provider, MaskingProvider)
    return provider

def inflatedEarthIntersection(cameraToPixelDirection, cameraPos, earthInflation=110, earthModel='wgs84'):
    """
    Return the intersection points with an inflated earth
    when shooting rays originating at `cameraPos` and 
    going in the direction `cameraToPixelDirection`.
    
    When `earthModel` is 'sphere' then the shape of the inflated earth
    is described by a sphere with radius `earthRadius + earthInflation`.
    
    When `earthModel` is 'wgs84' then the shape of the inflated earth
    is described by an ellipsoid with an equatorial axis of
    `wgs84A + earthInflation` and a polar axis of `wgs84B + earthInflation`
    where `wgs84A` and `wgs84B` refers to the 
    `World Geodetic System 84 <https://en.wikipedia.org/wiki/World_Geodetic_System>`_.
    
    :param cameraToPixelDirection: direction vectors in ICRS from camera location to pixel/sky location, shape (n,3)
    :param cameraPos: ndarray of xyz J2000 coordinates in km
    :param earthInflation: in km, how much to expand the earth when intersecting, use 0 for earth intersection
    :param earthModel: earth model, either 'wgs84' or 'sphere'
    """
    assert ((cameraToPixelDirection.ndim == 1 and cameraToPixelDirection.shape[0] == 3) or
            (cameraToPixelDirection.ndim == 2 and cameraToPixelDirection.shape[1] == 3))
    t0 = time.time()
    if earthModel == 'wgs84':
        a = wgs84A + earthInflation
        b = wgs84B + earthInflation
        intersectionAurora = ellipsoidLineIntersection(a, b, cameraPos, cameraToPixelDirection)
        
    elif earthModel == 'sphere':
        earthRadius = const.R_earth.to(u.km).value
        sphereRadius = earthRadius + earthInflation
        intersectionAurora = sphereLineIntersection(sphereRadius, cameraPos, cameraToPixelDirection)
        
    else:
        raise ValueError('unsupported earth model: ' + earthModel)
    print('inflatedEarthIntersection:', time.time()-t0, 's')    
    return intersectionAurora

class _SMMapping(GenericMapping):
    @property
    def cameraFootpoint(self):
        mlat, mlt = j2000ToMLatMLT([self.cameraPosGCRS], self.photoTime)
        smlon = mltToSmLon(mlt)
        return Location(mlat[0], smlon[0])

def convertMappingToSM(mapping):
    """
    Return a new mapping with the coordinates transformed
    to solar magnetic latitudes and longitudes.
    
    .. warning:: This function is not intended for regular use.
        If you need to access MLat/MLT coordinates, use
        the :attr:`BaseMapping.mLatMlt` and :attr:`BaseMapping.mLatMltCenter`
        attributes.
        
    See :func:`auromat.draw.drawStereographicMLatMLT` for an application
    of this function.
    
    :param BaseMapping mapping: 
        A mapping where latitudes and longitudes are given
        in the standard geodetic coordinate system.  
    :rtype: BaseMapping
    """
    mlat, mlt = mapping.mLatMlt
    mlat, mlt = mlat.data, mlt.data
    smlons = mltToSmLon(mlt)
    mlatCenter, mltCenter = mapping.mLatMltCenter
    mlatCenter, mltCenter = mlatCenter.data, mltCenter.data
    smlonsCenter = mltToSmLon(mltCenter)
    newMapping = _SMMapping(mlat, smlons, mlatCenter, smlonsCenter, 
                            mapping.elevation, mapping.altitude, 
                            mapping.img, mapping.cameraPosGCRS, 
                            mapping.photoTime, mapping.identifier)
    return newMapping

def convertSMMappingToGeo(mapping):
    """ Inverse operation to :func:`convertMappingToSM`. """
    smlats, smlons = mapping.lats.data, mapping.lons.data
    smlatsCenter, smlonsCenter = mapping.latsCenter.data, mapping.lonsCenter.data
    lats, lons = smToLatLon(smlats, smlons, mapping.photoTime)
    latsCenter, lonsCenter = smToLatLon(smlatsCenter, smlonsCenter, mapping.photoTime)    
    newMapping = GenericMapping(lats, lons, latsCenter, lonsCenter, 
                                mapping.elevation, mapping.altitude, 
                                mapping.img, mapping.cameraPosGCRS, 
                                mapping.photoTime, mapping.identifier)
    return newMapping
