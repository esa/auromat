# Copyright European Space Agency, 2013

from __future__ import division, print_function

from collections import namedtuple
import datetime
import os
import fnmatch

import numpy as np
import numpy.ma as ma
from numpy.core.umath_tests import matrix_multiply

from astropy.coordinates.angles import Angle
import astropy.units as u

from auromat.coordinates.geodesic import wgs84A, wgs84B
from auromat.mapping.mapping import BaseMapping, BoundingBox, GenericMapping,\
    BaseMappingProvider, MappingCollection, \
    inflatedEarthIntersection, DefaultRGBMixin, sanitize_data
from auromat.coordinates.transform import Y, Z, rotation_matrix, geodetic2EcefZero,\
    ecef2Geodetic, spherical_to_cartesian, latLonToJ2000
import auromat.utils
from auromat.util.decorators import lazy_property, inherit_docs
from auromat.util.image import loadImage

fileDateTimeFormat = '%y%m%d_%H%M%S'

# NOTE: xc, yc, k are relative to a 512x512 image
# NOTE: xc is vertical, yc is horizontal!
CalibrationData = namedtuple('CalibrationData', ['station', 'validFrom', 'validTo',
                                                 'lat', 'lon', 'xc', 'yc', 
                                                 'k', 'rotation',
                                                 'boundingBoxSimple'])

@inherit_docs
class MIRACLEMappingProvider(BaseMappingProvider):
    
    def __init__(self, imageFolder, altitude=110, simple=False, maxTimeOffset=5):
        """
        
        :param imageFolder: path to folder containing images and cal.txt file
        :param altitude: the altitude in km onto which the image was mapped (e.g. 110)
                               Note that for simple=True, this is always 110km
        :param bool simple: whether to use simple mapping (constant lat-lon grid) or
                            intersection-based mapping
        :param maxTimeOffset: in seconds
        
        """
        self.imageFolder = imageFolder
        self.altitude = altitude
        self.simple = simple
        self.imageFileExtension = 'jpg'
        self.maxTimeOffset = maxTimeOffset
        
        fileNames = os.listdir(imageFolder)
        imageFileNames = sorted(fnmatch.filter(fileNames, '*.' + self.imageFileExtension))
        imageStations = [f[:3] for f in imageFileNames]
        self.imageDates = [datetime.datetime.strptime(f[3:16], fileDateTimeFormat) for f in imageFileNames]
        
        self.images = dict() # station -> [(filename,date)]
        for station in set(imageStations):
            self.images[station] = [(f,d) for (f,s,d) in zip(imageFileNames, imageStations, self.imageDates) if s == station]
    
    def __len__(self):
        return len(self.imageDates)
    
    @property
    def range(self):
        dates = sorted(self.imageDates)
        return dates[0], dates[-1]
    
    def contains(self, date):
        for images in self.images.values():
            dates = [d for (_,d) in images]
            
            # search for image at given time
            idx = auromat.utils.findNearest(dates, date)
            offset = abs(dates[idx]-date).total_seconds()
            if offset <= self.maxTimeOffset:
                return True
        
        return False
        
    def get(self, date):
        mappings = []
        
        # for each station find the image closest to date+-maxTimeOffset
        for images in self.images.values():
            dates = [d for (_,d) in images]
            
            # search for image at given time
            idx = auromat.utils.findNearest(dates, date)
            offset = abs(dates[idx]-date).total_seconds()
            if offset <= self.maxTimeOffset:
                imagePath = os.path.join(self.imageFolder, images[idx][0])
                mappings.append(getMapping(imagePath, self.altitude, self.simple))

        ident =  'MIRACLE.' + date.strftime('%Y.%m.%d.%H.%M.%S')            
        mappingColl = MappingCollection(mappings, identifier=ident, mayOverlap=True)
        return mappingColl
        
    def getById(self, identifier):
        raise NotImplementedError
    
    def getSequence(self, dateBegin=None, dateEnd=None):
        raise NotImplementedError

@sanitize_data
@inherit_docs
class MIRACLEMapping(DefaultRGBMixin, BaseMapping):
    """
    A mapping defined using an image and calibration data from FMI MIRACLE.
    """
    
    def __init__(self, calData, imagePath, photoTime, alti, simple=False):
        """
        
        :param CalibrationData calData:
        :param photoTime: datetime object
        :param alti: the altitude in km onto which the image was mapped (e.g. 110)
                     Note that for simple=True, this is always 110km
        :param bool simple: whether to use simple mapping (constant lat-lon grid) or
                            intersection-based mapping
        """
        
        # TODO SOD ASI images include wavelength in filename, add to identifier?
        identifier = calData.station + '.' + photoTime.strftime('%Y.%m.%d.%H.%M.%S')
        
        xASI,yASI,zASI = geodetic2EcefZero(np.deg2rad(calData.lat), np.deg2rad(calData.lon))
        self.cameraPosGEO = [xASI, yASI, zASI]

        cameraPosGCRS = latLonToJ2000(calData.lat, calData.lon, 0, photoTime)
        
        alti = 110 if simple or alti is None else alti
        
        BaseMapping.__init__(self, alti, cameraPosGCRS, photoTime, identifier)
        
        self._calData = calData
        self._simple = simple
        self._imagePath = imagePath
        self._img = None
        self._img_unmasked = None
    
    @property
    def img(self):
        if self._img is None:
            self._img = ma.masked_array(self._img_unmasked)   
        return self._img
    
    @property
    def img_unmasked(self):
        if self._img_unmasked is None:
            rgb = loadImage(self._imagePath)
            if rgb.shape[0] != rgb.shape[1]:
                # contains caption below image, cut it off
                rgb = rgb[:rgb.shape[1],:]
                assert rgb.shape == (rgb.shape[1],rgb.shape[1],3)
            self._img_unmasked = rgb
        return self._img_unmasked
    
    @property
    def lats(self):
        lats, _ = self._latsLonsCorner
        return lats
        
    @property
    def lons(self):
        _, lons = self._latsLonsCorner
        return lons
    
    @property
    def latsCenter(self):
        lats, _ = self._latsLonsCenter
        return lats
        
    @property
    def lonsCenter(self):
        _, lons = self._latsLonsCenter
        return lons
        
    @lazy_property
    def _latsLonsCorner(self):
        lats, lons = self._calculateLatsLons(center=False)
        lats = ma.masked_invalid(lats, copy=False)
        lons = ma.masked_invalid(lons, copy=False)
        return lats, lons
    
    @lazy_property
    def _latsLonsCenter(self):
        lats, lons = self._calculateLatsLons(center=True)
        lats = ma.masked_invalid(lats, copy=False)
        lons = ma.masked_invalid(lons, copy=False)
        return lats, lons
    
    def _calculateLatsLons(self, center=True):
        if self._simple:
            bb = self._calData.boundingBoxSimple
            
            deltaLat = (bb.latNorth - bb.latSouth)/self.img.shape[0]
            deltaLon = (bb.lonEast - bb.lonWest)/self.img.shape[1]
            
            if center:
                latSpace = np.linspace(bb.latNorth-deltaLat/2, bb.latSouth+deltaLat/2, self.img.shape[0])
                lonSpace = np.linspace(bb.lonWest+deltaLon/2, bb.lonEast-deltaLon/2, self.img.shape[0])
            else:
                latSpace = np.linspace(bb.latNorth, bb.latSouth, self.img.shape[0]+1)
                lonSpace = np.linspace(bb.lonWest, bb.lonEast, self.img.shape[0]+1)
                
            #lats, lons = np.meshgrid(latSpace, lonSpace, indexing='ij') # 'indexing' not supported in np 1.6
            lats, lons = np.dstack(np.meshgrid(latSpace, lonSpace)).T
        else:
            intersectionInflated = self.intersectionInflatedCenter if center else self.intersectionInflatedCorner
            xyz = intersectionInflated.reshape(-1,3)
            lats, lons = ecef2Geodetic(xyz[:,0], xyz[:,1], xyz[:,2], wgs84A, wgs84B)
            np.rad2deg(lats, lats)
            np.rad2deg(lons, lons)
            lats = lats.reshape(intersectionInflated.shape[0], intersectionInflated.shape[1])
            lons = lons.reshape(intersectionInflated.shape[0], intersectionInflated.shape[1])
        
        return lats, lons
        
    @lazy_property
    def cameraToPixelCornerDirection(self):
        """
        Direction vector for each pixel corner.
        """        
        vecs = self._calculateCameraToPixelDirection(self.elevationCorner, self.azimuthCorner)
        return vecs
    
    @lazy_property
    def cameraToPixelCenterDirection(self):
        """
        Direction vector for each pixel center.
        """
        _, el = self.azElCenter # don't use self.elevation here, conflicts with sanitization decorator
        vecs = self._calculateCameraToPixelDirection(el, self.azimuthCenter)
        return vecs
    
    def _calculateCameraToPixelDirection(self, el, az):
        el = np.deg2rad(el)
        az = np.deg2rad(-(az-180))
        
        x,y,z = spherical_to_cartesian(1,el,az)

        vecs = np.dstack((x,y,z))
        
        # simple spherical latitude rotation works here because
        # the latitude is the geodetic latitude which is the
        # angle between the normal and the equatorial plane
        matLat = rotation_matrix(np.deg2rad(90 - self._calData.lat), Y)[:3,:3]
        matLon = rotation_matrix(np.deg2rad(-self._calData.lon), Z)[:3,:3]
        mat = np.dot(matLon, matLat) # rotate latitude first, then longitude
        
        vecs = vecs.reshape(el.shape[0]*el.shape[1], 3)
        vecsRot = matrix_multiply(mat, vecs[...,np.newaxis]).reshape(el.shape[0], el.shape[1], 3)
                
        return vecsRot
                
    @lazy_property
    def intersectionInflatedCorner(self):
        """Returns the point of intersection with the inflated earth for each pixel corner."""        
        intersectionInflated = inflatedEarthIntersection(self.cameraToPixelCornerDirection.reshape(-1,3),
                                                         self.cameraPosGEO, self.altitude)
        return intersectionInflated.reshape(self.cameraToPixelCornerDirection.shape)
    
    @lazy_property
    def intersectionInflatedCenter(self):
        """Returns the point of intersection with the inflated earth for each pixel center."""
        intersectionInflated = inflatedEarthIntersection(self.cameraToPixelCenterDirection.reshape(-1,3),
                                                         self.cameraPosGEO, self.altitude)
        return intersectionInflated.reshape(self.cameraToPixelCenterDirection.shape)

    @property
    def elevation(self):
        _, el = self.azElCenter
        return el
    
    @property
    def elevationCorner(self):
        _, el = self.azElCorner
        return el
    
    @property
    def azimuthCenter(self):
        az, _ = self.azElCenter
        return az

    @property
    def azimuthCorner(self):
        az, _ = self.azElCorner
        return az
    
    @lazy_property
    def azElCorner(self):
        """
        Azimuth and elevation for each pixel corner.
        """
        az, el = self.calculateAzEl(center=False)
        az = ma.masked_invalid(az, copy=False)
        el = ma.masked_invalid(el, copy=False)
        return az, el
    
    @lazy_property
    def azElCenter(self):
        """
        Azimuth and elevation for each pixel center.
        """
        az, el = self.calculateAzEl(center=True)
        az = ma.masked_invalid(az, copy=False)
        el = ma.masked_invalid(el, copy=False)
        return az, el
    
    def calculateAzEl(self, center=True):
        w = self.img_unmasked.shape[0]
        xc = self._calData.xc
        yc = self._calData.yc
        k = self._calData.k
        
        refW = 512
        if w != refW:
            xc *= w/refW
            yc *= w/refW
            k *= w/refW
        
        w_ = w if center else w+1
        
        center_ = np.array([xc,yc])
        ind = np.indices((w_, w_))
        # TODO verify that the (0,0) is really the upper left corner of the upper left pixel
        if center:
            ind += 0.5
        ind = np.dstack((ind[0], ind[1])).reshape((w_)*(w_), 2)
        vecs = ind - center_
        
        northVec = [-1,0] # points to top of image
        northVecs = np.repeat([northVec], len(vecs), axis=0)        
        az = auromat.utils.signedAngleBetween(vecs, northVecs).reshape(w_,w_)
        az -= self._calData.rotation
        az = Angle(az * u.rad).wrap_at(360 * u.deg).degree
        
        distCenter = np.sqrt((vecs*vecs).sum(axis=1)).reshape(w_,w_)
        z = distCenter/k
        np.rad2deg(z, z)
        elev = 90 - z
        
        return az, elev
               
    def createResampled(self, lats, lons, latsCenter, lonsCenter, elevation, img):
        mapping = GenericMapping(lats, lons, latsCenter, lonsCenter, elevation, self.altitude, img, 
                                 self.cameraPosGCRS, self.photoTime, self.identifier)
        return mapping
    
def getMapping(imagePath, alti=110, simple=False):
    filename = os.path.basename(imagePath)
    # e.g. KEV120304_172100_____4000.jpg
    station = filename[:3]
    date = datetime.datetime.strptime(filename[3:16], fileDateTimeFormat)
    
    calPath = os.path.join(os.path.dirname(imagePath), 'cal.txt')
    calData = getCalibrationData(calPath, station, date)
    
    mapping = MIRACLEMapping(calData, imagePath, date, alti, simple=simple)
    mapping = mapping.maskedByElevation(0.1) # .1 to account for rounding errors
    return mapping

def getCalibrationData(path, station, date):
    
    calDataEntries = np.loadtxt(path, dtype={'names': ('station', 'lat', 'lon', 'from', 'to', 
                                                       'xc', 'yc', 'k', 'rotation', 'lat+', 'lat-', 
                                                       'lon-', 'lon+', 'i1', 'i2', 'i3'),
                                             'formats': ('S3', 'f8', 'f8', 'f8', 'f8', 
                                                         'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 
                                                         'f8', 'f8', 'b1', 'b1', 'b1')})
    for entry in calDataEntries:
        if entry['station'] != station:
            continue
        
        fromDateY = int(entry['from'])
        fromDateM = int((entry['from']-fromDateY)*12 + 1)
        toDateY = int(entry['to'])
        toDateM = int((entry['to']-toDateY)*12 + 1)
        
        fromDate = datetime.datetime(fromDateY, fromDateM, 1)
        toDate = datetime.datetime(toDateY, toDateM+1, 1) # easier than using end of month
        
        if not fromDate <= date <= toDate:
            continue
        
        lat = entry['lat']
        lon = entry['lon']
        
        bbSimple = BoundingBox(latSouth=lat+entry['lat-'], 
                               lonWest=lon+entry['lon-'], 
                               latNorth=lat+entry['lat+'], 
                               lonEast=lon+entry['lon+'])
        
        calData = CalibrationData(station=entry['station'], validFrom=fromDate, validTo=toDate, 
                                  lat=lat, lon=lon, 
                                  xc=entry['xc'], yc=entry['yc'], k=entry['k'],
                                  rotation=entry['rotation'], boundingBoxSimple=bbSimple)
        return calData
        
    raise ValueError('No MIRACLE calibration data found for ' + station + ' station')

# TODO MIRACLEMapping is missing in sphinx docs
__all__ = map(lambda f: f.__name__,
              [MIRACLEMappingProvider, MIRACLEMapping, getMapping])