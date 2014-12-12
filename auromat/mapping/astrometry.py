# Copyright European Space Agency, 2013

from __future__ import division, absolute_import, print_function

import time
import numpy as np
import numpy.ma as ma

import auromat.utils
from auromat.util.decorators import lazy_property, inherit_docs
from auromat.mapping.mapping import BaseMapping, GenericMapping, inflatedEarthIntersection
from auromat.coordinates.transform import j2000ToLatLon, j2000ToMLatMLT,\
    spherical_to_cartesian
from auromat.utils import vectorLengths
from auromat.coordinates.wcs import pix2world

@inherit_docs
class BaseAstrometryMapping(BaseMapping):
    """
    A mapping which calculates its coordinates based on the camera position and its WCS definition.
    """

    def __init__(self, wcsHeader, alti, cameraPosGCRS, photoTime, identifier, metadata={},
                 fastCenterCalculation=False):
        """
        
        :param alti: mapping altitude in km
        :param fastCenterCalculation:
            Calculates center coordinates directly from the mean of the
            corner coordinates.
        """
        BaseMapping.__init__(self, alti, cameraPosGCRS, photoTime, identifier, metadata)
        self._wcsHeader = wcsHeader
        self.fastCenterCalculation = fastCenterCalculation
        if fastCenterCalculation:
            # When calculating center coordinates from corners it is already assured
            # that the centers are only defined when all its corners are defined.
            # This then carries over to the elevation calculation.
            # Note that img must be masked manually in the subclass!
            self.isSanitized = True
            
        self._mlatmlt = None
        self._mlatmltCenter = None

    @property
    def wcsHeader(self):
        return self._wcsHeader
        
    @lazy_property
    def cameraToPixelCornerDirection(self):
        """
        Direction vector for each pixel corner.
        """
        return pixelDirection(self.wcsHeader, corner=True)
    
    @lazy_property
    def cameraToPixelCenterDirection(self):
        """
        Direction vector for each pixel center.
        """
        if self.fastCenterCalculation:
            return self._calcCenters(self.cameraToPixelCornerDirection)
        else:
            return pixelDirection(self.wcsHeader, corner=False)
        
    @property
    def ra(self):
        """
        Right ascension for each pixel center.
        For debugging purposes only!
        """
        photoWidth, photoHeight = self.wcsHeader['IMAGEW'], self.wcsHeader['IMAGEH']
        ra, _ = pix2world(self.wcsHeader, photoWidth, photoHeight)
        return ra
    
    @property
    def dec(self):
        """
        Descension for each pixel center.
        For debugging purposes only!
        """
        photoWidth, photoHeight = self.wcsHeader['IMAGEW'], self.wcsHeader['IMAGEH']
        _, dec = pix2world(self.wcsHeader, photoWidth, photoHeight)
        return dec
    
    @lazy_property
    def intersectionInflatedCorner(self):
        """
        Returns the point of intersection with the inflated earth for each pixel corner.
        """
        intersectionInflated = inflatedEarthIntersection(self.cameraToPixelCornerDirection.reshape(-1,3),
                                                         self.cameraPosGCRS, self.altitude)
        return intersectionInflated.reshape(self.cameraToPixelCornerDirection.shape)

    @lazy_property
    def intersectionInflatedCenter(self):
        """
        Returns the point of intersection with the inflated earth for each pixel center.
        """
        if self.fastCenterCalculation:
            intersectionInflated = self._calcCenters(self.intersectionInflatedCorner)
        else:
            intersectionInflated = inflatedEarthIntersection(self.cameraToPixelCenterDirection.reshape(-1,3),
                                                             self.cameraPosGCRS, self.altitude)
            intersectionInflated = intersectionInflated.reshape(self.cameraToPixelCenterDirection.shape)
        return intersectionInflated
    
    @property
    def distance(self):
        """
        Distance for each pixel center between camera and intersection point.
        For debugging purposes only!
        """
        vectors = (self.intersectionInflatedCenter - self.cameraPosGCRS).reshape(-1, 3)
        lens = vectorLengths(vectors).reshape(self.intersectionInflatedCenter.shape[0], self.intersectionInflatedCenter.shape[1])
        return lens
    
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
        latDeg, lonDeg = j2000ToLatLon(self.intersectionInflatedCorner.reshape(-1,3), self.photoTime)
        latDeg = latDeg.reshape(self.intersectionInflatedCorner.shape[0], self.intersectionInflatedCorner.shape[1])
        lonDeg = lonDeg.reshape(self.intersectionInflatedCorner.shape[0], self.intersectionInflatedCorner.shape[1])
        latDeg, lonDeg = ma.masked_invalid(latDeg, copy=False), ma.masked_invalid(lonDeg, copy=False)
        return latDeg, lonDeg
    
    @lazy_property
    def _latsLonsCenter(self):
        latDeg, lonDeg = j2000ToLatLon(self.intersectionInflatedCenter.reshape(-1,3), self.photoTime)
        latDeg = latDeg.reshape(self.intersectionInflatedCenter.shape[0], self.intersectionInflatedCenter.shape[1])
        lonDeg = lonDeg.reshape(self.intersectionInflatedCenter.shape[0], self.intersectionInflatedCenter.shape[1])
        latDeg, lonDeg = ma.masked_invalid(latDeg, copy=False), ma.masked_invalid(lonDeg, copy=False)
        return latDeg, lonDeg
    
    @staticmethod
    def _calcCenters(corners):
        centers = corners[:-1,:-1] + corners[:-1,1:]
        centers += corners[1:,1:]
        centers += corners[1:,:-1]
        centers /= 4
        return centers
    
    def setDirty(self):
        """
        Overrides BaseMapping.setDirty()
        """
        self._mlatmlt = None
        self._mlatmltCenter = None        
        super(BaseAstrometryMapping, self).setDirty()
    
    @property
    def mLatMlt(self):
        """
        Overrides BaseMapping.mLatMlt. 
        We directly use the J2000 coordinates (instead of GEO) as source here to minimize
        numerical errors caused by additional transformations and to gain some speed.
        """
        if self._mlatmlt is None:
            mlat, mlt = j2000ToMLatMLT(self.intersectionInflatedCorner.reshape(-1,3), self.photoTime)
            mlat = mlat.reshape(self.intersectionInflatedCorner.shape[0], self.intersectionInflatedCorner.shape[1])
            mlt = mlt.reshape(self.intersectionInflatedCorner.shape[0], self.intersectionInflatedCorner.shape[1])
            mask = ma.getmaskarray(self.lats)
            self._mlatmlt = ma.masked_array(mlat, mask), ma.masked_array(mlt, mask)
        return self._mlatmlt
    
    @property
    def mLatMltCenter(self):
        """
        Overrides BaseMapping.mLatMltCenter.
        We directly use the J2000 coordinates (instead of Lat/Lon->GEO) as source here to minimize
        numerical errors caused by additional transformations and to gain some speed.
        """
        if self._mlatmltCenter is None:
            mlat, mlt = j2000ToMLatMLT(self.intersectionInflatedCenter.reshape(-1,3), self.photoTime)
            mlat = mlat.reshape(self.intersectionInflatedCenter.shape[0], self.intersectionInflatedCenter.shape[1])
            mlt = mlt.reshape(self.intersectionInflatedCenter.shape[0], self.intersectionInflatedCenter.shape[1])
            mask = ma.getmaskarray(self.latsCenter)
            self._mlatmltCenter = ma.masked_array(mlat, mask), ma.masked_array(mlt, mask)
        return self._mlatmltCenter
        
    @lazy_property
    def elevation(self):
        """
        Elevation for each pixel center. Angles are between 0 (horizon) and 90 (nadir) degrees.
        """
        pixelToCameraDirection = -self.cameraToPixelCenterDirection
        intersectionUnit = auromat.utils.unitVectors(self.intersectionInflatedCenter.reshape(-1,3))
        alpha = auromat.utils.angleBetween(pixelToCameraDirection.reshape(-1,3), intersectionUnit)
        alpha = alpha.reshape(self.intersectionInflatedCenter.shape[0], self.intersectionInflatedCenter.shape[1])
        np.rad2deg(alpha, alpha)
        np.subtract(90, alpha, alpha)
        elevation = ma.masked_invalid(alpha, copy=False)
        return elevation
    
    def createResampled(self, lats, lons, latsCenter, lonsCenter, elevation, img):
        mapping = GenericMapping(lats, lons, latsCenter, lonsCenter, elevation, self.altitude, img, 
                                 self.cameraPosGCRS, self.photoTime, self.identifier,
                                 metadata=self.metadata)
        return mapping
 
class ImageMaskAstrometryMixin(object):
    """
    Helper mixin class which handles the fastCenterCalculation=True case for
    images.
    
    Has to be applied as last step in the hierarchy, e.g.:
    
    MyClass(ImageMaskAstrometryMixin, FileImageMixin, BaseSpacecraftMapping)
    """
    
    def __init__(self):
        self._img = None
    
    @property
    def img(self):
        if self._img is None:
            img = super(ImageMaskAstrometryMixin, self).img
            if self.fastCenterCalculation:
                # see BaseAstrometryMapping.__init__ for why this is needed
                mask = ma.getmaskarray(self.latsCenter)
                imgMask = np.repeat(mask[:,:,None], 3, 2)
                img = ma.masked_array(img.data, mask=imgMask)
            self._img = img
        return self._img

def pixelDirection(fitsWcsHeader, corner=True):
    """
    Calculates the direction vector in ICRS for each pixel corner or center, given a WCS solution.
    
    Technically, the returned cartesian ICRS coordinates would have to be converted
    to GCRS/J2000 for the earth-intersection calculations,
    but this would involve distances which aren't available here.
    The solution is that we pretend that the ICRS coordinates are GCRS coordinates. This is acceptable because
    the error is lower than pixel resolution. 
    (ICRF/GCRS diff is around 0.01" while ISS photography is around 20-100"/px)
    
    :param dictionary fitsWcsHeader: must also contain IMAGEW, IMAGEH in pixels
    :rtype: unit direction vector array of shape (IMAGEH+1, IMAGEW+1, 3) if corner==True,
            otherwise (IMAGEH, IMAGEW, 3) 
    """
    photoWidth, photoHeight = fitsWcsHeader['IMAGEW'], fitsWcsHeader['IMAGEH']

    t0 = time.time()   
    camToPixelDirection = pix2world(fitsWcsHeader, photoWidth, photoHeight, corner=corner, ascartesian=True)
    print('pix2world:', time.time()-t0, 's')
    
    assert np.all(np.array(camToPixelDirection.shape) == np.array([photoHeight + corner, photoWidth + corner, 3])), \
           '{} != {}'.format(camToPixelDirection.shape, [photoHeight + corner, photoWidth + corner, 3])
    
    return camToPixelDirection
