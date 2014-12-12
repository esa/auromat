# Copyright European Space Agency, 2013

from __future__ import division, absolute_import, print_function

from six.moves.urllib.error import HTTPError
from datetime import datetime, timedelta
import os

import numpy as np
import numpy.ma as ma
from spacepy import pycdf

from auromat.mapping.mapping import BaseMapping, MappingCollection,\
    BaseMappingProvider, sanitize_data
import auromat.utils
from auromat.util.decorators import inherit_docs
from auromat.coordinates.geodesic import wgs84A, wgs84B
from auromat.coordinates.transform import geodetic2Ecef, geodetic2EcefZero,\
    ecef2Geodetic, latLonToJ2000
from auromat.coordinates.intersection import ellipsoidLineIntersection
from auromat.util.os import touch
from auromat.util.url import downloadFile

stations = ['atha', 'chbg', 'ekat', 'fsim', 'fsmi', 'fykn',
            'gako', 'gbay', 'gill', 'inuv', 'kapu', 'kian',
            'kuuj', 'mcgr', 'nrsq', 'pgeo', 'pina', 'rank',
            'snap', 'snkq', 'talo', 'tpas', 'whit', 'yknf']

L1BaseUrl = 'http://themis.ssl.berkeley.edu/data/themis/thg/l1/asi/'
L2BaseUrl = 'http://themis.ssl.berkeley.edu/data/themis/thg/l2/asi/cal/'

L1Prefix = '{station}/{year}/{month}/'
L1Filename = 'thg_l1_asf_{station}_{date}_v01.cdf'
L2Filename = 'thg_l2_asc_{station}_19700101_v01.cdf'

@inherit_docs
class ThemisMappingProvider(BaseMappingProvider):
    
    def __init__(self, cdfL1CacheFolder, cdfL2CacheFolder, altitude=110, 
                 minBrightness=None, maxBrightness=None,
                 offline=False):
        BaseMappingProvider.__init__(self, maxTimeOffset=2)
        self.offline = offline
        if not os.path.exists(cdfL1CacheFolder) and not offline:
            os.makedirs(cdfL1CacheFolder)
        if not os.path.exists(cdfL2CacheFolder) and not offline:
            os.makedirs(cdfL2CacheFolder)
        self.cdfL1CacheFolder = cdfL1CacheFolder
        self.cdfL2CacheFolder = cdfL2CacheFolder
        self.altitude = altitude
        self.minBrightness = minBrightness
        self.maxBrightness = maxBrightness
        
    @property
    def range(self):
        raise NotImplementedError
        
    def contains(self, date):
        for station in stations:
            if getL1Data(self.cdfL1CacheFolder, station, date, maxTimeOffset=self.maxTimeOffset):
                return True
        return False
    
    def download(self, dateBegin, dateEnd):
        """
        Download data in the given interval.
        An error is raised if self.offline is True and the data has not
        been downloaded yet.
        """
        if not (dateBegin and dateEnd):
            raise ValueError('start and end dates must be given')
        if dateBegin > dateEnd:
            raise ValueError('start date must be earlier than end date')
        
        # cache everything between the given date range
        # split range into 1h resolution dates and download all
        dateBegin = datetime(*dateBegin.timetuple()[:4])
        dateEnd = datetime(*dateEnd.timetuple()[:4])
        hours = int((dateEnd-dateBegin).total_seconds())//60//60
        dates = [dateBegin + timedelta(hours=h) for h in range(hours+1)]
        
        for station in stations:
            if self.offline:
                if not hasL2Data(cdfL2CacheFolder, station):
                    raise RuntimeError('offline=True but L2 data not cached yet')            
            else:
                downloadL2Data(self.cdfL2CacheFolder, station)
            
            for date in dates:
                if self.offline:
                    if hasL1Data(cdfL1CacheFolder, station, date) is False:
                        raise RuntimeError('offline=True but L1 data not cached yet')
                else:
                    downloadL1Data(self.cdfL1CacheFolder, station, date)
    
    def get(self, date):
        mappings = getMappings(date, self.cdfL1CacheFolder, self.cdfL2CacheFolder, 
                               self.altitude, self.maxTimeOffset, self.minBrightness, self.maxBrightness,
                               offline=self.offline)
        if mappings.empty:
            raise ValueError('No THEMIS mappings found at ' + str(date) + ' +- ' + str(self.maxTimeOffset) + 's')
        return mappings
    
    def getById(self, identifier):
        raise NotImplementedError
    
    def getSequence(self, dateBegin, dateEnd):
        raise NotImplementedError
    
@sanitize_data
@inherit_docs
class ThemisMapping(BaseMapping):
    def __init__(self, lats, lons, latsCenter, lonsCenter, elev, alti, img, cameraPosGCRS, photoTime, 
                 station, minBrightness=None, maxBrightness=None):
        """
        
        :param lats: (h+1,w+1) in degrees
        :param lons: (h+1,w+1) in degrees
        :param latsCenter: (h,w) in degrees
        :param lonsCenter: (h,w) in degrees
        :param elev: elevation in degrees for each pixel center (h,w), can be None  
        :param alti: the altitude in km onto which the image was mapped (e.g. 110)
        :param img: masked array, (h,w) or (h,w,3) in [0,255] as int or float, can have NaN's
        :param cameraPosGCRS: [x,y,z] in km
        :param photoTime: datetime object
        :param string station: 
        :param minBrightness: 
        :param maxBrightness:
        """
        assert img.ndim == 2
        h, w = img.shape[0], img.shape[1]
        assert lats.shape == lons.shape == (h+1, w+1)
        assert elev is None or elev.shape == (img.shape[0], img.shape[1])
        
        # adapted from web filenames: RANK.2013.09.26.05.03.gif
        identifier = station + '.' + photoTime.strftime('%Y.%m.%d.%H.%M.%S')
        
        BaseMapping.__init__(self, alti, cameraPosGCRS, photoTime, identifier)
        self._img = img[:,:,None]
        self._lats = ma.masked_invalid(lats, copy=False)
        self._lons = ma.masked_invalid(lons, copy=False)
        self._latsCenter = ma.masked_invalid(latsCenter, copy=False)
        self._lonsCenter = ma.masked_invalid(lonsCenter, copy=False)
        self._elevation = ma.masked_invalid(elev, copy=False)
        # allow to be changed from outside:
        self.minBrightness = minBrightness
        self.maxBrightness = maxBrightness
    
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
    
    @property
    def img(self):
        return self._img
    
    @property
    def img_unmasked(self):
        return self._img.data
    
    @property
    def rgb(self):
        img = self.brightness_scaled(self.img)    
        rgb = np.repeat(img, 3, 2)
        rgb = np.require(rgb, dtype=np.uint8)
        return rgb
    
    @property
    def rgb_unmasked(self):
        img = self.brightness_scaled(self.img_unmasked)
        rgb = np.repeat(img, 3, 2)
        rgb = np.require(rgb, dtype=np.uint8)
        return rgb
        
    def brightness_scaled(self, img):
        # brightness scaling, taken from thm_asi_create_mosaic.pro
        if self.minBrightness is not None or self.maxBrightness is not None:
            img = bytscl(self.img, min_=self.minBrightness, max_=self.maxBrightness, top=255)
        else:
            med = np.median(self.img[self.img > 1])
            img = self.img/med*64
            img = np.minimum(img, 255)
        return img
             
        
    def createResampled(self, lats, lons, latsCenter, lonsCenter, elevation, img):
        mapping = ThemisMapping(lats, lons, latsCenter, lonsCenter, elevation, self.altitude, img, 
                                self.cameraPosGCRS, self.photoTime, 
                                self.minBrightness, self.maxBrightness)
        return mapping
    
def bytscl(array, max_=None , min_=None , top=255 ):
    """
    :see: https://github.com/mperrin/misc_astro/blob/master/idlastro_ports/idlbase.py
    :license: https://github.com/mperrin/misc_astro/blob/master/LICENSE.rst
    """
    # see http://star.pst.qub.ac.uk/idl/BYTSCL.html
    # note that IDL uses slightly different formulae for bytscaling floats and ints. 
    # here we apply only the FLOAT formula...

    if max_ is None: max_ = np.nanmax(array)
    if min_ is None: min_ = np.nanmin(array)
    
    return np.maximum(np.minimum(  
        ((top+0.9999)*(array-min_)/(max_-min_)).astype(np.int16)
        , top),0)
        
def reproject(latLonASI, latsRef, lonsRef, heightRef, heightNew):
    """
    
    :param latLonASI: tuple of latitude,longitude of ASI
    :param latsRef: latitudes of pixel corners for reference height in degrees
    :param lonsRef: longitudes of pixel corners for reference height in degrees
    :param heightRef: reference height in km (above ground)
    :param heightNew: new height in km (above ground)
    :rtype: tuple of reprojected latitudes and longitudes 
    """
    latASI, lonASI = latLonASI
    xASI,yASI,zASI = geodetic2EcefZero(np.deg2rad(latASI), np.deg2rad(lonASI))
    
    # reconstruct direction vector as seen from ASI
    x,y,z = geodetic2Ecef(np.deg2rad(latsRef), np.deg2rad(lonsRef), heightRef)
    x -= xASI
    y -= yASI
    z -= zASI
    
    # reproject
    a = wgs84A + heightNew
    b = wgs84B + heightNew
    direction = np.transpose([x,y,z])
    intersection = ellipsoidLineIntersection(a, b, [xASI,yASI,zASI], direction.reshape(-1,3))
    intersection = intersection.reshape(direction.shape)
    x, y, z = intersection.transpose()
    latsNew, lonsNew = ecef2Geodetic(x, y, z, wgs84A, wgs84B)
    np.rad2deg(latsNew, latsNew)
    np.rad2deg(lonsNew, lonsNew)
    return latsNew, lonsNew

def maskByL2(mask, img):
    """
    Masks image pixels using the L2 mask.
    
    WARNING: Don't use this method as the L2 masks seem to contain
             inconsistent data (interpretation of 0 and 1 mixed up in one case)
    
    :param mask: L2 mask of shape (w,w)
    :param ndarray img: grayscale img (w,w)
    :rtype: ndarray<float32> with masked pixels set to NaN
    """
    img = img.astype(np.float32)
    img[mask==1] = np.nan
    
    return img

def hasL2Data(cdfL2CacheFolder, station):
    filename = L2Filename.format(station=station)
    path = os.path.join(cdfL2CacheFolder, filename)
    return os.path.exists(path)

def downloadL2Data(cdfL2CacheFolder, station):   
    if hasL2Data(cdfL2CacheFolder, station):
        return
    
    # download remotely, if available
    filename = L2Filename.format(station=station)
    path = os.path.join(cdfL2CacheFolder, filename)
    url = L2BaseUrl + filename
    downloadFile(url, path)

def getL2Data(cdfL2CacheFolder, station):
    """
    
    :param cdfL2CacheFolder: full path to folder where L2 CDF files are/should be stored
    :param station: name of the station in lower case
    :rtype: tuple (latASI,lonASI), az, el, latsRef, lonsRef, heightsRef (km)
    :raises: DownloadError (in case of network/IO problems)
    """    
    filename = L2Filename.format(station=station)
    path = os.path.join(cdfL2CacheFolder, filename)
        
    with pycdf.CDF(path) as cdf:
        latASI = cdf['thg_asc_' + station + '_glat'][...] # station coords
        lonASI = cdf['thg_asc_' + station + '_glon'][...]
        
        # coordinates are only defined for useful pixels, rest is nan
        az = cdf['thg_asf_' + station + '_azim'][0] # azimuth of pixel center in degrees (256,256)
        el = cdf['thg_asf_' + station + '_elev'][0] # elevation of pixel center in degrees (256,256)
        
        latsRef = cdf['thg_asf_' + station + '_glat'][0] # lats in degrees of pixel corners for multiple aurora heights (257,257,3)
        lonsRef = cdf['thg_asf_' + station + '_glon'][0] # lons, dito
        
        heightsRef = cdf['thg_asf_' + station + '_alti'][...] # reference heights in meters
        
        # the masks don't contain useful and consistent data, ignore for now
        #mask = cdf['thg_asf_' + station + '_mask'][0] # mask (0 or 1) for every pixel (256,256)
        
        # multiply and flat multiply are 1.0, offset is 2500 for all stations for every pixel
#        multiply = cdf['thg_asf_' + station + '_multiply'][...]
#        flat = cdf['thg_asf_' + station + '_flat'][...]
#        offset = cdf['thg_asf_' + station + '_offset'][...]

    
    # turn (257,257,3) into (3,257,257) to make it more convenient
    latsRef = np.rollaxis(latsRef, 2)
    lonsRef = np.rollaxis(lonsRef, 2)
    
    return (latASI,lonASI), az, el, latsRef, lonsRef, heightsRef/1000

def getL1Filename(station, date):
    return L1Filename.format(station=station, date=date.strftime('%Y%m%d%H'))

def hasL1Data(cdfL1CacheFolder, station, date, retry404After = timedelta(days=30)):
    filename = getL1Filename(station, date)
    path = os.path.join(cdfL1CacheFolder, filename)
    if os.path.exists(path):
        return True
    
    path404 = path + '.404'
    if os.path.exists(path404):
        mtime = datetime.fromtimestamp(os.path.getmtime(path404))
        if datetime.now() - mtime > retry404After:
            os.remove(path404)
        else:
            return '404'
    return False

def downloadL1Data(cdfL1CacheFolder, station, date, retry404After = timedelta(days=30)):
    """
    
    :param datetime.timedelta retry404After: amount of time after which 404'd requests are retried
    """
    # check if it exists in the cache
    status = hasL1Data(cdfL1CacheFolder, station, date)
    if status is True:
        return True
    if status == '404':
        return False
    # if false, try downloading
    
    filename = getL1Filename(station, date)
    path = os.path.join(cdfL1CacheFolder, filename)
         
    # download remotely, if available
    url = L1BaseUrl + L1Prefix.format(station=station, year=date.strftime('%Y'),
                                      month=date.strftime('%m')) + filename
    
    path404 = path + '.404'
    try:
        downloadFile(url, path, unifyErrors=False)
        
    except HTTPError as e:
        if e.code == 404:
            touch(path404)
        return False
            
    except Exception as e: # e.g. no disk space left
        print(repr(e))
        return False
    
    return True

def getL1Data(cdfL1CacheFolder, station, date, maxTimeOffset=2):
    """
    Loads a single image (if available) corresponding to the given date.
    
    :param cdfL1CacheFolder:
    :param station:
    :param date:
    :param maxTimeOffset: maximum difference in seconds between nearest image and given date
    """    
    filename = getL1Filename(station, date)
    path = os.path.join(cdfL1CacheFolder, filename)
    
    with pycdf.CDF(path) as cdf:
        # search for image at given time
        epoch = cdf['thg_asf_' + station + '_epoch']
        idx = auromat.utils.findNearest(epoch, date)
        offset = abs(epoch[idx]-date).total_seconds()
        if offset > maxTimeOffset:
            print('image time offset too big:', offset, 'seconds')
            return None, None
        return cdf['thg_asf_' + station][idx], epoch[idx]

def mappingSingleASI(station, date, cdfL1CacheFolder, cdfL2CacheFolder, 
                     maxTimeOffset=2, altitude=110,
                     minBrightness=None, maxBrightness=None, offline=False):
    if offline and hasL1Data(cdfL1CacheFolder, station, date) is False:
        raise RuntimeError('offline=True but L1 data not cached yet')
    
    if not downloadL1Data(cdfL1CacheFolder, station, date):
        return None
    
    img, imgDate = getL1Data(cdfL1CacheFolder, station, date, maxTimeOffset)
    if img is None:
        return None
    
    img = ma.masked_array(img)
    
    if not offline:
        downloadL2Data(cdfL2CacheFolder, station)
    latLonASI, _, el, latsRef, lonsRef, heightsRef = getL2Data(cdfL2CacheFolder, station) 
    
    if altitude*1000 in heightsRef:
        refIdx = np.where(heightsRef==altitude*1000)[0][0]
        lats, lons = latsRef[refIdx], lonsRef[refIdx]
    else:
        # TODO reprojection leads to border artifacts (long polygons) for minElevationToPlot <= 3
        refIdx = 0
        lats, lons = reproject(latLonASI, latsRef[refIdx], lonsRef[refIdx], heightsRef[refIdx], altitude)
    
    # THEMIS mappings don't span across the discontinuity
    # so it is easy to calculate the pixel centers
    latsCenter = (lats[:-1,:-1] + lats[1:,:-1] + lats[:-1,1:] + lats[1:,1:])/4
    lonsCenter = (lons[:-1,:-1] + lons[1:,:-1] + lons[:-1,1:] + lons[1:,1:])/4
    
#    img = maskByL2(mask, img)
        
    # for THEMIS, downsampling is only useful if the resolution should be *much* lower
    # otherwise downsampling with similar pixel density just introduces additional visible artifacts 
    #lats, lons, img = auromat.resample.resample(lats, lons, img, skipDiscontinuityCheck=True, skipPoleCheck=True)
    
    # 2500 is the _offset value which is equal in all THEMIS L2 files for all pixels.
    # The multipliers _multiply and _flat are all 1.0 still. Should this change in the future,
    # then it has to be changed here as well (by properly applying the matrices). For now,
    # this is a fast short-cut.
    img -= 2500
    
    latASI, lonASI = latLonASI    
    cameraPosGCRS = latLonToJ2000(latASI, lonASI, 0, imgDate)
        
    mapping = ThemisMapping(lats, lons, latsCenter, lonsCenter, el, altitude, img, cameraPosGCRS, 
                            imgDate, station, minBrightness, maxBrightness)
    
    # THEMIS L2 data seems to be partly wrong at very low elevation angles.
    # Therefore we already apply a mask here, otherwise a possible resampling
    # step by the enduser would lead to errors.
    mapping = mapping.maskedByElevation(1)
    
    return mapping
    
def getMappings(photoTime, cdfL1CacheFolder, cdfL2CacheFolder, 
                altitude=110, maxTimeOffset=2, minBrightness=None, maxBrightness=None,
                offline=False):
    """
    Returns a mapping collection for all available ASIs.
    """
    mappings = []
    
    for station in stations:
        mapping = mappingSingleASI(station, photoTime, cdfL1CacheFolder, cdfL2CacheFolder,
                                   maxTimeOffset=maxTimeOffset, altitude=altitude,
                                   minBrightness=minBrightness, maxBrightness=maxBrightness,
                                   offline=offline)
        if mapping is None:
            continue
        
        mappings.append(mapping)
        
    identifier = 'THEMIS.' + photoTime.strftime('%Y.%m.%d.%H.%M.%S')

    mappingColl = MappingCollection(mappings, identifier, mayOverlap=True)
    return mappingColl


if __name__ == '__main__':
    import auromat.draw; 'OPTIONAL'
    
    cdfL1CacheFolder = '/home/mriecher/data/themis/l1'
    cdfL2CacheFolder = '/home/mriecher/data/themis/l2'
    
    date = datetime(2012,2,4,7,56,26)
    
    mappings = getMappings(date, cdfL1CacheFolder, cdfL2CacheFolder)
    mappings = mappings.maskedByElevation(10)
        
    auromat.draw.saveFig('themis_stereo.png', auromat.draw.drawStereographic(mappings))
