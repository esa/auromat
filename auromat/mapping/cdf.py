# Copyright European Space Agency, 2013

from __future__ import division, print_function

import os
import collections

import numpy as np
import numpy.ma as ma

from spacepy import pycdf

from auromat.mapping.mapping import BaseMappingProvider, \
    sanitize_data, BaseMapping, ArrayImageMixin, GenericMapping
from auromat.util.decorators import inherit_docs
import auromat.utils

@inherit_docs
class CDFMappingProvider(BaseMappingProvider):
    def __init__(self, cdfPaths, maxTimeOffset=3):
        self.cdfPaths = cdfPaths
        self.maxTimeOffset = maxTimeOffset
        
        # create mapping from date to (path index, cdf index)
        # cdf index is the index inside the cdf file
        # currently we only store 1 image in each, but this could change
        # e.g. storing multiple downsampled images in one file
        
        datemap = {}
        for path_idx, path in enumerate(cdfPaths):
            with pycdf.CDF(path) as root:
                dates = root['Epoch'][:]
                for cdf_idx, date in enumerate(dates):
                    if date in datemap:
                        raise ValueError('The date ' + str(date) + ' is appearing twice ' +
                                         'in the CDF files ' + path + ' and ' +
                                         cdfPaths[datemap[date][0]])
                    datemap[date] = (path_idx, cdf_idx)
        
        self.datemap = collections.OrderedDict(sorted(datemap.items()))
                     
    def __len__(self):
        return len(self.datemap)

    @property
    def range(self):
        return list(self.datemap.keys())[0], list(self.datemap.keys())[-1]
    
    def contains(self, date):
        dates = list(self.datemap.keys())
        idx = auromat.utils.findNearest(dates, date)
        offset = abs(dates[idx]-date).total_seconds()
        return offset <= self.maxTimeOffset
        
    def get(self, date):
        dates = list(self.datemap.keys())
        idx = auromat.utils.findNearest(dates, date)
        offset = abs(dates[idx]-date).total_seconds()
        if offset > self.maxTimeOffset:
            raise ValueError('Closest mapping found at ' + str(dates[idx]) + 
                             ' but offset > ' + str(self.maxTimeOffset) + ' seconds, ' +
                             'requested: ' + str(date))
        path_idx, cdf_idx = self.datemap[dates[idx]]
        return CDFMapping(self.cdfPaths[path_idx], cdf_idx)
        
    def getById(self, identifier):
        raise NotImplementedError
    
    def getSequence(self, dateBegin=None, dateEnd=None):
        if not dateBegin:
            dateBegin = self.range[0]
        if not dateEnd:
            dateEnd = self.range[1]
        dates = filter(lambda d: dateBegin <= d <= dateEnd, self.datemap.keys())
        for date in dates:
            path_idx, cdf_idx = self.datemap[date]
            yield CDFMapping(self.cdfPaths[path_idx], cdf_idx)

@sanitize_data
@inherit_docs
class CDFMapping(ArrayImageMixin, BaseMapping):
    
    # variable names in the cdf files as produced by auromat.export.cdf
    var_altitude = 'altitude'
    var_cameraPos = 'camera_pos'
    var_photoTime = 'Epoch'
    var_img = 'img'
    var_img_red = 'img_red'
    var_img_green = 'img_green'
    var_img_blue = 'img_blue'
    var_latsCenter = 'lat'
    var_lonsCenter = 'lon'
    var_zenithAngle = 'zenith_angle'
    
    def __init__(self, cdfPath, i=0):
        with pycdf.CDF(cdfPath) as root:
            var = root
            altitude = var[self.var_altitude][...]/1000
            cameraPosGCRS = var[self.var_cameraPos][i]
            photoTime = var[self.var_photoTime][i]
            
            # for three channels (RGB), each channel is stored as a
            # separate variable: img_red, img_green, img_blue
            # for grayscale, the single variable is called 'img'
            try:
                fillval = var[self.var_img].attrs['FILLVAL']
                img = np.atleast_3d(var[self.var_img][i])
                img = _convertImgDtype(img, fillval)
            except:
                fillval = var[self.var_img_red].attrs['FILLVAL']
                img_red = _convertImgDtype(var[self.var_img_red][i], fillval)
                img_green = _convertImgDtype(var[self.var_img_green][i], fillval)
                img_blue = _convertImgDtype(var[self.var_img_blue][i], fillval)
                img = ma.dstack((img_red, img_green, img_blue))
                        
            latsCenter = var[self.var_latsCenter][i]
            lonsCenter = var[self.var_lonsCenter][i]
            lats = var[var[self.var_latsCenter].attrs['bounds']][i]
            lons = var[var[self.var_lonsCenter].attrs['bounds']][i]
            
            # TODO read in MLat/MLT as well if available
                        
            self._latsCenter = ma.masked_invalid(latsCenter)
            self._lonsCenter = ma.masked_invalid(lonsCenter)
            self._lats = ma.masked_invalid(lats)
            self._lons = ma.masked_invalid(lons)
            self._elevation = ma.masked_invalid(90 - var[self.var_zenithAngle][i])
            
            metadata = root.attrs
            
            assert var[self.var_altitude].attrs['UNITS'] == 'meters'
            assert var[self.var_cameraPos].attrs['UNITS'] == 'kilometers'
        
        identifier = os.path.splitext(os.path.basename(cdfPath))[0]
        BaseMapping.__init__(self, altitude, cameraPosGCRS, photoTime, identifier, metadata=metadata)
        ArrayImageMixin.__init__(self, img)

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
    
def _convertImgDtype(arr, fillval):
    if arr.dtype in [np.uint8, np.uint16, np.uint32]:
        return arr
    
    arr = ma.masked_equal(arr, fillval, copy=False)
    
    if arr.dtype == np.int16:
        assert 0 <= np.min(arr) <= np.max(arr) <= np.iinfo(np.uint8).max
        return arr.astype(np.uint8)
    
    elif arr.dtype == np.int32:
        assert 0 <= np.min(arr) <= np.max(arr) <= np.iinfo(np.uint16).max
        return arr.astype(np.uint16)
    
    elif arr.dtype == np.int64:
        assert 0 <= np.min(arr) <= np.max(arr) <= np.iinfo(np.uint32).max
        return arr.astype(np.uint32)
        
    else:
        raise NotImplementedError('Data type not supported: ' + str(arr.dtype))
    