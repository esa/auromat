# Copyright European Space Agency, 2013

from __future__ import division, absolute_import, print_function

import os.path

import numpy as np
import numpy.ma as ma

from netCDF4 import Dataset

from auromat.mapping.mapping import BaseMappingProvider, \
    sanitize_data, BaseMapping, ArrayImageMixin, GenericMapping
from auromat.util.decorators import inherit_docs
from datetime import datetime
from numpy.testing.utils import assert_array_equal
import collections
import auromat.utils

# TODO remove code duplication with cdf module

@inherit_docs
class NetCDFMappingProvider(BaseMappingProvider):
    def __init__(self, cdfPaths, maxTimeOffset=3):
        self.cdfPaths = cdfPaths
        self.maxTimeOffset = maxTimeOffset
        
        # create mapping from date to path index        
        datemap = {}
        for path_idx, path in enumerate(cdfPaths):
            with Dataset(path, 'r') as root:
                date = _readDate(root.variables['time'])
                if date in datemap:
                    raise ValueError('The date ' + str(date) + ' is appearing twice ' +
                                     'in the NetCDF files ' + path + ' and ' +
                                     cdfPaths[datemap[date][0]])
                datemap[date] = path_idx
        
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
        path_idx = self.datemap[dates[idx]]
        return NetCDFMapping(self.cdfPaths[path_idx])
        
    def getById(self, identifier):
        raise NotImplementedError
    
    def getSequence(self, dateBegin=None, dateEnd=None):
        if not dateBegin:
            dateBegin = self.range[0]
        if not dateEnd:
            dateEnd = self.range[1]
        dates = filter(lambda d: dateBegin <= d <= dateEnd, self.datemap.keys())
        for date in dates:
            path_idx = self.datemap[date]
            yield NetCDFMapping(self.cdfPaths[path_idx])

@sanitize_data
@inherit_docs
class NetCDFMapping(ArrayImageMixin, BaseMapping):
    
    # variable names in the netcdf files as produced by auromat.export.netcdf
    var_altitude = 'altitude'
    var_cameraPos = 'camera_pos'
    var_photoTime = 'time'
    var_img = 'img'
    var_img_red = 'img_red'
    var_img_green = 'img_green'
    var_img_blue = 'img_blue'
    var_latsCenter = 'lat'
    var_lonsCenter = 'lon'
    var_zenithAngle = 'zenith_angle'
    
    def __init__(self, cdfPath):
        with Dataset(cdfPath, 'r') as root:
            var = root.variables
            altitude = var[self.var_altitude][:]/1000
            cameraPosGCRS = var[self.var_cameraPos][:]
            photoTime = _readDate(var[self.var_photoTime])
            
            # for three channels (RGB), each channel is stored as a
            # separate variable: img_red, img_green, img_blue
            # for grayscale, the single variable is called 'img'
            try:
                img = np.atleast_3d(var[self.var_img][:])
                img = _convertImgDtype(img)
            except:
                img_red = _convertImgDtype(var[self.var_img_red][:])
                img_green = _convertImgDtype(var[self.var_img_green][:])
                img_blue = _convertImgDtype(var[self.var_img_blue][:])
                img = ma.dstack((img_red, img_green, img_blue))
                        
            latsCenter = var[self.var_latsCenter][:]
            lonsCenter = var[self.var_lonsCenter][:]
            latBounds = var[var[self.var_latsCenter].bounds][:]
            lonBounds = var[var[self.var_lonsCenter].bounds][:]
            
            # TODO read in MLat/MLT as well if available
            
            if latsCenter.ndim == 1:
                latsCenter, lonsCenter = np.dstack(np.meshgrid(latsCenter, lonsCenter)).T
                
                assert np.all(latBounds[:-1,1] == latBounds[1:,0])
                assert np.all(lonBounds[:-1,1] == lonBounds[1:,0])
                latBounds = np.concatenate((latBounds[:,0], [latBounds[-1,1]]))
                lonBounds = np.concatenate((lonBounds[:,0], [lonBounds[-1,1]]))
                lats, lons = np.dstack(np.meshgrid(latBounds, lonBounds)).T
            else:
                lats = np.empty((latsCenter.shape[0]+1, latsCenter.shape[1]+1), latBounds.dtype)
                lons = np.empty_like(lats)
                
                for grid, bounds in [(lats, latBounds), (lons, lonBounds)]:
                    # have to use numpy's assert to handle NaN's correctly
                    assert_array_equal(bounds[:-1,:-1,2], bounds[:-1,1:,3])
                    assert_array_equal(bounds[:-1,:-1,2], bounds[1:,1:,0])
                    assert_array_equal(bounds[:-1,:-1,2], bounds[1:,:-1,1])
                    grid[:-1,:-1] = bounds[:,:,0]
                    grid[-1,:-1] = bounds[-1,:,3]
                    grid[:-1,-1] = bounds[:,-1,1]
                    grid[-1,-1] = bounds[-1,-1,2]
            
            self._latsCenter = ma.masked_invalid(latsCenter)
            self._lonsCenter = ma.masked_invalid(lonsCenter)
            self._lats = ma.masked_invalid(lats)
            self._lons = ma.masked_invalid(lons)
            self._elevation = ma.masked_invalid(90 - var[self.var_zenithAngle][:])
            
            metadata = root.__dict__ # ordered dict of all global attributes
            
            assert var[self.var_altitude].units == 'meters'
            assert var[self.var_cameraPos].units == 'kilometers'
        
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
    
def _convertImgDtype(arr):
    if arr.dtype in [np.uint8, np.uint16]:
        return arr
    
    elif arr.dtype == np.int16:
        assert 0 <= np.min(arr) <= np.max(arr) <= np.iinfo(np.uint8).max
        return arr.astype(np.uint8)
    
    elif arr.dtype == np.int32:
        assert 0 <= np.min(arr) <= np.max(arr) <= np.iinfo(np.uint16).max
        return arr.astype(np.uint16)
        
    else:
        raise NotImplementedError('Data type not supported: ' + str(arr.dtype))

def _readDate(date_var):
    # we don't use netcdf4.num2date here because it only has 1sec-resolution
    assert date_var.units == 'seconds since 1970-01-01 00:00:00'
    return datetime.utcfromtimestamp(date_var[:])
