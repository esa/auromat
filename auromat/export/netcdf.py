# Copyright European Space Agency, 2013

from __future__ import division, absolute_import, print_function

"""
This module exports :class:`auromat.mapping.mapping.BaseMapping` objects
into the netCDF file format following the CF 1.6 and NODC conventions.
Each mapping is exported as a single self-contained file.

See http://cfconventions.org/1.6.html,
http://www.nodc.noaa.gov/data/formats/netcdf/grid.cdl and
http://wiki.esipfed.org/index.php/NetCDF-CF_File_Examples_for_Satellite_Swath_Data
for details.
"""

import sys
from datetime import datetime
import numpy as np
from netCDF4 import Dataset

from auromat.coordinates.transform import northGeomagneticPoleLocation
from auromat.mapping.mapping import isPlateCarree

def write(outputPath, mapping, metadata={}, includeBounds=True,
          includeMagCoords=True, includeGeoCoords=True,
          use1dIfPossible=True, compress=True):
    """
    
    :param str outputPath:
    :param auromat.mapping.mapping.BaseMapping mapping:
    :param dict metadata: additional metadata, overwrites mapping.metadata entries if existing,
                          a dictionary of root attributes, e.g.::
                          
                            {'Project': '..',
                             'Source_name': '..',
                             'Discipline': 'Space Physics>Magnetospheric Science',
                             'Descriptor': '..'
                            }
                            
                          See http://wiki.esipfed.org/index.php/Attribute_Convention_for_Data_Discovery.
    :param bool includeBounds: stores the coordinates of each pixel corner (in addition to the center)
    :param bool includeMagCoords: include geomagnetic latitude-magnetic local time coordinates
    :param bool compress: use compression for variables
    """
    if not includeGeoCoords:
        raise ValueError('Geodetic coordinates cannot be disabled for netCDF as they are essential to the format')
    
    with Dataset(outputPath, 'w', format='NETCDF4') as root:
        # ROOT ATTRIBUTES
        root.Conventions = 'CF-1.6'
        
        metadata = dict(list(mapping.metadata.items()) + list(metadata.items()))
        for k,v in metadata.items():
            if isinstance(v, bool):
                v = np.uint8(v)
            try:
                setattr(root, k, v)
            except TypeError:
                print('Cannot store global attribute "{}" with value {}'.format(k,repr(v)), file=sys.stderr) 
                raise
            
        isLatLonPlateCarree = use1dIfPossible and isPlateCarree(mapping.lats, mapping.lons)
        isMLatMltPlateCarree = use1dIfPossible and isPlateCarree(*mapping.mLatMlt) if includeMagCoords else None
        
        # NODC conventions:
        root.geospatial_lat_min = mapping.boundingBox.latSouth
        root.geospatial_lat_max = mapping.boundingBox.latNorth
        root.geospatial_lon_min = mapping.boundingBox.lonWest
        root.geospatial_lon_max = mapping.boundingBox.lonEast
        root.geospatial_lat_units = 'degrees_north'
        root.geospatial_lon_units = 'degrees_east'
                
        # DIMENSIONS
        # NOTE There is a bug preventing that a dimension is named equally as a variable.
        #      Therefore we append an 's' to the dimension names where it matters.
        w = mapping.img.shape[1]
        h = mapping.img.shape[0]
        if isLatLonPlateCarree:
            root.createDimension('lats', h)
            root.createDimension('lons', w)
        if isMLatMltPlateCarree:
            root.createDimension('mlats', h)
            root.createDimension('mlts', w)
        if not isLatLonPlateCarree or isMLatMltPlateCarree is False: # leave as "is False"! see above
            root.createDimension('y', h)
            root.createDimension('x', w)
        if includeBounds:
            if isLatLonPlateCarree or isMLatMltPlateCarree:
                root.createDimension('vertex2', 2) # for 1D coordinate arrays
            if not isLatLonPlateCarree or isMLatMltPlateCarree is False:
                root.createDimension('vertex4', 4) # for 2D coordinate arrays        
        root.createDimension('channel', mapping.img.shape[2])
        root.createDimension('xyz', 3)
        
        # COORDINATE VARIABLES
        
        # It seems that netcdf-CF defines time as in POSIX timestamps (without leap seconds).
        # see https://github.com/Unidata/netcdf4-python/issues/280
        # This means that a dataset covering a leap second with 1 second resolution will have
        # one data point appearing twice when the leap second happens.
        # Supporting leap seconds would require more effort though, as Python's datetime has the
        # same problem, not to mention what camera EXIF times would be like.
        time = root.createVariable('time', np.float64)
        time.units = 'seconds since 1970-01-01 00:00:00'
        time.calendar = 'gregorian'
        time.standard_name = 'time'
        time.axis = 'T'
        time.long_name = ''
        time.comment = ''
        time[:] = _unix(mapping.photoTime)
        # Note: Adding time bounds using the EXIF exposure time doesn't make sense here.
        #       The reason is that this would require very accurate EXIF times from
        #       the beginning. For the ISS mappings, this is not the case.
        
        # Note that chunk sizes are chosen such that they are compatible to GDAL
        # see http://trac.osgeo.org/gdal/ticket/4513#comment:18
        
        # Note that for all float variables we use NaN to indicate missing values (instead of _FillValue).
        # see http://www.unidata.ucar.edu/software/netcdf/docs/BestPractices.html
                     
        # lat lon coordinates have a time dimension here as each non-resampled mapping has its lat lon arrays.
        if isLatLonPlateCarree:
            # Note that we use the unmasked arrays here to stay compatible with the CF conventions
            # which say that coordinate arrays should not have missing values. This also makes it more compatible
            # with standard software.
            latsCenter, lonsCenter = mapping.latsCenter.data[:,0], mapping.lonsCenter.data[0,:]
            
            lat = root.createVariable('lat', np.float64, ('lats'), zlib=compress)
            lat.actual_range = np.float64([latsCenter[-1], latsCenter[0]])
            lat[:] = latsCenter
            
            lon = root.createVariable('lon', np.float64, ('lons'), zlib=compress)
            lon.actual_range = np.float64([lonsCenter[0], lonsCenter[-1]])
            lon[:] = lonsCenter
            
            # Note that plate carree is not yet defined as a recognized projection within CF. 
            # Therefore we cannot use a more specific crs.grid_mapping_name value.
            
        else:
            # auxiliary 2D coordinate variables with non-standard projection
            # see http://cfconventions.org/1.6.html#idp5561056  
            
            # IMPORTANT:
            # netCDF-CF 1.6 does not allow missing values for coordinate arrays.
            # There is a proposal to add support for it in the future though:
            # https://cf-pcmdi.llnl.gov/trac/ticket/85
            # This means that we violate the CF conventions slightly, but as our irregular (curvilinear)
            # grid is not defined for all image pixels there is no alternative here (at least for
            # the original non-resampled mappings).
            
            lat = root.createVariable('lat', np.float64, ('y', 'x'), zlib=compress, chunksizes=(1, w))
            lat.actual_range = np.float64([np.min(mapping.latsCenter), np.max(mapping.latsCenter)])
            lat[:] = mapping.latsCenter.filled(np.nan)
            
            lon = root.createVariable('lon', np.float64, ('y', 'x'), zlib=compress, chunksizes=(1, w))
            lon.actual_range = np.float64([np.min(mapping.lonsCenter), np.max(mapping.lonsCenter)])
            lon[:] = mapping.lonsCenter.filled(np.nan)
            
        lat.units = 'degrees_north'
        lat.valid_min = np.float64(-90)
        lat.valid_max = np.float64(90)
        lat.standard_name = 'latitude'
        lat.axis = 'Y'
        lat.long_name = 'Latitude'
        lat.comment = 'Geodetic latitude'
        
        lon.units = 'degrees_east'
        lon.valid_min = np.float64(-180)
        lon.valid_max = np.float64(180)
        lon.standard_name = 'longitude'
        lon.axis = 'X'
        lon.long_name = 'Longitude'
        lon.comment = 'Geodetic longitude'
                
        altitude = root.createVariable('altitude', np.int32)
        altitude.units = 'meters'
        altitude.standard_name = 'height_above_reference_ellipsoid'
        altitude.axis = 'Z'
        altitude.long_name = ''
        #altitude.comment = ''
        altitude[:] = mapping.altitude * 1000

        if includeBounds:
            # http://cfconventions.org/1.6.html#cell-boundaries
            lat.bounds = 'lat_bounds'
            lon.bounds = 'lon_bounds'
            
            if isLatLonPlateCarree:
                lat_bounds = root.createVariable('lat_bounds', np.float64, ('lats', 'vertex2'), 
                                                 zlib=compress, chunksizes=(h, 2))
                lat_bounds[:] = _bounds1d(mapping.lats.data[:,0])
                                
                lon_bounds = root.createVariable('lon_bounds', np.float64, ('lons', 'vertex2'),
                                                 zlib=compress, chunksizes=(w, 2))
                lon_bounds[:] = _bounds1d(mapping.lons.data[0,:])
            else:         
                lat_bounds = root.createVariable('lat_bounds', np.float64, ('y', 'x', 'vertex4'), 
                                                 zlib=compress, chunksizes=(1, w, 4))
                lat_bounds[:] = _bounds2d(mapping.lats.filled(np.nan))
                
                lon_bounds = root.createVariable('lon_bounds', np.float64, ('y', 'x', 'vertex4'),
                                                 zlib=compress, chunksizes=(1, w, 4))
                lon_bounds[:] = _bounds2d(mapping.lons.filled(np.nan))
        
        if includeMagCoords:
            # The CF 1.6 standard has no convention to define coordinates in different systems and
            # somehow relate them to the existing primary coordinates.
            # Therefore the following is non-standard.
            
            mlats, mlts = mapping.mLatMltCenter
            
            if isMLatMltPlateCarree:
                mlatsCenter, mltsCenter = mlats.data[:,0], mlts.data[0,:]
                
                mlat = root.createVariable('mlat', np.float64, ('mlats'), zlib=compress)
                mlat.actual_range = np.float64([mlatsCenter[-1], mlatsCenter[0]])
                mlat[:] = mlatsCenter
                
                mlt = root.createVariable('mlt', np.float64, ('mlts'), zlib=compress)
                mlt.actual_range = np.float64([mltsCenter[0], mltsCenter[-1]])
                mlt[:] = mltsCenter
                
            else:  
                mlat = root.createVariable('mlat', np.float64, ('y', 'x'), zlib=compress, chunksizes=(1, w))
                mlat.actual_range =  np.float64([np.min(mlats), np.max(mlats)])
                mlat[:] = mlats.filled(np.nan)
                
                mlt = root.createVariable('mlt', np.float64, ('y', 'x'), zlib=compress, chunksizes=(1, w))
                mlt.actual_range = np.float64([np.min(mlts), np.max(mlts)])                
                mlt[:] = mlts.filled(np.nan)
              
            #mlat.standard_name = 'grid_latitude'
            mlat.long_name = 'Geomagnetic latitude'  
            mlat.units = 'degrees'
            mlat.valid_min = np.float64(-90)
            mlat.valid_max = np.float64(90)
            # There is no standard attribute to define the coordinate system on the coordinate
            # variables themselves. We use the non-standard 'crs' attribute for that purpose.
            mlat.crs = 'mcrs'
            
            mlt.long_name = 'Magnetic local time'
            mlt.units = 'hours'
            mlt.valid_min = np.float64(0)
            mlt.valid_max = np.float64(24)
            mlt.crs = 'mcrs'
                
            
            if includeBounds:
                mlat.bounds = 'mlat_bounds'
                mlt.bounds = 'mlt_bounds'
                
                mlats, mlts = mapping.mLatMlt
                
                if isMLatMltPlateCarree:
                    mlat_bounds = root.createVariable('mlat_bounds', np.float64, ('mlats', 'vertex2'), 
                                                      zlib=compress, chunksizes=(h, 2))
                    mlat_bounds[:] = _bounds1d(mlats.data[:,0])                
                    
                    mlt_bounds = root.createVariable('mlt_bounds', np.float64, ('mlts', 'vertex2'), 
                                                     zlib=compress, chunksizes=(w, 2))
                    mlt_bounds[:] = _bounds1d(mlts.data[0,:])
                else:                                
                    mlat_bounds = root.createVariable('mlat_bounds', np.float64, ('y', 'x', 'vertex4'), 
                                                      zlib=compress, chunksizes=(1, w, 4))
                    mlat_bounds[:] = _bounds2d(mlats.filled(np.nan))
                    
                    mlt_bounds = root.createVariable('mlt_bounds', np.float64, ('y', 'x', 'vertex4'), 
                                                     zlib=compress, chunksizes=(1, w, 4))
                    mlt_bounds[:] = _bounds2d(mlts.filled(np.nan))
            
            magPoleLat, magPoleLon = northGeomagneticPoleLocation(mapping.photoTime)
            
            # This is all non-standard.
            mcrs = root.createVariable('mcrs', np.int8) # holds no actual data
            mcrs.north_geomagnetic_pole_lat = magPoleLat
            mcrs.north_geomagnetic_pole_lon = magPoleLon
            mcrs.comment = 'Geocentric MLat/MLT system based on the given geomagnetic pole position'

        # DATA VARIABLES
        y = 'lats' if isLatLonPlateCarree else 'y'
        x = 'lons' if isLatLonPlateCarree else 'x'

        # netcdf doesn't support unsigned types except byte
        # we generically convert to int types and ignore the case of uint8 with no masked pixels
        imgDtypeMap = {np.dtype(np.uint8): np.int16,
                       np.dtype(np.uint16): np.int32,
                       }
        if not mapping.img.dtype in imgDtypeMap.keys():
            raise NotImplementedError('Image data type not supported: ' + str(mapping.img.dtype))
        
        imgDtype = imgDtypeMap[mapping.img.dtype]
        imgFillval = imgDtype(np.iinfo(imgDtype).min)
        img_ = mapping.img.astype(imgDtype).filled(imgFillval)
        
        if img_.shape[2] == 1:
            bands = ['img']
        elif img_.shape[2] == 3:
            bands = ['img_red', 'img_green', 'img_blue']
        else:
            raise NotImplementedError
        
        for i, band in enumerate(bands):
            img = root.createVariable(band, imgDtype, (y, x), fill_value=imgFillval, 
                                      zlib=compress, chunksizes=(1, w))
            img.units = 'unitless'
    
            # cell_methods is not set for img as it is too meaningless and ambiguous.
            # It is not clear whether the value of a pixel is really the 'mean' or
            # something else over time and area.
            # 'area' (equally 'x: y:') depends on the camera sensor type and the method for
            # generating pixels out of raw sensor data. E.g. for Bayer-filter cameras
            # demosaicing algorithms interpolate the pixel colours.
            img.valid_min = imgDtype(np.iinfo(mapping.img.dtype).min)
            img.valid_max = imgDtype(np.iinfo(mapping.img.dtype).max)
            img.actual_range =  imgDtype([np.min(mapping.img[:,:,i]), np.max(mapping.img[:,:,i])])

            #img.long_name = ''
            img.coordinates = 'altitude time' if isLatLonPlateCarree else 'lat lon altitude time'
            img.grid_mapping = 'crs'
            img[:] = img_[:,:,i]
        
        # netCDF-CF doesn't know elevation angle, but zenith angle
        # so we use that instead to follow the conventions
        zena = 90 - mapping.elevation
        zenith_angle = root.createVariable('zenith_angle', np.float32, (y, x), 
                                           zlib=compress, chunksizes=(1, w))
        zenith_angle.units = 'degrees'
        zenith_angle.cell_methods = 'time: lat: lon: point' if isLatLonPlateCarree else 'time: y: x: point'
        zenith_angle.valid_min = np.float32(0)
        zenith_angle.valid_max = np.float32(90)
        zenith_angle.actual_range = [np.min(zena), np.max(zena)]
        # zenith_angle: 0=zenith, 90=horizon
        zenith_angle.standard_name = 'zenith_angle'
        zenith_angle.coordinates = 'altitude time' if isLatLonPlateCarree else 'lat lon altitude time'
        zenith_angle.grid_mapping = 'crs'
        zenith_angle.long_name = 'Absolute sensor zenith angle'
        zenith_angle[:] = zena.filled(np.nan)
        
        cameraPos = root.createVariable('camera_pos', np.float64, ('xyz'))
        cameraPos.units = 'kilometers'
        cameraPos.cell_methods = 'time: point'
        cameraPos.coordinates = 'time'
        cameraPos.long_name = 'Camera position in cartesian GCRS coordinates'
        cameraPos.comment = 'Axis order: xyz'
        cameraPos[:] = mapping.cameraPosGCRS
        
        crs = root.createVariable('crs', np.int8) # holds no actual data
        crs.grid_mapping_name = 'latitude_longitude' # = unknown projection lat/lon coordinate system
        crs.semi_major_axis = 6378137.0
        crs.inverse_flattening = 298.257223563
        crs.comment = 'Geographic Coordinate System, WGS 84'

def _bounds1d(arr):
    assert arr.ndim == 1
    arr = arr[:,None]
    bounds = np.concatenate((arr[:-1],arr[1:]), axis=1)
    assert bounds.shape == (arr.shape[0]-1, 2)
    return bounds

def _bounds2d(arr):
    assert arr.ndim == 2
    arr = arr[:,:,None]
    bounds = np.concatenate((
                arr[0:-1, 0:-1],
                arr[0:-1, 1:  ],
                arr[1:  , 1:  ],
                arr[1:  , 0:-1],
                ), axis=2)
    assert bounds.shape == (arr.shape[0]-1, arr.shape[1]-1, 4)
    return bounds

def _unix(dt):
    return (dt - datetime(1970, 1, 1)).total_seconds()

if __name__ == '__main__':
    import os
    from auromat.mapping.spacecraft import getMapping
    
    m = getMapping('/home/mriecher/data/arrrgh/img/ISS030-E/102100/ISS030-E-102170.jpg',
                   '/home/mriecher/data/arrrgh/wcs/ISS030-E/102100/ISS030-E-102170_1727.wcs')

    path = '/user_data/mriecher/test.nc'
    if os.path.exists(path):
        os.remove(path)

    write(path, m, includeBounds=True, includeMagCoords=True, zlib=True)
    