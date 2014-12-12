# Copyright European Space Agency, 2013

"""
This module exports :class:`auromat.mapping.mapping.BaseMapping` objects
into NASA's CDF file format following the ISTP/IACG guidelines.
Each mapping is exported as a single self-contained file.

See http://cdf.gsfc.nasa.gov/html/CDF_docs.html and
http://spdf.gsfc.nasa.gov/sp_use_of_cdf.html for details.

Where no guidelines exist, the ones from the :mod:`auromat.export.netcdf`
module are used, in particular the `geospatial_*` attributes for describing
the bounding box, and the `crs` attributes for describing the coordinate
systems.
"""

from __future__ import division, absolute_import, print_function

import numpy as np
from spacepy import pycdf

from auromat.coordinates.transform import northGeomagneticPoleLocation
import sys

def write(outputPath, mapping, metadata={}, includeBounds=True, 
          includeMagCoords=True, includeGeoCoords=True, 
          compress=True, useTT2000=True):
    """
    
    :param str outputPath:
    :param auromat.mapping.mapping.BaseMapping mapping:
    :param dict metadata: additional metadata, overwrites `mapping.metadata` entries if existing,
                          a dictionary of root attributes, e.g.::
                          
                            {'Project': '..',
                             'Source_name': '..',
                             'Discipline': 'Space Physics>Magnetospheric Science',
                             'Descriptor': '..'
                            }
                          
                          See http://spdf.gsfc.nasa.gov/istp_guide/gattributes.html.
    :param bool includeBounds: stores the coordinates of each pixel corner (in addition to the center)
    :param bool includeMagCoords: include geomagnetic latitude-magnetic local time coordinates
    :param bool includeGeoCoords: include geodetic coordinates
    :param bool compress: use compression for variables
    :param bool useTT2000: 
        Whether to use the CDF_TIME_TT2000 type for storing times.
        If False, then the CDF_EPOCH type is used.
        See http://cdf.gsfc.nasa.gov/html/leapseconds.html for the advantages 
        of using CDF_TIME_TT2000.
        Note that this requires CDF 3.4.0 or higher both for reading and writing.
    """
    # for CDF_TIME_TT2000 backward compatible mode must be off
    pycdf.lib.set_backward(not useTT2000)
    
    if compress:
        compress = pycdf.const.GZIP_COMPRESSION
    else:
        compress = pycdf.const.NO_COMPRESSION
    
    with pycdf.CDF(outputPath, '') as root:            
        # ROOT ATTRIBUTES
        metadata = dict(list(mapping.metadata.items()) + list(metadata.items()))
        for k,v in metadata.items():
            if isinstance(v, bool):
                v = int(v)
            try:
                root.attrs[k] = v
            except TypeError:
                print('Cannot store global attribute "{}" with value {}'.format(k,repr(v)), file=sys.stderr) 
                raise
        
        # NODC conventions:
        root.attrs['geospatial_lat_min'] = mapping.boundingBox.latSouth
        root.attrs['geospatial_lat_max'] = mapping.boundingBox.latNorth
        root.attrs['geospatial_lon_min'] = mapping.boundingBox.lonWest
        root.attrs['geospatial_lon_max'] = mapping.boundingBox.lonEast
        root.attrs['geospatial_lat_units'] = 'degrees_north'
        root.attrs['geospatial_lon_units'] = 'degrees_east'
        
        # VARIABLES
        if useTT2000:
            root.new('Epoch', [mapping.photoTime], type=pycdf.const.CDF_TIME_TT2000)
        else:
            # uses CDF_EPOCH by default
            root['Epoch'] = [mapping.photoTime]
        time = root['Epoch']
        time.attrs['VAR_TYPE'] = 'support_data'
        
        if includeGeoCoords:
            root.new('lat', mapping.latsCenter[np.newaxis,:], compress=compress)
            lat = root['lat']
            lat.attrs['VAR_TYPE'] = 'data'
            lat.attrs['DEPEND_0'] = 'Epoch'
            lat.attrs['DEPEND_1'] = 'y_pixel'
            lat.attrs['DEPEND_2'] = 'x_pixel'
            lat.attrs['UNITS'] = 'degrees'
            lat.attrs['VALIDMIN'] = -90.0
            lat.attrs['VALIDMAX'] = 90.0
            lat.attrs['FIELDNAM'] = 'Latitude of pixel center'    
            lat.attrs['VAR_NOTES'] = 'Geodetic latitude'
            lat.attrs['crs'] = 'crs'
            
            root.new('lon', mapping.lonsCenter[np.newaxis,:], compress=compress)
            lon = root['lon']
            lon.attrs['VAR_TYPE'] = 'data'
            lon.attrs['DEPEND_0'] = 'Epoch'
            lon.attrs['DEPEND_1'] = 'y_pixel'
            lon.attrs['DEPEND_2'] = 'x_pixel'
            lon.attrs['UNITS'] = 'degrees'
            lon.attrs['VALIDMIN'] = -180.0
            lon.attrs['VALIDMAX'] = 180.0
            lon.attrs['FIELDNAM'] = 'Longitude of pixel center'    
            lon.attrs['VAR_NOTES'] = 'Geodetic longitude'
            lon.attrs['crs'] = 'crs'
            
            if includeBounds:
                lat.attrs['bounds'] = 'lat_bounds'
                lon.attrs['bounds'] = 'lon_bounds'
                
                root.new('lat_bounds', mapping.lats[np.newaxis,:], compress=compress)
                lat_bounds = root['lat_bounds']
                lat_bounds.attrs['VAR_TYPE'] = 'data'
                lat_bounds.attrs['DEPEND_0'] = 'Epoch'
                lat_bounds.attrs['DEPEND_1'] = 'y_corner'
                lat_bounds.attrs['DEPEND_2'] = 'x_corner'
                lat_bounds.attrs['UNITS'] = 'degrees'
                lat_bounds.attrs['VALIDMIN'] = -90.0
                lat_bounds.attrs['VALIDMAX'] = 90.0
                lat_bounds.attrs['FIELDNAM'] = 'Latitude of pixel corner'
                lat_bounds.attrs['VAR_NOTES'] = 'Geodetic latitude'
                lat_bounds.attrs['crs'] = 'crs'
                
                root.new('lon_bounds', mapping.lons[np.newaxis,:], compress=compress)
                lon_bounds = root['lon_bounds']
                lon_bounds.attrs['VAR_TYPE'] = 'data'
                lon_bounds.attrs['DEPEND_0'] = 'Epoch'
                lon_bounds.attrs['DEPEND_1'] = 'y_corner'
                lon_bounds.attrs['DEPEND_2'] = 'x_corner'
                lon_bounds.attrs['UNITS'] = 'degrees'
                lon_bounds.attrs['VALIDMIN'] = -180.0
                lon_bounds.attrs['VALIDMAX'] = 180.0
                lon_bounds.attrs['FIELDNAM'] = 'Longitude of pixel corner'    
                lon_bounds.attrs['VAR_NOTES'] = 'Geodetic longitude'
                lon_bounds.attrs['crs'] = 'crs'
        
        root.new('altitude', mapping.altitude * 1000, recVary=False)
        altitude = root['altitude']
        altitude.attrs['VAR_TYPE'] = 'support_data'
        altitude.attrs['UNITS'] = 'meters'
        altitude.attrs['FIELDNAM'] = 'Height above reference ellipsoid'
        altitude.attrs['crs'] = 'crs'
        
        if includeMagCoords:            
            mlats, mlts = mapping.mLatMltCenter
                        
            root.new('mlat', mlats[np.newaxis,:], compress=compress)
            mlat = root['mlat']
            mlat.attrs['VAR_TYPE'] = 'data'
            mlat.attrs['DEPEND_0'] = 'Epoch'
            mlat.attrs['DEPEND_1'] = 'y_pixel'
            mlat.attrs['DEPEND_2'] = 'x_pixel'
            mlat.attrs['UNITS'] = 'degrees'
            mlat.attrs['VALIDMIN'] = -90.0
            mlat.attrs['VALIDMAX'] = 90.0
            mlat.attrs['FIELDNAM'] = 'Geomagnetic latitude of pixel center'
            mlat.attrs['VAR_NOTES'] = ''
            mlat.attrs['crs'] = 'mcrs'
                        
            root.new('mlt', mlts[np.newaxis,:], compress=compress)
            mlt = root['mlt']
            mlt.attrs['VAR_TYPE'] = 'data'
            mlt.attrs['DEPEND_0'] = 'Epoch'
            mlt.attrs['DEPEND_1'] = 'y_center'
            mlt.attrs['DEPEND_2'] = 'x_center'
            mlt.attrs['UNITS'] = 'hours'
            mlt.attrs['VALIDMIN'] = 0.0
            mlt.attrs['VALIDMAX'] = 24.0
            mlt.attrs['FIELDNAM'] = 'Magnetic local time of pixel center'
            mlt.attrs['crs'] = 'mcrs'
            
            if includeBounds:
                mlat.attrs['bounds'] = 'mlat_bounds'
                mlt.attrs['bounds'] = 'mlt_bounds'
                
                mlats, mlts = mapping.mLatMlt
                
                root.new('mlat_bounds', mlats[np.newaxis,:], compress=compress)
                mlat_bounds = root['mlat_bounds']
                mlat_bounds.attrs['VAR_TYPE'] = 'data'
                mlat_bounds.attrs['DEPEND_0'] = 'Epoch'
                mlat_bounds.attrs['DEPEND_1'] = 'y_corner'
                mlat_bounds.attrs['DEPEND_2'] = 'x_corner'
                mlat_bounds.attrs['UNITS'] = 'degrees'
                mlat_bounds.attrs['VALIDMIN'] = -90.0
                mlat_bounds.attrs['VALIDMAX'] = 90.0
                mlat_bounds.attrs['FIELDNAM'] = 'Geomagnetic latitude of pixel corner'
                mlat_bounds.attrs['VAR_NOTES'] = ''
                mlat_bounds.attrs['crs'] = 'mcrs'
                            
                root.new('mlt_bounds', mlts[np.newaxis,:], compress=compress)
                mlt_bounds = root['mlt_bounds']
                mlt_bounds.attrs['VAR_TYPE'] = 'data'
                mlt_bounds.attrs['DEPEND_0'] = 'Epoch'
                mlt_bounds.attrs['DEPEND_1'] = 'y_corner'
                mlt_bounds.attrs['DEPEND_2'] = 'x_corner'
                mlt_bounds.attrs['UNITS'] = 'hours'
                mlt_bounds.attrs['VALIDMIN'] = 0.0
                mlt_bounds.attrs['VALIDMAX'] = 24.0
                mlt_bounds.attrs['FIELDNAM'] = 'Magnetic local time of pixel corner'
                mlt_bounds.attrs['crs'] = 'mcrs'
            
            magPoleLat, magPoleLon = northGeomagneticPoleLocation(mapping.photoTime)
            
            root.new('mcrs', 0, recVary=False) # holds no actual data
            mcrs = root['mcrs']
            mcrs.attrs['VAR_TYPE'] = 'support_data'
            mcrs.attrs['north_geomagnetic_pole_lat'] = magPoleLat
            mcrs.attrs['north_geomagnetic_pole_lon'] = magPoleLon
            mcrs.attrs['VAR_NOTES'] = 'Geocentric MLat/MLT system based on the given geomagnetic pole position'

        if np.any(mapping.img.mask):
            # CDF supports much more types than netCDF
            imgDtypeMap = {np.dtype(np.uint8): np.int16,  # no overhead (compared to a separate int8 mask)
                           np.dtype(np.uint16): np.int32, # 1 Byte overhead per pixel  
                           np.dtype(np.uint32): np.int64  # 2 Byte overhead per pixel
                           }
            if not mapping.img.dtype in imgDtypeMap.keys():
                raise NotImplementedError('Image data type not supported: ' + str(mapping.img.dtype))
            
            imgDtype = imgDtypeMap[mapping.img.dtype]
            imgFillval = imgDtype(np.iinfo(imgDtype).min)
            img_ = mapping.img.astype(imgDtype).filled(imgFillval)
        else:
            imgDtype = mapping.img.dtype
            img_ = img_.data
            imgFillval = None

        if img_.shape[2] == 1:
            bands = ['img']
        elif img_.shape[2] == 3:
            bands = ['img_red', 'img_green', 'img_blue']
        else:
            raise NotImplementedError

        for i, band in enumerate(bands):
            root.new(band, img_[np.newaxis,:,:,i], compress=compress)
            img = root[band]
            img.attrs['VAR_TYPE'] = 'data'
            img.attrs['DEPEND_0'] = 'Epoch'
            img.attrs['DEPEND_1'] = 'y_pixel'
            img.attrs['DEPEND_2'] = 'x_pixel'        
            img.attrs['FIELDNAM'] = ''
            img.attrs['VALIDMIN'] = np.iinfo(mapping.img.dtype).min
            img.attrs['VALIDMAX'] = np.iinfo(mapping.img.dtype).max
            if imgFillval:
                img.attrs['FILLVAL'] = imgFillval
            img.attrs['UNITS'] = 'unitless'
                    
        zen = 90 - mapping.elevation[np.newaxis,:].astype(np.float32)
        root.new('zenith_angle', zen, compress=compress)
        zenith_angle = root['zenith_angle']
        zenith_angle.attrs['VAR_TYPE'] = 'data'
        zenith_angle.attrs['DEPEND_0'] = 'Epoch'
        zenith_angle.attrs['DEPEND_1'] = 'y_pixel'
        zenith_angle.attrs['DEPEND_2'] = 'x_pixel'
        zenith_angle.attrs['UNITS'] = 'degrees'
        zenith_angle.attrs['VALIDMIN'] = 0.0
        zenith_angle.attrs['VALIDMAX'] = 90.0
        zenith_angle.attrs['FIELDNAM'] = 'Absolute sensor zenith angle of pixel center'

        root['camera_pos'] = np.array([mapping.cameraPosGCRS])
        cameraPos = root['camera_pos']
        cameraPos.attrs['VAR_TYPE'] = 'support_data'
        cameraPos.attrs['DEPEND_0'] = 'Epoch'
        cameraPos.attrs['UNITS'] = 'kilometers'
        cameraPos.attrs['FIELDNAM'] = 'Camera position in cartesian GCRS coordinates'
        cameraPos.attrs['VAR_NOTES'] = 'Axis order: xyz'

        root.new('crs', 0, recVary=False) # holds no actual data
        crs = root['crs']
        crs.attrs['VAR_TYPE'] = 'support_data'
        crs.attrs['semi_major_axis'] = 6378137.0
        crs.attrs['inverse_flattening'] = 298.257223563
        crs.attrs['VAR_NOTES'] = 'Geographic Coordinate System, WGS 84'
