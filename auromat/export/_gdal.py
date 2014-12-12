# Copyright European Space Agency, 2013

# The following code is commented out because nansat cannot currently be used 
# due to incompatible licensing. Once auromat is released under a real open-source
# license, this code should be uncommented again.


"""
Converts mappings to the GDAL data model such that they can be saved in various
raster geospatial file formats such as GeoTIFF, or running data conversions using
gdal tools like gdalwarp.
See http://www.gdal.org/formats_list.html for a list of supported formats.

GDAL's geolocation arrays are our irregular lat/lon arrays.
GDAL supports reading those from netCDF-CF files, but not writing them.
This is one of the reasons why we have a separate netCDF exporter.

Instead of using this module, the gdal tools (like gdalwarp) can also be used
directly on the exported netCDF-CF files. The advantage is that it may integrate
better into a shell based processing pipeline. The disadvantage is that it requires
saving the mappings as (quite huge) netCDF-CF files first.

NOTE: GDAL support is limited currently
      see http://trac.osgeo.org/gdal/ticket/4513#comment:23
"""

#from __future__ import division
#
#import numpy as np
#import nansat
#from osgeo.gdalconst import GCI_RedBand, GCI_GreenBand, GCI_BlueBand
#
## we use Nansat as a nice wrapper around GDAL
#
#def asGdalDataset(mapping):
#    n = asNansat(mapping)
#    dataset = n.vrt.dataset
#    return dataset
#
#def asNansat(mapping):
#    domain = nansat.Domain(lon=mapping.lonsCenter, lat=mapping.latsCenter)
#
#    elevParams = {'name': 'elevation',
#                  }
#    
#    n = nansat.Nansat(domain=domain, array=mapping.elevation, parameters=elevParams)
#    
#    # The min/max data values cannot be set programmatically in GDAL.
#    # The min/max of the data type is used here.
#    # Therefore we leave the img data type as it is and provide the mask
#    # as a separate band.
#    
#    # 0 = no data (opposite to numpy definition)
#    mask = (~mapping.img.mask).astype(np.int8)
#    mask[mask==1] = 64 # see nansat.mosaic
#    # this is not a real GDAL mask band (CreateMaskBand) but is recognized by nansat
#    n.addBand(mask, parameters={'name': 'mask'})
#    
#    #n.vrt.set_subsetMask(self, maskDs, xOff, yOff, dstXSize, dstYSize)
#    
#    if mapping.img.shape[2] == 3: # RGB
#        n.addBand(mapping.img[:,:,0].data, parameters={'name': 'img_r'})
#        n.addBand(mapping.img[:,:,1].data, parameters={'name': 'img_g'})
#        n.addBand(mapping.img[:,:,2].data, parameters={'name': 'img_b'})
#        
#        r = n.get_GDALRasterBand('img_r')
#        r.SetRasterColorInterpretation(GCI_RedBand)
#        
#        g = n.get_GDALRasterBand('img_g')
#        g.SetRasterColorInterpretation(GCI_GreenBand)
#        
#        b = n.get_GDALRasterBand('img_b')
#        b.SetRasterColorInterpretation(GCI_BlueBand)           
#        
#    else: # something else
#        for i in range(mapping.img.shape[2]):
#            n.addBand(mapping.img[:,:,i].data, parameters={'name': 'img'})
#            
#    return n
