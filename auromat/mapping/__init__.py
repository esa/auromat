"""
The mapping package is the heart of this library. 
Its functionality can be broadly split up in three areas:

- (Down-)Loading images and calibration data from different providers
- Calculation of geodetic coordinates and elevation angles at arbitrary altitude 
  for an image from calibration data
- Accessing derived information about a georeferenced image 
  (e.g. bounding box and outline, centroid, pixel resolution)
- Transformation of coordinates (masking, resampling, coordinate system change)

The calibration data kinds currently supported are:

- `FITS WCS <http://fits.gsfc.nasa.gov/fits_wcs.html>`_ (header-only) 
  with camera position and image timestamp
- `MIRACLE <http://www.space.fmi.fi/MIRACLE/ASC/index.html>`_ calibration text files
- raw coordinate arrays (from memory, or CDF/netCDF files like 
  `THEMIS <http://themis.ssl.berkeley.edu>`_)

The data providers currently supported are:

- FITS WCS file provider (local only)
- `ESA ISS aurora archive <http://cosmos.esa.int/arrrgh>`_ 
  provider (web access, uses the FITS WCS file provider)
- THEMIS provider (web access)
- MIRACLE provider (local only)
- CDF file provider (local only, format as in :mod:`auromat.export.cdf`)
- netCDF file provider (local only, format as in :mod:`auromat.export.netcdf`)

The starting point is the :mod:`auromat.mapping.mapping` module on which all other
modules depend. 
"""