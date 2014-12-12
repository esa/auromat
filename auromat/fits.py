# Copyright European Space Agency, 2013

"""
A module for reading and writing specific FITS header values.
"""

from __future__ import division, absolute_import, print_function

from six import string_types
import warnings
from datetime import datetime, timedelta
import numpy as np
from math import sqrt, sin, cos, atan2
from astropy.io import fits
from astropy.wcs.wcs import WCS
import astropy.units as u
import astropy.coordinates as coord
import time
import sys
import astropy.utils.data
from astroquery.query import suspend_cache

try:
    from astroquery.vizier import Vizier
except ImportError as e:
    print(str(e))
    warnings.warn('astroquery not available, .fits.getCatalogStars cannot be used')

def readHeader(filePath):
    """ Return the primary FITS header. """
    return fits.getheader(filePath, checksum=True)
    
def writeHeader(filePath, header, overwrite=False):
    """
    Creates a new file containing the given FITS header.
    
    :param filePath: Path where FITS header will be written
    :param header: FITS header to write
    :param overwrite: Raises an error if False and filePath already exists
    """
    fits.writeto(filePath, None, header, clobber=overwrite, checksum=True)
    
def getPixelScale(header):
    """
    Returns the pixel scale in degrees/pixel based on the CD matrix of the WCS header.
    SIP distortion coefficients (if present) are not considered.
    """
    assert header['CUNIT1'] == 'deg' and header['CUNIT2'] == 'deg'
    cd11 = header['CD1_1']
    cd21 = header['CD2_1']
    scale = sqrt(cd11**2 + cd21**2)
    return scale

def getRotationAngle(header):
    """
    Returns the rotation (roll) angle in degrees based on the CD matrix of the WCS header.
    Angle is in [-180,180].
    """
    assert header['CUNIT1'] == 'deg' and header['CUNIT2'] == 'deg'
    cd11 = header['CD1_1']
    cd21 = header['CD2_1']

    rho_a = atan2(cd21, cd11)
    
    return np.rad2deg(rho_a)

def cd11cd21(scale, rotation):
    """
    Calculates CD11 and CD21 from the given pixel scale and rotation.
    
    :param scale: pixel scale in degrees/pixel
    :param rotation: rotation angle in degrees within [-180,180]
    :rtype: tuple (cd11, cd21)
    """
    rho = np.deg2rad(rotation)
    cd11 = scale * cos(rho)
    cd21 = scale * sin(rho)
    return cd11, cd21

def setCdMatrix(header, scale, rotation):
    """
    Sets the CD matrix from the given pixel scale and rotation.
    
    :param header:
    :param scale: pixel scale in degrees/pixel
    :param rotation: rotation angle in degrees within [-180,180]
    """
    cd11, cd21 = cd11cd21(scale, rotation)
    header['CD1_1'] = cd11
    header['CD1_2'] = -cd21
    header['CD2_1'] = cd21
    header['CD2_2'] = cd11

def getRadius(header, extend=0):
    """
    Returns the radius in degrees of the circle which encloses the image.
    Degrees are based on the CD matrix of the WCS header. 
    SIP distortion coefficients (if present) are not considered.
    
    :param header: must also contain IMAGEW and IMAGEH
    :param extend: how much to extend the circle in percent [0,1]
    """
    diagPx = sqrt(header['IMAGEW']**2 + header['IMAGEH']**2)
    radiusPx = diagPx/2 * (1+extend)
    radiusDeg = getPixelScale(header) * radiusPx
    return radiusDeg

def getCenterRADec(header):
    """
    Returns RA,Dec for the image center.
    
    :param header: must also contain IMAGEW and IMAGEH
    :rtype: tuple (ra,dec) in degrees, ra is [0,360], dec is [-90,90]
    """
    w = header['IMAGEW']
    h = header['IMAGEH']
    del header['NAXIS'] # emits a warning in WCS constructor if present (as it's 0 for .wcs files)
    return WCS(header).all_pix2world(w/2, h/2, 0, ra_dec_order=True)

def setCenterRADec(header, ra, dec):
    """
    Set the WCS reference point celestial coordinates.
    The reference point is in the image center.
    
    :param header: FITS header
    :param ra: in degrees within [0,360]
    :param dec: in degrees within [-90,90]
    """
    assert 0 <= ra <= 360
    assert -90 <= dec <= 90
    w = header['IMAGEW']
    h = header['IMAGEH']
    header['CRPIX1'] = int(w//2 + 1) # FITS is 1-based
    header['CRPIX2'] = int(h//2 + 1)
    header['CRVAL1'] = ra
    header['CRVAL2'] = dec
    
def readQuadMatch(fitsMatchPath):
    """
    
    :param fitsMatchPath: path to .match file written by astrometry.net
    :rtype: array of pixel coordinates of the stars in the quad, shape (n,2) in [x,y] order
    """
    with fits.open(fitsMatchPath) as hdulist:
        data = hdulist[1].data[0]
        starCount = data.field('DIMQUADS')
        starPxCoords = data.field('QUADPIX').reshape(-1,2)[:starCount]
        return starPxCoords
    
def readCorr(corrPath):
    """
    Return corresponding sources and reference stars as found by astrometry.net
    from the given .corr file.
    
    :param str corrPath: path to .corr file written by astrometry.net
    :rtype: tuple (xField, yField, xIndex, yIndex)
    """
    with fits.open(corrPath) as hdulist:
        data = hdulist[1].data
        xField = data.field('field_x')
        yField = data.field('field_y')
        xIndex = data.field('index_x')
        yIndex = data.field('index_y')
    
    return xField, yField, xIndex, yIndex

def readXy(fitsXyPath, sort=False, sortKey='FLUX', sortReverse=True, retSortField=False):
    """
    X,Y position of sources/stars with origin (0,0)
    
    .axy = extracted sources from image, includes flux
    .xyls = stars from reference catalog, no flux
    
    :raise KeyError: if sort is True and the sortKey doesn't exist 
    """
    with fits.open(fitsXyPath) as hdulist:
        data = hdulist[1].data
        x = data.field('X')-1 # FITS has (1,1) origin
        y = data.field('Y')-1
    if sort:
        flux = data.field(sortKey)
        fluxSort = np.argsort(flux)
        if sortReverse:
            fluxSort = fluxSort[::-1]            
        x = x[fluxSort]
        y = y[fluxSort]
    
    if sort and retSortField:
        return x,y,flux
    else:
        return x,y

def recomputeXylsPixelPositions(originalXylsPath, originalWcsPath, newWcsPathOrHeader):
    """
    Return pixel coordinates valid for `newWcsPathOrHeader` for
    the reference stars found in `originalXylsPath` (belonging to `originalWcsPath`).
    
    :rtype: tuple (x,y) with x and y being ndarrays
    """
    # Step 1: compute RA,Dec of reference stars (as this is not stored in .xyls)
    originalWCS = WCS(readHeader(originalWcsPath))
    x, y = readXy(originalXylsPath)
    ra, dec = originalWCS.all_pix2world(x, y, 0)
    
    # Step 2: compute pixel positions of reference stars in new WCS solution
    if isinstance(newWcsPathOrHeader, string_types):
        newWCS = WCS(readHeader(newWcsPathOrHeader))
    else:
        newWCS = WCS(newWcsPathOrHeader)

    # all_world2pix raised a NoConvergenc error
    # As we don't use SIP, we don't need to use all_world2pix.
    # wcs_world2pix doesn't support any distortion correction. 
    xNew, yNew = newWCS.wcs_world2pix(ra, dec, 0)
    
    return xNew,yNew

def getCatalogStars(header, limit=500, maxVmag=None, retVmag=False, retry=1):
    """
    Queries the Vizier catalog and retrieves stars for the sky area
    as defined by the given WCS header.
    
    :param header: FITS WCS header, must include IMAGEW and IMAGEH
    :param limit: maximum number of stars to return (optional)
    :param maxVmag: maximum magnitude of stars (optional)
    :param retVmag: if true, include Vmag in the result tuple
    :param retry: how many times to retry in case of errors (e.g. network problems)
    :rtype: tuple (x, y) or (x, y, vmag) sorted by decreasing brightness, origin (0,0)
            Note that vmag is a masked array and can contain masked values.
    """
    column_filters = {}
    if maxVmag:
        column_filters['VTmag'] = '<' + str(maxVmag)
        
    w, h = header['IMAGEW'], header['IMAGEH']
                
    # Step 1: query stars in tycho-2 online catalog, ordered by Vmag
    # We add a small border here. This is useful for
    # circling stars in an image, such that half circles
    # are drawn at the image corners instead of suddenly disappearing
    # circles.
    catalog = 'I/259/tyc2'
    centerRa, centerDec = getCenterRADec(header)
    border = 0.01 * w
    radiusBorder = getPixelScale(header)*border
    
    radius = getRadius(header) + radiusBorder
    if limit:
        # we have to query more stars as our search region is a circle
        # and we are filtering stars out afterwards
        row_limit = limit + int(limit*1.4)
    else:
        row_limit = -1
    print('Querying Vizier...')
    v = Vizier(columns=['_RAJ2000', '_DEJ2000', '+VTmag'],
               column_filters=column_filters, 
               row_limit=row_limit)
    try:
        result = v.query_region(coord.SkyCoord(ra=centerRa, dec=centerDec,
                                               unit=(u.deg, u.deg),
                                               frame='icrs'), 
                                radius=radius*u.deg, catalog=catalog)[0]
    except Exception as e:
        if retry > 0:
            print(repr(e))
            print('retrying...')
            time.sleep(2)
            # astroquery may have stored a corrupt response in its cache,
            # so we try again without using the cache
            # see https://github.com/astropy/astroquery/issues/465
            with suspend_cache(Vizier):
                return getCatalogStars(header, limit, maxVmag, retVmag, retry-1)
        print('Vizier query_region: ra={}, dec={}, radius={}, column_filters={}, row_limit={}, catalog={}'.
              format(centerRa, centerDec, radius, column_filters, row_limit, catalog),
              file=sys.stderr)
        raise
        
    vmag = result['VTmag']
    ra = result['_RAJ2000']
    dec = result['_DEJ2000']
    
    print(len(vmag), 'stars received')
    
    # Step 2: compute pixel coordinates for stars
    wcs = WCS(header)
    x, y = wcs.wcs_world2pix(ra, dec, 0)
    
    # Step 3: remove stars outside the image bounds
    # As above, we leave a small border here.
    inside = (-border <= y) & (y < h+border) & (-border <= x) & (x < w+border)
    x = x[inside]
    y = y[inside]
    vmag = vmag[inside]
    print(len(vmag), 'stars left after filtering')
    
    # Step 4: apply limit by removing the faintest stars
    if limit:
        x = x[:limit]
        y = y[:limit]
        vmag = vmag[:limit]
        
    if retVmag:
        return x, y, vmag
    else:
        return x, y
    
def writeXyls(path, x, y, vmag=None, clobber=False):
    """
    Writes an .xyls file as produced by astrometry.net.
    The input data can be retrieved using getCatalogStars.
    
    :param path: path to 
    :param x: x coordinates of stars, origin 0
    :param y: y coordinates of stars, origin 0
    :param vmag: if masked array, then masked values become nan
    :param clobber: overwrite output file if it already exists
    """
    assert len(x) == len(y)
    if vmag is not None:
        assert len(x) == len(vmag)
    x = x+1
    y = y+1
    colX = fits.Column(name='X', format='1D', array=x)
    colY = fits.Column(name='Y', format='1D', array=y)
    cols = [colX, colY]
    if vmag is not None:
        colVmag = fits.Column(name='Vmag', format='1D', array=vmag)
        cols.append(colVmag)
    tbhdu = fits.BinTableHDU.from_columns(cols)
    prihdr = fits.Header()
    prihdr['AN_FILE'] = ('XYLS', 'Astrometry.net file type')
    prihdu = fits.PrimaryHDU(header=prihdr)
    thdulist = fits.HDUList([prihdu, tbhdu])
    thdulist.writeto(path, checksum=True, clobber=clobber)
    
def getNoradId(header):
    """
    Returns the NORAD ID as a five-char string from the NORADID card of a FITS header,
    or None if the card doesn't exist.
    """
    noradId = header.get('NORADID')
    if noradId is not None:
        noradId = int(noradId)
    return noradId
    
def setNoradId(header, noradId):
    noradId = str(int(noradId))
    
    if header.get('NORADID') is None:
        header['HISTORY'] = 'NORADID added by auromat Python library'
    
    header['NORADID'] = (noradId, 'NORAD ID of spacecraft')

def getPhotoTime(header):
    """
    Return the phototime as found in DATE-OBS or None if not existing.
    
    :rtype: datetime.datetime | None
    """
    dateobs = header.get('DATE-OBS')
    if dateobs is None:
        return None
    else:
        try:
            d = datetime.strptime(dateobs, '%Y-%m-%dT%H:%M:%S.%f')
        except:
            d = datetime.strptime(dateobs, '%Y-%m-%dT%H:%M:%S')
        return d

def getShiftedPhotoTime(header):
    """
    Returns the corrected photo time or if not available, the
    original photo time.
    
    :rtype: datetime.datetime | None
    """
    _, d, _ = getShiftedSpacecraftPosition(header)
    if d is None:
        d = getPhotoTime(header)
    return d

def getSpacecraftPosition(header):
    """
    Returns the spacecraft position at the original photo time
    in cartesian GCRS coordinates.
    :rtype: tuple([x,y,z], date)
    """
    date = getPhotoTime(header)
    x = header.get('POSX')
    if x is None or date is None:
        return None, None
    y = header['POSY']
    z = header['POSZ']
    return np.array([x,y,z]), date

def setSpacecraftPosition(header, xyz, date):
    """
    Add the spacecraft position at the original photo time
    in cartesian GCRS coordinates to header cards
    POSX, POSY, POSZ, DATE-OBS.
    
    :param ndarray xyz: [x,y,z] position of spacecraft at 'date'
    :param datetime date: the original photo time
    """
    x,y,z = xyz
       
    if header.get('POSX') is None:
        header['HISTORY'] = 'POS* & DATE-OBS added by auromat Python library'
        
    header['POSX'] = (x, 'X coordinate of spacecraft in GCRS at DATE-OBS')
    header['POSY'] = (y, 'Y coordinate of spacecraft in GCRS at DATE-OBS')
    header['POSZ'] = (z, 'Z coordinate of spacecraft in GCRS at DATE-OBS')
    dateStr = date.strftime('%Y-%m-%dT%H:%M:%S.%f')
    header['DATE-OBS'] = (dateStr, 'EXIF timestamp of the photograph')

def getShiftedSpacecraftPosition(header):
    """
    Returns the spacecraft position at the corrected photo time
    in cartesian GCRS coordinates.
    :rtype: tuple([x,y,z], datetime date, timedelta delta)
    """
    date = getPhotoTime(header)
    shift = header.get('DATESHIF')
    x = header.get('POSXSHIF')
    if x is None or date is None or shift is None:
        return None, None, None
    y = header['POSYSHIF']
    z = header['POSZSHIF']
    delta = timedelta(seconds=shift)
    shiftedDate = date + delta
    return np.array([x,y,z]), shiftedDate, delta
    
def setShiftedSpacecraftPosition(header, xyz, shiftedDate):
    """
    Add the spacecraft position at the given shifted photo time
    in cartesian GCRS coordinates to header cards
    POSXSHIF, POSYSHIF, POSZSHIF, DATESHIF.
    
    :param ndarray xyz: [x,y,z] position of spacecraft at 'date'
    :param datetime shiftedDate: the shifted photo time
    """
    x,y,z = xyz
    
    date = getPhotoTime(header)
    if date is None:
        raise ValueError('DATE-OBS needs to be set before setting the shifted date')
    delta = (shiftedDate - date).total_seconds()
    
    if header.get('POSXSHIF') is None:
        header['HISTORY'] = 'POS*SHIF & DATESHIF added by auromat Python library'
        
    header['POSXSHIF'] = (x, 'X coordinate of spacecraft in GCRS at DATESHIF')
    header['POSYSHIF'] = (y, 'Y coordinate of spacecraft in GCRS at DATESHIF')
    header['POSZSHIF'] = (z, 'Z coordinate of spacecraft in GCRS at DATESHIF')
    header['DATESHIF'] = (delta, 'DATE-OBS shift in seconds')
