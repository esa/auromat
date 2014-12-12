# Copyright European Space Agency, 2013

"""
This module provides various functions for visualizing mappings,
astrometric solutions, and distortion profiles.

The return value is nearly always a tuple consisting of
a matplotlib figure and axes object. This can then be further
modified or saved as an image file with :func:`saveFig`.
"""

from __future__ import division, absolute_import, print_function

from six import reraise, string_types
import os
import sys
import warnings
import datetime
from functools import partial
from auromat.util.coroutine import coroutine, throw

import numpy as np
import numpy.ma as ma
from astropy.coordinates.angles import Angle
import astropy.units as u
from scipy.spatial import Delaunay

import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
from matplotlib.mlab import poly_between
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import PolyCollection
from mpl_toolkits.basemap import Basemap
import brewer2mpl
from astropy.wcs.wcs import WCS
from auromat.fits import readHeader
import itertools
from matplotlib.patches import Polygon
from math import modf
blue_red = brewer2mpl.get_map('RdBu', 'Diverging', 11, reverse=True).mpl_colormap


from auromat.draw_helpers import overlapPolygons,\
    generatePolygonsFromMapping, \
    _generatePolygonsFromMappingOrCollection, _addFigureBottomTitle,\
    _convertMappingsToSM, _formatMLT, ensureContinuousPath,\
    loadFigImage, _saveFig, _getMplColors, _setMplColors, _circles, ColorMode
from auromat.resample import plateCarreeResolution, _resample
from auromat.utils import outline
import auromat.resources
import auromat.resample
import auromat.fits
from auromat.coordinates.geodesic import Location
from auromat.coordinates import geodesic, constellations
from auromat.mapping.mapping import BoundingBox
try:
    import auromat.util.lensdistortion
    import lensfunpy
except Exception as e:
    print(str(e))
    warnings.warn('lensfunpy not found, reduced functionality in auromat.draw')
    
DatePlotTitleFormat = '%Y-%m-%d %H:%M:%S.%f UTC'
DatePlotTitleWithoutUTCFormat = '%Y-%m-%d %H:%M:%S.%f'
            
def drawPlot(mapping):
    """
    Draws a single mapping onto a Longitude/Latitude plot equivalent
    to a rectilinear map projection.
    
    :rtype: tuple(Figure, Axes)
    """
    verts, colors = generatePolygonsFromMapping(mapping, ColorMode.matplotlib)
    
    if mapping.boundingBox.containsDiscontinuity:
        verts[:,:,1] = Angle((verts[:,:,1] + 180) * u.deg).wrap_at(180 * u.deg).degree
            
    fig, ax = plt.subplots()

    ax.set_xlabel('Longitude (deg)')
    ax.set_ylabel('Latitude (deg)')      

    verts = overlapPolygons(verts)
    
    verts = np.dstack((verts[:,:,1],verts[:,:,0])) # swap axes such that x=longitude

    coll = PolyCollection(verts, facecolors=colors, edgecolors='none', zorder=2,
                          rasterized=True)
    ax.add_collection(coll)
    ax.autoscale()
    
    if mapping.boundingBox.containsDiscontinuity:
        def fixAxis(x,pos):
            deg = Angle((x + 180) * u.deg).wrap_at(180 * u.deg).degree
            s = '{:g}'.format(deg) # removes trailing .0
            return s.replace('-', u'\u2212') # uses unicode minus (longer)
            # Note: this should rather make use of ScalarFormatter but I couldn't figure it out
        ax.xaxis.set_major_formatter(FuncFormatter(fixAxis))

    return fig, ax

def drawKmlImage(mapping):
    """
    Draws a mapping onto a figure without borders for use within Google Earth.
    The extents are identical to the bounding box of the mapping.
    
    Note: When saving, make sure to use `transparent=True` to make the areas
    outside the valid data transparent.
    
    :rtype: tuple(Figure, Axes)
    """
    boundingBox = mapping.boundingBox
    verts, colors = generatePolygonsFromMapping(mapping, ColorMode.matplotlib)
        
    if boundingBox.containsDiscontinuity:
        verts[:,:,1] = Angle((verts[:,:,1] + 180) * u.deg).wrap_at(180 * u.deg).degree
        lonWest = Angle((boundingBox.lonWest + 180) * u.deg).wrap_at(180 * u.deg).degree
        lonEast = Angle((boundingBox.lonEast + 180) * u.deg).wrap_at(180 * u.deg).degree
        boundingBox = BoundingBox(boundingBox.latSouth, lonWest, boundingBox.latNorth, lonEast)
        
    fig = plt.figure()
    fig.set_size_inches(1, 1)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.set_autoscale_on(False)
    ax.set_xlim(boundingBox.lonWest, boundingBox.lonEast)
    ax.set_ylim(boundingBox.latSouth, boundingBox.latNorth)
    fig.add_axes(ax)

    verts = overlapPolygons(verts)
    
    verts = np.dstack((verts[:,:,1],verts[:,:,0])) # swap axes such that x=longitude

    coll = PolyCollection(verts, facecolors=colors, edgecolors='none') # expects [r,g,b]
    ax.add_collection(coll)
    
    return fig, ax

def drawStereographic(mappings, lat0=None, lon0=None, width=None, height=None, 
                      boundingBox=None,
                      sizeFactor=1, lineDelta=10,
                      labelsParallels=[True, False, False, False], labelsMeridians=[False, False, False, True],
                      fmtParallels=None, fmtMeridians=None, bottomTitle=None,
                      drawlsmask=True,
                      drawCities=False, cityAlpha=0.6, figWidth=8.0, figHeight=5.0, rasterized=True):
    """
    Draws one or more mappings onto a stereographic map.
    
    Note that when drawing multiple mappings, the resulting bounding box should
    span less than 180 degrees in longitude.
    
    :param mappings: either a single BaseMapping, a MappingCollection, or a list of
                     both types
    :param degree|None lat0: latitude of image center
    :param degree|None lon0: longitude of image center
    :param number|None width: width of map in km
    :param number|None height: height of map in km
    :param BoundingBox boundingBox: determine lat0,lon0,width,heigth from this bounding box
    :param number sizeFactor: how much to zoom out, 1 = everything just fits on the map
    :param int lineDelta: draw a parallel/meridian every lineDelta degrees
    :param labelsParallels: a boolean list [left,right,top,bottom] that determines on which sides
                            to draw labels for parallels
    :param labelsMeridians: a boolean list [left,right,top,bottom] that determines on which sides
                            to draw labels for meridians
    :param str|function fmtParallels: format string or function for parallel labels
    :param str|function fmtMeridians: format string or function for meridian labels
    :param str bottomTitle: adds a title below the figure
    :param bool drawlsmask: whether to draw land-sea mask
    :param bool drawCities: whether to draw city markers
    :param number cityAlpha: the alpha value with which to draw the city markers
    :param number figWidth: figure width in inches
    :param number figHeight: figure height in inches
    :param bool rasterized: whether to rasterize the polygon mesh when saved as SVG
    :rtype: tuple(Figure, Axes, Basemap)
    """
    if isinstance(mappings, list):
        vertsArr = []
        colorsArr = []
        for mapping in mappings:
            vertsArr_, colorsArr_ = _generatePolygonsFromMappingOrCollection(mapping)
            vertsArr.extend(vertsArr_)
            colorsArr.extend(colorsArr_)
        boundingBoxes = [mapping.boundingBox for mapping in mappings]
        photoTimes = [mapping.photoTime for mapping in mappings]
        photoTimeRange = (min(photoTimes), max(photoTimes))
    else:
        vertsArr, colorsArr = _generatePolygonsFromMappingOrCollection(mappings)
        boundingBoxes = [mappings.boundingBox]
        photoTimeRange = (mappings.photoTime, mappings.photoTime)
    
    if lat0 is None or lon0 is None or width is None or height is None:
        if boundingBox is None:
            boundingBox = BoundingBox.mergedBoundingBoxes(boundingBoxes)
        if lat0 is None:
            lat0 = boundingBox.center.lat
        if lon0 is None:
            lon0 = boundingBox.center.lon

        if width is None:
            width = boundingBox.size.width * 1.05 * sizeFactor
        if height is None:
            height = boundingBox.size.height * 1.05 * sizeFactor
    
    fig, ax, m = _prepareStereographicBasemap(figWidth, figHeight, lat0, lon0, width, height,
                                              lineDelta=lineDelta, labelsParallels=labelsParallels,
                                              labelsMeridians=labelsMeridians, drawlsmask=drawlsmask,
                                              fmtMeridians=fmtMeridians, fmtParallels=fmtParallels)

    if photoTimeRange[0] == photoTimeRange[1]:
        fig.suptitle(photoTimeRange[0].strftime(DatePlotTitleFormat))
    else:
        fig.suptitle(photoTimeRange[0].strftime(DatePlotTitleFormat) + ' - ' + 
                     photoTimeRange[1].strftime(DatePlotTitleFormat))
        
    if bottomTitle:
        _addFigureBottomTitle(fig, bottomTitle)
    
    _drawPolygons(m, ax, vertsArr, colorsArr, rasterized=rasterized)
    if drawCities:
        _drawCities(ax, m, alpha=cityAlpha)
    return fig, ax, m
    
def drawStereographicMLatMLT(mappings, *args, **kwargs):
    """
    Draws one or more mappings in the MLat/MLT coordinate system with stereographic projection.
    
    See :func:`drawStereographic` for parameter descriptions.
    
    :rtype: tuple(Figure, Axes, Basemap)
    """
    mappings = _convertMappingsToSM(mappings)
    
    bottomTitle = kwargs.pop('bottomTitle', 'MLat/MLT')
    fmtMeridians = kwargs.pop('fmtMeridians', partial(_formatMLT, format_='%H:%M'))
    # as we use a different reference frame we cannot draw any shapes based on the
    # standard geographic coordinates
    kwargs['drawCities'] = False
    kwargs['drawlsmask'] = False
    return drawStereographic(mappings, *args, fmtMeridians=fmtMeridians, bottomTitle=bottomTitle, **kwargs)

def drawMLatMLTPolar(mappings, lineDelta=10, boundingBox=None, figWidth=8.0, figHeight=5.0, rasterized=True):
    """
    Draws one or more mappings in the MLat/MLT coordinate system with 
    polar azimuthal equidistant projection.
    
    :param mappings: either a single BaseMapping, a MappingCollection, or a list of
                     both types
    :param int lineDelta: draw a parallel/meridian every lineDelta degrees
    :param BoundingBox boundingBox: determines the latitude range from this bounding box    
    :param number figWidth: figure width in inches
    :param number figHeight: figure height in inches
    :param bool rasterized: whether to rasterize the polygon mesh when saved as SVG
    :rtype: tuple(Figure, Axes, Basemap)
    """
  
    def coordsFn(mapping):
        mlat, mlt = mapping.mLatMlt
        # [0,24] -> [-180,180]
        smlon = mlt-12
        smlon /= (24/360)
        return mlat, smlon
  
    if isinstance(mappings, list):
        vertsArr = []
        colorsArr = []
        for mapping in mappings:
            vertsArr_, colorsArr_ = _generatePolygonsFromMappingOrCollection(mapping, coordsFn=coordsFn)
            vertsArr.extend(vertsArr_)
            colorsArr.extend(colorsArr_)
        photoTimes = [mapping.photoTime for mapping in mappings]
        photoTimeRange = (min(photoTimes), max(photoTimes))
    else:
        vertsArr, colorsArr = _generatePolygonsFromMappingOrCollection(mappings, coordsFn=coordsFn)
        photoTimeRange = (mappings.photoTime, mappings.photoTime)
    
    if boundingBox is None:
        latSouth = min(np.min(v[:,0,0]) for v in vertsArr)
        latNorth = max(np.max(v[:,0,0]) for v in vertsArr)
    else:
        latSouth = boundingBox.latSouth
        latNorth = boundingBox.latNorth
        
    fig, ax, m = _preparePolarBasemap(figWidth, figHeight, latSouth, latNorth, 
                                      lon_0=180, round_=True,
                                      drawlsmask=False, drawgrid=False)
    
    if photoTimeRange[0] == photoTimeRange[1]:
        fig.suptitle(photoTimeRange[0].strftime(DatePlotTitleFormat))
    else:
        fig.suptitle(photoTimeRange[0].strftime(DatePlotTitleFormat) + ' - ' +
                     photoTimeRange[1].strftime(DatePlotTitleFormat))

    # bottom title
    fig.text(0.5, 0.02, 'MLat/MLT', horizontalalignment='center')
    
    m.drawparallels(np.arange(-90,90,lineDelta))
    
    # draw latitude labels along 9h line
    if (latSouth+latNorth)/2 > 0: # north pole projection
        latonplot = lambda lat: m.boundinglat <= lat
    else: # south pole projection
        latonplot = lambda lat: lat <= m.boundinglat
    lon = -45
    for lat in filter(latonplot, range(-80, 81, 10)):
        x,y = m(lon,lat)
        x,y = ax.transLimits.transform((x,y))
        x += 0.03
        ax.text(x, y, str(lat), horizontalalignment='left', verticalalignment='center', 
                transform=ax.transAxes)
    
    m.drawmeridians(list(range(-180,180,45)), labels=[True, True, True, True], fmt=_formatMLT)
    
    fig._applyMeridiansParallelsColors = True
    
    _drawPolygons(m, ax, vertsArr, colorsArr, rasterized=rasterized)
    return fig, ax, m

def _prepareStereographicBasemap(figWidth, figHeight, lat0, lon0, mapWidth, mapHeight, lineDelta=10,
                                 drawlsmask=True, drawgrid=True,
                                 labelsParallels=[True,False,False,False], labelsMeridians=[False,False,False,True],
                                 fmtParallels=None, fmtMeridians=None):
    assert mapWidth > 0 and mapHeight > 0
    print('centering map at', lat0, lon0)
    print('sizing map at', mapWidth, mapHeight, 'km')

    fig = plt.figure(figsize=(figWidth,figHeight))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    
    width = mapWidth*1000
    height = mapHeight*1000
    
    try:
        m = Basemap(width=width, height=height, # in meters
                    resolution='l', projection='stere', ellps='WGS84',
                    lat_0=lat0, lon_0=lon0, ax=ax)
    except:
        # add some debug information for reproducing the issue
        _, e, tb = sys.exc_info()
        new_exc = Exception('{}: {}; width={}, height={}, lat_0={}, lon_0={}'.\
                            format(e.__class__.__name__, e, width, height, lat0, lon0))
        reraise(new_exc.__class__, new_exc, tb)
        
    
    if drawlsmask:
        m.drawlsmask(ocean_color='0.8', land_color='0.6', lakes=False, resolution='l',grid=1.25) # lower res: resolution='c'
    
    if drawgrid:
        # TODO find good line frequency automatically
        kw = {}
        if fmtParallels:
            kw['fmt'] = fmtParallels  
        m.drawparallels(np.arange(-80.,81.,lineDelta), labels=labelsParallels, **kw) # labelstyle="+/-"
        
        kw = {}
        if fmtMeridians:
            kw['fmt'] = fmtMeridians
        m.drawmeridians(np.arange(-180.,181.,lineDelta), labels=labelsMeridians, **kw)
        
        fig._applyMeridiansParallelsColors = True
            
    return fig, ax, m

def _preparePolarBasemap(figWidth, figHeight, latSouth, latNorth, lineDelta=10, lon_0=180,
                         drawlsmask=True, drawgrid=True, round_=False,
                         labelsParallels=[True,False,False,False], labelsMeridians=[False,False,False,True]):
    """
    
    :param lon_0: lon_0 is at 6-o'clock
    """

    fig = plt.figure(figsize=(figWidth,figHeight))
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
        
    if (latSouth+latNorth)/2 > 0:
        # north pole
        projection='npaeqd'
        boundinglat = latSouth - 5
    else:
        # south pole
        projection='spaeqd'
        boundinglat = latNorth + 5
        
    m = Basemap(projection=projection, boundinglat=boundinglat, lon_0=lon_0,
                round=round_, resolution='l', ax=ax)

    if round_:
        # note that the actual color is set in draw_helpers.setMplColors
        m.drawmapboundary(fill_color='none')
        # see draw_utils._setMplColors
        fig._applyRoundPolarMapColors = True
        
    if drawlsmask:
        m.drawlsmask(ocean_color='0.8', land_color='0.6', lakes=False, resolution='l',grid=1.25) # lower res: resolution='c' 
    
    if drawgrid:
        # TODO find good line frequency automatically
        m.drawparallels(np.arange(-80.,81.,lineDelta),labels=labelsParallels) # labelstyle="+/-"
        m.drawmeridians(np.arange(-180.,181.,lineDelta),labels=labelsMeridians)
        fig._applyMeridiansParallelsColors = True
    
    return fig, ax, m

def _drawCities(ax, m, shapefile=None, color='red', alpha=0.6, moreCities=True):
    scale = '10' if moreCities else '50'
    if shapefile is None:
        resPath = os.path.dirname(auromat.resources.__file__)
        shapefile = os.path.join(resPath, 'ne_' + scale + 'm_populated_places_simple')
    m.readshapefile(shapefile, 'cities')
    natscale = np.array([i['natscale'] for i in m.cities_info])
    x,y = list(zip(*m.cities))
    xy = np.transpose([x,y])
    c = np.logical_and(m.xmin < xy[:,0], 
                       np.logical_and(xy[:,0] < m.xmax, 
                                      np.logical_and(m.ymin < xy[:,1], xy[:,1] < m.ymax)))
    xy = xy[c]
    natscale = natscale[c]/10
    
    s = ax.scatter(xy[:,0], xy[:,1], natscale, color, marker='o', edgecolors='none', zorder=10, alpha=alpha)
    # so that it can be accessed when outputting as svg
    s.set_gid('cities')
    
def _drawPolygons(m, ax, verts, colors, rasterized=True):
    """
    Draws one or more polygon meshes on a map.
    
    :param verts: list of (n, 4, 2) arrays in degrees (:,:,0 = lat)
    :param colors: list of (n, 3) arrays in [0,1]
    """    
    # see http://matplotlib.org/basemap/users/mapcoords.html
    verts_ = np.concatenate(verts)
    colors_ = np.concatenate(colors)
    if len(verts_) == 0:
        return
    
    lats = verts_[:,:,0].ravel()
    lons = verts_[:,:,1].ravel()
    xpt, ypt = m(lons,lats)
    verts_ = np.dstack((xpt.reshape(-1,4),ypt.reshape(-1,4)))

    verts_ = overlapPolygons(verts_)                                            
    coll = PolyCollection(verts_, facecolors=colors_, edgecolors='none', zorder=2,
                          rasterized=rasterized) # expects [r,g,b]
    ax.add_collection(coll)

def drawHorizon(astrometryMapping, color='blue', channel=None, lineThickness=2,
                useMappingAltitude=False, figax=None):
    """
    Draws the earth horizon of an unresampled astrometry-based mapping
    on top of the underlying image of the mapping.
    
    :param astrometryMapping: the unresampled mapping
    :type astrometryMapping: auromat.mapping.astrometry.BaseAstrometryMapping
    :param color: earth horizon line color (matplotlib color spec)
    :param str channel: if given (r|g|b) then draw only the given channel of the image
    :param lineThickness: line width
    :param figax: a (Figure,Axes) tuple which shall be used instead of creating a new one;
                  if given, then the mapping image is not drawn
    :rtype: tuple(Figure,Axes)
    """
    if figax:
        fig, ax = figax
    else:
        im = astrometryMapping.rgb_unmasked
        if channel:
            if channel == 'red':
                im = im[:,:,0]
            elif channel == 'green':
                im = im[:,:,1]
            elif channel == 'blue':
                im = im[:,:,2]
        fig, ax = loadFigImage(im)
    
    if astrometryMapping.isConsistent():
        if useMappingAltitude:
            intersection = astrometryMapping.intersectionInflatedCenter
        else:
            intersection = astrometryMapping.intersectsEarth
        
        curve = _getIntersectionCurve(intersection)
        ax.plot(curve[:,0], curve[:,1], color=color, lw=lineThickness)   

    return fig, ax

def _getIntersectionCurve(intersection):
    """
    :param intersection: either (h+1,w+1,3) with intersection points or as boolean intersects array (h+1,w+1)
                         Note that the array must include both intersected and non-intersected elements. 
    :rtype: (n,2) ndarray of curve points
    """
    assert (intersection.dtype == bool and intersection.ndim == 2) or (intersection.ndim == 3)
 
    if intersection.dtype == bool:
        intersects = intersection
    else:
        intersects = ~np.isnan(intersection[:,:,0])
        
    outl = outline(intersects)
    # filter out points along the image border
    yfilt = np.logical_and(0 < outl[:,1], outl[:,1] < intersection.shape[0]-1)
    xfilt = np.logical_and(0 < outl[:,0], outl[:,0] < intersection.shape[1]-1)
    curve = outl[np.logical_and(yfilt, xfilt)]
                
    # Now the curve might end up as two logical path segments, because
    # the initial start of the path may have been in the middle of the curve,
    # depending on where findContours of OpenCV starts.
    # To correct that, we find the jump and swap both curve parts.
    curve = ensureContinuousPath(curve)
    return curve

def drawHistogram(hist, vlines=[], xlabel=None, ylabel=None, **kw):
    """
    Draws a histogram in line-style with unspecified bin boundaries.
    
    :param hist: a one-dimensional array of histogram y-values 
    :param vlines: a list of (value,color) tuples describing vertical lines to draw
    :param str xlabel: label of x axis
    :param str ylabel: label of y axis
    :rtype: tuple(Figure,Axes)
    """
    bins = np.arange(len(hist))

    fig, ax = drawLinePlot(bins, hist, xlabel, ylabel, **kw)
    ax.set_xlim([0, bins[-1]])
    
    for (i,color) in vlines:
        ax.axvline(i, color=color)
        
    return fig, ax

def drawHeatmaps(mapping, pathPrefix='heat_', xlabel='Image Width (px)', ylabel='Image Height (px)',
                 bgcolor='white', widthPx=None, dpi=None):
    """
    Draws latitude, longitude, elevation, azimuth (if available, e.g. MIRACLE mappings)
    as blue-red heatmaps for the given mapping.
    
    The resulting filenames are:
    
    - `<pathPrefix>lats.png`
    - `<pathPrefix>lons.png`
    - `<pathPrefix>elevation.png`
    - `<pathPrefix>azimuth.png`
    
    :param str pathPrefix: path prefix for the images that are created
    :param bgcolor: plot background color (matplotlib color spec)
    :param widthPx: width in pixels of the saved images
    :param dpi: dpi of the saved images
    """
    save = partial(saveFig, bgcolor=bgcolor, widthPx=widthPx, dpi=dpi)
    save(pathPrefix + 'lats.png', drawHeatmap(mapping.lats, cbLabel='Latitude (deg)', 
                                              xlabel=xlabel, ylabel=ylabel))
    save(pathPrefix + 'lons.png', drawHeatmap(mapping.lons, cbLabel='Longitude (deg)',
                                              xlabel=xlabel, ylabel=ylabel))
    save(pathPrefix + 'elevation.png', drawHeatmap(mapping.elevation, cbLabel='Elevation angle (deg)', 
                                                   xlabel=xlabel, ylabel=ylabel))
    
    try: # not all mappings have azimuth data
        save(pathPrefix + 'azimuth.png', drawHeatmap(mapping.azimuthCenter, xlabel=xlabel, ylabel=ylabel))
    except:
        pass

def drawHeatmap(data, cbLabel=None, xlabel=None, ylabel=None, figax=None, alpha=1.0):
    """
    Draws a 2d-data array as blue-red heatmap.
    
    :param data: 2D array, can contains NaN's and/or be a masked numpy array
    :param cbLabel: data label, written alongside the colorbar
    :param figax: a (Figure,Axes) tuple to reuse
    :param alpha: alpha value of the drawn heatmap
    :rtype: tuple(Figure,Axes)
    """
    data = ma.masked_invalid(data, copy=False)
    fig, ax = plt.subplots() if figax is None else figax
    im = ax.imshow(data, cmap=blue_red, alpha=alpha)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(im, cax=cax)
    if cbLabel is not None:
        cb.set_label(cbLabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    # TODO fix colors
#    _setMplColors(None, cax, colors)
    return fig, ax

@coroutine
def drawScanLinesCo(outputFile, widthPx=None, dpi=None, transparent=False, bgcolor='black',
                    lat0=None, lon0=None, mapWidth=None, mapHeight=None,
                    fmtParallels=None, fmtMeridians=None, drawlsmask=True, bottomTitle=None,
                    labelsParallels=[True, False, False, False], labelsMeridians=[False, False, False, True],
                    lineDelta=10, figWidth=8.0, figHeight=5.0,
                    arcsecPerPx=100, lineWidthFactor=1):
    """
    Draws a scanline from each given mapping such that the resulting stereographic map
    gives a temporal and spatial varying overview of all mappings.
    Scanlines are spherical rectangles perpendicular to the 'flying' direction of the mapping
    centroids when seen on a map.
    
    Note that the centroid of each mapping becomes the center of each corresponding scanline, 
    therefore the sequence should be masked by elevation before-hand with a sensible value.
    
    The input to this coroutine can either be a sequence of unresampled mappings or
    a dictionary containing `mapping.properties` in the 'props' key and the masked and
    resampled mapping in the 'mapping' key. In the latter case, the `arcsecPerPx` parameter
    is ignored.
    
    Example:
    
    >>> from auromat.util.coroutine import broadcast
    >>> seq = ...retrieve mapping sequence
    >>> broadcast(seq, drawScanLinesCo('test.png'))
    
    :param number arcsecPerPx: spherical resolution used for resampling
    :param number lineWidthFactor: how much to widen the scanline width relative to the
                                   the computed width derived from the first and second mapping;
                                   useful if mappings in-between the sequence are missing
    
    For all other parameters, see :func:`saveFig` and :func:`drawStereographic`.
    """
    try:
        currentMapping = yield
    except GeneratorExit:
        raise ValueError('mapping sequence too short, need at least 2 mappings')
    
    try:
        currentMappingProps = currentMapping['props']
        currentMapping = currentMapping['mapping']
    except (KeyError,TypeError):
        currentMappingProps = currentMapping.properties
        pxPerDeg = plateCarreeResolution(currentMappingProps.boundingBox, arcsecPerPx)
        currentMapping = auromat.resample.resample(currentMapping, pxPerDeg=pxPerDeg)
    
    bb = currentMappingProps.boundingBox
    # constant height of rotated scanline box
    # it is bigger than needed, just to be sure that it contains as much as possible
    height = geodesic.distance(bb.topLeft, bb.bottomRight)*1.5
    
    # constant width of rotated bounding box
    # determined from first and second mapping
    width = None
    
         
    # the amount of azimuthCenter to add to the camera footpoint azimuthCenter to face the mapping centroid   
    deltaAzimuth = None
    # the distance between the camera footpoint and the mapping centroid
    deltaDistance = None
    
    # The reason for calculating the centroid azimuthCenter track from the camera footpoints
    # is smoothness. The direct calculation of azimuths from two centroids is too noisy
    # and cannot be used for creating the scan lines.
    # The assumption made is that the camera doesn't move (tilt etc.) throughout the sequence.
    
    photoTimes = []
    lineBoundingBoxes = []
    centroids = []
    azimuths = []
    maxHeight = 0 # maximum measured height across all scanlines 
    
    vertsArr = []
    colorsArr = []
    
    done = False
    while not done:
        try:
            nextMapping = yield
        except GeneratorExit:
            done = True
            if not photoTimes:
                raise ValueError('mapping sequence too short, need at least 2 mappings')
        
        photoTimes.append(currentMappingProps.photoTime)
        centroids.append(currentMappingProps.centroid)
        
        if not done:
            try:
                nextMappingProps = nextMapping['props']
                nextMapping = nextMapping['mapping']
            except (KeyError,TypeError):
                nextMappingProps = nextMapping.properties
                # Important: Resampling is done after all values like centroid etc.
                # have been stored using the non-resampled mapping (higher accuracy).
                # For drawing, we need to resample though to reduce polygon counts.
                
                # Note that we use the same pxPerDeg for all mappings to stay on the same grid
                nextMapping = auromat.resample.resample(nextMapping, pxPerDeg=pxPerDeg)
                
            azCamFoot = geodesic.course(currentMappingProps.cameraFootpoint, nextMappingProps.cameraFootpoint)
        
        if width is None:
            # The width must be at least as big such that a polygon/pixel
            # can fit inside, no matter how the scanline is oriented in regards
            # to the regular lat/lon grid. The diagonal size of the first polygon
            # multiplied by 3 is used to approximate this. The factor was determined
            # by trial-and-error until no polygons of any scanline were left out
            # anymore. The exact value would have to use the maximum polygon size across
            # the scanlines of all mappings, which would require 2 passes and hence
            # double the time. As this plot merely serves for overview purposes, 
            # the approximation is considered good enough.
            verts, _ = generatePolygonsFromMapping(currentMapping)
            p1 = Location(verts[0][0][0], verts[0][0][1])
            p2 = Location(verts[0][2][0], verts[0][2][1])
            diagonalDistancePixel = geodesic.distance(p1, p2)
            
            consecutiveMappingDistance = geodesic.distance(currentMappingProps.centroid, nextMappingProps.centroid)
            width = max(diagonalDistancePixel*3, consecutiveMappingDistance) * lineWidthFactor
            
            deltaDistance = geodesic.distance(currentMappingProps.cameraFootpoint, currentMappingProps.centroid)
            
            azCamFootToCentroid = geodesic.course(currentMappingProps.cameraFootpoint, currentMappingProps.centroid)
            deltaAzimuth = azCamFoot - azCamFootToCentroid
            
        if not done:
            azCamFootToCentroid = azCamFoot - deltaAzimuth
            centroidFromAzCurrent = geodesic.destination(currentMappingProps.cameraFootpoint, azCamFootToCentroid, deltaDistance)
            centroidFromAzNext = geodesic.destination(nextMappingProps.cameraFootpoint, azCamFootToCentroid, deltaDistance)
            az = geodesic.course(centroidFromAzCurrent, centroidFromAzNext)
        else:
            # reuse the azimuth from the previous two frames
            # it shouldn't change that much from the last one
            pass
        
        azimuths.append(az)
                    
        middleRight = geodesic.destination(currentMappingProps.centroid, az, width/2)
        middleLeft = geodesic.destination(currentMappingProps.centroid, az+180, width/2)
        
        topLeft = geodesic.destination(middleLeft, az-90, height/2)
        bottomLeft = geodesic.destination(middleLeft, az+90, height/2)
        topRight = geodesic.destination(middleRight, az-90, height/2)
        bottomRight = geodesic.destination(middleRight, az+90, height/2)
        
        # skip last point of each geodesic line so that there are no duplicate points
        polygon = np.concatenate((geodesic.line(topLeft, topRight)[:-1],
                                  geodesic.line(topRight, bottomRight)[:-1],
                                  geodesic.line(bottomRight, bottomLeft)[:-1],
                                  geodesic.line(bottomLeft, topLeft)[:-1]))
        
        currentMapping = currentMapping.maskedByPolygon(polygon)
        
        lineBoundingBoxes.append(currentMapping.boundingBox)
        
        # approximate real height of scanline
        # the commented code uses OpenCV
#        rect = cv.minAreaRect(np.require(currentMapping.outline, np.float32, 'C').reshape(-1,1,2))
#        box = cv.cv.BoxPoints(rect)
#        distance1 = geodesic.distance(Location(*box[0]), Location(*box[1]))
#        distance2 = geodesic.distance(Location(*box[1]), Location(*box[2]))
#        currentHeight = max(distance1, distance2)
        # Using minAreaRect would be the most accurate, but currently only OpenCV
        # includes that as a function. We approximate it by using the diagonal
        # of the bounding box. This may lead to slightly larger heights than the actual heights.
        currentHeight = geodesic.distance(currentMapping.boundingBox.topLeft, 
                                          currentMapping.boundingBox.bottomRight)
        if currentHeight > maxHeight:
            maxHeight = currentHeight

        verts, colors = generatePolygonsFromMapping(currentMapping, ColorMode.matplotlib)
        vertsArr.append(verts)
        colorsArr.append(colors)
        
#        print('vertsarr:', sum(a.nbytes/1024/1024 for a in vertsArr), 'MB')
#        print('colorsarr:', sum(a.nbytes/1024/1024 for a in colorsArr), 'MB')
        
        currentMapping = nextMapping
        currentMappingProps = nextMappingProps
        
    debug = False
    
    if len(vertsArr) == 0:
        raise ValueError('mapping sequence too short')
                    
    # calculate time axis coordinates
    # the ticks are distributed with the assumption that there are no big gaps
    # in regards to available mappings over the sequence time range
    
    # first, the axis line itself
    mappingCount = len(vertsArr)
    tickCount = 4
    axisLineResolution = 10
    axisLinePointCount = mappingCount//axisLineResolution
    axisLinePointCount = max(tickCount, axisLinePointCount)
    axisLineDistance = maxHeight/2*1.1 
    lineIndices = np.round(np.linspace(0, mappingCount-1, axisLinePointCount)).astype(np.int)

    axisLine = [geodesic.destination(centroids[i], azimuths[i]-90, axisLineDistance) 
                for i in lineIndices]
    
    # then, ticks and labels
    labelDistance = axisLineDistance*1.2
    tickIndices = np.round(np.linspace(0, mappingCount-1, tickCount)).astype(np.int)
    tickLines = [(geodesic.destination(centroids[i], azimuths[i]-90, axisLineDistance),
                  geodesic.destination(centroids[i], azimuths[i]-90, axisLineDistance*1.04))
                 for i in tickIndices]
    tickLabels = [(geodesic.destination(centroids[i], azimuths[i]-90, labelDistance),
                   photoTimes[i])
                  for i in tickIndices]

    estimatedLabelEdge = labelDistance*1.1
    tickLabelsEdges = [geodesic.destination(centroids[i], azimuths[i]-90, estimatedLabelEdge)           
                       for i in tickIndices]
    timeAxisBoundingBox = BoundingBox.minimumBoundingBox([[p.lat,p.lon] for p in tickLabelsEdges])
    
    # TODO account for text width of first/last label 
    
    boundingBoxes = lineBoundingBoxes + [timeAxisBoundingBox]
    
    # TODO duplicate code (see drawStereographic)
#        if lat0 is None or lon0 is None or mapWidth is None or mapHeight is None:
    
    combinedBoundingBox = BoundingBox.mergedBoundingBoxes(boundingBoxes)
    if lat0 is None or debug:
        lat0 = combinedBoundingBox.center.lat
    if lon0 is None or debug:
        lon0 = combinedBoundingBox.center.lon

    if mapWidth is None or debug:
        mapWidth = combinedBoundingBox.size.width * 1.05
    if mapHeight is None or debug:
        mapHeight = combinedBoundingBox.size.height * 1.05
    
    # finally, draw everything
    fig, ax, m = _prepareStereographicBasemap(figWidth, figHeight, lat0, lon0, mapWidth, mapHeight,
                                 lineDelta=lineDelta, labelsParallels=labelsParallels, labelsMeridians=labelsMeridians,
                                 drawlsmask=drawlsmask, fmtParallels=fmtParallels, fmtMeridians=fmtMeridians)
    
    photoTimeRange = (min(photoTimes), max(photoTimes))

    if photoTimeRange[0] == photoTimeRange[1]:
        fig.suptitle(photoTimeRange[0].strftime(DatePlotTitleFormat))
    else:
        fig.suptitle(photoTimeRange[0].strftime(DatePlotTitleWithoutUTCFormat) + ' - ' + 
                     photoTimeRange[1].strftime(DatePlotTitleFormat))
        
    if bottomTitle:
        _addFigureBottomTitle(fig, bottomTitle)
    
    _drawPolygons(m, ax, vertsArr, colorsArr)
    
    if mappingCount > 0:
        x, y = m([p.lon for p in axisLine], [p.lat for p in axisLine])
        ax.plot(x, y)

        for (p1,p2) in tickLines:
            lats = [p1.lat, p2.lat]
            lons = [p1.lon, p2.lon]
            x, y = m(lons, lats)
            ax.plot(x, y)
        
        for (bottomCenter, date) in tickLabels:
            x, y = m(bottomCenter.lon, bottomCenter.lat)
            timeStr = date.strftime('%H:%M:%S')
            ax.text(x, y, timeStr, ha='center', va='bottom') #fontsize=12
    
    saveFig(outputFile, (fig,ax), bgcolor=bgcolor, transparent=transparent, widthPx=widthPx, dpi=dpi)

@coroutine
def drawScanLinesMLatMLTCo(*args, **kwargs):
    """
    As :func:`drawScanLinesCo` but with MLat/MLT as coordinate system.
    """
    fmtMeridians = kwargs.pop('fmtMeridians', partial(_formatMLT, format_='%H:%M'))
    bottomTitle = kwargs.pop('bottomTitle', 'MLat/MLT')
    # as we use a different reference frame we cannot draw any shapes based on the
    # standard geographic coordinates
    kwargs['drawlsmask'] = False
    drawFn = drawScanLinesCo(*args, fmtMeridians=fmtMeridians, bottomTitle=bottomTitle, **kwargs)
    
    try:
        while True:
            try:
                drawFn.send(_convertMappingToSM((yield)))
            except GeneratorExit:
                break
        
        drawFn.close()
    except:
        throw(drawFn, *sys.exc_info()) # will raise back to us    

def _convertMappingToSM(mapping):
    try:
        return {'props': mapping['props_sm'],
                'mapping': mapping['mapping_sm']}
    except KeyError:
        return _convertMappingsToSM(mapping)
        
@coroutine
def drawAzimuthPlotsCo(outputFileAzCentroid=None, outputFileAzCentroidFromCam=None,
                       outputFileAzCamFootToCentroid=None,
                       outputFileLatLonCentroid=None, outputFileLatLonCentroidFromAz=None,
                       outputFileLatLonCamFoot=None,
                       widthPx=None, dpi=None, transparent=False, bgcolor='black'):
    """
    Draws various plots based on the centroid of each mapping.
    
    :param outputFileAzCentroid: azimuth ('flying' direction) of the centroids 
    :param outputFileAzCentroidFromCam: azimuth ('flying' direction) of the recalculated centroids
                                        (using camera footpoints) 
    :param outputFileLatLonCentroid: latitude-longitude plot of centroids
    :param outputFileLatLonCentroidFromAz: latitude-longitude plot of recalculated centroids
                                           (using camera footpoints) 
    :param outputFileLatLonCamFoot: latitude-longitude plot of camera footpoints
    
    See :func:`saveFig` for other parameters.
    """
    try:
        mapping = yield
    except GeneratorExit:
        raise ValueError('mapping sequence too short')
    
    try:
        currentMappingProps = mapping['props']
    except (KeyError,TypeError):
        currentMappingProps = mapping.properties
    del mapping
            
    # the amount of azimuthCenter to add to the camera footpoint azimuthCenter to face the mapping centroid   
    deltaAzimuth = None
    # the distance between the camera footpoint and the mapping centroid
    deltaDistance = None
    
    # The reason for calculating the centroid azimuthCenter track from the camera footpoints
    # is smoothness. The direct calculation of azimuths from two centroids is too noisy
    # and cannot be used for creating the scan lines.
    
    boundingBoxes = []
    camFootpoints = []
    centroids = []
    centroidsFromAz = []
    azimuthsCamFootToCentroid = []
    azimuthsCentroid = []
    azimuthsCentroidFromCam = []
    
    photoTimes = []
        
    while True:
        try:
            mapping = yield
        except GeneratorExit:
            break
        
        try:
            nextMappingProps = mapping['props']
        except (KeyError,TypeError):
            nextMappingProps = mapping.properties
        del mapping
        
        boundingBoxes.append(currentMappingProps.boundingBox)
        centroids.append(currentMappingProps.centroid)
        photoTimes.append(currentMappingProps.photoTime)
        camFootpoints.append(currentMappingProps.cameraFootpoint)
        
        azCamFoot = geodesic.course(currentMappingProps.cameraFootpoint, nextMappingProps.cameraFootpoint)
        
        if deltaDistance is None:
            deltaDistance = geodesic.distance(currentMappingProps.cameraFootpoint, currentMappingProps.centroid)
            
            azCamFootToCentroid = geodesic.course(currentMappingProps.cameraFootpoint, currentMappingProps.centroid)
            deltaAzimuth = azCamFoot - azCamFootToCentroid
        
        
        # B. determine azimuthCenter using centroid
        azimuthsCentroid.append(geodesic.course(currentMappingProps.centroid, nextMappingProps.centroid))
        
        # C. determine azimuthCenter using recalculated centroid (from camera footpoint)
        azCamFootToCentroid = azCamFoot - deltaAzimuth # TODO check, possibly wrap
        centroidFromAzCurrent = geodesic.destination(currentMappingProps.cameraFootpoint, azCamFootToCentroid, deltaDistance)
        centroidFromAzNext = geodesic.destination(nextMappingProps.cameraFootpoint, azCamFootToCentroid, deltaDistance)
        centroidsFromAz.append(centroidFromAzCurrent)
        azimuthsCentroidFromCam.append(geodesic.course(centroidFromAzCurrent, centroidFromAzNext))
        azimuthsCamFootToCentroid.append(azCamFootToCentroid)
        
        currentMappingProps = nextMappingProps
            
    save = partial(saveFig, bgcolor=bgcolor, transparent=transparent, widthPx=widthPx, dpi=dpi)

    # B. draw time-azimuthCenter plot using centroids
    if outputFileAzCentroid:
        figax = drawLinePlot(photoTimes, azimuthsCentroid, 
                              xlabel='Time', 
                              ylabel='Azimuth ($^\circ$) using centroid')
        save(outputFileAzCentroid, figax)
    
    # C. draw time-azimuthCenter plot using recalculated centroids
    if outputFileAzCentroidFromCam:
        figax = drawLinePlot(photoTimes, azimuthsCentroidFromCam, 
                              xlabel='Time', 
                              ylabel='Azimuth ($^\circ$) using recalculated centroid')
        save(outputFileAzCentroidFromCam, figax)
                
    # B2. draw lat-lon line plot using centroids
    if outputFileLatLonCentroid:
        lats = [c.lat for c in centroids]
        lons = [c.lon for c in centroids]
        figax = drawLinePlot(lons, lats, 
                              xlabel='Longitude ($^\circ$) of centroid', 
                              ylabel='Latitude ($^\circ$) of centroid')
        save(outputFileLatLonCentroid, figax)
    
    # C2. draw lat-lon line plot using recalculated centroids
    if outputFileLatLonCentroidFromAz:
        lats = [c.lat for c in centroidsFromAz]
        lons = [c.lon for c in centroidsFromAz]
        figax = drawLinePlot(lons, lats, 
                              xlabel='Longitude ($^\circ$) of recalculated centroid', 
                              ylabel='Latitude ($^\circ$) of recalculated centroid')
        save(outputFileLatLonCentroidFromAz, figax)
    
    # draw lat-lon line plot of camera footpoints
    if outputFileLatLonCamFoot:
        lats = [c.lat for c in camFootpoints]
        lons = [c.lon for c in camFootpoints]
        figax = drawLinePlot(lons, lats, 
                              xlabel='Longitude ($^\circ$) of camera footpoint', 
                              ylabel='Latitude ($^\circ$) of camera footpoint')
        save(outputFileLatLonCamFoot, figax)

def drawLinePlot(x, y, xlabel, ylabel, title=None, linecolor=None, linewidth=None, retline=False, **kw):
    """
    Draws a line plot.
    
    :param array-like x: x values, can be datetime 
    :param array-like y: y values
    :param str xlabel: label of x axis
    :param str ylabel: label of y axis
    :param str title: plot title
    :param linecolor: line color (matplotlib color spec)
    :param linewidth: line width
    :param bool retline: if True, then the Line object is returned as last tuple element
    :param kw: additional keywords are handed over to matplotlib's `ax.plot` function
    :rtype: tuple (Figure,Axes[,Line])
    """
    fig,ax = plt.subplots()
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if linecolor is not None:
        kw['color'] = linecolor
        
    if len(x)>0 and isinstance(x[0], datetime.datetime): 
        x_ = mpl.dates.date2num(x)
        line, = ax.plot_date(x_, y, fmt='b-', **kw)
        if max(d.microsecond for d in x) == 0:
            ax.xaxis.get_major_formatter().scaled[1/(24.*60.)] = '%H:%M:%S' # don't show float seconds
        fig.autofmt_xdate()
    else:    
        line, = ax.plot(x, y, **kw)
        
    if linewidth:
        line.set_linewidth(linewidth)
        
    if retline:
        return fig, ax, line
    else:
        return fig, ax
    
def _drawScatterPlot(x, y, xlabel, ylabel, title=None, pointcolor=None, retscatter=False, **kw):
    fig,ax = plt.subplots()
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    scatter = ax.scatter(x, y, color=pointcolor, **kw)
    if retscatter:
        return fig, ax, scatter
    else:
        return fig, ax

def drawLensDistortionDisplacementPlot(mod):
    """
    Draws the displacement in pixels caused by lens distortion as a 2D heatmap.
    
    :param lensfunpy.Modifier mod: the Modifier object
    :rtype: tuple(Figure,Axes) 
    """
    distances = auromat.util.lensdistortion.lensDistortionPixelDistances(mod=mod)
    
    calib = mod.lens.interpolate_distortion(mod.focal_length)
    model, params = _formatDistortionParams(calib)
    
    cbLabel = 'Displacement (px)'
    fig, ax = drawHeatmap(distances, cbLabel=cbLabel)
    
    ax.set_xlabel('Image Width (px)')
    ax.set_ylabel('Image Height (px)')
    ax.set_title(model + '(' + params + ')')
    
    return fig, ax
    
def drawLensDistortionDerivativePlot(mod):
    """
    Draws the lens distortion derivative.
    This visualizes barrel and pincushion distortion over the sensor radius.
    
    :param lensfunpy.Modifier mod: the Modifier object
    :rtype: tuple(Figure,Axes)
    """
    # derivatives of (rd - ru)/ru
    # see http://sourceforge.net/p/lensfun/mailman/message/32205273/
    def ptlens(ru, a, b, c):
        return 3*a*ru**2 + 2*b*ru + c
    
    def poly3(ru, k1):
        return 2*k1*ru
    
    def poly5(ru, k1, k2):
        return 2*k1*ru + 4*k2*ru**3
    
    # get the internal models interpolated for the given focal length
    calib = mod.lens.interpolate_distortion(mod.focal_length)
    
    if calib.model == lensfunpy.DistortionModel.PTLENS:
        a,b,c = calib.terms
        rd1 = partial(ptlens, a=a, b=b, c=c)    
        
    elif calib.model == lensfunpy.DistortionModel.POLY3:
        k1,_,_ = calib.terms
        rd1 = partial(poly3, k1=k1)
        
    elif calib.model == lensfunpy.DistortionModel.POLY5:
        k1,k2,_ = calib.terms
        rd1 = partial(poly5, k1=k1, k2=k2)
        
    else:
        raise NotImplementedError
    
    model, params = _formatDistortionParams(calib)
    
    # calculate the sensor's (half) height, as it is used for scaling within lensfun
    # TODO doesn't take into account the aspect ratio yet
    wFX = 36
    hFX = 24
    dFX = np.sqrt(wFX**2 + hFX**2)
    alpha = np.arcsin(wFX/dFX)
    d = dFX*mod.lens.crop_factor
    h = np.cos(alpha)*d
    
    sensorHalfHeight = h/2
    sensorHalfDiagonal = d/2
    
    X = np.linspace(0, sensorHalfDiagonal, 100)
    Xscaled = X/sensorHalfHeight
    
    fig, ax = drawLinePlot(X, 
                           rd1(Xscaled)*sensorHalfHeight,
                           xlabel='$h\;(\mathrm{mm})$',
                           ylabel='$dD/dh\;(\mathrm{mm}^{-1})$',
                           title=model + '(' + params + ')',
                           )
    ax.set_xlim([0, sensorHalfDiagonal])
    # shade y>0 and y<0 with colors, to indicate pincushion vs. barrel distortion
    ymin, ymax = ax.get_ylim()
    ax.autoscale(False)
    polyPincushion = poly_between([0,sensorHalfDiagonal], 0, ymax)
    polyPincushionArtist = ax.fill(*polyPincushion, facecolor='PeachPuff')[0]
    polyBarrel = poly_between([0,sensorHalfDiagonal], ymin, 0)
    polyBarrelArtist = ax.fill(*polyBarrel, facecolor='LightBlue')[0]

    ax.legend([polyPincushionArtist, polyBarrelArtist], 
              ['pincushion', 'barrel'],
              loc='lower right')
    
    return fig, ax

def _formatDistortionParams(calib):
    pround = partial(round, ndigits=7)
    if calib.model == lensfunpy.DistortionModel.PTLENS:
        model = 'ptlens'
        a,b,c = calib.terms
        params = '{}, {}, {}'.format(pround(a), pround(b), pround(c))
        
    elif calib.model == lensfunpy.DistortionModel.POLY3:
        model = 'poly3'
        k1,_,_ = calib.terms
        params = '{}'.format(pround(k1))
        
    elif calib.model == lensfunpy.DistortionModel.POLY5:
        model = 'poly5'
        k1,k2,_ = calib.terms
        params = '{}, {}'.format(pround(k1),pround(k2))
        
    else:
        raise NotImplementedError
    
    return model, params

def drawReferenceStars(mapping, **kw):
    """
    Draws reference stars on top of the original image of an unresampled 
    astrometry-based mapping by using the Tycho-2 catalogue of the Vizier web service.
    
    :param auromat.mapping.astrometry.BaseAstrometryMapping mapping: the mapping
    :rtype: tuple(Figure,Axes)
    
    For other parameters, see :func:`drawIndxPlot`.
    """
    return drawIndxPlot(mapping.rgb_unmasked, wcsPath=mapping.wcsHeader,
                        useWebCatalog=True, **kw)

def drawIndxPlot(imagePath, axyPath=None, xylsPath=None, matchPath=None,
                 wcsPath=None, newWcsPathOrHeader=None, maxAxyObjects=40, mask=None,
                 useWebCatalog=False, webCatalogLimit=1000, xylsSources=None, scale=5,
                 figax=None):
    """
    Draws reference and/or image-extracted stars from astrometry.net's .axy and .xyls files,
    or for reference stars as well using the Vizier webservice with the Tycho-2 catalogue.
    
    The matching star quad as found by astrometry.net and stored in .match files can also be
    visualized.
    
    :param str|array imagePath: the image to use as background, either a path or
                                an RGB/grayscale image array of uint8/uint16 type
    :param str axyPath: path to .axy file containing extracted stars
    :param str xylsPath: path to .xyls file containing reference stars
    :param str matchPath: path to .match file containing the matched quad (see astrometry.net docs)
    :param wcsPath, newWcsPathOrHeader: if given together with xlysPath, 
                    then the xyls pixel positions are recalculated
    :param useWebCatalog: if True and wcsPath is given, then astroquery
                          is used to obtain reference stars
    :param webCatalogLimit: the maximum number of stars to fetch from the web catalog 
    :param float scale: the size of the circles drawn around stars
    :param int maxAxyObjects: the maximum number of extracted stars to draw
    :param array-like xylsSources: an x,y pixel coordinate array to use as reference stars 
    :param figax: an existing (Figure,Axes) tuple to use; if given, the background image is not drawn;
                  see :func:`auromat.draw_helpers.loadFigImage`
    :param mask: boolean array of the size of the image, True=don't draw stars here
    :rtype: tuple(Figure,Axes)
    """
    assert axyPath or xylsPath or xylsSources or useWebCatalog
    if axyPath:
        xSource, ySource, fluxSource = auromat.fits.readXy(axyPath, sort=True, retSortField=True)
        sources = list(zip(xSource, ySource, fluxSource))
    
    if xylsSources:
        refSources = xylsSources
    elif xylsPath:
        if wcsPath:
            assert newWcsPathOrHeader
            xRefSource, yRefSource = auromat.fits.recomputeXylsPixelPositions(xylsPath, wcsPath,
                                                                              newWcsPathOrHeader)
        else:
            xRefSource, yRefSource = auromat.fits.readXy(xylsPath)
        refSources = list(zip(xRefSource, yRefSource))
    elif useWebCatalog:
        assert wcsPath, 'useWebCatalog can only be used together with wcsPath'
        if isinstance(wcsPath, string_types):
            header = auromat.fits.readHeader(wcsPath)
        else:
            header = wcsPath
        xRefSource, yRefSource = auromat.fits.getCatalogStars(header, limit=webCatalogLimit)
        refSources = list(zip(xRefSource, yRefSource))
    else:
        refSources = None
    
    if matchPath:
        quad = auromat.fits.readQuadMatch(matchPath)
    
    if figax:
        fig, ax = figax[0], figax[1]
    else:
        fig, ax = loadFigImage(imagePath)
    
    if mask is not None:
        if axyPath:
            sources = [x_y for x_y in sources if not mask[x_y[1],x_y[0]]]
        if refSources:
            # The reference sources can include stars slightly outside the image bounds.
            # We clip such sources to the image bounds for checking against the mask.
            clipped = np.array(refSources)
            clipped[:,0] = np.clip(clipped[:,0], 0, mask.shape[1]-1)
            clipped[:,1] = np.clip(clipped[:,1], 0, mask.shape[0]-1)
            refSources = [ref_clip[0] for ref_clip in zip(refSources, clipped) 
                          if not mask[ref_clip[1][1], ref_clip[1][0]]]           
    
    # use only the brightest extracted sources    
    # (catalog stars are already nicely distributed and not overcrowded)
    if axyPath and maxAxyObjects:
        sources = sources[:maxAxyObjects]
    
    redRadius = int(round(6*scale))
    greenRadius = int(round(4*scale))
    thickness = int(round(2*scale))
    
    if axyPath:
        sources = np.asarray(sources)
        if len(sources) > 0:
            if refSources:
                # disable alpha to make matches more clear
                alpha = np.ones(len(sources))
            else:
                flux = sources[:,2]
                flux_min, flux_max = np.min(flux), np.max(flux)
                # scale flux to [0.5,1] so that it can be used as alpha
                alpha = 0.7*(flux-flux_min)/(flux_max-flux_min) + 0.3
            colors = np.zeros((len(sources), 4), float)
            colors[:,0] = 1 # red
            colors[:,3] = alpha
            _circles(sources[:,0], sources[:,1], redRadius, ax, edgecolor=colors, lw=thickness, facecolor='')
    
    if refSources:
        refSources = np.asarray(refSources)
        if len(refSources) > 0:
            _circles(refSources[:,0], refSources[:,1], greenRadius, ax, edgecolor='lime', lw=thickness, facecolor='')
    
    if matchPath:
        # close the polygon
        quad = np.vstack((quad, [quad[0]]))
        ax.plot(quad[:,0], quad[:,1], c='lime', lw=max(1, thickness//2))
    
    return fig, ax

def drawConstellations(figax, wcsPath, clipPoly=None, labels=True, lineStyles=('-',':'),
                       lineThickness=5, fontsize=70, colors=None, alpha=0.5, padding=10):
    """ Plot constellation patterns.

    This does not provide extraordinarily high precision; it's just a cosmetic display that sketches out
    the appropriate stick figures on the sky.

    Based on IDL constellations.pro by Marshall Perrin.
    Constellation data is taken from Xephem, courtesy of Elwood Downey. 
    
    See :mod:`auromat.coordinates.constellations` module for licence text.

    :see: https://github.com/mperrin/misc_astro/blob/master/constellations.py
    :param figax: figimage, as produced by :func:`auromat.draw_helpers.loadFigImage`
    :param wcsPath: path to FITS file containing WCS header
    :param clipPoly: (n,2) x,y polygon points; restricts drawing to the region of the polygon
    :param label: whether to draw labels
    :param lineStyles: tuple(standard style, dotted style)
    :param lineThickness:
    :param string|list|dict colors: 
             color of lines and labels, can be:
              - single color name
              - list of color names which will be cycled
              - mapping from constellation name to color name
              The default is a list of 9 colours.
    :param alpha: alpha of lines and labels
    :param padding: spacing at line ends in pixels
    """
    header = readHeader(wcsPath)
    w, h = header['IMAGEW'], header['IMAGEH']
    wcs = WCS(header)
    ax = figax[1]
    labelBorder = w*0.05
    
    if isinstance(colors, dict):
        constellationColor = lambda name: colors[name]
    else:
        if not colors:
            colors = itertools.cycle(['white', 'lime', 'red', 'orange', 'cyan', 'magenta', 'lightblue', 'hotpink', 'yellow'])
        elif isinstance(colors, string_types):
            colors = itertools.cycle([colors])
        else:
            colors = itertools.cycle(colors)
        constellationColor = lambda _: next(colors)
    
    if clipPoly is not None:
        clip_path = Polygon(clipPoly, visible=False)
        # have to add the polygon, otherwise strange things happen
        ax.add_patch(clip_path)
        clip_path_contains = clip_path.get_path().contains_point
    else:
        clip_path = None
        clip_path_contains = None
    
    for name, point_list in constellations.data.items():
        points = np.asarray(point_list)
        
        drawtype = points[:,0]
        ra_degrees = points[:,1] / 1800. * 15 # 15 = hours to degrees
        dec_degrees = points[:,2] / 60.
                
        x, y = wcs.wcs_world2pix(ra_degrees, dec_degrees, 0)
        
        # With the TAN WCS projection there is only half of the celestial plane
        # defined, which means that for some RA/Dec values there is no
        # equivalent pixel value. Still, wcs_world2pix returns bogus coordinates
        # in these cases instead of NaN. To identify these bogus results, we
        # convert x,y back to RA,Dec and see if there is a mismatch.
        # TODO fixed in master, rework once astropy 0.4.2 is released
        #   (see https://github.com/astropy/astropy/pull/2965)
        ra_check, dec_check = wcs.wcs_pix2world(x, y, 0)
        if not np.allclose(ra_degrees, ra_check) or not np.allclose(dec_degrees, dec_check):
            continue
                
        if np.all((x<0) | (x>=w) | (y<0) | (y>=h)):
            continue
        
        if clip_path and not any(clip_path_contains([x_,y_]) for x_,y_ in zip(x,y)):
            # this check is necessary as the label logic expects at least one star
            # inside the clipping area
            continue
                
        color_ = constellationColor(name)
        
        for i in range(0, len(points)):
            if drawtype[i] == 0:    
                continue # don't draw lines, just move for type 0
            
            ls = lineStyles[1] if drawtype[i] == 2 else lineStyles[0]
            
            if padding > 0:
                originalStart = np.array([x[i-1], y[i-1]])
                originalEnd = np.array([x[i], y[i]])
                vec = originalEnd - originalStart
                length = np.sqrt((vec*vec).sum(axis=0))
                direction = vec/length
                newStart = originalStart + direction*padding
                newEnd = originalStart + direction*(length-padding)
                
                ax.plot([newStart[0],newEnd[0]], [newStart[1],newEnd[1]], linestyle=ls,
                        clip_path=clip_path, 
                        lw=lineThickness, color=color_, alpha=alpha)
            else:
                ax.plot(x[i-1:i+1], y[i-1:i+1], linestyle=ls,
                        clip_path=clip_path, 
                        lw=lineThickness, color=color_, alpha=alpha)
            
        if labels:
            name = name.replace('_', ' ')
            xl = np.unique(x)
            yl = np.unique(y)
            labelx = np.clip(np.mean(xl), labelBorder, w-labelBorder)
            labely = np.clip(np.mean(yl), labelBorder, h-labelBorder)
            # check if label is outside clipping area
            if clip_path and not clip_path_contains([labelx,labely]):
                # find all stars that are inside the clip path and use only them for positioning
                xyl = filter(lambda x_y: x_y[0]>=0 and x_y[0]<w and x_y[1]>=0 and x_y[1]<h and 
                                         clip_path_contains([x_y[0],x_y[1]]), zip(x,y))
                xl = np.unique([x_ for x_,_ in xyl])
                yl = np.unique([y_ for _,y_ in xyl])
                labelx = np.clip(np.mean(xl), labelBorder, w-labelBorder)
                labely = np.clip(np.mean(yl), labelBorder, h-labelBorder)
            ax.text(labelx, labely, name, fontsize=fontsize, color=color_, 
                    horizontalalignment='center', verticalalignment='center', alpha=alpha,
                    clip_path=clip_path) 

    return figax

def getFixedConstellationColors(colors=None):
    """
    Tries to determine a color for each constellation such that neighboring
    constellations have different colors.
    
    :param colors: set of base colors to use (optional)
    :return: a mapping from constellation names to color strings
    :rtype: dict
    """
    if colors is None:
        colors = {'white', 'lime', 'red', 'orange', 'cyan', 'magenta', 'lightblue', 'hotpink', 'yellow'}
    else:
        colors = set(colors)
        
    def find_neighbors(x, tri):
        """see https://stackoverflow.com/a/17811731/60982"""
        return list(set(indx for simplex in tri.vertices if x in simplex for indx in simplex if indx !=x))
    
    keys = list(constellations.data.keys())
    # we use the "middle" point of each constellation as input for neighbor analysis
    points = np.array([p[len(p)//2][1:] for p in constellations.data.values()])
    tri = Delaunay(points)
    constellationColors = {} # mapping from constellation index to color name
    for i in range(len(points)):
        # determine colors that are already used
        indices = find_neighbors(i, tri) + [i]
        indicesDone = list(filter(lambda i: i in constellationColors, indices))
        colorsDone = map(lambda i: constellationColors[i], indicesDone)
        remainingColors = colors - set(colorsDone)
        remainingIndices = set(indices) - set(indicesDone)
        for i, color in zip(remainingIndices, remainingColors):
            constellationColors[i] = color
    
    constellationColors = {keys[i]: c for i,c in constellationColors.items()}
    return constellationColors 

def drawParallelsAndMeridians(mapping, boundingBox=None, labelLon=None, labelLat=None, 
                              lineThickness=5, color='white', alpha=0.5, figax=None):
    """
    Draws parallels and meridians on top of an unresampled mapping.
    
    :param auromat.mapping.mapping.BaseMapping mapping: the unresampled mapping
    :param boundingBox: the bounding box for which to draw parallels/meridians (optional)                        
    :param labelLon: the longitude along which the latitude labels are drawn (optional)
    :param labelLat: the latitude along which the longitude labels are drawn (optional) 
    :param lineThickness: line width
    :param color: color of the parallels and meridians (matplotlib color spec)
    :param alpha: alpha value of the parallels and meridians
    :param figax: figimage, as produced by :func:`auromat.draw_helpers.loadFigImage`
    """
    # The idea is to resample the pixel coordinates into a regular grid such
    # that we have a lookup table from lat/lon to pixel position.

    if boundingBox is None:
        boundingBox = mapping.boundingBox
    
    # First, we interpolate to a fine-grained resolution.
    # When drawing, we choose certain integer parallels and meridians out of that.
    # The means that most of the interpolated values are not actually used.
    interpolateEvery = 0.2 # degrees
    pxPerDeg = 1/interpolateEvery
    assert modf(pxPerDeg)[0] == 0, '1 mod interpolateEvery = 0 is not satisfied'
    
    h,w = mapping.latsCenter.shape[0:2]
    # data is x,y pixel coordinates, (h,w,2)
    data = np.dstack(np.meshgrid(np.arange(w), np.arange(h))).astype(np.float)
        
    _,_,latGrid,lonGrid,pxcoords = _resample(mapping.latsCenter.filled(np.nan), 
                                             mapping.lonsCenter.filled(np.nan), mapping.altitude,
                                             data, lambda: mapping.outline, boundingBox,
                                             (pxPerDeg,pxPerDeg), boundingBox.containsDiscontinuity, 
                                             boundingBox.containsPole, method='mean')
    
    assert pxcoords.ndim == 3 and pxcoords.shape[2] == 2
    
    drawEvery = 2 # degrees, must be an integer
    
    if figax:
        fig, ax = figax[0], figax[1]
    else:
        fig, ax = loadFigImage(mapping.rgb_unmasked)
        
    # draw parallels
    # we want to draw integer degrees only, therefore we first identify these
    isLatInteger = np.abs(latGrid[:,0]-np.round(latGrid[:,0]).astype(int)) < 1e-10        
    latIndices = isLatInteger.nonzero()[0]
    if len(latIndices) > 0:
        # retain only those latitudes that are on fixed grid, e.g. -180,-175,-170 etc for drawEvery=5
        gridCheck = np.array([abs(latGrid[i,0]/drawEvery) for i in latIndices])
        latIndices = latIndices[np.abs(gridCheck-np.round(gridCheck).astype(int)) < 1e-10]
    
    for i in latIndices:
        ax.plot(pxcoords[i,:,0], pxcoords[i,:,1], 
                linestyle='--', lw=lineThickness, color=color, alpha=alpha)

    # draw meridians
    isLonInteger = np.abs(lonGrid[0,:]-np.round(lonGrid[0,:]).astype(int)) < 1e-10
    lonIndices = isLonInteger.nonzero()[0]
    if len(lonIndices) > 0:
        # retain only those longitudes that are on fixed grid, e.g. -90,-85,-80 etc for drawEvery=5
        gridCheck = np.array([abs(lonGrid[0,i]/drawEvery) for i in lonIndices])
        lonIndices = lonIndices[np.abs(gridCheck-np.round(gridCheck).astype(int)) < 1e-10]
    
    for i in lonIndices:
        ax.plot(pxcoords[:,i,0], pxcoords[:,i,1], 
                linestyle='--', lw=lineThickness, color=color, alpha=alpha)
    
    # draw labels
    # we find the parallel and meridian that is closest to the mapping centroid
    # and use that for labelling
    # to get a point which is more in the center, we first mask the far away areas
    if labelLat is None or labelLon is None:
        mapping = mapping.maskedByElevation(10)
    if labelLat is None:
        labelLat = mapping.centroid.lat
    if labelLon is None:
        labelLon = mapping.centroid.lon
    
    axtext = partial(ax.text, fontsize=70, color=color, alpha=alpha,
                     horizontalalignment='center', verticalalignment='center')

    # the longitude where latitude labels are drawn
    # we draw labels between two longitudes
    if len(lonIndices) > 0:
        if boundingBox.containsDiscontinuity:
            angle = 180
            lonsRot = Angle((lonGrid[0,lonIndices] + angle) * u.deg).wrap_at(angle * u.deg).degree
            labelLonRot = Angle((labelLon + angle) * u.deg).wrap_at(angle * u.deg).degree
            distances = lonsRot-labelLonRot
        else:
            distances = lonGrid[0,lonIndices]-labelLon
        labelLonLeftIdx = np.argmin(np.abs(distances))
        labelLonRightIdx = labelLonLeftIdx + 1 if labelLonLeftIdx + 1 < len(lonIndices) else labelLonLeftIdx
        labelLonGridMiddleIdx = (lonIndices[labelLonLeftIdx] + lonIndices[labelLonRightIdx])//2
        
        for i in filter(lambda i: not np.isnan(pxcoords[i,labelLonGridMiddleIdx,0]), latIndices):
            lat = latGrid[i,0]
            if lat>0:
                direction = 'N'
            elif lat<0:
                direction = 'S'
            else:
                direction = ''
            latStr = u'%i\N{DEGREE SIGN}%s' % (np.round(abs(lat)), direction)
            axtext(pxcoords[i,labelLonGridMiddleIdx,0], pxcoords[i,labelLonGridMiddleIdx,1], latStr)
    
    # the latitude where longitude labels are drawn
    if len(latIndices) > 0:
        labelLatLeftIdx = np.argmin(np.abs(latGrid[latIndices,0]-labelLat))
        labelLatRightIdx = labelLatLeftIdx + 1 if labelLatLeftIdx + 1 < len(latIndices) else labelLatLeftIdx
        labelLatGridMiddleIdx = (latIndices[labelLatLeftIdx] + latIndices[labelLatRightIdx])//2
        
        for i in filter(lambda i: not np.isnan(pxcoords[labelLatGridMiddleIdx,i,0]), lonIndices):
            lon = lonGrid[0,i]
            if lon>0:
                direction = 'E'
            elif lon<0:
                direction = 'W'
            else:
                direction = ''
            lonStr = u'%i\N{DEGREE SIGN}%s' % (np.round(abs(lon)), direction)
            axtext(pxcoords[labelLatGridMiddleIdx,i,0], pxcoords[labelLatGridMiddleIdx,i,1], lonStr)
    
    return fig, ax      

def drawDate(figax, mapping, color='white'):
    """
    Writes the mapping date in the top center of the image.
    
    :param figax: figimage, as produced by :func:`draw_helpers.loadFigImage`
    """
    ax = figax[1]
    fontsize = ax.get_xlim()[1] * 0.016
    ax.text(0.5, 0.98, mapping.photoTime.strftime(DatePlotTitleFormat),
            fontsize=fontsize, color=color,
            horizontalalignment='center', verticalalignment='top',
            transform=ax.transAxes)

def _readCorrs(corrPaths):
    """
    
    :param str|iterable corrPaths: paths of .corr files written by astrometry.net
    """
    if isinstance(corrPaths, str):
        corrPaths = [corrPaths]
        
    xFieldArr, yFieldArr = [], []
    xIndexArr, yIndexArr = [], [] 
    for corrPath in corrPaths:
        xField, yField, xIndex, yIndex = auromat.fits.readCorr(corrPath)
        xFieldArr.append(xField)
        yFieldArr.append(yField)
        xIndexArr.append(xIndex)
        yIndexArr.append(yIndex)
    
    return xFieldArr, yFieldArr, xIndexArr, yIndexArr

def drawCorrPlot(corrPaths,
                 title=r'Distances between corresponding stars: $\mu={:0.2f}$, $\sigma={:0.2f}$'):
    """
    Draws a histogram of distances between corresponding stars from an astrometry.net .corr file.
    
    :param corrPaths: path to .corr file or list of paths (histograms are merged)
    :rtype: tuple(Figure,Axes)
    """
    xFieldArr, yFieldArr, xIndexArr, yIndexArr = _readCorrs(corrPaths)
    
    xField = np.concatenate(xFieldArr)
    yField = np.concatenate(yFieldArr)
    xIndex = np.concatenate(xIndexArr)
    yIndex = np.concatenate(yIndexArr)
    
    field = np.transpose([xField, yField])
    index = np.transpose([xIndex, yIndex])
    dist = auromat.utils.vectorLengths(field - index)
    mean, std = np.mean(dist), np.std(dist)
    
    colors = _getMplColors()
    fig,ax = plt.subplots()
    _setMplColors(fig, ax, colors)
    ax.set_xlabel('Distance (pixels)')
    ax.set_ylabel('Count')
    ax.set_title(title.format(mean, std))
    ax.hist(dist, bins=100)
    
    return fig, ax
    
def drawCorrSeqPlot(corrPaths, x=None, xlabel='Frame', retline=False):
    """
    Draws the mean distance of corresponding stars from astrometry.net's .corr files
    for each given file.
    
    :param corrPaths: list of .corr paths
    :param x: x axis values
    :rtype: tuple(Figure,Axes[,Line])
    """
    if x is None:
        x = list(range(len(corrPaths)))
    assert len(x) == len(corrPaths)
    xFieldArr, yFieldArr, xIndexArr, yIndexArr = _readCorrs(corrPaths)
    
    means = []
    stds = []
    for xField, yField, xIndex, yIndex in zip(xFieldArr, yFieldArr, xIndexArr, yIndexArr):    
        field = np.transpose([xField, yField])
        index = np.transpose([xIndex, yIndex])
        dist = auromat.utils.vectorLengths(field - index)
        means.append(np.mean(dist))
        stds.append(np.std(dist))
        
    fig,ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Mean distance (pixels)')
    ax.set_title(r'Distances between corresponding stars')
    plotline, _, _ = ax.errorbar(x, means, stds, linestyle='None', marker='s')
    ax.set_xlim(x[0]-1, x[-1]+1)
    
    if retline:
        return fig, ax, plotline
    else:
        return fig, ax

def _getHeaders(wcsPathsOrHeadersOrMappings):
    wcsPathsOrHeadersOrMappings = list(wcsPathsOrHeadersOrMappings)
    if not wcsPathsOrHeadersOrMappings:
        return []
    if isinstance(wcsPathsOrHeadersOrMappings[0], string_types):
        headers = list(map(auromat.fits.readHeader, wcsPathsOrHeadersOrMappings))
    elif hasattr(wcsPathsOrHeadersOrMappings[0], 'wcsHeader'):
        headers = [m.wcsHeader for m in wcsPathsOrHeadersOrMappings]
    else:
        headers = wcsPathsOrHeadersOrMappings
    return headers

def drawAstrometryPixelScales(wcsHeaders, x=None, 
                              xlabel='Frame - 1st Frame', ylabel='Pixel scale (arcsec/px)',
                              title='Celestial Sphere Pixel Scale ($\sqrt{CD_{11}^2 + CD_{21}^2}$)',
                              retscatter=False, **kw):
    """
    Draws a scatter plot of the celestial pixel scales of the given astrometric solutions.
    
    :param wcsHeaders: iterable of either WCS header objects or paths to .wcs files or astrometry-based mappings
    :param x: values to use for the x axis; if not given, then x simply increase from 0 to len(wcsHeaders)
    :param bool retscatter: whether to return the `PathCollection` of scatter points
    :rtype: tuple(Figure,Axes[,PathCollection])
    """
    wcsHeaders = list(wcsHeaders)
    if x is None:
        x = list(range(len(wcsHeaders)))
    assert len(x) == len(wcsHeaders)
    
    wcsHeaders = _getHeaders(wcsHeaders)
    pixelScales = list(map(auromat.fits.getPixelScale, wcsHeaders))
    arcsec = (pixelScales*u.deg).to(u.arcsec).value
    
    fig, ax, scatter = _drawScatterPlot(x, arcsec,
                                        xlabel=xlabel, 
                                        ylabel=ylabel, 
                                        title=title,
                                        s=10, retscatter=True, **kw)

    mean = np.mean(arcsec)
    median = np.median(arcsec)
    stddev = np.std(arcsec)
        
    # axhline is not supported by mpld3 yet
    # but we can emulate it with hlines
    #meanline = ax.axhline(mean, color='red')
    #medianline = ax.axhline(median, color='blue')
    meanline = ax.hlines([mean], 0, x[-1]*2, color='red')
    medianline = ax.hlines([median], 0, x[-1]*2, color='blue')
    
    ax.legend([meanline, medianline], 
              ['mean, $\sigma={:0.4f}$'.format(stddev), 'median'],
              loc='upper right', frameon=False)
    
    ax.set_xlim(0, len(wcsHeaders))
    
    if retscatter:
        return fig, ax, scatter
    else:
        return fig, ax
    
def drawAstrometryRotationAngles(wcsHeaders, x=None, 
                                 xlabel='Time', ylabel='Rotation angle (deg)',
                                 title=r'Rotation Angle ($\operatorname{atan}(CD_{21},CD_{11})$)',
                                 retline=False, **kw):
    """
    Draws a line plot of the celestial rotation angles of the given astrometric solutions.
    
    :param wcsHeaders: iterable of either WCS header objects or paths to .wcs files or astrometry-based mappings
    :param x: values to use for the x axis; if not given, then x will be the time axis
    :param bool retline: whether to return the matplotlib `Line` object
    :param kw: additional keywords are handed over to :func:`drawLinePlot`
    :rtype: tuple(Figure,Axes[,Line])
    """
    wcsHeaders = _getHeaders(wcsHeaders)
    
    if x is None:
        x = list(map(auromat.fits.getPhotoTime, wcsHeaders))
    assert len(x) == len(wcsHeaders)
    
    angles = np.asarray(list(map(auromat.fits.getRotationAngle, wcsHeaders)))
    containsDiscontinuity = np.max(angles) - np.min(angles) > 100
    if containsDiscontinuity:
        angles = Angle((angles + 180) * u.deg).wrap_at(180 * u.deg).degree
    
    fig, ax, line = drawLinePlot(x, angles,
                                 xlabel=xlabel, ylabel=ylabel, title=title,
                                 retline=True, **kw)
    
    if containsDiscontinuity:
        def fixAxis(x,pos):
            deg = Angle((x + 180) * u.deg).wrap_at(180 * u.deg).degree
            s = '{:g}'.format(deg) # removes trailing .0
            return s.replace('-', u'\u2212') # uses unicode minus (longer)
            # Note: this should rather make use of ScalarFormatter but I couldn't figure it out
        ax.yaxis.set_major_formatter(FuncFormatter(fixAxis))
    
    if retline:
        return fig, ax, line
    else:
        return fig, ax
    
def drawCD11CD21(wcsHeaders, xlabel='$CD_{11}$', ylabel='$CD_{21}$',
                 title='WCS Transformation Matrix Values',
                 retline=False, **kw):
    """
    Draws a line plot of the CD11 and CD21 values of the given astrometric solutions
    together with a reference circle derived from a fixed (median) celestial pixel scale.
    
    :param wcsHeaders: iterable of either WCS header objects or paths to .wcs files or astrometry-based mappings
    :param bool retline: whether to return the matplotlib `Line` object
    :param kw: additional keywords are handed over to :func:`drawLinePlot`
    :rtype: tuple(Figure,Axes[,Line])
    """
    wcsHeaders = _getHeaders(wcsHeaders)
    cd11 = [h['CD1_1'] for h in wcsHeaders]
    cd21 = [h['CD2_1'] for h in wcsHeaders]
    
    pixelScales = list(map(auromat.fits.getPixelScale, wcsHeaders))
    pixelScaleMedian = np.median(pixelScales)
    
    fig, ax, line = drawLinePlot(cd11, cd21,
                                 xlabel=xlabel, ylabel=ylabel, 
                                 title=title, retline=True, **kw)
    
    circle = plt.Circle((0,0), pixelScaleMedian, fill=False)
    ax.add_patch(circle)
    
    pixelScaleMedianArcSec = (pixelScaleMedian*u.deg).to(u.arcsec).value
    
    ax.legend([circle], 
              ['{0:0.2f}'.format(pixelScaleMedianArcSec) + ' arcsec/px (median)'],
              loc='upper right', frameon=False)
    
    if retline:
        return fig, ax, line
    else:
        return fig, ax
    
def drawRaDec(wcsHeaders,
              xlabel='Right ascension (deg)', ylabel='Declination (deg)',
              title='Equatorial Coordinates of Image Centers', 
              retline=False, **kw):
    """
    Draws a line plot of the right ascension and declination values of the given astrometric solutions.
    
    :note: Assumes that CRPIX1,CRPIX2 are the image center.
    :param wcsHeaders: iterable of either WCS header objects or paths to .wcs files or astrometry-based mappings
    :param bool retline: whether to return the matplotlib `Line` object
    :param kw: additional keywords are handed over to :func:`drawLinePlot`
    :rtype: tuple(Figure,Axes[,Line])
    """
    wcsHeaders = _getHeaders(wcsHeaders)
    ra = [h['CRVAL1'] for h in wcsHeaders]
    dec = [h['CRVAL2'] for h in wcsHeaders]
    
    return drawLinePlot(ra, dec,
                        xlabel=xlabel, ylabel=ylabel, 
                        title=title, retline=retline, **kw)
    
def drawRightAscension(wcsHeaders, x=None,
                       xlabel='Time', ylabel='Right ascension (deg)',
                       title='Right Ascension of Image Centers', retline=False, **kw):
    """
    Draws a line plot of the right ascension values of the given astrometric solutions.
    
    :note: Assumes that CRPIX1,CRPIX2 are the image center.
    :param wcsHeaders: iterable of either WCS header objects or paths to .wcs files or astrometry-based mappings
    :param x: values to use for the x axis; if not given, then x will be the time axis
    :param bool retline: whether to return the matplotlib `Line` object
    :param kw: additional keywords are handed over to :func:`drawLinePlot`
    :rtype: tuple(Figure,Axes[,Line])
    """
    wcsHeaders = _getHeaders(wcsHeaders)
    
    if x is None:
        x = map(auromat.fits.getPhotoTime, wcsHeaders)
    assert len(x) == len(wcsHeaders)
    
    ra = [h['CRVAL1'] for h in wcsHeaders]
    return drawLinePlot(x, ra,
                        xlabel=xlabel, ylabel=ylabel, 
                        title=title, retline=True, **kw)
    
def drawDeclination(wcsHeaders, x=None,
                    xlabel='Time', ylabel='Declination (deg)',
                    title='Declination of Image Centers', retline=False, **kw):
    """
    Draws a line plot of the declination values of the given astrometric solutions.
    
    :note: Assumes that CRPIX1,CRPIX2 are the image center.
    :param wcsHeaders: iterable of either WCS header objects or paths to .wcs files or astrometry-based mappings
    :param x: values to use for the x axis; if not given, then x will be the time axis
    :param bool retline: whether to return the matplotlib `Line` object
    :param kw: additional keywords are handed over to :func:`drawLinePlot`
    :rtype: tuple(Figure,Axes[,Line])
    """  
    wcsHeaders = _getHeaders(wcsHeaders)
    
    if x is None:
        x = map(auromat.fits.getPhotoTime, wcsHeaders)
    assert len(x) == len(wcsHeaders)
    
    dec = [h['CRVAL2'] for h in wcsHeaders]
    
    return drawLinePlot(x, dec,
                        xlabel=xlabel, ylabel=ylabel, 
                        title=title, retline=True, **kw)

def drawCameraFootpoints(mappings,
                         xlabel='Longitude (deg)', ylabel='Latitude (deg)',
                         title='Camera Footpoints', 
                         retline=False, **kw):
    """
    Draws a line plot of the camera footpoints of the given mappings.
    
    :param mapping: iterable of mappings
    :param bool retline: whether to return the matplotlib `Line` object
    :param kw: additional keywords are handed over to :func:`drawLinePlot`
    :rtype: tuple(Figure,Axes[,Line])
    """
    footpoints = [m.cameraFootpoint for m in mappings]
    lat = [f.lat for f in footpoints]
    lon = [f.lon for f in footpoints]
    
    return drawLinePlot(lon, lat,
                        xlabel=xlabel, ylabel=ylabel, 
                        title=title, retline=retline, **kw)

def setColors(figax, bgcolor='white', transparent=False):
    """
    Changes colors of all plot elements to a given color scheme.
    
    :param figax: `Figure` object, or array-like with `Figure`,`Axes` as first two elements 
    :param str bgcolor: background color of the plot, 'white' or 'black'
    :param bool transparent: whether the background outside the plot bounds should be transparent
    """
    try:
        fig, ax = figax[0], figax[1]
    except:
        fig = figax
        ax = fig.gca()
    _setMplColors(fig, ax, _getMplColors(bgcolor, transparent))

def saveFig(outputFile, figax, bgcolor='white', transparent=False, widthPx=None, heightPx=None, dpi=None, format_=None,
            dontClose=False):
    """
    Saves a matplotlib `Figure` to a path on disk or a `File`-like object.
    
    :note: Only one of `widthPx`, `heightPx`, and `dpi` must be defined. If none is given, then default
           values from matplotlib are used.
    :param outputFile: path to file, or `File`-like object
    :param figax: `Figure` object, or array-like with `Figure`,`Axes` as first two elements 
    :param str bgcolor: background color of the plot, 'white' or 'black'
    :param bool transparent: whether the background outside the plot bounds should be transparent
    :param int widthPx: width in pixels
    :param int heightPx: height in pixels
    :param number dpi: dpi in inches
    :param str format_: file format, e.g. 'jpg', 'png', 'svg'; will be derived from extension if `outputFile`
                        is a path name
    :param bool dontClose: whether to close the `Figure` or leave it open for further processing;
                           default is to close it; use `plt.close(fig)` for manual closing
    """
    try:
        fig, ax = figax[0], figax[1]
    except:
        fig = figax
        ax = fig.gca()
    setColors((fig,ax), bgcolor, transparent)
    return _saveFig(outputFile, fig,
                    widthPx=widthPx, heightPx=heightPx, dpi=dpi,
                    format_=format_, dontClose=dontClose)
    
if __name__ == '__main__':
    fig, ax = plt.subplots()
    colors = getFixedConstellationColors({'lime', 'red', 'orange', 'cyan', 'magenta', 'lightblue', 'hotpink', 'yellow'})
    for name, points in constellations.data.items():
        points = np.asarray(points)
        drawtype = points[:,0]
        ra_degrees = points[:,1] / 1800. * 15 # 15 = hours to degrees
        dec_degrees = points[:,2] / 60.
        for i in range(0, len(points)):
            if drawtype[i] == 0:    
                continue
            ax.plot(ra_degrees[i-1:i+1], dec_degrees[i-1:i+1], linestyle='-',
                    color=colors[name])
    
    saveFig('test.svg',(fig,ax))
        
    print(getFixedConstellationColors())