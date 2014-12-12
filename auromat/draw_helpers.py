# Copyright European Space Agency, 2013

"""
Internal module used by the .draw module.
"""

from __future__ import division, absolute_import, print_function

from six import integer_types, string_types
import warnings
import datetime
from collections import namedtuple

import numpy as np
import numpy.ma as ma
from numpy.core.umath_tests import inner1d # vector-wise dot product

from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import brewer2mpl
from matplotlib.collections import PatchCollection
blue_red = brewer2mpl.get_map('RdBu', 'Diverging', 11, reverse=True).mpl_colormap

from auromat.util.image import image2mpl, loadImage
from auromat.mapping.mapping import BaseMapping, MappingCollection,\
    convertMappingToSM

class ColorMode:
    matplotlib = 'matplotlib'

def createPolygonsAndColors(latDeg, lonDeg, rgb, colorMode=None):
    """
    Returns polygons (in lat/lon coords) and a color for each polygon.
    
    :param latDeg: latitude for each pixel corner (h+1,w+1)
    :param lonDeg: longitude for each pixel corner (h+1,w+1)
    :param rgb: RGB array of (h,w,3) shape
    :param colorMode: 'matplotlib' normalizes colors to [0,1]
    :rtype: verts of shape (h*w,4,2), colors of shape (h*w,3)
    """
    latLonDeg = ma.dstack((latDeg,lonDeg))
    
    # adapted from matplotlib.collections.QuadMesh.convert_mesh_to_paths
    verts = ma.concatenate((
                latLonDeg[0:-1, 0:-1],
                latLonDeg[0:-1, 1:  ],
                latLonDeg[1:  , 1:  ],
                latLonDeg[1:  , 0:-1],
                #latLonDeg[0:-1, 0:-1] # matplotlib automatically closes the polygon
                ), axis=2)
    verts = verts.reshape(rgb.shape[0] * rgb.shape[1], 4, 2)
    
    if colorMode == ColorMode.matplotlib:
        rgb = image2mpl(rgb)
    
    colors = rgb.reshape(-1,3)
    
    return verts, colors
    
def filterNanPolygons(verts, colors):
    # when color is defined, then all its corners are also defined
    # according to BaseMapping guarantees, therefore we filter using color
    if ma.isMaskedArray(colors):
        hasNans = ma.getmaskarray(colors)[:,0]
    else:
        hasNans = np.isnan(colors[:,0])
    verts = verts[~hasNans]
    colors = colors[~hasNans]
    # we don't need masked arrays anymore after all invalid values are gone
    if ma.isMaskedArray(verts):
        verts = verts.data
    if ma.isMaskedArray(colors):
        colors = colors.data
    #assert np.all(~np.isnan(verts))
    #assert np.all(~np.isnan(colors))
    return verts, colors

def generatePolygonsFromMapping(mapping, colorMode=None, coordsFn=None):
    """
    :param colorMode: 'matplotlib' normalizes colors to [0,1]
    """
    coordsFn = _coordsFn(coordsFn)
    lats, lons = coordsFn(mapping)
    rgb = mapping.rgb
    verts, colors = createPolygonsAndColors(lats, lons, rgb, colorMode)
    verts, colors = filterNanPolygons(verts, colors)
    return verts, colors
    
def overlapPolygons(verts):
    """
    See https://github.com/matplotlib/matplotlib/issues/2823
    
    NOTE: the bigger the polygons, the higher the resulting error
    """
    # TODO overlapping leads to visible borders, esp. in resampled data
    # FIXME this will cause trouble in 180deg discontinuity area with fixDiscontinuityPolys
    a = 1.0 # the factor should be much lower but matplotlib seems to need it...
    verts[:,1,:] = verts[:,1,:] + a*(verts[:,1,:]-verts[:,0,:])
    verts[:,2,:] = verts[:,2,:] + a*(verts[:,2,:]-verts[:,3,:])
    verts[:,2,:] = verts[:,2,:] + a*(verts[:,2,:]-verts[:,1,:])
    verts[:,3,:] = verts[:,3,:] + a*(verts[:,3,:]-verts[:,0,:])
    return verts

def fixDiscontinuityPolys(verts, colors):
    """
    Splits polygons which cross the 180deg discontinuity into two parts.
    
    NOTE: The vertices must lie on a regular lat/lon grid!
    
    Currently not used, see comment on _drawPolygons.
     
    :param verts: (n,4,2) in degrees (:,:,0 = lat)
    :param colors: (n,3)
    """
    length = np.abs(verts[:,3,1] - verts[:,0,1])
    polysWithDiscont = verts[length>180]
    print(polysWithDiscont)
    return verts, colors

def _coordsFn(coordsFn):
    if coordsFn is None:
        coordsFn = lambda mapping: (mapping.lats, mapping.lons)
    return coordsFn

def _generatePolygonsFromMappingOrCollection(mappings, coordsFn=None):
    """
    replicates THEMIS drawing (higher elevation pixels have priority when mappings
    overlap) for MappingCollection's which have mayOverlap==True
        
    :param mappings: BaseMapping or MappingCollection
    :rtype: tuple (vertsArr, colorsArr)
    """
    coordsFn = _coordsFn(coordsFn)
        
    if isinstance(mappings, BaseMapping):
        mapping = mappings
        verts, colors = generatePolygonsFromMapping(mapping, ColorMode.matplotlib, coordsFn=coordsFn)
        vertsArr, colorsArr = [verts], [colors]
    
    elif isinstance(mappings, MappingCollection):
        vertsArr = []
        colorsArr = []
        elevationArr = []
        for mapping in mappings.mappings:
            if mappings.mayOverlap:
                # don't apply filtering yet, need to relate to elevation
                lats, lons = coordsFn(mapping)
                verts, colors = createPolygonsAndColors(lats, lons, mapping.rgb, ColorMode.matplotlib)
            else:
                verts, colors = generatePolygonsFromMapping(mapping, ColorMode.matplotlib, coordsFn=coordsFn)
            vertsArr.append(verts)
            colorsArr.append(colors)
            if mappings.mayOverlap:
                el = mapping.elevation.ravel()
                assert len(el) == len(verts) == len(colors)
                elevationArr.append(el)
        
        if mappings.mayOverlap and len(mappings)>0:
            # The polygons are sorted according to increasing elevation. If they
            # are drawn in the same order then the outermost parts of an ASI image will
            # be overdrawn by higher-elevation polygons of overlapping ASIs. The reasoning
            # is that low-elevation pixels are less important (due to the projection error)
            # than high-elevation pixels.
            verts = np.concatenate(vertsArr)
            colors = np.concatenate(colorsArr)
            elevation = np.concatenate(elevationArr)

            elSorted = np.argsort(elevation)
            verts, colors = filterNanPolygons(verts[elSorted], colors[elSorted])
            
            vertsArr = [verts]
            colorsArr = [colors]
    else:
        raise ValueError
    return vertsArr, colorsArr

def _addFigureBottomTitle(fig, title):
    fig.text(0.5, 0.02, title, horizontalalignment='center')
   
def _convertMappingsToSM(mappings):
    # for drawing, we pretend that Mlat/MLT are geographic coordinates
    # that way, we can reuse the regular drawing logic (esp. bounding box calculation etc.)    
    # convertMapping and convertMappingOrCollection map MLat/MLT to lat/lon coordinates   
    def convertMappingOrCollection(mapping):
        if isinstance(mapping, BaseMapping):
            newMapping = convertMappingToSM(mapping)
        elif isinstance(mapping, MappingCollection):
            mappings_ = list(map(convertMappingToSM, mapping.mappings))
            newMapping = MappingCollection(mappings_, mapping.identifier, mapping.mayOverlap)
        else:
            raise ValueError
        return newMapping
    
    if isinstance(mappings, list):
        newMappings = []
        for mapping in mappings:
            newMapping = convertMappingOrCollection(mapping)    
            newMappings.append(newMapping)
    else:
        newMappings = convertMappingOrCollection(mappings)
    
    return newMappings
    
def _formatMLT(smlon, format_='%H:%M:%S'):
    eps = 1e-10
    if smlon > 180:
        smlon = smlon-360
    mlt = smlon*(24/360) + 12
    if abs(mlt - 24) < eps:
        mlt = 0
    hours = int(mlt)
    minutes = int((mlt - hours)*60)
    seconds = int(((mlt - hours)*60 - minutes)*60)
    micros = int((((mlt - hours)*60 - minutes)*60 - seconds)*1e6)
    # round 02:59:59.999 up
    if micros > 999900:
        # this should be written more nicely..
        if seconds == 59:
            seconds = 0
            if minutes == 59:
                minutes = 0
                if hours == 23:
                    hours = 0
                else:
                    hours += 1
            else:
                minutes += 1
        else:
            seconds += 1
        micros = 0
    timestr = datetime.time(hours, minutes, seconds, micros).strftime(format_)
    return timestr

def _circles(x, y, s, ax, **kwargs):
    """
    Draw circles with given radius.

    :param x,y: coordinates of circles
    :type x,y: scalar or array_like, shape (n,)
    :param s: radius of circles
    :type s: scalar or array_like, shape (n,)
    :param Axes ax: axes to draw onto
    :param kwargs: additional keywords handed over to `~matplotlib.collections.Collection`
        eg. alpha, edgecolors, facecolors, linewidths, linestyles, norm, cmap
    :rtype: matplotlib.collections.PathCollection
    """
    if isinstance(x, integer_types) or isinstance(x, float):
        patches = [Circle((x, y), s),]
    elif isinstance(s, integer_types) or isinstance(s, float):
        patches = [Circle((x_,y_), s) for x_,y_ in zip(x,y)]
    else:
        patches = [Circle((x_,y_), s_) for x_,y_,s_ in zip(x,y,s)]
    collection = PatchCollection(patches, **kwargs)

    ax.add_collection(collection)
    return collection

def ensureContinuousPath(points):
    """
    Reorders a maximum of two logical segments in the given path whose points must
    be at most one (also diagonal) pixel apart.
    
    For example, the first segment goes from the middle to the right, and the
    second from the left to the middle. This function detects and swaps the two
    segments such that one continuous segment results.
    
    :param points: shape (n,2)
    :rtype: points with segments swapped, or original array if no swap necessary
    """
    vecs = points[1:] - points[:-1]
    lenSq = inner1d(vecs, vecs)
    jumps = lenSq > 2
    if np.any(jumps):
        jumpIdx = np.argmax(jumps)
        return np.concatenate((points[jumpIdx+1:], points[:jumpIdx+1]))
    else:
        return points

def _saveFig(outputFile, fig, widthPx=None, heightPx=None, dpi=None, format_=None, dontClose=False):
    assert not(widthPx and heightPx), 'Either specify widthPx OR heightPx, not both'
    assert not((widthPx or heightPx) and dpi), 'Either specify dpi OR widthPx OR heightPx'
    
    if widthPx:
        dpi = widthPx / fig.get_figwidth()
    elif heightPx:
        dpi = heightPx / fig.get_figheight()
    else:
        dpi = fig.get_dpi()
        
    fig.savefig(outputFile, dpi=dpi, facecolor=fig.get_facecolor(), format=format_)
        
    if not dontClose:
        plt.close(fig)

def loadFigImage(im):
    """
    Return a matplotlib `Figure` with the given raster image spanning the plot extents
    and with data coordinates equal to pixel coordinates. All axis and labels are
    hidden.
    
    :param im: either path to image file, or RGB image array in uint8 or uint16 type
    :rtype: tuple(Figure,Axes)
    """
    if isinstance(im, string_types):
        im = loadImage(im)
    im = image2mpl(im)
    h,w = im.shape[0], im.shape[1]
    dpi = 80
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.invert_yaxis()
    ax.set_axis_off()
    fig.add_axes(ax)
    if im.ndim == 2:
        fig.figimage(im, cmap=cm.gray)
    else:
        fig.figimage(im)
    fig._dontSetColors = True

    return fig, ax

def _setMplColors(fig, ax, colors):
    if fig and hasattr(fig, '_dontSetColors'):
        return        
    if fig:
        fig.patch.set_facecolor(colors.facecolor)
    ax.set_axis_bgcolor(colors.facecolor)
    ax.patch.set_facecolor(colors.facecolor)
    
    ax.spines['bottom'].set_color(colors.textcolor)
    ax.spines['top'].set_color(colors.textcolor)
    ax.spines['left'].set_color(colors.textcolor)
    ax.spines['right'].set_color(colors.textcolor)
    for i in ax.get_xticklabels(): i.set_color(colors.textcolor)
    for i in ax.get_yticklabels(): i.set_color(colors.textcolor)
    for t in ax.xaxis.get_ticklines(): t.set_color(colors.textcolor)
    for t in ax.yaxis.get_ticklines(): t.set_color(colors.textcolor)
    for t in ax.xaxis.get_minorticklines(): t.set_color(colors.textcolor)
    for t in ax.yaxis.get_minorticklines(): t.set_color(colors.textcolor)
    ax.xaxis.label.set_color(colors.textcolor)
    ax.yaxis.label.set_color(colors.textcolor)
    for t in fig.texts: t.set_color(colors.textcolor)
    for t in ax.findobj(Text): t.set_color(colors.textcolor)
    
    if fig and hasattr(fig, '_applyMeridiansParallelsColors'):
        for t in ax.findobj(Line2D): t.set_color(colors.textcolor)
    
    # circles of polar maps
    if fig and hasattr(fig, '_applyRoundPolarMapColors'):
        for c in ax.findobj(Circle):
            c.set_edgecolor(colors.textcolor)
        
MplColors = namedtuple('MplColors', ['facecolor', 'textcolor'])

def _getMplColors(bgcolor='white', transparent=False):
    textcolor = 'white' if bgcolor == 'black' else 'black'
    facecolor = 'none' if transparent else bgcolor
    return MplColors(facecolor, textcolor)
