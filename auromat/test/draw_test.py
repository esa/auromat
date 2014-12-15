# Copyright European Space Agency, 2013

from __future__ import print_function

import os
from six.moves import map
import unittest
from numpy.testing.utils import assert_equal
from nose.plugins.attrib import attr

from auromat.mapping.spacecraft import getMapping
from auromat.resample import resample
from auromat.mapping import miracle
from auromat.mapping.mapping import BoundingBox, GenericMapping
from auromat.utils import outline
from auromat.util import coroutine
import numpy as np

try:
    from auromat.draw import drawMLatMLTPolar, drawStereographic,\
        drawStereographicMLatMLT, drawHeatmap, drawHeatmaps,\
        drawHorizon, drawIndxPlot, saveFig, drawConstellations,\
        drawParallelsAndMeridians, drawScanLinesCo,\
        drawScanLinesMLatMLTCo
    from mpl_toolkits.basemap import Basemap
    import matplotlib.pyplot as plt
    from auromat.draw_helpers import loadFigImage
except ImportError as e:
    print(repr(e))
    # import is optional so that the test module import doesn't fail if
    # the tests are not actually run and matplotlib is not installed

@attr('slow')
class Test(unittest.TestCase):
    def _testHorizonImage(self):
        m = _getMappingNorth()
        saveFig('horizon.jpg', drawHorizon(m))
        
    def _testColors(self):
        m = _getMappingNorth()
        m = resample(m, arcsecPerPx=100, method='mean')
        saveFig('test_white.png', drawStereographic(m, bottomTitle='foo'), bgcolor='white')
        saveFig('test_black.png', drawStereographic(m, bottomTitle='foo'), bgcolor='black')
        saveFig('test_polar_white.png', drawMLatMLTPolar(m), bgcolor='white')
        saveFig('test_polar_black.png', drawMLatMLTPolar(m), bgcolor='black')
               
    def testIndxPlot(self):
        imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
        wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
        figax = drawIndxPlot(imagePath, wcsPath=wcsPath, useWebCatalog=True, webCatalogLimit=200)
        saveFig('test_indx.png', figax)
        
    def _testConstellationPlot(self):
        imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
        wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
        
        m = getMapping(imagePath, wcsPath)
        figax = loadFigImage(imagePath)
        drawConstellations(figax, wcsPath, clipPoly=outline(~m.intersectsEarth))
        saveFig('test_constellations.jpg', figax)
    
    def testParallelsMeridiansPlot(self):
        imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
        wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
        m = getMapping(imagePath, wcsPath, altitude=0, fastCenterCalculation=True)
        figax = loadFigImage(imagePath)
        drawParallelsAndMeridians(m, figax=figax)
        drawConstellations(figax, wcsPath, clipPoly=outline(~m.intersectsEarth))
        saveFig('test_parallelsmeridians.jpg', figax)
    
    def _testParallelsMeridiansPlotOptimized(self):
        imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
        wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
        # As we precalculated the bounding box and the lat/lon label position
        # we only need to access the 'latCenter', 'lonCenter' and 'intersectsEarth'
        # attributes of the spacecraft mapping. This means that we don't need
        # any sanitization applied on the corner coordinate arrays ('lats','lons').
        # For this reason, and because we don't use masking, we set nosanitize=True
        # and can cut the required time in half. 
        m = getMapping(imagePath, wcsPath, altitude=0, fastCenterCalculation=False, nosanitize=True)
        
        # the following values must be precalculated to use the optimization
        # the bounding box at altitude=0 without any masking
        bb = BoundingBox(latSouth=49.3401757697, lonWest=-116.368770925, latNorth=64.288454984, lonEast=-91.8890098192)
        # center of bounding box at altitude=0 when masked below 10deg elevation
        labelLat, labelLon = 53.133, -98.684
        
        figax = loadFigImage(imagePath)
        drawParallelsAndMeridians(figax, m, boundingBox=bb, labelLat=labelLat, labelLon=labelLon)
        drawConstellations(figax, wcsPath, clipPoly=outline(~m.intersectsEarth))
        saveFig('test_parallelsmeridians_optimized.jpg', figax)
    
    def _testBBPole2(self):
        bb = BoundingBox(latSouth=35.3446724767, lonWest=-180.0, latNorth=90.0, lonEast=180.0)
        m = _getMappingNorth()
        m = resample(m, arcsecPerPx=100, method='mean')
        saveFig('test_bb.png', drawStereographic(m, boundingBox=bb))
    
    def _testStereographicMap(self):
        m = _getMappingSouth()
        m = m.maskedByElevation(10)
        m = resample(m, arcsecPerPx=100, method='mean')
        saveFig('test_stereo.svg', drawStereographic(m), dpi=200)
    
    def _testMLatMLTPolarMapNorthPole(self):
        m = _getMappingNorth()
        m.checkGuarantees()
        #m = m.maskedByElevation(10)
        #m.checkGuarantees()
        m = resample(m, arcsecPerPx=100, method='mean')
        m.checkGuarantees()
        saveFig('test_mlatmlt_polar_north.png', drawMLatMLTPolar(m))
         
    def _testMLatMLTPolarMapBothHemispheres(self):
        m = _getMappingSouth()
        m.checkGuarantees()
        #m = m.maskedByElevation(10)
        #m.checkGuarantees()
        m = resample(m, arcsecPerPx=100, method='mean')
        m.checkGuarantees()
        seqbb = BoundingBox(latSouth=-61.150846231, lonWest=142.622725698, latNorth=6.84984918353, lonEast=-116.820615123)
        saveFig('test_mlatmlt_polar_seqbb.png', drawMLatMLTPolar(m, boundingBox=seqbb))
               
    def _testScanlineMap(self):
        imagePath = getResourcePath('ISS029-E-8492.jpg')
        import gc, psutil
        proc = psutil.Process()
        
        from auromat.mapping.mapping import BaseMapping 
        def _getmapping(frame):
            gc.collect()
            if frame > 8494:
                # at this place we only have exactly 1 resampled mapping floating around
                assert_equal(len(list(filter(lambda o: isinstance(o, BaseMapping), gc.get_objects()))), 1)                
                
            print('Process memory:', proc.get_memory_info()[0]/1024/1024, 'MB')
            wcsPath = getResourcePath('seq/ISS029-E-%i.wcs' % frame)
            mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True, identifier=str(frame))
            mapping = mapping.maskedByElevation(10)
            # save some memory by removing additional data
            mapping = GenericMapping.fromMapping(mapping)
            print('SENDING MAPPING', frame, 'TO COROUTINES')            
            return mapping
        
        co = drawScanLinesCo('test_scanlines.png', arcsecPerPx=100, bgcolor='white')
        mappings = map(_getmapping, range(8493,8503))
        coroutine.broadcast(mappings, co)
                
    def _testScanlineMapBug(self):       
        """
        Not handling degenerate contours led to exceptions in auromat.utils.polygonArea
        as the polygon had just 0 or 1 vertices.
        See also the comment in auromat.utils.outline_skimage. 
        """
        imagePath = getResourcePath('ISS029-E-8492.jpg')
        def seq():
            for frame in range(229356,229360):
                wcsPath = getResourcePath('seq2/ISS030-E-%i.wcs' % frame)
                mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
                mapping = mapping.maskedByElevation(10)
                print('SENDING MAPPING', frame, 'TO COROUTINES')
                yield mapping
        
        co = drawScanLinesMLatMLTCo('test_scanlines2.png', arcsecPerPx=100, lineWidthFactor=3, bgcolor='white')
        coroutine.broadcast(seq(), co)
        
    def _testScanlineMapBug2(self):       
        """
        polygon masking was wrong, didn't honour existing mask
        note that this test has to be manually run and visually inspected
        """
        imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
        def seq():
            for frame in range(102170,102172):
                wcsPath = getResourcePath('seq3/ISS030-E-%i.wcs' % frame)
                mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
                mapping = mapping.maskedByElevation(10)
                print('SENDING MAPPING', frame, 'TO COROUTINES')
                yield mapping
        
        co = drawScanLinesCo('test_scanlines3.png', arcsecPerPx=100, lineWidthFactor=3, bgcolor='white')
        coroutine.broadcast(seq(), co) 
               
    def _testHoleBug(self):
        """
        Triggers a (now fixed) bug that occured when 'mean' resampling led to
        single-polygon holes where surrounding polygons were well-defined.
        A hole is a polygon such that all corners are defined and just the polygon
        center data was missing (color etc.). The issue was that those holes were
        not filtered and led to subsequent errors. The fix was to filter not by whether
        all corners are defined but whether the color is defined as this also guarantees
        that the corners are defined.        
        """
        m = _getMappingNorth()
        m = resample(m, arcsecPerPx=200, method='mean')
        # saving as svg without rasterization will lead to errors when trying to use
        # polygons with NaN colors; saving as png only leads to black areas!
        saveFig('test_stereo.svg', drawStereographic(m, rasterized=False))
    
    def _testBasemapBug(self):
        """
        Reproduce the following error:
        File "../mpl_toolkits/basemap/__init__.py", line 1079, in __init__ self._readboundarydata('gshhs',as_polygons=True) 
        File "../mpl_toolkits/basemap/__init__.py", line 1422, in _readboundarydata if not poly.is_valid(): poly=poly.fix() 
        File "_geoslib.pyx", line 234, in _geoslib.BaseGeometry.fix (src/_geoslib.c:2043) 
        File "_geoslib.pyx", line 334, in _geoslib.Polygon.__init__ (src/_geoslib.c:3233) 
        IndexError: index -1 is out of bounds for axis 0 with size 0
        
        see https://github.com/matplotlib/basemap/issues/161
        """
        # it seems that the error only occurs for certain combinations of values,
        # e.g. lon_0=-2 works
        Basemap(width=1000000, height=1000000, projection='stere', lat_0=-88, lon_0=-3)

    def _testMplCrashSimple(self):
        """
        reproduces https://github.com/matplotlib/matplotlib/issues/3304
        Note that this test only fails when an exception is added in mpl's
        _backend_agg.cpp, see https://github.com/matplotlib/matplotlib/issues/3304#issuecomment-50455754
        
        When putting an object outside of the plot limits while using svg and rasterized=True, 
        then apparently matplotlib tries to create a zero size rasterized image which leads to an
        uninitialized pointer being assigned as the resulting svg string buffer.
        """
        fig, ax = plt.subplots()
        circle1=plt.Circle((-10,10),rasterized=True)
        ax.add_artist(circle1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.savefig('test.svg')
        plt.close(fig)
       
    
    def _testMiracle(self):
        m = _getMiracleMapping()
        saveFig('heat_intersection.png', drawHeatmap(m.intersectionInflatedCorner[...,0]))
        drawHeatmaps(m)
        m.checkGuarantees()
        m = m.maskedByElevation(10)
        m.checkGuarantees()
        m = resample(m, arcsecPerPx=100, method='mean')
        m.checkGuarantees()
        saveFig('test_stereo.png', drawStereographic(m))

def getResourcePath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/" + name)

def _getMappingNorth():
    imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
    wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
    mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
    return mapping

def _getMappingSouth():
    imagePath = getResourcePath('ISS029-E-8492.jpg')
    wcsPath = getResourcePath('ISS029-E-8492.wcs')
    mapping = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
    return mapping

def _getMiracleMapping():
    imagePath = getResourcePath('SOD120304_171900_557_1000.jpg')
    mapping = miracle.getMapping(imagePath)
    return mapping


def _testCoords(offset):
    n = 10
    sp, step = np.linspace(offset,offset+10, num=n, retstep=True)
    coord = np.tile(sp, n).reshape(n,n).astype(np.float32)

    r = n*0.4
    y,x = np.ogrid[-r: r+1, -r: r+1]
    mask = x**2+y**2 <= r**2

    coord[mask] = -coord[mask]
    coord[coord>0] = np.nan
    coord[mask] = -coord[mask]
    
    coordCenter = coord[:-1,:-1] + step/2
    
    return coord, coordCenter
    
def testCoords():
    # no discontinuity or pole
    lats, latsCenter = _testCoords(70)
    lats = lats.T
    latsCenter = latsCenter.T
    
    lons, lonsCenter = _testCoords(160)
    
    return lats, lons, latsCenter, lonsCenter
from astropy.coordinates import Angle
import astropy.units as u
def testCoordsDiscontinuity():
    lats, latsCenter = _testCoords(70)
    lats = lats.T
    latsCenter = latsCenter.T
    
    lons, lonsCenter = _testCoords(160)
    
    lons = Angle((lons + 15) * u.deg).wrap_at(180 * u.deg).degree
    lonsCenter = Angle((lonsCenter + 15) * u.deg).wrap_at(180 * u.deg).degree
    return lats, lons, latsCenter, lonsCenter