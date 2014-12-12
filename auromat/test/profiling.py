# Copyright European Space Agency, 2013

from __future__ import print_function

import os.path
import cProfile
from auromat.mapping.spacecraft import getMapping
from auromat.resample import resample

try:
    import pyinstrument
except ImportError:
    print('pyinstrument not available, will use cProfile only')
    pyinstrument = None

def cprofile_profile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats(sort='time')
    return profiled_func

def pyinstrument_profile(func):
    def profiled_func(*args, **kwargs):
        try:
            profile = pyinstrument.Profiler()
        except:
            profile = pyinstrument.Profiler(use_signal=False)
        try:
            profile.start()
            result = func(*args, **kwargs)
            profile.stop()
            return result
        finally:
            print(profile.output_text(color=True))
    return profiled_func

def profile(profiler):
    imagePath = getResourcePath('ISS030-E-102170_dc.jpg')
    wcsPath = getResourcePath('ISS030-E-102170_dc.wcs')
    m = getMapping(imagePath, wcsPath, fastCenterCalculation=True)
    profiler(_profileLatsLons)(m)
    profiler(_profileBoundingBox)(m)
    profiler(_profileMeanResampling)(m)

def _profileLatsLons(m):
    print('profiling lats/lons/elevation/img calculation')
    m.lats
    m.latsCenter
    m.elevation
    m.img

def _profileBoundingBox(m):
    print('profiling bounding box calculation')
    m.boundingBox

def _profileMeanResampling(m):
    print('profiling mean resampling')
    resample(m, method='mean')

def getResourcePath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/" + name)

if __name__ == '__main__':
    profile(cprofile_profile)
    
    if pyinstrument is not None:
        profile(pyinstrument_profile)
