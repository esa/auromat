# Copyright European Space Agency, 2013

from __future__ import print_function

import timeit

setup = """
import numpy as np
from astropy.coordinates.distances import spherical_to_cartesian, cartesian_to_spherical
from auromat.coordinates.transform import _ecef2GeodeticOptimized_np, _ecef2GeodeticOptimized_ne
from auromat.coordinates.transform import _spherical_to_cartesian_np, _spherical_to_cartesian_ne
from auromat.coordinates.transform import _cartesian_to_spherical_np, _cartesian_to_spherical_ne

N = 10*1000*1000
x = np.random.rand(N)
y = np.random.rand(N)
z = np.random.rand(N)
"""

setup2 = setup + """
from auromat.coordinates.intersection import _ellipsoidLineIntersection_np, _ellipsoidLineIntersection_ne
from auromat.coordinates.intersection import _ellipsoidLineIntersects_np, _ellipsoidLineIntersects_ne

lineDirection = np.array([x,y,z]).T
a = b = 2
"""

if __name__ == "__main__":
    
    print(min(timeit.repeat('_ecef2GeodeticOptimized_np(x, y, z)', setup, number=1)), 'vs.',\
          min(timeit.repeat('_ecef2GeodeticOptimized_ne(x, y, z)', setup, number=1)))
    
#    print min(timeit.repeat('spherical_to_cartesian(x, y, z)', setup, number=1)), 'vs.',\
#          min(timeit.repeat('_spherical_to_cartesian_np(x, y, z)', setup, number=1)), 'vs.',\
#          min(timeit.repeat('_spherical_to_cartesian_ne(x, y, z)', setup, number=1))

#    print min(timeit.repeat('cartesian_to_spherical(x, y, z)', setup, number=1)), 'vs.',\
#          min(timeit.repeat('_cartesian_to_spherical_np(x, y, z)', setup, number=1)), 'vs.',\
#          min(timeit.repeat('_cartesian_to_spherical_ne(x, y, z)', setup, number=1))
          
#    print min(timeit.repeat('_ellipsoidLineIntersection_np(a, b, [0,0,0], lineDirection)', setup2, number=1)), 'vs.',\
#          min(timeit.repeat('_ellipsoidLineIntersection_ne(a, b, [0,0,0], lineDirection)', setup2, number=1))
          
#    print min(timeit.repeat('_ellipsoidLineIntersects_np(a, b, [0,0,0], lineDirection)', setup2, number=1)), 'vs.',\
#          min(timeit.repeat('_ellipsoidLineIntersects_ne(a, b, [0,0,0], lineDirection)', setup2, number=1))