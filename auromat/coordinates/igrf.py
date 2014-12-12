# Copyright European Space Agency, 2013

from __future__ import division, absolute_import, print_function

from math import floor, ceil

__all__ = []

NUM_IGRF_YEARS_DEFINED = 24

IGRF_DEFINED_UNTIL_YEAR = 1900 + (NUM_IGRF_YEARS_DEFINED-1)*5

__doc__ = """
This module defines the first three IGRF coefficients `g01`, `g11`, and `h11`
for the years 1900 until {}. They are used by the :mod:`auromat.coordinates.transform`
module.

Note that the coefficients have to be updated here if needed.
When updating, the variables `NUM_IGRF_YEARS_DEFINED`, `g01`, `g11`, and `h11`
have to be changed.
""".format(IGRF_DEFINED_UNTIL_YEAR)

g01 = [-31543, -31464, -31354, -31212, -31060, -30926, -30805, -30715,
       -30654, -30594, -30554, -30500, -30421, -30334, -30220, -30100,
       -29992, -29873, -29775, -29692, -29619.4, -29554.63, -29496.5,
       -29439.5]

g11 = [-2298, -2298, -2297, -2306, -2317, -2318, -2316, -2306, -2292, -2285,
       -2250, -2215, -2169, -2119, -2068, -2013, -1956, -1905, -1848, -1784,
       -1728.2, -1669.05, -1585.9, -1502.4]

h11 = [5922, 5909, 5898, 5875, 5845, 5817, 5808, 5812, 5821, 5810, 5815,
       5820, 5791, 5776, 5737, 5675, 5604, 5500, 5406, 5306, 5186.1, 5077.99, 
       4945.1, 4801.1]

assert len(g01) == len(g11) == len(h11) == NUM_IGRF_YEARS_DEFINED

def calcG01(fracYearIndex, fracYear):
    _checkFracYearIndex(fracYearIndex)
    return (g01[int(floor(fracYearIndex))]*(1.0-fracYear) + 
            g01[int(ceil(fracYearIndex))]*fracYear)

def calcG11(fracYearIndex, fracYear):
    _checkFracYearIndex(fracYearIndex)
    return (g11[int(floor(fracYearIndex))]*(1.0-fracYear) + 
            g11[int(ceil(fracYearIndex))]*fracYear)

def calcH11(fracYearIndex, fracYear):        
    _checkFracYearIndex(fracYearIndex)
    return (h11[int(floor(fracYearIndex))]*(1.0-fracYear) + 
            h11[int(ceil(fracYearIndex))]*fracYear)
    
def _checkFracYearIndex(fracYearIndex):
    if fracYearIndex >= NUM_IGRF_YEARS_DEFINED-1:
        raise ValueError("ERROR: Specified year is greater than IGRF implementation, "
                         "please update coefficients in auromat.coordinates.igrf module")
    