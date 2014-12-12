# Copyright European Space Agency, 2013

from __future__ import absolute_import, division

import astropy.time

def containsLeapSecond(d1, d2):
    """
    Checks whether the given datetime range contains a leap second.
    
    :param datetime.datetime d1: start date
    :param datetime.datetime d2: end date
    :rtype: bool
    """
    eps = 1e-6
    diffWithoutLeap = (d2-d1).total_seconds()
    diffWithLeap = (astropy.time.Time(d2, scale='utc') - astropy.time.Time(d1, scale='utc')).sec
    containsLeap = abs(diffWithoutLeap - diffWithLeap) > eps
    return containsLeap
        