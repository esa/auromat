# Copyright European Space Agency, 2013

from __future__ import division, absolute_import, print_function

import warnings
import calendar
import datetime

import numpy as np

from astropy import units
from astropy import coordinates as coord

import ephem

class EphemerisCalculator:
    """
    Calculates J2000 positions using a set of TLEs for a *single* satellite.
    TLEs are read once into a string to serve as a cache.
    
    Use the :mod:`~auromat.coordinates.spacetrack` module for downloading
    TLE data.
    
    Note that this module assumes continuous TLE data as created by
    the :mod:`~auromat.coordinates.spacetrack` module. 
    
    Example::
    
      from datetime import datetime
      from auromat.coordinates.spacetrack import Spacetrack
      from auromat.coordinates.ephem import EphemerisCalculator
      path = 'iss_tles.txt'
      st = Spacetrack('user', 'pass')
      st.updateTLEs(25544, path)
      iss = EphemerisCalculator(path)
      j2000_xyz = iss(datetime(2012,1,1,12,30,0))
      print(j2000_xyz)
      
    """
    def __init__(self, tleFilePath):
        """
        :param string tleFilePath: path to file containing TLE sets
        """
        with open(tleFilePath, 'r') as t:
            self.tles = t.readlines()
                
        assert len(self.tles) > 0
        assert len(self.tles) % 2 == 0
         
    def __call__(self, date):
        """
        Return the J2000 position of the space object for the given date.
        
        :param datetime.datetime date:
        :return: x,y,z position as array
        :rtype: ndarray of shape (3,)
        """
        return self.getPosition(date)
    
    @property
    def noradId(self):
        """
        The NORAD ID as read from the first TLE.
        
        :rtype: str
        """
        return self.tles[0][2:7]

    @property
    def firstEpoch(self):
        """
        The earliest epoch in the TLE file.
        
        :rtype: datetime.datetime
        """
        tle = ephem.readtle('foo', self.tles[0], self.tles[1])
        date = self._toDateTime(tle._epoch)
        return date
    
    @property
    def lastEpoch(self):
        """
        The latest epoch in the TLE file.
        
        :rtype: datetime.datetime
        """
        tle = ephem.readtle('foo', self.tles[-2], self.tles[-1])
        date = self._toDateTime(tle._epoch)
        return date
    
    def contains(self, date):
        """
        Return whether firstEpoch <= date <= lastEpoch holds. 
        
        :type date: datetime.datetime
        """
        return self.firstEpoch <= date <= self.lastEpoch
    
    @staticmethod
    def _toDateTime(ephemDate):
        """
        Converts PyEphem dates to datetime objects.
        """
        timestamp = calendar.timegm(ephemDate.tuple())
        return datetime.datetime.utcfromtimestamp(timestamp)
    
    def getTLE(self, date):
        date = ephem.Date(date)
        
        tleCount = len(self.tles) // 2
        lo = 0
        hi = tleCount
        while lo < hi: # equals bisect.bisect_left
            mid = (lo+hi)//2
            tle = ephem.readtle('foo', self.tles[mid*2], self.tles[mid*2 + 1])
            if tle._epoch < date:
                lo = mid+1
            else:
                hi = mid

        tleIdx = lo-1 # equals find_lt (rightmost TLE with epoch less than given date)

        return self.tles[tleIdx*2], self.tles[tleIdx*2 + 1]
       
    def getPosition(self, date):
        """
        Return the J2000 position of the space object for the given date.
        
        Note that you can also use the call operator::
        
          iss = EphemerisCalculator(path)
          iss(date)
        
        :type date: datetime.datetime
        :rtype: ndarray of xyz J2000 coordinates in km
        """
        tlelines = self.getTLE(date)
        tle = ephem.readtle('foo', tlelines[0], tlelines[1])
        
        date = ephem.Date(date)
        
        if tle._epoch > date:
            raise Exception("The epoch of the earliest available TLE is AFTER the requested date. " +
                            "Are you missing historic TLE data?")
        
        if (date - tle._epoch > 24*ephem.hour):
            warnings.warn('closest TLE epoch is ' + str((date - tle._epoch)*24) + 'h away from photo time')                          
                
        tle.compute(date)
        
        # see http://stackoverflow.com/q/19426505/60982
        earthRadiusPyEphem = 6378.16 * units.km
        distanceEarthCenter = tle.elevation*units.m + earthRadiusPyEphem
        
        xyz = coord.distances.spherical_to_cartesian(distanceEarthCenter.to(units.km).value, tle.a_dec, tle.a_ra)
        return np.array(xyz)
        
    def _getPositionX(self, date):
        """
        experimental: uses sgp4 library
        
        NOTE: returns TEME coordinates (true equator mean equinox)

        :param datetime date:
        :rtype: GCRSCoordinates
        """
        from sgp4.earth_gravity import wgs72
        from sgp4.io import twoline2rv
        from sgp4.ext import jday

        jdate = jday(date.year, date.month, date.day, date.hour, date.minute, date.second)

        tleCount = len(self.tles) // 2
        lo = 0
        hi = tleCount
        while lo < hi: # equals bisect.bisect_left
            mid = (lo+hi)//2
            tle = twoline2rv(self.tles[mid*2], self.tles[mid*2 + 1], wgs72)
            if tle.jdsatepoch < jdate:
                lo = mid+1
            else:
                hi = mid
        
        tleIdx = lo-1 # equals find_lt (rightmost TLE with epoch less than given date)
        tle = twoline2rv(self.tles[tleIdx*2], self.tles[tleIdx*2 + 1], wgs72)

        if tle.jdsatepoch > jdate:
            raise Exception("The epoch of the earliest available TLE is AFTER the requested date. " +
                            "Are you missing historic TLE data?")
        
        if (jdate - tle.jdsatepoch > 10*ephem.hour):
            warnings.warn('closest TLE epoch is ' + str((date - tle.jdsatepoch)*24) + 'h away from photo time')
        
        position, _ = tle.propagate(date.year, date.month, date.day, date.hour, date.minute, date.second)

        x,y,z = units.km.to(units.pc, position)
        
        # FIXME this is wrong, conversion TEME-J2000 is missing
        # -> need skyfield library for that, but too heavy currently
        #   see https://github.com/brandon-rhodes/python-skyfield/issues/31
        return np.array([x,y,z]) 