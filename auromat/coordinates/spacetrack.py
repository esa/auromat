# Copyright European Space Agency, 2013

from __future__ import division, absolute_import, print_function

import logging
import os
from six.moves.urllib.parse import urlencode
from six.moves import range
from datetime import datetime, timedelta

import ephem

from auromat.coordinates.ephem import EphemerisCalculator
from auromat.util.os import makedirs
import auromat.util.url

baseUrl = 'https://www.space-track.org/'
authUrl = baseUrl + 'ajaxauth/login'
queryPrefix = 'basicspacedata/query/'

class Spacetrack:
    """
    Downloads TLE data from http://space-track.org and stores them as files
    on disk (one file per NORAD ID).
    
    Use the :mod:`auromat.coordinates.ephem` module for calculating
    coordinates based on the TLE files.
    """
    
    def __init__(self, user, password, minUpdateInterval=timedelta(days=30)):
        """
        
        :param str user: space-track.org user name
        :param str password: space-track.org password
        :param datetime.timedelta minUpdateInterval:
            The minimum amount of time after which an update of a cached
            TLE file is performed when using one of the updateTLEs* methods.
            The time is relative to the file modification dates.
        """
        self.user = user
        self.password = password
        self.minUpdateInterval = minUpdateInterval
    
    def updateTLEsFor(self, noradId, tlePath, observationDate):
        """
        Updates the TLEs such that the given observation date is covered.
        If no suitable TLEs are available (newest TLE Epoch < observationDate),
        an exception is raised.
        
        Note that TLEs are only appended, which means that TLE files
        which were produced outside this class and which only contain
        a certain time period will not be prepended with older TLEs.
        
        :param str|int noradId:
        :param str tlePath:
        :param datetime.datetime observationDate:
        :raise: DownloadError: on any network error
        :raise: ValueError: if the downloaded TLEs could not be correctly read
        :raise: ValueError: if no TLEs were found for the given date
        """
        noradId = int(noradId)
        
        if not os.path.exists(tlePath):
            self.updateTLEs(noradId, tlePath)
            newTleFile = True
        else:
            newTleFile = False

        observationTimeOutsideTLERange = False
        
        ephemCalculator = EphemerisCalculator(tlePath)
        if ephemCalculator.lastEpoch < observationDate:
            if not newTleFile:
                newTlesAdded = self.updateTLEs(noradId, tlePath)
                if newTlesAdded:
                    ephemCalculator = EphemerisCalculator(tlePath)
                    if ephemCalculator.lastEpoch < observationDate:
                        observationTimeOutsideTLERange = True
                else:
                    observationTimeOutsideTLERange = True
            else:
                observationTimeOutsideTLERange = True
        
        elif ephemCalculator.firstEpoch > observationDate:
            # Here we could check whether there are older TLEs available than already stored.
            # We don't do it as this won't be the case if this class was used for creating the
            # existing TLE file. (assuming that spacetrack doesn't release older TLEs after newer
            # ones are released)
            observationTimeOutsideTLERange = True
        
        if observationTimeOutsideTLERange:
            raise ValueError('No TLEs (NORAD: ' + str(noradId) + ') available >= ' + str(observationDate))
    
    def updateTLEs(self, noradId, tlePath):
        """
        Updates the TLEs to the latest available data.
        
        :param str|int noradId:
        :param str tlePath:
        :return: True, if new TLEs were added, False otherwise
        :raise: DownloadError: on any network error
        :raise: ValueError: if the downloaded TLEs could not be correctly read
        """
        if os.path.exists(tlePath):
            mtime = datetime.fromtimestamp(os.path.getmtime(tlePath))
            if datetime.now() - mtime < self.minUpdateInterval:
                return False
            
            # read latest available epoch
            with open(tlePath, 'r') as t:
                tles = t.readlines()
            lastTle = ephem.readtle('foo', tles[-2], tles[-1])
            year,month,day,h,m,s = lastTle._epoch.tuple()
            date = datetime(year,month,day,h,m,int(round(s))).strftime('%Y-%m-%d %H:%M:%S')
        else:
            date = '0000-00-00'
            tles = []

        query = 'class/tle/NORAD_CAT_ID/%s/EPOCH/>%s/orderby/EPOCH asc/format/tle' % (noradId,date)
        response = self.query(query)
        newTles = response.splitlines()
                
        if len(newTles) == 0:
            return False
        
        if len(newTles) % 2 != 0:
            raise ValueError('The number of returned TLE lines from space-track.org is not a multiple of 2')
        
        # filter out TLEs where the checksum is missing
        # e.g. within the ISS TLEs sporadically between 2001-2004
        # Note that appending a 0 as checksum isn't enough to satisfy pyephem,
        # but it would be enough for the sgp4 library.
        # TODO recalculate checksum
        newTles = [line for line in newTles if len(line) == 69]
                
        tleCount = len(newTles) // 2
        for i in range(tleCount):
            try:
                ephem.readtle('foo', newTles[i*2], newTles[i*2 + 1])
            except Exception as e:
                raise ValueError("The following TLE couldn't be read: [" + 
                                  newTles[i*2] + ', ' + newTles[i*2+1] + '] (reason: ' + repr(e) + ')')
        
        makedirs(os.path.dirname(tlePath))
        with open(tlePath, 'a') as t:
            t.write('\n'.join(newTles))
            t.write('\n')
        
        return True
        
    def query(self, query):
        """
        Query spacetrack and return the result as string. 
        
        :raise DownloadError: on any network error
        """
        queryUrl = baseUrl + queryPrefix + query
        logging.info('Querying ' + queryUrl)
        
        data = urlencode({'identity': self.user, 'password': self.password, 'query': queryUrl})
        res = auromat.util.url.downloadResource(authUrl, lambda r: r.read(), data=data)
        return res

if __name__ == '__main__':
    import sys
    
    assert len(sys.argv) == 5, 'Syntax: spacetrack.py user pass noradid tlepath' 
        
    st = Spacetrack(user=sys.argv[1], password=sys.argv[2])
    if st.updateTLEs(int(sys.argv[3]), sys.argv[4]):
        print('new TLEs were added')
    else:
        print('no updates available or download/read error')
        