# Copyright European Space Agency, 2013

"""
This module contains functions for calculating and working with
geodetic coordinates.
"""

from __future__ import division
from six.moves import range

import logging
from collections import namedtuple

import numpy as np
from geographiclib.geodesic import Geodesic
from geographiclib.constants import Constants


wgs84A = Constants.WGS84_a/1000
wgs84B = wgs84A * (1 - Constants.WGS84_f)

Location = namedtuple('Location', ['lat','lon']) # in degrees

def distance(location1, location2):
    """
    Return the shortest distance in meters between two locations.
    """
    data = Geodesic.WGS84.Inverse(location1.lat, location1.lon, 
                                  location2.lat, location2.lon, 
                                  Geodesic.DISTANCE)
    d = data['s12']
    return d

def angularDistance(location1, location2):
    """
    Return the shortest angular distance in degrees
    on an auxiliary sphere between two locations.
    """
    data = Geodesic.WGS84.Inverse(location1.lat, location1.lon, 
                                  location2.lat, location2.lon, 
                                  Geodesic.EMPTY)
    a = data['a12']
    return a

def line(location1, location2, resolution=1000):
    """
    Return points on the given geodesic in the given resolution.
    If the distance between the two locations is smaller than
    the requested resolution, then the line points consist only
    of the unchanged start and end locations.
    
    :type location1: Location
    :type location2: Location
    :param resolution: in meters
    :rtype: ndarray of shape (n, 2) with [lat,lon] order
    :return: line points in degrees
    """
    data = Geodesic.WGS84.Inverse(location1.lat, location1.lon, 
                                  location2.lat, location2.lon, 
                                  Geodesic.AZIMUTH | Geodesic.DISTANCE)
    az = data['azi1']
    d = data['s12']
    line = Geodesic.WGS84.Line(location1.lat, location1.lon, az, 
                               Geodesic.LATITUDE | Geodesic.LONGITUDE)
    
    num = d//resolution
    if num < 2:
        logging.warn('Geodesic line has less than two points at ' + str(resolution) + 'm resolution' +
                     'for a line length of ' + str(d) + 'm. The input points are returned as-is.')
        return np.array([[location1.lat, location1.lon], [location2.lat, location2.lon]])
    
    latLons = []
    ds = np.linspace(0, d, num)
    for dStartToPoint in ds:
        data = line.Position(dStartToPoint, Geodesic.LATITUDE | Geodesic.LONGITUDE)
        latLons.append([data['lat2'], data['lon2']])
    return np.array(latLons)

def destination(location, azimuth, distance):
    """
    Return the location when starting at `location` and
    travelling in direction `azimuth` for `distance` meters.
    
    :param lon, lat, azimuth: in degrees
    :param distance: in meters
    :rtype: :class:`Location`
    """
    data = Geodesic.WGS84.Direct(location.lat, location.lon, azimuth, distance)
    lon, lat = data['lon2'], data['lat2']
    return Location(lat, lon)

def intermediate(location1, location2, f=0.5):
    """
    Return the location when travelling `f * distance(location1,location2)` meters
    starting at `location1` in direction of `location2`.
    
    :type location1: Location
    :type location2: Location
    :param f: fraction to travel from `location1` to `location2`
    :rtype: :class:`Location`
    """
    data = Geodesic.WGS84.Inverse(location1.lat, location1.lon, 
                                  location2.lat, location2.lon, 
                                  Geodesic.AZIMUTH | Geodesic.DISTANCE)
    d = data['s12']
    az = data['azi1']
    data2 = Geodesic.WGS84.Direct(location1.lat, location1.lon, az, d*f, Geodesic.LATITUDE | Geodesic.LONGITUDE)
    lon, lat = data2['lon2'], data2['lat2']
    return Location(lat, lon)
    
def course(location1, location2):
    """
    Return the course/azimuth in degrees when travelling from `location1` to `location2`.
    """
    data = Geodesic.WGS84.Inverse(location1.lat, location1.lon, 
                                  location2.lat, location2.lon, 
                                  Geodesic.AZIMUTH)
    az = data['azi1']
    return az

def _courseDelta(a1, a2):
    """
    angles in degrees
    see http://www.element84.com/follow-up-to-determining-if-a-spherical-polygon-contains-a-pole.html
    """
    if a2 < a1:
        a2 += 360

    left_turn_amount = a2 - a1

    if left_turn_amount == 180:
        return 0
    elif left_turn_amount > 180:
        return left_turn_amount - 360
    else:
        return left_turn_amount
    
def _courseDeltaSum(points):
    """
    returns delta sum in integer degrees (either -360, -180, 0, 180, or 360)
    
    see:
    http://www.element84.com/determining-if-a-spherical-polygon-contains-a-pole.html
    http://www.element84.com/follow-up-to-determining-if-a-spherical-polygon-contains-a-pole.html
    
    Note: points must be in degrees as lat, lon pairs
    Note: points must be ordered such that they form a non-intersecting polygon
    Note: the first point must not be repeated at the end
    """
    # calculate initial and ending courses for each arc
    points = np.asarray(points)
    assert points.ndim == 2 and points.shape[1] == 2
    
    arcs = len(points)-1
    courses = np.empty(arcs*2)
    
    for i in range(arcs):
        lon1, lat1 = points[i,1], points[i,0]
        lon2, lat2 = points[i+1,1], points[i+1,0]
        courses[2*i] = course(Location(lat1,lon1), Location(lat2,lon2))
        courses[2*i+1] = course(Location(lat2,lon2), Location(lat1,lon1)) + 180

    # calculate course deltas
    deltas = np.empty(arcs*2)
    deltas[0] = _courseDelta(courses[arcs*2-1], courses[0])
    for i in range(1, arcs*2):
        deltas[i] = _courseDelta(courses[i-1], courses[i])
    
    deltaSum = np.around(np.sum(deltas), decimals=1)
    assert deltaSum in [-360,-180,0,180,360]
    return deltaSum

def containsOrCrossesPole(points):
    """
    Return whether the given polygon contains or crosses one of the poles.
    
    :param points: ordered points forming a non-intersecting unclosed polygon
    :type points: ndarray of shape (n,2) with lat,lon coordinates in degrees
    :rtype: bool 
    """
    deltaSum = _courseDeltaSum(points)

    containsPole = False
    if abs(deltaSum) == 360:
        logging.debug("pole: no")
    elif abs(deltaSum) == 180:
        containsPole = True
        logging.debug("pole: CROSSED")
    elif deltaSum == 0:
        containsPole = True
        logging.debug("pole: CONTAINED")
    return containsPole
