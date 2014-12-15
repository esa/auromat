# Copyright European Space Agency, 2013

from __future__ import absolute_import, print_function

from nose.plugins.attrib import attr
import os
from time import sleep
from auromat.mapping.spacecraft import getMappingSequence

@attr('slow')
def parallelTest():
    n = 5
    imagePaths = [getResourcePath('ISS030-E-102170_dc.jpg')]*n
    wcsPaths = [getResourcePath('ISS030-E-102170_dc.wcs')]*n
    for mapping in getMappingSequence(imagePaths, wcsPaths, parallel=True):
        # simulate heavy work
        print(mapping.arcSecPerPx)
        sleep(10)
        
    
def getResourcePath(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/" + name)

if __name__ == '__main__':
    parallelTest()