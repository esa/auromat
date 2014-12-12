# Copyright European Space Agency, 2013

from __future__ import absolute_import

import os
import errno

def touch(path):
    """
    Create new file if it doesn't exist or just update mtime to NOW.
    """
    with open(path, 'a'):
        os.utime(path, None)
        
def makedirs(*paths):
    """
    Recursively creates folders if not already existing.
    """
    for path in paths:
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise