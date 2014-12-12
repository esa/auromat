# Copyright European Space Agency, 2013

from __future__ import absolute_import, print_function

from six import reraise
from six.moves.urllib.error import HTTPError
from six.moves.urllib.request import urlopen
import shutil
import os
import json
import sys

# monkey-patch HTTPError and add a __repr__ method
# so that it doesn't display as 'HTTPError()' but like __str__
# as e.g. 'HTTP Error 500: msg'
HTTPError.__repr__ = HTTPError.__str__

def urlResponseCode(url):
    """
    Return the response code of the server without downloading the
    actual data.
    """
    try:
        code = _urlResponseCode(url)
    except: # try again once in case of network problems
        code = _urlResponseCode(url)
    return code

def _urlResponseCode(url):
    try:
        connection = urlopen(url)
        code = connection.getcode()
        connection.close()
    except HTTPError as e:
        code = e.getcode()
    return code

class DownloadError(Exception):
    pass

def downloadFile(url, path, unifyErrors=True):
    """
    Download a single resource and store it as file to disk.
    On download errors (except 404), the download is retried once,
    after that an exception is raised.
    """
    def saveToDisk(req):
        tmpPath = path + '.tmp'
        with open(tmpPath,'wb') as fp:
            shutil.copyfileobj(req, fp)
        os.rename(tmpPath, path)
    downloadResource(url, saveToDisk, unifyErrors=unifyErrors)
    
def downloadJSON(url, unifyErrors=True, data=None, **kw):
    """
    Parse and return the JSON document at the given URL.
    Any additional keywords are given to json.load unchanged.    
    """
    def asjson(req):
        return json.load(req, **kw)
    return downloadResource(url, asjson, data=data, unifyErrors=unifyErrors)

def downloadResource(url, fn, data=None, unifyErrors=True):
    """
    Download a single resource and call `fn` on it.
    On download errors (except 404), the download is retried once,
    after that an exception is raised.
    """
    retry = False
    try:
        return _downloadResource(url, fn, data=data)
    except HTTPError as e:
        if e.code == 404:
            if unifyErrors:
                raise DownloadError(e)
            else:
                raise
        else:
            retry = True
    except IOError as e:
        if unifyErrors:
            raise DownloadError(e)
        else:
            raise
    except:
        retry = True
    
    if retry:
        print('download error, retrying once')
        try:
            return _downloadResource(url, fn, data=data)
        except:
            if unifyErrors:
                _, e, tb = sys.exc_info()
                new_exc = DownloadError('{}: {}'.format(e.__class__.__name__, e))
                reraise(new_exc.__class__, new_exc, tb)
            else:
                raise
    
def _downloadResource(url, fn, data=None):
    print('downloading', url, end='')
    try:        
        req = urlopen(url, data=data) # throws also on 404
        res = fn(req)
        print('-> done')
        return res
    except Exception as e:
        # 404, network problem, IO error, ...
        print('->', str(e))
        raise
        
def downloadFiles(urls, paths, retFailures=False):
    """
    Downloads multiple resources and stores them to disk at the given `paths`,
    ignoring already existing files
    on disk. On download errors (except 404), the download is retried once.
    If retFailures is False, then True is returned if all files
    could be downloaded successfully, otherwise False.
    If retFailures is True, then a tuple (bool, failures) is
    returned which additionally contains the urls (with exceptions)
    that couldn't be downloaded.
    """
    failures = []
    for url, path in zip(urls, paths):
        if os.path.exists(path):
            continue
        try:
            downloadFile(url, path, unifyErrors=False)
        except Exception as e:
            failures.append((url, e))
                
    if retFailures:
        return len(failures) == 0, failures
    else:
        return len(failures) == 0
