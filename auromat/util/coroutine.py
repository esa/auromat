# Copyright European Space Agency, 2013

from __future__ import print_function

from six import reraise, PY3
from functools import wraps
import sys

# see http://slideshare.net/emptysquare/nyc-python-meetup-coroutines-2013-0416

def coroutine(func):
    @wraps(func)
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        next(cr)
        return cr
    return start

def broadcast(iterable, *targets):
    """
    Sends items to multiple coroutines.
    
    If one coroutine raises an exception, then all other coroutines
    are killed as well and the exception is re-raised to the caller.
    
    :param iterable: the items to send to the coroutines
    :param targets: coroutines to send iterables to
    """
    _sendTo(iterable, _broadcast(targets))

@coroutine
def _broadcast(targets):
    exc = None
    
    if len(targets) == 1:
        # optimized code path which doesn't hold a reference
        # to the yielded item -> conserves memory if item
        # is replaced/transformed in target
        target = targets[0]
        while True:
            try:
                try:
                    target.send((yield))
                except GeneratorExit:
                    break
                except StopIteration:
                    pass
                except:
                    exc = sys.exc_info()
                    raise GeneratorExit
            except GeneratorExit:
                break
    else:
        while True:
            try:
                item = (yield)
                for target in targets:
                    try:
                        target.send(item)
                    except (GeneratorExit, StopIteration):
                        pass
                    except:
                        exc = sys.exc_info()
                        raise GeneratorExit
            except GeneratorExit:
                break
            except:
                exc = sys.exc_info()
                break
        
        # in case item is big, remove reference to it
        # to conserve memory for the closing phase
        try:
            del item
        except NameError:
            pass
        
    if exc is None:
        for target in targets:
            try:
                target.close()
            except (GeneratorExit, StopIteration):
                pass
            except:
                exc = sys.exc_info()
                break
    
    if exc is not None:
        etype, e, tb = exc
        
        # kill other coroutines
        for target in targets:
            try:
                throw(target, etype, e, tb)
            except:
                pass
        
        reraise(etype, e, tb)
         
def _sendTo(iterable, target):
    try:
        try:
            while True:
                # important: we use next() so that we don't have
                # to store a reference in a local variable in a for-loop
                # (see _broadcast())
                target.send(next(iterable))
        except StopIteration:
            pass
        target.close()
    except:
        # if generator raised, then we kill the target with the same exception
        etype, ei, tb = sys.exc_info()
        throw(target, etype, ei, tb) # will raise back to us

def throw(target, etype, e, tb):
    if PY3:
        target.throw(etype(e).with_traceback(tb))
    else:
        target.throw(etype, e, tb)

# Example use
if __name__ == '__main__':
    from time import sleep
    
    @coroutine
    def grep(pattern, e):
        print("Looking for %s" % pattern)
        try:
            line = yield
            line = yield
        except GeneratorExit:
            raise RuntimeError('not enough items!')
                    
        while True:
            sleep(1)
            try:
                line = yield
            except GeneratorExit:
                print('close was called, no more items left')
                break
            
            if pattern in line:
                print(line)
                
            if e:
                raise ValueError('I am a bad consumer')
         
        print('end')

    # main limitation: no single return value (only for each item)

    def gen():
        for i in range(10):
            yield str(i)
            if i == 5:
                raise ValueError('I am a bad generator')

    consumers = [grep('foo', True), grep('bar', False)]
    broadcast(["fooooo","barrrr"]*3, consumers)
    #broadcast(["fooooo"], consumers)
    #broadcast(gen(), consumers)

    sleep(5)
    print('program exit')