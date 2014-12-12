# Copyright European Space Agency, 2013

from functools import wraps

def lazy_property(fn):
    """
    Caches the result of a property.
    """
    attr_name = '_lazy_' + fn.__name__
    @property
    @wraps(fn)
    def _lazyprop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazyprop

def inherit_docs(cls):
    """
    Inherits docstrings from base classes
    for all overridden methods and properties in `cls` which lack a docstring.
    """
    # see https://stackoverflow.com/a/23964187
    for name in dir(cls):
        func = getattr(cls, name)
        if func.__doc__: 
            continue
        for parent in cls.mro()[1:]:
            if not hasattr(parent, name):
                continue
            doc = getattr(parent, name).__doc__
            if not doc: 
                continue
            try:
                # __doc__'s of properties are read-only.
                # The work-around below wraps the property into a new property.
                if isinstance(func, property):
                    # We don't want to introduce new properties, therefore check
                    # if cls owns it or search where it's coming from.
                    # With that approach (using dir(cls) instead of var(cls))
                    # we also handle the mix-in class case.
                    wrapped = property(func.fget, func.fset, func.fdel, doc)
                    clss = list(filter(lambda c: name in vars(c).keys() and not getattr(c, name).__doc__, cls.mro()))
                    setattr(clss[0], name, wrapped)
                else:
                    try:
                        func = func.__func__ # for instancemethod's
                    except:
                        pass
                    func.__doc__ = doc
            except: # some __doc__'s are not writable
                pass
            break
    return cls