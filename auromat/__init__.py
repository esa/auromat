"""
The auromat package is split up in several packages and modules each covering
different aspects of its functionality.

The :mod:`auromat.cli` package contains command-line tools that will be installed as `auromat-name`
for a module name `auromat.cli.name`. It is not intended to be used from within
Python code.

The :mod:`auromat.coordinates` package allows to determine coordinates of space objects, 
to convert existing coordinates into other reference frames,
to calculate intersection points between a ray and an ellipsoid, and to perform geodesic
calculations. It does not depend on the `mapping` objects used in other parts of
the :mod:`auromat` package and can therefore be easily re-used for other purposes.

The :mod:`auromat.solving` package contains modules to blindly georeference
astronaut photography pointing to earth using the starfield in the images.
The use of this package is not required for working with already georeferenced
images such as the ones from the ESA-ISS archive.

The :mod:`auromat.mapping` package contains various modules for reading
and working with georeferenced data, e.g. as produced by the :mod:`auromat.solving`
package or other available in forms like the THEMIS or ESA-ISS web archives.
It also contains the
:class:`~auromat.mapping.mapping.BaseMapping` main class which other packages depend
on for accessing georeferenced image data through a defined interface.

The :mod:`auromat.export` package can export any :class:`~auromat.mapping.mapping.BaseMapping`
into a self-contained CDF or netCDF file, suitable for further processing in different
software.

The :mod:`auromat.util` package contains independent generic helper functions not strictly related
to the main functions of this library. No module has a dependency to another part of
this library.

The :mod:`auromat` package also contains various submodules, e.g. for 
resampling (:mod:`auromat.resample`) and visualizing (:mod:`auromat.draw`)
mappings, as well as some modules which did not fit well into any other bigger package.
"""

from __future__ import absolute_import

from ._version import __version__, __version_info__

import warnings
try:
    import matplotlib as mpl
    # Agg is the only backend supporting the rasterized option so we can't just use cairo.
    mpl.use('Agg') # headless backend
except ImportError:
    pass
else:
    from distutils.version import LooseVersion
    if LooseVersion(mpl.__version__) < '1.4.0':
        warnings.warn('Your matplotlib version has a memory leak which occurs when saving maps ' +
                      'in SVG format. Use >= 1.4.0 if you need to draw many maps in a row. ' +
                      'More details are at https://github.com/matplotlib/matplotlib/issues/3197')