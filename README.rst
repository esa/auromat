AUROra MApping Toolkit
======================

.. image:: https://travis-ci.org/esa/auromat.svg?branch=master
    :target: https://travis-ci.org/esa/auromat
    :alt: Linux Build Status

Installation under Linux
------------------------

The following assumes Ubuntu, but should be similar for other distributions.

Before installing auromat, some system libraries have to be installed.

    sudo apt-get install libraw-dev liblensfun-dev

If you want to use THEMIS data or export in CDF format you have to
install NASA's CDF library, see http://cdf.gsfc.nasa.gov for details.
Also, for using the CDF library in Python we need the spacepy library.
As this is not yet released on PyPI, you have to install it manually using:

    git clone --depth 1 http://git.code.sf.net/p/spacepy/code spacepy
    cd spacepy && python setup.py install --user && cd ..

If you want to export in netCDF format:

    sudo apt-get install libnetcdf-dev libhdf5-serial-dev
 
Now, install auromat with:

    pip install --user auromat[cdf,netcdf]

Use auromat[cdf] or auromat[netcdf] to leave out CDF or netCDF support, respectively.

Installation under Mac OS X
---------------------------

First, install Homebrew if you don't have it yet:

    ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

Then, install some native software with Homebrew:

    brew install git python

If you want to use THEMIS data or export in CDF format you have to
install NASA's CDF library, see http://cdf.gsfc.nasa.gov for details.
Also, for using the CDF library in Python we need the spacepy library.
As this is not yet released on PyPI, you have to install it manually using:

    git clone --depth 1 http://git.code.sf.net/p/spacepy/code spacepy
    cd spacepy && python setup.py install --user && cd ..

If you want to export in netCDF format:

    brew install netcdf hdf5

Now, install auromat with:

    pip install --user auromat[cdf,netcdf]

Use auromat[cdf] or auromat[netcdf] to leave out CDF or netCDF support, respectively.

Installation under Windows
--------------------------

If you need to use THEMIS data or export in CDF format, then you need to use
Python 2.7 for 32 bit. The Python library that is used for handling CDF files
(SpacePy) is currently only available for Python 2.6 and 2.7 for 32 bit.

For Python 3.3 and lower, you have to install the package manager pip,
see http://pip.readthedocs.org/en/latest/installing.html for instructions.

Some required Python packages (as of late 2014) don't offer Windows binary
wheels on PyPI yet. Therefore, you have to install them manually:

Please install numpy, scipy, scikit-image, astropy, and pyephem from
http://www.lfd.uci.edu/~gohlke/pythonlibs/

If you want to use THEMIS data or export in CDF format you have to
install NASA's CDF library (32 bit version), see http://cdf.gsfc.nasa.gov for details.
Also, for using the CDF library in Python you need the SpacePy library.
You can download an installer from
http://sourceforge.net/projects/spacepy/files/spacepy

If you want to export in netCDF format please install the netCDF4 library from:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#netcdf4

Now, install auromat with:

    pip install --user auromat[cdf,netcdf]

Use auromat[cdf] or auromat[netcdf] to leave out CDF or netCDF support, respectively.

Advanced functionality
----------------------

The following software can be installed if you want to georeference images yourself
and not use the available data providers. Note that the complete workflow is not as
straight-forward for certain data sources, e.g. to correctly georeference ISS images
you have to consider inaccurate camera timestamps and possibly create missing lens distortion
profiles.

If you want to determine astrometric solutions yourself using the auromat.solving package,
you need to install astrometry.net, see http://astrometry.net/use.html. Make sure the
bin/ folder is in your PATH so that auromat can find it.

If you want to automatically mask the starfield of an image using the auromat.solving.masking
module, please install on Ubuntu:

    sudo apt-get install libopencv-imgproc-dev python-opencv
    
on Mac OS X, please follow 
http://jjyap.wordpress.com/2014/05/24/installing-opencv-2-4-9-on-mac-osx-with-python-support/

on Windows, install from http://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv

If you want to correct lens distortion in an image with the lensfun database
using EXIF data extracted from the image, please install on Ubuntu:

    sudo apt-get install libimage-exiftool-perl
    
on Mac OS X, 
    
    brew install exiftool
    
on Windows, extract the zip archive from http://www.sno.phy.queensu.ca/~phil/exiftool/
into a folder and put it in your PATH so that auromat can find exiftool.
