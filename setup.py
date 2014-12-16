# Copyright European Space Agency, 2013

from setuptools import setup, find_packages
import re

# version handling from https://stackoverflow.com/a/7071358
VERSIONFILE="auromat/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name = 'auromat',
    description = 'AUROra MApping Toolkit',
    long_description = open('README.rst').read(),
    version = verstr,
    author = 'Maik Riechert',
    author_email = 'mriecher@cosmos.esa.int, awalsh@sciops.esa.int',
    license = 'ESCL - Type 1',
    url = 'https://github.com/esa/auromat',
    classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Intended Audience :: Science/Research',
      'Natural Language :: English',
      'Programming Language :: Python :: 2',
      'Programming Language :: Python :: 3',
      'Operating System :: OS Independent',
      'Topic :: Scientific/Engineering',
      'Topic :: Software Development :: Libraries',
    ],
    packages = find_packages(),
    install_requires=['six',
                      'futures',
                      'papy',
                      'numpy>=1.6',
                      'scipy>=0.9',
                      'matplotlib', # required by auromat.utils
                      'brewer2mpl',
                      'numexpr',
                      'scikit-image',
                      'pillow', # used by scikit-image
                      'psutil',
                      'ExifRead',
                      'geographiclib',
                      'astropy>=0.4.1',
                      'astroquery',
                      'pyephem>=3.7.5.2',
                      'lensfunpy>=1.1.0',
                      'rawpy>=0.1',
                      ],
    extras_require = {
        'netcdf': ['netCDF4'], # for netCDF export
        'cdf': ['spacepy'], # for CDF export and reading THEMIS CDF files
    },
    package_data = {
        '': ['*.dbf', '*.shp', '*.shx'],
    },
    entry_points = {
        'console_scripts': [
            'auromat-download = auromat.cli.download:main',
            'auromat-convert = auromat.cli.convert:main',
        ]
    }
)