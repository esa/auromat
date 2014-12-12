"""
This package allows to georeference images using the camera position and
the starfield within the image. It should be generic enough to work with
any kind of images but was developed with ISS aurora images in mind.
It tries to automatically detect where the starfield in the image is
(see :mod:`auromat.solving.masking` modules) and then
runs astrometry.net to extract the stars and find the pointing and scale of the
image (see :mod:`auromat.solving.solving` module).

For images taken from spacecrafts (like the ISS) it contains two convenience
modules: :mod:`~auromat.solving.spacecraft` and :mod:`~auromat.solving.eol`.
The :mod:`~auromat.solving.spacecraft` module
determines the spacecraft position (=camera position) at the time
the image was taken using TLE data from space-track.org, given the NORAD ID of the spacecraft.
The :mod:`~auromat.solving.eol` module allows to easily download images
from `NASA's Earth Observation website <http://eol.jsc.nasa.gov>`_ in JPEG and RAW format.

The resulting calibration is a FITS WCS header from which the geodetic
pixel coordinates can be calculated from. The counterpart modules in the
mapping package are :mod:`auromat.mapping.astrometry` and :mod:`auromat.mapping.spacecraft`.
The :mod:`auromat.mapping.iss` module relies on those two modules as well and
is tailored towards the ESA ISS archive containing such images and calibrations.
"""