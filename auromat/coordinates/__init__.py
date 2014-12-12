"""
This package contains modules to determine coordinates of space objects 
(:mod:`~auromat.coordinates.spacetrack` and :mod:`~auromat.coordinates.ephem`), 
to convert existing coordinates into other reference frames 
(:mod:`~auromat.coordinates.transform` and :mod:`~auromat.coordinates.igrf`),
to calculate intersection points between a ray and an ellipsoid
(:mod:`~auromat.coordinates.intersection`), and to perform geodesic
calculations (:mod:`~auromat.coordinates.geodesic`).

This package does not depend on the `mapping` objects used in other parts of
the :mod:`auromat` package and can therefore be re-used generically for
other purposes.
"""