# -*- coding: utf-8 -*-
"""
=====================
Animation
=====================
"""

from __future__ import annotations

import numpy as np

from .transformation import Transformable, Transformation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Moveable(Transformable):
    """Moveables are time-dependent transformations.

    Moveable objects may change their position and orientation in space during simulation runtime.
    """

    __velocity: np.ndarray

    def __init__(self, pose: Transformation | None = None, velocity: np.ndarray | None = None) -> None:
        """
        Args:

            pose (Transformation, optional):
                Initial pose of the moveable with respect to its reference coordinate frame.
                By default, a unit transformation is assumed.

            velocity (np.ndarray, optional):
                Initial velocity of the moveable in local coordinates.
                By default, the moveable is assumed to be resting.
        """

        # Initialize base class
        Transformable.__init__(self, pose)

        # Initialize class attributes
        self.velocity = np.zeros(3, dtype=float) if velocity is None else velocity

    @property
    def velocity(self) -> np.ndarray:
        """Current velocity of the moveable.

        Returns: Cartesian velocity vector in m/s.
        """

        return self.__velocity

    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        self.__velocity = value
