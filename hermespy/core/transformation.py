# -*- coding: utf-8 -*-
"""
=================================
Coordinate System Transformations
=================================
"""

from __future__ import annotations
from math import atan
from typing import List, Optional

import numpy as np

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Transformation(np.ndarray):

    @property
    def position(self) -> np.ndarray:

        return self[:3, 3]

    @property
    def orientation_rpy(self) -> Tuple[float, float, float]:
        """Orientation in Roll, Pitch and Yaw Angles.
        
        Returns:
            Roll, Pitch and Yaw in Radians.
        """

        roll = attan(self[2, 1] / self[2, 2])
        pitch = atan(- self[2, 0] * (self[2, 1] + self[2, 2]) ** -.5)
        yaw = atan(self[1, 0] / self[0, 0])

        return roll, pitch, yaw

class Transformable(object):
    """Representation of a Coordinate Frame within a Kinematic Chain."""

    __reference_frame: Optional[Transformable]
    __linked_frames: List[Transformable]

    def __init__(self) -> None:
        
        self.__reference_frame = None
        self.__linked_frames = []

    @property
    def position(self) -> np.ndarray:
        """Position of the Transformable.


        Cartesian offset to the reference coordinate frame.

        Returns:
            Cartesian position x, y, z in m.

        Raises:
            ValueError:
                If `position` is not a valid three-dimensional vector.
        """

        return self.__position


    @position.setter
    def position(self, value: np.ndarray) -> None:

        value = value.flatten()

        if len(value) != 3:
            raise ValueError("Value must be a three-dimensional vector")

        self.__position = value

    @property
    def orientation_rpy(self) -> Tuple[float, float, float]:

        return self.__orientation
    

    @property
    def reference(self) -> Optional[Transformable]:
        """Reference frame to this coordinate frame.

        Reference frames are the previous frame within the kinematic chain.

        Args:

            reference (Optional[Transformable]):
                The reference frame.
                `None` if the frame is considered a base coordinate frame.
        """

        return self.__reference_frame


    @reference.setter
    def reference(self, value: Transformable) -> None:

        # Remove the link to this frame from the respective reference if it is currently linked
        if self.reference is not None:
            self.reference.remove_link(self)

        # Establish new link to the requested reference frame
        self.__reference_frame = value
        value.add_link(self)

    @property
    def is_base(self) -> bool:
        """Is this frame a coordinate system base?

        Returns:
            Boolean indicator.
        """

        return self.__reference_frame is None


    def add_link(self, link: Transformable) -> None:
        """Establish a new link to acoordinate frame depending on this frame.


        Args:

            link (Transformable):
                The transformable frame to be registered.

        Raises:
            
            RuntimeError:
                If the `link` is already registered as a dependency.
        """

        if link in self.__linked_frames:
            raise RuntimeError("Transformable already registered as a link")

        self.__linked_frames.append(link)


    def remove_link(self, link: Transformable) -> None:
        """Remove an established link to this coordinate frame.
        
        Args:

            link (Transformable):
                The coordinate frame to be linked to this frame.

        Raises:

            RuntimeError:
                If the `link` is not currently registerd.
        """

        if link not in self.__linked_frames:
            raise RuntimeError("Transformable link not registered")

        self.__linked_frames.remove(link)
