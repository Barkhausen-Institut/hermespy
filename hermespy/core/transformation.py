# -*- coding: utf-8 -*-
"""
=================================
Coordinate System Transformations
=================================
"""

from __future__ import annotations
from abc import ABCMeta, abstractmethod
from functools import cached_property
from typing import overload, Set, Type

import numpy as np
from numba import jit
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode, Node

from .factory import Serializable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Direction(np.ndarray):
    """A cartesian unit norm vector pointing towards a direction."""

    @staticmethod
    @jit(nopython=True)
    def __from_spherical(angles: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Generate a cartesian unit vector from spherical coordinates.

        Args:

            angles (np.ndarray):
                Azimuth and zenith in radians.

        Returns: The unit vector.
        """

        cos = np.cos(angles)
        sin = np.sin(angles)

        unit_vector = np.array([sin[1] * cos[0], sin[1] * sin[0], cos[1]], dtype=np.float_)
        return unit_vector

    @classmethod
    def From_Spherical(cls: Type[Direction], azimuth: float, zenith: float) -> Direction:
        """Initialize a direction from spherical parameters.

        Args:

            azimuth (float):
                Azimuth angle in radians.


            zenith (float):
                Zenith angle in radians.

        Returns: The initialized direction.
        """

        direction = cls.__from_spherical(np.array([azimuth, zenith], dtype=float)).view(cls)
        return direction

    @jit(nopython=True)
    def __to_spherical(unit_vector: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Transform a unit vector to spherical coordinates.

        Args:

            unit_vector (np.ndarray):
                Cartesian numpy vector.

        Returns: An array representing azimuth and zenith angles in radians.
        """

        # Equation 7.1-8 of ETSI TR 138901 v17
        azimuth = np.arctan2(unit_vector[1], unit_vector[0])
        # azimuth = np.angle(unit_vector[0] + 1j * unit_vector[1])

        # Equation 7.1-7 of ETSI TR 138901 v17
        # Only valid if the direction has been normalized!!!!
        zenith = np.arccos(unit_vector[2])

        return np.array([azimuth, zenith], dtype=np.float_)

    def to_spherical(self) -> np.ndarray:
        """Represent the direction as spherical coordinates.

        Returns: An array representing azimuth and zenith angles in radians.
        """

        return self.__to_spherical().view(np.ndarray)

    @classmethod
    def From_Cartesian(
        cls: Type[Direction], vector: np.ndarray, normalize: bool = False
    ) -> Direction:
        """Initialize a direction from a cartesian vector.

        Raises:

            ValueError: If `vector` does not represent a valid cartesian vector.

        Returns: The initialized direction.
        """

        vector = vector.flatten().astype(np.float_)
        ndmin = vector.size
        if ndmin > 3:
            raise ValueError("Vector is not a valid cartesian vector")

        if normalize:
            norm = np.linalg.norm(vector)

            if norm == 0:
                raise ValueError("Zero-vectors cannot be normalized")

            vector /= norm

        unit_vector = np.zeros(3, dtype=np.float_)
        unit_vector[:ndmin] = vector

        return unit_vector.view(Direction)


class Transformation(np.ndarray, Serializable):
    """Coordinate system transformation."""

    yaml_tag = "Transformation"

    @property
    def translation(self) -> np.ndarray:
        return self[:3, 3].view(np.ndarray)

    @translation.setter
    def translation(self, value: np.ndarray) -> None:
        value = value.flatten()
        if len(value) != 3:
            raise ValueError("Translations must be three-dimensional cartesian vectors")

        self[:3, 3] = value

    @property
    def rotation_rpy(self) -> np.ndarray:
        """Orientation in Roll, Pitch and Yaw Angles.

        Returns: Roll, Pitch and Yaw in Radians.
        """

        c = (self[0, 0] ** 2 + self[1, 0] ** 2) ** 0.5

        if c != 0:
            rpy = np.arctan2(
                [self[2, 1] / c, -self[2, 0], self[1, 0] / c], [self[2, 2] / c, c, self[0, 0] / c]
            )

        else:
            rpy = np.array([np.arctan2(self[0, 1], self[1, 1]), 0.5 * np.pi, 0], dtype=np.float_)

        return rpy.view(np.ndarray)

    @rotation_rpy.setter
    def rotation_rpy(self, value: np.ndarray) -> None:
        self[:3, :3] = self._rotation_from_rpy(value)

    @classmethod
    def No(cls: Type[Transformation]) -> Transformation:
        return np.eye(4, 4, dtype=float).view(cls)

    @staticmethod
    @jit(nopython=True)
    def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Calculate a rotation matrix from roll pitch yaw angles.

        Args:

            rpy (np.ndarray):
                Numpy vector of length 3 representing roll pitch and yaw in radians.

        Returns:

            A :math:`3 \\times 3` numpy matrix representing the rotation.
        """

        # Compute rotational transformation portion
        cos = np.cos(rpy)
        sin = np.sin(rpy)
        rotation = np.array(
            [
                [
                    cos[2] * cos[1],
                    cos[2] * sin[1] * sin[0] - sin[2] * cos[0],
                    cos[2] * sin[1] * cos[0] + sin[2] * sin[0],
                ],
                [
                    sin[2] * cos[1],
                    sin[2] * sin[1] * sin[0] + cos[2] * cos[0],
                    sin[2] * sin[1] * cos[0] - cos[2] * sin[0],
                ],
                [-sin[1], cos[1] * sin[0], cos[1] * cos[0]],
            ],
            dtype=np.float_,
        )

        return rotation

    @classmethod
    def From_RPY(cls: Type[Transformation], rpy: np.ndarray, pos: np.ndarray) -> Transformation:
        # Generate empty transformation matrix
        transformation = np.empty((4, 4), dtype=float)

        # Compute rotational transformation portion
        transformation[:3, :3] = cls._rotation_from_rpy(rpy)

        # Copy translational transformation portion
        transformation[0:3, 3] = pos

        # Fill in the static portion
        transformation[3, :] = (0, 0, 0, 1)

        # Return by view
        return transformation.view(cls)

    @classmethod
    def From_Translation(cls: Type[Transformation], translation: np.ndarray) -> Transformation:
        translation = translation.flatten()
        ndim = len(translation)

        if ndim > 3:
            raise ValueError("Translation must be a valid three-dimensional cartesian vector")

        # Generate non-rotating transformation matrix
        transformation = np.eye(4, 4, dtype=float)

        # Copy translational transformation portion
        transformation[0:ndim, 3] = translation

        # Return by view
        return transformation.view(cls)

    def transform_position(self, position: np.ndarray) -> np.ndarray:
        """Transform a cartesian position.

        Args:

            position (np.ndarray):
                Numpy array representing the cartesian position.

        Returns: The transformed position.

        Raises:

            ValueError: If `position` is invalid.
        """

        position = position.flatten()
        ndmin = len(position)

        if ndmin > 3:
            raise ValueError("Position must be a valid cartesian numpy vector")

        factor = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        factor[:ndmin] = position

        transformed_position = self.view(np.ndarray) @ factor
        return transformed_position[:ndmin]

    def rotate_direction(self, direction: np.ndarray) -> Direction:
        """Rotate a cartesian direction.

        Args:

            direction (np.ndarray):
                A directional vector.

        Returns: The transformed direction.

        Raises:

            ValueError: If `direction` is invalid.
        """

        direction = direction.flatten()
        ndim = len(direction)
        if ndim > 3:
            raise ValueError("Direction must be a valid cartesian numpy vector")

        rotated_direction = np.tensordot(self[:ndim, :ndim], direction, axes=(1, 0))
        return Direction.From_Cartesian(rotated_direction)

    def transform_direction(self, direction: np.ndarray, normalize: bool = False) -> Direction:
        """Transform a direction.

        Args:

            direction (np.ndarray):
                Direction to be transformed.

            normalize (bool, optional):
                Normalize the resulting transformed direction to a unit norm vector.
                Disabled by default.

        Returns: The transformed direction.
        """

        return Direction.From_Cartesian(self.transform_position(direction), normalize=normalize)

    @classmethod
    def to_yaml(
        cls: Type[Transformation], representer: SafeRepresenter, node: Transformation
    ) -> MappingNode:
        state = {"translation": node.translation, "rotation": node.rotation_rpy}

        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(
        cls: Type[Transformation], constructor: SafeConstructor, node: Node
    ) -> Transformation:
        state = constructor.construct_mapping(node, deep=False)
        return cls.From_RPY(state.get("rotation", None), state.get("translation", None))


class TransformableLink(metaclass=ABCMeta):
    """Abstract base class of kinetmatic chain links."""

    __linked_frames: Set[TransformableLink]

    def __init__(self) -> None:
        self.__linked_frames = set()

    @property
    @abstractmethod
    def forwards_transformation(self) -> Transformation:
        """Transformation to convert local coordinates to global coordinates.

        Updated every time the kinematic chain's parameters change.

        Returns: The transformation.
        """
        ...  # pragma no cover

    @abstractmethod
    def _kinematics_updated(self) -> None:
        """Callback notifying the link that its kinematics have been updated."""
        ...  # pragma no cover

    @abstractmethod
    def set_base(self, base: TransformableLink | None) -> None:
        """Set the relative base coordinate frame of this link.

        Args:

            base (TransformableLink | None):
                The base to be coordinate frame to be set.
                `None` to detach the link.
        """
        ...  # pragma no cover

    @property
    def linked_frames(self) -> Set[TransformableLink]:
        return self.__linked_frames

    def add_link(self, link: Transformable) -> None:
        """Establish a new link to a coordinate frame depending on this frame.

        Args:

            link (Transformable):
                The transformable frame to be registered.
        """

        self.__linked_frames.add(link)
        link.set_base(self)

    def remove_link(self, link: Transformable, force_removal: bool = True) -> None:
        """Remove an established link to this coordinate frame.

        Args:

            link (Transformable):
                The coordinate frame to be linked to this frame.

            force_removal(bool, optional):
                Raise a RuntimeError if the link is not registered

        Raises:

            RuntimeError:
                If the `link` is not currently registerd and `force_removal` is enabled.
        """

        if link not in self.__linked_frames:
            if force_removal:
                raise RuntimeError("Transformable link not registered")

            else:
                return

        self.__linked_frames.remove(link)  # Remove the link from the list of links
        link.set_base(None)  # Notify the link that it's now detached


class TransformableBase(TransformableLink):
    """Base of kinematic chains."""

    @property  # @cached_property
    def forwards_transformation(self) -> Transformation:
        return Transformation.No()

    def _kinematics_updated(self) -> None:
        raise RuntimeError("Called base updated routine of a base, this should not be possibel")

    def set_base(self, base: TransformableLink | None) -> None:
        if base is not None:
            raise RuntimeError("A base link may not be assigned another link as base")


class Transformable(Serializable, TransformableLink):
    """Representation of a Coordinate Frame within a Kinematic Chain."""

    property_blacklist = {"pose"}

    __base: TransformableLink | None
    __pose: Transformation

    def __init__(self, pose: Transformation | None = None) -> None:
        """
        Args:

            pose (Transformation, optional):
                Transformation of the transformable with respect to its reference frame.
                By default, no transformation is considered, i.e. :meth:`Transformation.No`
        """

        # Init base class
        TransformableLink.__init__(self)

        self.__base = None
        self.pose = Transformation.No() if pose is None else pose

    @property
    def position(self) -> np.ndarray:
        """Position of the Transformable.

        Cartesian offset to the reference coordinate frame.

        Returns:
            Cartesian position x, y, z in m.

        Raises:

            ValueError: If `position` is not a valid three-dimensional vector.
        """

        return self.__pose.translation

    @position.setter
    def position(self, value: np.ndarray) -> None:
        if value.ndim != 1:
            raise ValueError("Position must be a vector")

        if len(value) > 3:
            raise ValueError("Position may have at most three dimensions")

        # Make vector 3D if less dimensions have been supplied
        if len(value) < 3:
            value = np.append(value, np.zeros(3 - len(value), dtype=float))

        self.__pose.translation = value
        self._kinematics_updated()

    @property
    def orientation(self) -> np.ndarray:
        """Orientation of the Transformable.

        Returns: The transformation in radians for roll, pitch and yaw.

        Raises:

            ValueError: If `orientation` is not a three-dimensional numpy vector.
        """

        return self.__pose.rotation_rpy

    @orientation.setter
    def orientation(self, value: np.ndarray) -> None:
        if value.ndim != 1 or len(value) != 3:
            raise ValueError("Orientation must be a three-dimensional vector")

        self.__pose.rotation_rpy = value
        self._kinematics_updated()

    def set_base(self, base: TransformableLink | None) -> None:
        if self.__base == base:
            return

        if self.__base is not None:
            self.__base.remove_link(self, force_removal=False)

        self.__base = base
        self._kinematics_updated()

        if self.__base is not None:
            self.__base.add_link(self)

    @property
    def global_position(self) -> np.ndarray:
        """Position of the represented object within the global coordinate system.

        Returns: Three-dimensional numpy vector representing the cartesian object coordinates.
        """

        return self.forwards_transformation.translation

    @property
    def global_orientation(self) -> np.ndarray:
        """Orientation of the represented object within the global coordinate system.

        Returns: Three-dimensional numpy vector representing roll, pitch and yaw in radians.
        """

        return self.forwards_transformation.rotation_rpy

    @property
    def is_base(self) -> bool:
        """Is this transformable acting as a base frame?

        Returns: Boolean indicator.
        """

        return self.__base is None

    @property
    def pose(self) -> Transformation:
        """Pose of the Transformable with respect to its reference link.

        Returns: The pose's transformation.
        """

        return self.__pose

    @pose.setter
    def pose(self, value: Transformation) -> None:
        self.__pose = value.copy()
        self._kinematics_updated()

    @cached_property
    def forwards_transformation(self) -> Transformation:
        if self.is_base:
            return self.pose

        transformation = self.__base.forwards_transformation @ self.pose
        return transformation.view(Transformation)

    @cached_property
    def backwards_transformation(self) -> Transformation:
        """Transformation to convert global coordinates to local coordinates.

        Updated every time the kinematic chain's parameters change.

        Returns: The transformation.
        """

        return np.linalg.inv(self.forwards_transformation).view(Transformation)

    @overload
    def to_local_coordinates(self, global_object: Transformable) -> Transformation:
        ...  # pragma no cover

    @overload
    def to_local_coordinates(self, global_object: Transformation) -> Transformation:
        ...  # pragma no cover

    @overload
    def to_local_coordinates(
        self, position: np.ndarray, orientation: np.ndarray | None = None
    ) -> Transformation:
        ...  # pragma no cover

    def to_local_coordinates(self, arg_0: Transformable | Transformation | np.ndarray, arg_1: np.ndarray | None = None) -> Transformation:  # type: ignore
        if isinstance(arg_0, Transformable):
            arg_0 = arg_0.forwards_transformation

        elif isinstance(arg_0, Transformation):
            ...  # pragma no cover

        elif isinstance(arg_0, np.ndarray):
            arg_1 = np.zeros(3, dtype=float) if arg_1 is None else arg_1
            arg_0 = Transformation.From_RPY(arg_1, arg_0)

        else:
            raise ValueError("Unknown type of first argument")

        local_transformation = self.backwards_transformation @ arg_0
        return local_transformation.view(Transformation)

    def _kinematics_updated(self) -> None:
        # Clear the cached forwards transformation if the base has been updated
        if "forwards_transformation" in self.__dict__:
            del self.forwards_transformation

        if "backwards_transformation" in self.__dict__:
            del self.backwards_transformation

        for frame in self.linked_frames:
            frame._kinematics_updated()
