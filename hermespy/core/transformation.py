# -*- coding: utf-8 -*-


from __future__ import annotations
from abc import ABCMeta, abstractmethod
from functools import cached_property
from typing import overload, Set, Type

import numpy as np
from numba import jit

from .factory import Serializable, SerializationProcess, DeserializationProcess

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
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

            angles:
                Azimuth and zenith in radians.

        Returns: The unit vector.
        """

        cos = np.cos(angles)
        sin = np.sin(angles)

        unit_vector = np.array([sin[1] * cos[0], sin[1] * sin[0], cos[1]], dtype=np.float64)
        return unit_vector

    @classmethod
    def From_Spherical(cls: Type[Direction], azimuth: float, zenith: float) -> Direction:
        """Initialize a direction from spherical parameters.

        Args:

            azimuth:
                Azimuth angle in radians.


            zenith:
                Zenith angle in radians.

        Returns: The initialized direction.
        """

        direction = cls.__from_spherical(np.array([azimuth, zenith], dtype=float)).view(cls)
        return direction

    @jit(nopython=True)
    def __to_spherical(unit_vector: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Transform a unit vector to spherical coordinates.

        Args:

            unit_vector: Cartesian numpy vector.

        Returns: An array representing azimuth and zenith angles in radians.
        """

        # Equation 7.1-8 of ETSI TR 138901 v17
        azimuth = np.arctan2(unit_vector[1], unit_vector[0])
        # azimuth = np.angle(unit_vector[0] + 1j * unit_vector[1])

        # Equation 7.1-7 of ETSI TR 138901 v17
        # Only valid if the direction has been normalized!!!!
        zenith = np.arccos(unit_vector[2])

        return np.array([azimuth, zenith], dtype=np.float64)

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

        vector = vector.flatten().astype(np.float64)
        ndmin = vector.size
        if ndmin > 3:
            raise ValueError("Vector is not a valid cartesian vector")

        if normalize:
            norm = np.linalg.norm(vector)

            if norm == 0:
                raise ValueError("Zero-vectors cannot be normalized")

            vector /= norm

        unit_vector = np.zeros(3, dtype=np.float64)
        unit_vector[:ndmin] = vector

        return unit_vector.view(Direction)


class Transformation(np.ndarray, Serializable):
    """Coordinate system transformation."""

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
            rpy = np.array([np.arctan2(self[0, 1], self[1, 1]), 0.5 * np.pi, 0], dtype=np.float64)

        return rpy.view(np.ndarray)

    @rotation_rpy.setter
    def rotation_rpy(self, value: np.ndarray) -> None:
        self[:3, :3] = self._rotation_from_rpy(value)

    @staticmethod
    @jit(nopython=True)
    def _rotation_from_quaternion(q: np.ndarray, normalize: bool) -> np.ndarray:  # pragma: no cover
        """Calculate a rotation matrix from a quaternion.

        Args:
            q: Quaternion in w, x, y, z representation.
            normalize: Normalize the quaternion before computing the rotation matrix.

        Returns: A :math:`3 \\times 3` numpy matrix representing the rotation.
        """

        q = q / np.linalg.norm(q) if normalize else q

        return np.array(
            [
                [
                    1 - 2 * (q[2] ** 2 + q[3] ** 2),
                    2 * (q[1] * q[2] - q[0] * q[3]),
                    2 * (q[0] * q[2] + q[1] * q[3]),
                ],
                [
                    2 * (q[1] * q[2] + q[0] * q[3]),
                    1 - 2 * (q[1] ** 2 + q[3] ** 2),
                    2 * (q[2] * q[3] - q[0] * q[1]),
                ],
                [
                    2 * (q[1] * q[3] - q[0] * q[2]),
                    2 * (q[0] * q[1] + q[2] * q[3]),
                    1 - 2 * (q[1] ** 2 + q[2] ** 2),
                ],
            ],
            dtype=np.float64,
        )

    @property
    def rotation_quaternion(self) -> np.ndarray:  # pragma: no cover
        """Orientation in Quaternion representation.

        A numpy vector representing w, x, y, z.
        """

        tr = np.trace(self[:3, :3])

        if tr > 0:
            S = 2 * (tr + 1.0) ** 0.5
            return np.array(
                [
                    0.25 / S,
                    (self[2, 1] - self[1, 2]) / S,
                    (self[0, 2] - self[2, 0]) / S,
                    (self[1, 0] - self[0, 1]) / S,
                ],
                dtype=np.float64,
            )

        if self[0, 0] > self[1, 1] and self[0, 0] > self[2, 2]:
            S = 2 * (1 + self[0, 0] - self[1, 1] - self[2, 2]) ** 0.5
            return np.array(
                [
                    (self[2, 1] - self[1, 2]) / S,
                    0.25 / S,
                    (self[0, 1] + self[1, 0]) / S,
                    (self[0, 2] + self[2, 0]) / S,
                ],
                dtype=np.float64,
            )

        if self[1, 1] > self[2, 2]:
            S = 2 * (1 + self[1, 1] - self[0, 0] - self[2, 2]) ** 0.5
            return np.array(
                [
                    (self[0, 2] - self[2, 0]) / S,
                    (self[0, 1] + self[1, 0]) / S,
                    0.25 / S,
                    (self[1, 2] + self[2, 1]) / S,
                ],
                dtype=np.float64,
            )

        S = 2 * (1 + self[2, 2] - self[0, 0] - self[1, 1]) ** 0.5
        return np.array(
            [
                (self[1, 0] - self[0, 1]) / S,
                (self[0, 2] + self[2, 0]) / S,
                (self[1, 2] + self[2, 1]) / S,
                0.25 / S,
            ],
            dtype=np.float64,
        )

    @rotation_quaternion.setter
    def rotation_quaternion(self, value: np.ndarray) -> None:
        self[:3, :3] = self._rotation_from_quaternion(value, False)

    @classmethod
    def No(cls: Type[Transformation]) -> Transformation:
        return np.eye(4, 4, dtype=float).view(cls)

    @staticmethod
    @jit(nopython=True)
    def _rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Calculate a rotation matrix from roll pitch yaw angles.

        Args:
            rpy: Numpy vector of length 3 representing roll pitch and yaw in radians.

        Returns:  A :math:`3 \\times 3` numpy matrix representing the rotation.
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
            dtype=np.float64,
        )

        return rotation

    @classmethod
    def From_Quaternion(
        cls: Type[Transformation],
        quaternion: np.ndarray,
        pos: np.ndarray,
        normalize_quaternion: bool = True,
    ) -> Transformation:
        """Initialize a transformation from a quaternion.

        Args:
            quaternion: Quaternion in w, x, y, z representation.
            pos: Cartesian position in m.
            normalize:
                Normalize the quaternion before computing the rotation matrix.
                Enabled by default.

        Returns: The initialized transformation.
        """

        # Generate empty transformation matrix
        transformation = np.empty((4, 4), dtype=float)

        # Compute rotational transformation portion
        transformation[:3, :3] = Transformation._rotation_from_quaternion(
            quaternion, normalize_quaternion
        )

        # Copy translational transformation portion
        transformation[0:3, 3] = pos

        # Fill in the static portion
        transformation[3, :] = (0, 0, 0, 1)

        # Return by view
        return transformation.view(cls)

    @classmethod
    def From_RPY(cls: Type[Transformation], rpy: np.ndarray, pos: np.ndarray) -> Transformation:
        """Initialize a transformation from roll pitch yaw angles.

        Args:
            rpy: Roll, pitch and yaw angles in radians.
            pos: Cartesian position in m.

        Returns: The initialized transformation.
        """

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
    def From_Translation(
        cls: Type[Transformation], translation: np.typing.ArrayLike
    ) -> Transformation:
        translation = np.asarray(translation).flatten()
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
            position: Numpy array representing the cartesian position.

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
            direction: A directional vector.

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
            direction: Direction to be transformed.
            normalize:
                Normalize the resulting transformed direction to a unit norm vector.
                Disabled by default.

        Returns: The transformed direction.
        """

        return Direction.From_Cartesian(self.rotate_direction(direction), normalize=normalize)

    def invert(self) -> Transformation:
        """Invert the transformation.

        Returns: The inverted transformation.
        """

        return np.linalg.inv(self).view(Transformation)

    def lookat(
        self,
        target: np.ndarray = np.array([0.0, 0.0, 0.0], float),
        up: np.ndarray = np.array([0.0, 1.0, 0.0], float),
    ) -> Transformation:
        """Rotate and loook at the given coordinates. Modifies `orientation` property.

        Args:
            target:
                Cartesean coordinates to look at.
                Defaults to np.array([0., 0., 0.], float)
            up:
                Global catesean sky vector.
                Defines the upward direction of the local viewport.
                Defaults to np.array([0., 1., 0.], float)

        Returns: This modified Transformation.
        """

        # Validate arguments
        target_ = np.asarray(target)
        if target_.shape != (3,):
            raise ValueError(
                f"Got target of an unexpected shape (expected (3,), got {target_.shape})"
            )
        up_ = np.asarray(up)
        if up_.shape != (3,):
            raise ValueError(f"Got up of an unexpected shape (expected (3,), got {up_.shape})")
        up_ /= np.linalg.norm(up_)

        # Calculate new orientation
        # forward vector
        pos = self.translation
        f = target_ - pos
        f_norm = np.linalg.norm(f)
        f = f / f_norm if f_norm != 0.0 else pos  # normalize
        # side/right vector
        s = np.cross(up_, f)
        s_norm = np.linalg.norm(s)
        s = s / s_norm if s_norm != 0.0 else up_  # normalize
        # up vector
        u = np.cross(f, s)
        # Calcualte the new transformation matrix
        self[:3, 0] = s
        self[:3, 1] = u
        self[:3, 2] = f
        self[3, :] = [0.0, 0.0, 0.0, 1.0]

        return self

    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_array(self, "matrix")

    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> Transformation:
        return process.deserialize_array("matrix", np.float64).view(Transformation)


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
            base:
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
            link:  The transformable frame to be registered.
        """

        self.__linked_frames.add(link)
        link.set_base(self)

    def remove_link(self, link: Transformable, force_removal: bool = True) -> None:
        """Remove an established link to this coordinate frame.

        Args:
            link: The coordinate frame to be linked to this frame.
            force_removal: Raise a RuntimeError if the link is not registered

        Raises:
            RuntimeError: If the `link` is not currently registerd and `force_removal` is enabled.
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

            pose:
                Transformation of the transformable with respect to its reference frame.
                By default, no transformation is considered.
        """

        # Init base class
        TransformableLink.__init__(self)

        self.__base = None
        self.pose = Transformation.No() if pose is None else pose

    @property
    def position(self) -> np.ndarray:
        """Position of the transformable.

        Cartesian x, y, z offset to the reference coordinate frame in m.

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

        The transformation in radians for roll, pitch and yaw.

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

        Three-dimensional numpy vector representing the cartesian object coordinates.
        """

        return self.forwards_transformation.translation

    @property
    def global_orientation(self) -> np.ndarray:
        """Orientation of the represented object within the global coordinate system.

        Three-dimensional numpy vector representing roll, pitch and yaw in radians.
        """

        return self.forwards_transformation.rotation_rpy

    @property
    def is_base(self) -> bool:
        """Is this transformable acting as a base frame?"""

        return self.__base is None

    @property
    def pose(self) -> Transformation:
        """Pose of the Transformable with respect to its reference link"""

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

        return self.forwards_transformation.invert()

    @overload
    def to_local_coordinates(
        self, global_object: Transformable
    ) -> Transformation: ...  # pragma no cover

    @overload
    def to_local_coordinates(
        self, global_object: Transformation
    ) -> Transformation: ...  # pragma no cover

    @overload
    def to_local_coordinates(
        self, position: np.ndarray, orientation: np.ndarray | None = None
    ) -> Transformation: ...  # pragma no cover

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

    def lookat(
        self,
        target: np.ndarray = np.array([0.0, 0.0, 0.0], float),
        up: np.ndarray = np.array([0.0, 1.0, 0.0], float),
    ) -> None:
        """Rotate and loook at the given coordinates. Modifies `orientation` property.

        Args:
            target:
                Cartesean coordinates to look at.
                Defaults to numpy.ndarray([0., 0., 0.], float).
            up:
                Global catesean sky vector.
                Defines the upward direction of the local viewport.
                Defaults to numpy.ndarray([0., 1., 0.], float).
        """

        self.pose.lookat(target, up)

    def _kinematics_updated(self) -> None:
        # Clear the cached forwards transformation if the base has been updated
        if "forwards_transformation" in self.__dict__:
            del self.forwards_transformation

        if "backwards_transformation" in self.__dict__:
            del self.backwards_transformation

        for frame in self.linked_frames:
            frame._kinematics_updated()

    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_array(self.pose, "pose")

    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> Transformable:
        return cls(process.deserialize_array("pose", np.float64).view(Transformation))
