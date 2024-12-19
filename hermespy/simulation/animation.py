# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Slerp, Rotation

from hermespy.core import Serializable, Transformation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TrajectorySample(object):
    """Dataclass for a single pose sample within a trajectory."""

    def __init__(self, timestamp: float, pose: Transformation, velocity: np.ndarray) -> None:
        """
        Args:

            timestamp (float): Time at which the trajectory was sampled in seconds.
            pose (Timestamp): Pose of the object at the given time.
            velocity (numpy.ndarray): Velocity of the object at the given time.
        """

        # Initialize class attributes
        self.__timestamp = timestamp
        self.__pose = pose
        self.__velocity = velocity

    @property
    def timestamp(self) -> float:
        """Time at which the trajectory was sampled in seconds."""

        return self.__timestamp

    @property
    def pose(self) -> Transformation:
        """Pose of the object at the given time."""

        return self.__pose

    @property
    def velocity(self) -> np.ndarray:
        """Velocity of the object at the given time."""

        return self.__velocity


class Trajectory(ABC):
    """Base class for motion trajectories of moveable objects within simulation scenarios."""

    # lookat attributes
    _lookat_flag: bool = False
    _lookat_target: Trajectory = None
    _lookat_up: np.ndarray = np.array([0.0, 1.0, 0.0], float)  # (3,), float

    def lookat(self, target: Trajectory, up: np.ndarray = np.array([0.0, 1.0, 0.0], float)) -> None:
        """Set a target to look at and track.

        Args:
            target (Trajectory): Target trajectory.
            up (numpy.ndarray): Up/sky/head/ceiling global unit vector. Defaults to [0., 1., 0.].
        """

        self._lookat_flag = True
        self._lookat_target = target
        self._lookat_up = up

    def lookat_disable(self) -> None:
        self._lookat_flag = False

    def lookat_enable(self) -> None:
        if self._lookat_target is None:
            raise RuntimeError('Cannot enable lookat whithout a target. Use the "lookat" method.')

        self._lookat_flag = True

    @property
    @abstractmethod
    def max_timestamp(self) -> float:
        """Maximum timestamp of the trajectory in seconds.

        For times greater than this value the represented object's pose is assumed
        to be constant.
        """
        ...  # pragma: no cover

    @abstractmethod
    def sample_velocity(self, timestamp: float) -> np.ndarray:
        """Sample the trajectory's velocity.

        Args:
            timestamp (float): Time at which to sample the trajectory in seconds.

        Returns: A sample of the trajectory's velocity (vector (3,) of floats).
        """
        ...  # pragma: no cover

    @abstractmethod
    def sample_translation(self, timestamp: float) -> np.ndarray:
        """Sample the trajectory's translation.

        Args:
            timestamp (float): Time at which to sample the trajectory in seconds.

        Returns: A sample of the trajectory's translation (vector (3,) of floats).
        """
        ...  # pragma: no cover

    @abstractmethod
    def sample_orientation(self, timestamp: float) -> np.ndarray:
        """Sample the trajectory's orientation. Does not consider lookat.

        Args:
            timestamp (float): Time at which to sample the trajectory in seconds.

        Returns: A sample of the trajectory's orientation matrix (matrix (3, 3) of float).
        """
        ...  # pragma: no cover

    def sample(self, timestamp: float) -> TrajectorySample:
        """Sample the trajectory at a given point in time.

        Args:
            timestamp (float): Time at which to sample the trajectory in seconds.

        Returns: A sample of the trajectory.
        """

        # Init transformation and sample position
        transformation = np.eye(4, 4, dtype=float).view(Transformation)
        transformation[:3, 3] = self.sample_translation(timestamp)

        # Sample orientation
        if self._lookat_flag:
            target_translation = self._lookat_target.sample_translation(timestamp)
            transformation = transformation.lookat(target_translation, self._lookat_up)
        else:
            transformation[:3, :3] = self.sample_orientation(timestamp)

        return TrajectorySample(timestamp, transformation, self.sample_velocity(timestamp))


class LinearTrajectory(Trajectory):
    """A helper class generating a linear trajectory between two poses."""

    __initial_pose: Transformation
    __final_pose: Transformation
    __duration: float
    __start: float

    def __init__(
        self,
        initial_pose: Transformation,
        final_pose: Transformation,
        duration: float,
        start: float = 0.0,
    ) -> None:

        # Verify initialization parameters
        if duration <= 0:
            raise ValueError("Duration must be greater than zero")

        if start < 0.0:
            raise ValueError("Start time must be non-negative")

        # Initialize class attributes
        self.__initial_pose = initial_pose
        self.__duration = duration
        self.__start = start

        # Infer velocity from start and end poses
        self.__velocity = (final_pose.translation - initial_pose.translation) / duration
        rotations = Rotation.from_matrix([initial_pose[:3, :3], final_pose[:3, :3]])
        self.__slerp = Slerp([start, start + duration], rotations)

    @property
    def max_timestamp(self) -> float:
        return self.__start + self.__duration

    def sample_velocity(self, timestamp: float) -> np.ndarray:
        if timestamp >= self.__start and timestamp < self.__start + self.__duration:
            return self.__velocity
        else:
            return np.zeros(3, np.float64)

    def sample_translation(self, timestamp: float) -> np.ndarray:
        t = np.clip(timestamp, self.__start, self.__start + self.__duration) - self.__start
        return self.__initial_pose.translation + t * self.__velocity

    def sample_orientation(self, timestamp: float) -> np.ndarray:
        t = np.clip(timestamp, self.__start, self.__start + self.__duration)
        return self.__slerp(t).as_matrix()


class StaticTrajectory(Serializable, Trajectory):
    """A helper class generating a static trajectory."""

    yaml_tag = "Static"

    __pose: Transformation
    __velocity: np.ndarray

    def __init__(
        self, pose: Transformation | None = None, velocity: np.ndarray | None = None
    ) -> None:

        # Initialize class attributes
        self.__pose = Transformation.No() if pose is None else pose
        self.__velocity = np.zeros(3, dtype=np.float64) if velocity is None else velocity

    @property
    def pose(self) -> Transformation:
        """Static pose of the object."""

        return self.__pose

    @property
    def velocity(self) -> np.ndarray:
        """Static velocity of the object."""

        return self.__velocity

    @property
    def max_timestamp(self) -> float:
        return 0.0

    def sample_velocity(self, timestamp: float) -> np.ndarray:
        return self.__velocity

    def sample_translation(self, timestamp: float) -> np.ndarray:
        return self.__pose.translation

    def sample_orientation(self, timestamp: float) -> np.ndarray:
        return self.__pose[:3, :3]

    def sample(self, timestamp: float) -> TrajectorySample:
        return TrajectorySample(timestamp, self.__pose, self.__velocity)


class Moveable(object):
    """Base class of moveable objects within simulation scenarios."""

    __trajectory: Trajectory

    def __init__(self, trajectory: Trajectory | None) -> None:
        """
        Args:

            trajectory (Trajectory, optional):
                Trajectory this object is following.
                If not provided, the object is assumed to be static.
        """

        # Initialize class attributes
        self.__trajectory = StaticTrajectory() if trajectory is None else trajectory

    @property
    def trajectory(self) -> Trajectory:
        """Motion trajectory this object is following."""

        return self.__trajectory

    @trajectory.setter
    def trajectory(self, trajectory: Trajectory) -> None:
        self.__trajectory = trajectory

    @property
    def max_trajectory_timestamp(self) -> float:
        """Maximum timestamp of this object's motion trajectory."""

        return self.__trajectory.max_timestamp


class BITrajectoryB(Trajectory):
    """Easter-egg class for writing a lower-case b as a trajectory."""

    def __init__(self, height: float, duration: float) -> None:
        """
        Args:

            height (float): Height of the b in meters.
            duration (float): Duration of the b in seconds.
        """

        # Initialize base class
        Trajectory.__init__(self)

        # Initialize class attributes
        self.__height = height
        self.__duration = duration

    @property
    def max_timestamp(self) -> float:
        return self.__duration

    def __start_end_point_time(self, timestamp: float) -> tuple:
        """Returns start start_point, end_point, start_time and end_time of the straight path section."""
        if timestamp > self.__duration * 4 / 5:
            start_point = self.__height * np.array([0.5, 0.5, 0], dtype=np.float64)
            end_point = self.__height * np.array([0, 0.5, 0], dtype=np.float64)
            start_time = self.__duration * 4 / 5
            end_time = self.__duration

        elif timestamp > self.__duration * 3 / 5:
            start_point = self.__height * np.array([0.5, 0, 0], dtype=np.float64)
            end_point = self.__height * np.array([0.5, 0.5, 0], dtype=np.float64)
            start_time = self.__duration * 3 / 5
            end_time = self.__duration * 4 / 5

        elif timestamp > self.__duration * 2 / 5:
            start_point = np.array([0, 0, 0], dtype=np.float64)
            end_point = self.__height * np.array([0.5, 0, 0], dtype=np.float64)
            start_time = self.__duration * 2 / 5
            end_time = self.__duration * 3 / 5

        else:
            start_point = np.array([0, self.__height, 0], dtype=np.float64)
            end_point = np.array([0, 0, 0], dtype=np.float64)
            start_time = 0
            end_time = self.__duration * 2 / 5

        return start_point, end_point, start_time, end_time

    def sample_velocity(self, timestamp: float) -> np.ndarray:
        start_point, end_point, start_time, end_time = self.__start_end_point_time(timestamp)
        if timestamp <= start_time or timestamp >= end_time:
            return np.zeros(3, np.float64)
        return (end_point - start_point) / end_time - start_time

    def sample_translation(self, timestamp: float) -> np.ndarray:
        start_point, _, start_time, end_time = self.__start_end_point_time(timestamp)
        if timestamp <= start_time or timestamp >= end_time:
            return np.array([0, 0.5 * self.__height, 0], np.float64)
        return start_point + self.sample_velocity(timestamp) * (timestamp - start_time)

    def sample_orientation(self, timestamp: float) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], float)
