# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from hermespy.core import Serializable, Transformation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
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
            velocity (np.ndarray): Velocity of the object at the given time.
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

    @property
    @abstractmethod
    def max_timestamp(self) -> float:
        """Maximum timestamp of the trajectory in seconds.

        For times greater than this value the represented object's pose is assumed
        to be constant.
        """
        ...  # pragma: no cover

    @abstractmethod
    def sample(self, timestamp: float) -> TrajectorySample:
        """Sample the trajectory at a given point in time.

        Args:

            timestamp (float): Time at which to sample the trajectory in seconds.

        Returns: A sample of the trajectory.
        """
        ...  # pragma: no cover


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
        self.__final_pose = final_pose
        self.__duration = duration
        self.__start = start

        # Infer velocity from start and end poses
        self.__velocity = (final_pose.translation - initial_pose.translation) / duration
        self.__initial_quaternion = initial_pose.rotation_quaternion
        self.__quaternion_velocity = (
            final_pose.rotation_quaternion - initial_pose.rotation_quaternion
        ) / duration

    @property
    def max_timestamp(self) -> float:
        return self.__start + self.__duration

    def sample(self, timestamp: float) -> TrajectorySample:

        # If the timestamp is outside the trajectory, return the initial or final pose
        if timestamp < self.__start:
            return TrajectorySample(timestamp, self.__initial_pose, np.zeros(3, dtype=np.float_))

        if timestamp >= self.__start + self.__duration:
            return TrajectorySample(timestamp, self.__final_pose, np.zeros(3, dtype=np.float_))

        # Interpolate orientation and position
        t = timestamp - self.__start
        orientation = self.__initial_quaternion + t * self.__quaternion_velocity
        translation = self.__initial_pose.translation + t * self.__velocity
        transformation = Transformation.From_Quaternion(orientation, translation)

        return TrajectorySample(timestamp, transformation, self.__velocity)


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
        self.__velocity = np.zeros(3, dtype=np.float_) if velocity is None else velocity

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

    def sample(self, timestamp: float) -> TrajectorySample:

        if timestamp >= self.__duration:
            return TrajectorySample(
                timestamp,
                Transformation.From_Translation(
                    np.array([0, 0.5 * self.__height, 0], dtype=np.float_)
                ),
                np.zeros(3, dtype=np.float_),
            )

        if timestamp > self.__duration * 4 / 5:
            start_point = self.__height * np.array([0.5, 0.5, 0], dtype=np.float_)
            end_point = self.__height * np.array([0, 0.5, 0], dtype=np.float_)
            start_time = self.__duration * 4 / 5
            end_time = self.__duration

        elif timestamp > self.__duration * 3 / 5:
            start_point = self.__height * np.array([0.5, 0, 0], dtype=np.float_)
            end_point = self.__height * np.array([0.5, 0.5, 0], dtype=np.float_)
            start_time = self.__duration * 3 / 5
            end_time = self.__duration * 4 / 5

        elif timestamp > self.__duration * 2 / 5:
            start_point = np.array([0, 0, 0], dtype=np.float_)
            end_point = self.__height * np.array([0.5, 0, 0], dtype=np.float_)
            start_time = self.__duration * 2 / 5
            end_time = self.__duration * 3 / 5

        else:
            start_point = np.array([0, self.__height, 0], dtype=np.float_)
            end_point = np.array([0, 0, 0], dtype=np.float_)
            start_time = 0
            end_time = self.__duration * 2 / 5

        leg_duration = end_time - start_time
        velocity = (end_point - start_point) / leg_duration
        interpolated_position = start_point + velocity * (timestamp - start_time)

        return TrajectorySample(
            timestamp, Transformation.From_Translation(interpolated_position), velocity
        )
