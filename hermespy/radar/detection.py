# -*- coding: utf-8 -*-
"""
===============
Radar Detection
===============
"""

from __future__ import annotations
from abc import abstractmethod
from math import cos, sin
from typing import List, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter

from hermespy.core import Serializable, Visualizable
from .cube import RadarCube

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PointDetection(object):
    """A single radar point detection."""

    __position: np.ndarray  # Cartesian position of the detection in m
    __velocity: np.ndarray  # Velocity of the detection in m/s
    __power: float  # Power of the detection

    def __init__(self, position: np.ndarray, velocity: np.ndarray, power: float) -> None:
        """
        Args:

            position (np.ndarray):
                Cartesian position of the detection in cartesian coordinates.

            velocity (np.ndarray):
                Velocity vector of the detection in m/s

            power (float):
                Power of the detection.

        Raises:
            ValueError:
                If `position` is not three-dimensional.
                If `velocity` is not three-dimensional.
                If `power` is smaller or equal to zero.
        """

        if position.ndim != 1 or len(position) != 3:
            raise ValueError("Position must be a three-dimensional vector.")

        if velocity.ndim != 1 or len(velocity) != 3:
            raise ValueError("Velocity must be a three-dimensional vector.")

        if power <= 0.0:
            raise ValueError("Detected power must be greater than zero")

        self.__position = position
        self.__velocity = velocity
        self.__power = power

    @classmethod
    def FromSpherical(cls: Type[PointDetection], zenith: float, azimuth: float, velocity: float, range: float, power: float) -> PointDetection:
        """Generate a point detection from radar cube spherical coordinates.

        Args:

            zenith (float):
                Zenith angle in Radians.

            azimuth (float):
                Azimuth angle in Radians.

            velocity (float):
                Velocity from / to the coordinate system origin in m/s.

            range (float):
                Point distance to coordiante system origin in m/s.

            power (float):
                Point power indicator.
        """

        normal_vector = np.array([cos(azimuth) * sin(zenith), sin(azimuth) * sin(zenith), cos(zenith)], dtype=float)

        position = range * normal_vector
        velocity_vector = velocity * normal_vector

        return cls(position=position, velocity=velocity_vector, power=power)

    @property
    def position(self) -> np.ndarray:
        """Position of the detection.

        Returns:
            np.ndarray: Cartesian position in m.
        """

        return self.__position

    @property
    def velocity(self) -> np.ndarray:
        """Velocity of the detection.

        Returns:
            np.ndarray: Velocity vector in m/s.
        """

        return self.__velocity

    @property
    def power(self) -> float:
        """Detected power.

        Returns:
            float: Power.
        """

        return self.__power


class RadarPointCloud(Visualizable):
    """A sparse radar point cloud."""

    __points: List[PointDetection]
    __max_range: float

    def __init__(self, max_range: float = 0.0) -> None:
        """

        Args:

            max_range (float, optional):
                Maximal represented range.
                Used for visualization.

        Raises:

            ValueError: For `max_range` smaller than zero.
        """

        if max_range < 0.0:
            raise ValueError(f"Maximal represented range must be greater or equal to zero (not {max_range})")

        # Initialize base class
        Visualizable.__init__(self)

        # Initialize class attributes
        self.__points = []
        self.__max_range = max_range

    @property
    def max_range(self) -> float:
        """Maximal represented range.

        Returns:
            Maximal range in m.
            Zero if unknown.
        """

        return self.__max_range

    @property
    def points(self) -> List[PointDetection]:
        """Points contained within the represented point cloud.

        Returns: List of represented points.
        """

        return self.__points

    @property
    def num_points(self) -> int:
        """Number of points contained within this point cloud.

        Returns:

            The number of points.
        """

        return len(self.__points)

    def add_point(self, point: PointDetection) -> None:
        """Add a new detection to the point cloud.

        Args:

            point (PointDetection):
                The detection to be added.
        """

        self.__points.append(point)

    @property
    def title(self) -> str:
        return "Radar Point Coud"

    def _new_axes(self) -> Tuple[plt.Figure, plt.Axes]:
        figure = plt.figure()
        axis = figure.add_subplot(projection="3d")

        return figure, axis

    def _plot(self, axes: plt.Axes) -> None:
        for point in self.points:
            position = point.position
            axes.scatter(position[0], position[1], position[2], marker="o", c=point.power, cmap="Greens")

        # Configure axes
        axes.set_xlim((-self.max_range, self.max_range))
        axes.set_ylim((0, self.max_range))
        axes.set_zlim((-self.max_range, self.max_range))
        axes.set_xlabel("X [m]")
        axes.set_ylabel("Z [m]")
        axes.set_zlabel("Y [m]")


class RadarDetector(object):
    """Base class for radar detection algorithms.

    Radar detection algorithms convert a radar cube to a radar point cloud.
    """

    @abstractmethod
    def detect(self, cube: RadarCube) -> RadarPointCloud:
        """Generate a point cloud from a radar cube.

        Args:

            cube (RadarCube):
                The radar cube to be processed.

        Returns:

            The resulting (usually sparse) point cloud.
        """
        ...  # Pragma no cover


class ThresholdDetector(RadarDetector, Serializable):
    """Extract points by a power threshold."""

    yaml_tag = "Threshold"

    __min_power: float  # Minmally required point power
    __normalize: bool
    __peak_detection: bool

    def __init__(self, min_power: float, normalize: bool = True, peak_detection: bool = True) -> None:
        """
        Args:

            min_power (float):
                Minmally required point power.

            normalize (bool, optional):
                Normalize the power during detection, so that min_power
                becomes a relative value betwee zero and one.
                Enabled by default.

            peak_detection (bool, optional):
                Run a peak detection algorithm to only extract points at power peaks.
                Enabled by default.
        """

        self.min_power = min_power
        self.__normalize = normalize
        self.__peak_detection = peak_detection

    @property
    def min_power(self) -> float:
        """Minimally required point power.

        Returns:

            Power (linear).

        Raises:

            ValueError: On powers smaller or equal to zero.
        """

        return self.__min_power

    @min_power.setter
    def min_power(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Point power threshold must be greater than zero")

        self.__min_power = value

    @property
    def normalize(self) -> bool:
        """Normalize cube power before detection.

        Returns: Enabled flag.
        """

        return self.__normalize

    @property
    def peak_detection(self) -> bool:
        """Run a peak search before detection.

        Returns: Enabled flag.
        """

        return self.__peak_detection

    def detect(self, cube: RadarCube) -> RadarPointCloud:
        # Extract cube data and normalize if the respective flag is enabled
        cube_max = cube.data.max()
        cube_data: np.ndarray = (cube.data - cube.data.min()) / cube_max if self.normalize and cube_max > 0.0 else cube.data

        # Filter the raw cube data for peaks
        if self.peak_detection:
            footprint = np.array([1, 1, 10])
            cube_data = maximum_filter(cube_data, footprint)

        # Apply the thershold and extract the coordinates matching the requirements
        detection_point_indices = np.argwhere(cube_data >= self.min_power)

        # Transform the detections to a point cloud
        cloud = RadarPointCloud(cube.range_bins.max())
        for point_indices in detection_point_indices:
            # angles = cube.angle_bins[point_indices[0]]
            # ToDo: Add angular domains
            azimuth, zenith = cube.angle_bins[point_indices[0]]
            velocity = cube.velocity_bins[point_indices[1]]
            range = cube.range_bins[point_indices[2]]
            power_indicator = cube.data[tuple(point_indices)]

            cloud.add_point(PointDetection.FromSpherical(zenith, azimuth, velocity, range, power_indicator))

        return cloud


class MaxDetector(RadarDetector, Serializable):
    """Extracts the maximum point from the radar cube."""

    yaml_tag = "Max"
    """YAML serialization tag."""

    def detect(self, cube: RadarCube) -> RadarPointCloud:
        # Find the maximum point
        point_index = np.unravel_index(np.argmax(cube.data), cube.data.shape)
        point_power = cube.data[point_index]

        # Return an empty point cloud if no maximum was found
        if point_power <= 0.0:
            return RadarPointCloud()

        angles_of_arrival = cube.angle_bins[point_index[0], :]
        velocity = cube.velocity_bins[point_index[1]]
        range = cube.range_bins[point_index[2]]

        cloud = RadarPointCloud(cube.range_bins.max())
        cloud.add_point(PointDetection.FromSpherical(angles_of_arrival[0], angles_of_arrival[1], velocity, range, point_power))

        return cloud
