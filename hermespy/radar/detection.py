# -*- coding: utf-8 -*-
"""
===============
Radar Detection
===============
"""

from __future__ import annotations
from abc import abstractmethod
from calendar import c
from math import cos, sin
from typing import List

import numpy as np
from scipy.ndimage.filters import maximum_filter

from hermespy.core import Serializable
from .cube import RadarCube

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PointDetection(object):
    """A single radar point detection."""

    __position: np.ndarray      # Cartesian position of the detection in m
    __velocity: np.ndarray      # Velocity of the detection in m/s
    __power: float              # Power of the detection

    def __init__(self,
                 position: np.ndarray,
                 velocity: np.ndarray,
                 power: float) -> None:
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

        if power <= 0.:
            raise ValueError("Detected power must be greater than zero")

        self.__position = position
        self.__velocity = velocity
        self.__power = power
        
    @classmethod
    def FromSpherical(cls,
                      zenith: float,
                      azimuth: float,
                      velocity: float,
                      range: float,
                      power: float) -> PointDetection:
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
        
        normal_vector = np.array([cos(azimuth) * sin(zenith),
                                  sin(azimuth) * sin(zenith),
                                  cos(zenith)], dtype=float)
        
        position = range * normal_vector
        velocity = velocity * normal_vector
        
        return cls(position=position, velocity=velocity, power=power)
        
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
    

class RadarPointCloud(object):
    """A sparse radar point cloud."""
    
    points: List[PointDetection]
    
    def __init__(self) -> None:
        
        self.points = []
        
    @property
    def num_points(self) -> int:
        """Number of points contained within this point cloud.
        
        Returns:
        
            The number of points.
        """
        
        return len(self.points)
        
    def add_point(self, point: PointDetection) -> None:
        """Add a new detection to the point cloud.
        
        Args:
        
            point (PointDetection):
                The detection to be added.
        """
        
        self.points.append(point)


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
    
    yaml_tag = u'Threshold'
    """YAML serialization tag."""
    
    __min_power: float      # Minmally required point power
    
    def __init__(self,
                 min_power: float) -> None:
        """Args:
        
            min_power (float):
                Minmally required point power.
        """
        
        self.min_power = min_power
    
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
        
        if value <= 0.:
            raise ValueError("Point power threshold must be greater than zero")
        
        self.__min_power = value
        
    def detect(self, cube: RadarCube) -> RadarPointCloud:
        
        # Filter the raw cube data for peaks
        footprint = np.array([1, 1, 10])
        filtered_image = maximum_filter(cube.data, footprint)
        
        # Apply the thershold and extract the coordinates matching the requirements
        detection_point_indices = np.argwhere(filtered_image >= self.min_power)
        
        # Transform the detections to a point cloud
        cloud = RadarPointCloud()
        for point_indices in detection_point_indices:
            
            # angles = cube.angle_bins[point_indices[0]]
            # ToDo: Add angular domains
            zenith = 0.
            azimuth = 0.
            velocity = cube.velocity_bins[point_indices[1]]
            range = cube.range_bins[point_indices[2]]
            power_indicator = cube.data[tuple(point_indices)]
            
            cloud.add_point(PointDetection.FromSpherical(zenith, azimuth, velocity, range, power_indicator))
            
        return cloud


class MaxDetector(RadarDetector, Serializable):
    """Extracts the maximum point from the radar cube."""
    
    yaml_tag = u'Max'
    """YAML serialization tag."""
    
    def detect(self, cube: RadarCube) -> RadarPointCloud:
        
        # Find the maximum point
        point_index = np.unravel_index(np.argmax(cube.data), cube.data.shape)
        point_power = cube.data[point_index]
        
        # Return an empty point cloud if no maximum was found
        if point_power <= 0.:
            return RadarPointCloud()
        
        angles_of_arrival = cube.angle_bins[point_index[0], :]
        velocity = cube.velocity_bins[point_index[1]]
        range = cube.range_bins[point_index[2]]
        
        cloud = RadarPointCloud()
        cloud.add_point(PointDetection.FromSpherical(angles_of_arrival[0], angles_of_arrival[1], velocity, range, point_power))
        
        return cloud
