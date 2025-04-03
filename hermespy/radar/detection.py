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
from typing_extensions import override

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from scipy.ndimage import generate_binary_structure, maximum_filter
from scipy.signal import convolve

from hermespy.core import (
    ScatterVisualization,
    Serializable,
    VAT,
    Visualizable,
    SerializationProcess,
    DeserializationProcess,
)
from .cube import RadarCube

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Egor Achkasov"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PointDetection(Serializable):
    """A single radar point detection."""

    __position: np.ndarray  # Cartesian position of the detection in m
    __velocity: np.ndarray  # Velocity of the detection in m/s
    __power: float  # Power of the detection

    def __init__(self, position: np.ndarray, velocity: np.ndarray, power: float) -> None:
        """
        Args:

            position:
                Cartesian position of the detection in cartesian coordinates.

            velocity:
                Velocity vector of the detection in m/s

            power:
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
    def FromSpherical(
        cls: Type[PointDetection],
        zenith: float,
        azimuth: float,
        range: float,
        velocity: float,
        power: float,
    ) -> PointDetection:
        """Generate a point detection from radar cube spherical coordinates.

        Args:

            zenith:
                Zenith angle in Radians.

            azimuth:
                Azimuth angle in Radians.

            range:
                Point distance to coordiante system origin in m/s.

            velocity:
                Velocity from / to the coordinate system origin in m/s.

            power:
                Point power indicator.
        """

        normal_vector = np.array(
            [cos(azimuth) * sin(zenith), sin(azimuth) * sin(zenith), cos(zenith)], dtype=float
        )

        position = range * normal_vector
        velocity_vector = velocity * normal_vector

        return cls(position=position, velocity=velocity_vector, power=power)

    @property
    def position(self) -> np.ndarray:
        """Position of the detection.

        Returns: Cartesian position in m.
        """

        return self.__position

    @property
    def velocity(self) -> np.ndarray:
        """Velocity of the detection.

        Returns: Velocity vector in m/s.
        """

        return self.__velocity

    @property
    def power(self) -> float:
        """Detected power.

        Returns:
            float: Power.
        """

        return self.__power

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, PointDetection):
            return NotImplemented
        return (
            bool(np.all(self.position == __value.position))
            and bool(np.all(self.velocity == __value.velocity))
            and self.power == __value.power
        )

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_array(self.position, "position")
        process.serialize_array(self.velocity, "velocity")
        process.serialize_floating(self.power, "power")

    @classmethod
    @override
    def Deserialize(cls: Type[PointDetection], process: DeserializationProcess) -> PointDetection:
        position = process.deserialize_array("position", np.float64)
        velocity = process.deserialize_array("velocity", np.float64)
        power = process.deserialize_floating("power")
        return cls(position, velocity, power)


class RadarPointCloud(Serializable, Visualizable[ScatterVisualization]):
    """A sparse radar point cloud."""

    __points: List[PointDetection]
    __max_range: float

    def __init__(self, max_range: float = 0.0) -> None:
        """

        Args:

            max_range:
                Maximal represented range.
                Used for visualization.

        Raises:

            ValueError: For `max_range` smaller than zero.
        """

        if max_range <= 0.0:
            raise ValueError(
                f"Maximal represented range must be greater than zero (not {max_range})"
            )

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

        Returns: The number of points.
        """

        return len(self.__points)

    def add_point(self, point: PointDetection) -> None:
        """Add a new detection to the point cloud.

        Args:

            point: The detection to be added.
        """

        self.__points.append(point)

    @property
    def title(self) -> str:
        return "Radar Point Coud"

    def create_figure(self, **kwargs) -> Tuple[plt.Figure, VAT]:
        figure, axes = plt.subplots(1, 1, squeeze=False, subplot_kw={"projection": "3d"})
        return figure, axes

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> ScatterVisualization:
        # Configure axes
        ax: Axes3D = axes.flat[0]
        ax.set_xlim((-self.max_range, self.max_range))
        ax.set_ylim((0, self.max_range))
        ax.set_zlim((-self.max_range, self.max_range))
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Z [m]")
        ax.set_zlabel("Y [m]")

        paths = ax.plot([], [], [], marker="o", linestyle="None")
        return ScatterVisualization(figure, axes, paths)

    def _update_visualization(self, visualization: ScatterVisualization, **kwargs) -> None:
        # Collect point data
        point_positions = np.array([point.position for point in self.points])

        # Update visualization
        path = visualization.paths[0]
        path.set_data(point_positions[:, 0], point_positions[:, 1])
        path.set_3d_properties(point_positions[:, 2])

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.max_range, "max_range")
        process.serialize_object_sequence(self.points, "points")

    @classmethod
    @override
    def Deserialize(cls: Type[RadarPointCloud], process: DeserializationProcess) -> RadarPointCloud:
        pcl = cls(process.deserialize_floating("max_range"))
        points = process.deserialize_object_sequence("points", PointDetection)
        for point in points:
            pcl.add_point(point)
        return pcl


class RadarDetector(Serializable):
    """Base class for radar detection algorithms.

    Radar detection algorithms convert a radar cube to a radar point cloud.
    """

    @abstractmethod
    def detect(self, cube: RadarCube) -> RadarPointCloud:
        """Generate a point cloud from a radar cube.

        Args:

            cube: The radar cube to be processed.

        Returns: The resulting (usually sparse) point cloud.
        """
        ...  # pragma: no cover


class ThresholdDetector(RadarDetector):
    """Extract points by a power threshold."""

    __min_power: float  # Minmally required point power
    __normalize: bool
    __peak_detection: bool

    def __init__(
        self, min_power: float, normalize: bool = True, peak_detection: bool = True
    ) -> None:
        """
        Args:

            min_power:
                Minmally required point power.

            normalize:
                Normalize the power during detection, so that min_power
                becomes a relative value betwee zero and one.
                Enabled by default.

            peak_detection:
                Run a peak detection algorithm to only extract points at power peaks.
                Enabled by default.
        """

        self.min_power = min_power
        self.__normalize = normalize
        self.__peak_detection = peak_detection

    @property
    def min_power(self) -> float:
        """Minimally required point power.

        Returns: Power (linear).

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

    @peak_detection.setter
    def peak_detection(self, value: bool) -> None:
        self.__peak_detection = value

    @override
    def detect(self, cube: RadarCube) -> RadarPointCloud:
        # Extract cube data and normalize if the respective flag is enabled
        cube_max = cube.data.max()
        cube_data: np.ndarray = (
            (cube.data - cube.data.min()) / cube_max
            if self.normalize and cube_max > 0.0
            else cube.data
        )

        # Filter the raw cube data for peaks
        if self.peak_detection:
            footprint = generate_binary_structure(3, 1)
            candidates = maximum_filter(cube_data, footprint=footprint) == cube_data
            detection_point_indices = np.argwhere(
                np.logical_and(candidates, cube_data >= self.min_power)
            )

        else:
            # Apply the thershold and extract the coordinates matching the requirements
            detection_point_indices = np.argwhere(cube_data >= self.min_power)

        # Transform the detections to a point cloud
        cloud = RadarPointCloud(cube.range_bins.max())
        for point_indices in detection_point_indices:
            azimuth, zenith = cube.angle_bins[point_indices[0]]
            velocity = cube.velocity_bins[point_indices[1]]
            range = cube.range_bins[point_indices[2]]
            power_indicator = cube.data[tuple(point_indices)]

            cloud.add_point(
                PointDetection.FromSpherical(zenith, azimuth, range, velocity, power_indicator)
            )

        return cloud

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.min_power, "min_power")
        process.serialize_integer(self.normalize, "normalize")
        process.serialize_integer(self.peak_detection, "peak_detection")

    @classmethod
    @override
    def Deserialize(
        cls: Type[ThresholdDetector], process: DeserializationProcess
    ) -> ThresholdDetector:
        min_power = process.deserialize_floating("min_power")
        normalize = bool(process.deserialize_integer("normalize"))
        peak_detection = bool(process.deserialize_integer("peak_detection"))
        return cls(min_power, normalize, peak_detection)


class MaxDetector(RadarDetector, Serializable):
    """Extracts the maximum point from the radar cube."""

    @override
    def detect(self, cube: RadarCube) -> RadarPointCloud:
        # Find the maximum point
        point_index = np.unravel_index(np.argmax(cube.data), cube.data.shape)
        point_power = cube.data[point_index]

        # Return an empty point cloud if no maximum was found
        if point_power <= 0.0:
            return RadarPointCloud(cube.range_bins.max())

        angles_of_arrival = cube.angle_bins[point_index[0], :]
        velocity = cube.velocity_bins[point_index[1]]
        range = cube.range_bins[point_index[2]]

        cloud = RadarPointCloud(cube.range_bins.max())
        cloud.add_point(
            PointDetection.FromSpherical(
                angles_of_arrival[0], angles_of_arrival[1], range, velocity, point_power
            )
        )

        return cloud

    @override
    def serialize(self, process: SerializationProcess) -> None:
        pass

    @classmethod
    @override
    def Deserialize(cls: Type[MaxDetector], process: DeserializationProcess) -> MaxDetector:
        return cls()


class CFARDetector(RadarDetector, Serializable):
    """Constant False Alarm Rate Detector.

    Performs a two-dimensional detection over a :class:`RadarCube<hermespy.radar.cube.RadarCube>`'s range and doppler-frequency domain.
    Refer to :footcite:t:`1983:Rohling` for more information.

    The implemented detection kernel is of size

    .. math::

        \\left(1 + 2 N_{\\mathrm{T, R}} + 2 N_{\\mathrm{G, R}}\\right) \\times \\left(1 + 2 N_{\\mathrm{T, D}} + 2 N_{\\mathrm{G, D}}\\right)

    as depicted in the following figure

    .. figure:: /images/cfardetector_kernel_example.png
        :alt: CFAR Detector kernel window example
        :scale: 45%

    where `c` indicates the cell under test, `g` the guard cells and `t` the training cells.
    For each cell under test, the detector calculates the noise threshold over the available training cells

    .. math::

        P_{r,d} = \sum_{\\substack{
            i \\in \\lbrace \\mathbb{Z} | N_\\mathrm{G, R} < \\lVert r + i \\rVert \\leq N_{\\mathrm{T, R}} \\rbrace \\\\
            j \\in \\lbrace \\mathbb{Z} | N_\\mathrm{G, D} < \\lVert d + j \\rVert \\leq N_{\\mathrm{T, D}} \\rbrace
        }}
        x_{r+i,d+j}

    and makes a detection descision

    .. math::

        x_{r,d} > \\left( P_{\\mathrm{Fa}}^{\\frac{-1}{2 N_{\\mathrm{T, R}}+ 2 N_{\\mathrm{T, D}}}} - 1 \\right) P_{r,d}

    based on the configured probability of false alarm :math:`P_{\\mathrm{Fa}}`.
    """

    def __init__(self, num_training_cells: tuple, num_guard_cells: tuple, pfa: float) -> None:
        """
        CFAR Detector over the radar cube.

        Args:

            num_training_cells:
                Number of training cells in a kernel slice. Must be a pair of even ints.

            num_guard_cells:
                Number of guard cells in a kernel slice. Must be a pair of even ints.

            pfa:
                Probability of False Alarm. Must be in the interval of [0, 1].
        """

        self.num_training_cells = num_training_cells
        self.num_guard_cells = num_guard_cells
        self.pfa = pfa

    @property
    def num_training_cells(self) -> tuple:
        """Number of training cells in the detection kernel slice.

        Denoted by :math:`N_{\\mathrm{T, R}}` and :math:`N_{\\mathrm{T, D}}`.
        """

        return self.__num_training_cells

    @num_training_cells.setter
    def num_training_cells(self, value: tuple) -> None:
        if (
            not isinstance(value, tuple)
            or (len(value) != 2)
            or not isinstance(value[0], (int, np.int64))
            or not isinstance(value[1], (int, np.int64))
        ):
            raise ValueError("num_training_cells must be a tuple of two ints")
        if (value[0] <= 0) or (value[1] <= 0):
            raise ValueError("Number of training cells must be greater than zero")

        self.__num_training_cells = value

    @property
    def num_guard_cells(self) -> tuple:
        """Number of guard cells in the detection kernel slice.

        Denoted by :math:`N_{\\mathrm{G, R}}` and :math:`N_{\\mathrm{G, D}}`.
        """

        return self.__num_guard_cells

    @num_guard_cells.setter
    def num_guard_cells(self, value: tuple) -> None:
        if (
            not isinstance(value, tuple)
            or (len(value) != 2)
            or not isinstance(value[0], (int, np.int64))
            or not isinstance(value[1], (int, np.int64))
        ):
            raise ValueError("num_guard_cells must be a tuple of two ints")
        if (value[0] < 0) or (value[1] < 0):
            raise ValueError("Number of training cells must be non-negative")

        self.__num_guard_cells = value

    @property
    def pfa(self) -> float:
        """Probability of False Alarm.

        Denoted by :math:`P_{\\mathrm{Fa}}`.
        """

        return self.__pfa

    @pfa.setter
    def pfa(self, value: float) -> None:
        if (value < 0) or (value > 1):
            raise ValueError("Probability of false alarm must be in [0, 1]")

        self.__pfa = value

    @override
    def detect(self, cube: RadarCube) -> RadarPointCloud:
        # window is
        # [half of training cells][half of guard cells][CUT][half of guard cells][half of training cells]
        window_size_x = 2 * self.num_training_cells[0] + 2 * self.num_guard_cells[0] + 1
        window_size_y = 2 * self.num_training_cells[1] + 2 * self.num_guard_cells[1] + 1

        # check if data is bigger then the window
        if (cube.data.shape[1] < window_size_x) or (cube.data.shape[2] < window_size_y):
            raise ValueError(
                "Radar cube 2nd and 3rd dimensions must be bigger then num_training_cells + num_guard_cells + 1"
            )

        # prepare convolution kernel
        kernel = np.ones((1, window_size_x, window_size_y))
        kernel[
            :,
            self.num_training_cells[0] : -self.num_training_cells[0],
            self.num_training_cells[1] : -self.num_training_cells[1],
        ] = 0

        # calculate noise threshold matrix
        noise_threshold = convolve(cube.data, kernel, mode="same", method="direct")
        threshold_normalization = convolve(
            np.ones(cube.data.shape), kernel, mode="same", method="direct"
        )

        # detect
        threshold_factor = self.pfa ** (-1 / threshold_normalization) - 1
        detection_point_indices = np.argwhere(
            np.abs(cube.data) > np.abs(noise_threshold) * threshold_factor
        )

        # Transform the detections to a point cloud
        cloud = RadarPointCloud(cube.range_bins.max())
        for point_indices in detection_point_indices:
            azimuth, zenith = cube.angle_bins[point_indices[0]]
            velocity = cube.velocity_bins[point_indices[1]]
            range = cube.range_bins[point_indices[2]]
            power_indicator = cube.data[tuple(point_indices)]

            cloud.add_point(
                PointDetection.FromSpherical(zenith, azimuth, range, velocity, power_indicator)
            )

        return cloud

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_array(np.asarray(self.num_training_cells), "num_training_cells")
        process.serialize_array(np.asarray(self.num_guard_cells), "num_guard_cells")
        process.serialize_floating(self.pfa, "pfa")

    @classmethod
    @override
    def Deserialize(cls: Type[CFARDetector], process: DeserializationProcess) -> CFARDetector:
        num_training_cells = tuple(process.deserialize_array("num_training_cells", np.int64))
        num_guard_cells = tuple(process.deserialize_array("num_guard_cells", np.int64))
        pfa = process.deserialize_floating("pfa")
        return cls(num_training_cells, num_guard_cells, pfa)
