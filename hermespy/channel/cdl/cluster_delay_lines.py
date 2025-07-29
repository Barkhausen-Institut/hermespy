# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from functools import cache, cached_property
from math import ceil, sin, cos, sqrt
from typing import Generator, Generic, Literal, List, Set, Tuple, TypeVar
from typing_extensions import override

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections.polar import PolarAxes
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from scipy.constants import pi, speed_of_light

from hermespy.core import (
    AntennaMode,
    ChannelStateInformation,
    ChannelStateFormat,
    DeserializationProcess,
    Direction,
    Executable,
    Serializable,
    SerializableEnum,
    SerializationProcess,
    ScatterVisualization,
    SignalBlock,
    StemVisualization,
    VAT,
    VisualizableAttribute,
)
from hermespy.tools import db2lin
from ..channel import (
    Channel,
    ChannelRealization,
    ChannelSample,
    LinkState,
    ChannelSampleHook,
    InterpolationMode,
)
from ..consistent import (
    ConsistentGaussian,
    ConsistentRealization,
    ConsistentUniform,
    ConsistentGenerator,
    ConsistentSample,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DelayNormalization(SerializableEnum):
    """Normalization routine applied to a set of sampled delays.

    Configuration option to :class:.ClusterDelayLineBase models.
    """

    ZERO = 0
    """Normalize the delays, so that the minimal delay is zero."""

    TOF = 1
    """The minimal delay is the time of flight between two devices."""


class _PowerDelayVisualization(VisualizableAttribute[StemVisualization]):
    """Visualization of cluster delay sample power delay profile."""

    __sample: ClusterDelayLineSample  # Sample to be visualized

    def __init__(self, sample: ClusterDelayLineSample) -> None:
        self.__sample = sample

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> StemVisualization:

        _axes: plt.Axes = axes[0, 0]
        colors = self._get_color_cycle()

        container = _axes.stem(
            self.__sample.cluster_delays + self.__sample.delay_offset,
            self.__sample.cluster_powers,
            markerfmt="x",
        )
        container.stemlines.set_colors(colors)
        container.markerline.set_color("white")

        _axes.set_yscale("log")
        _axes.xaxis.set_label_text("Delay")
        _axes.yaxis.set_label_text("Power")
        _axes.set_xlim(0, self.__sample.max_delay)
        _axes.set_ylim(self.__sample.cluster_powers.min(), 1)

        return StemVisualization(figure, axes, container)

    def _update_visualization(self, visualization: StemVisualization, **kwargs) -> None:

        # Update stem markers
        visualization.container.markerline.set_data(
            self.__sample.cluster_delays + self.__sample.delay_offset, self.__sample.cluster_powers
        )

        # Update stem lines leading to markers
        visualization.container.stemlines.set_paths(
            [
                np.array([[xx, 0.0], [xx, yy]])
                for (xx, yy) in zip(
                    self.__sample.cluster_delays + self.__sample.delay_offset,
                    self.__sample.cluster_powers,
                )
            ]
        )

        # Update axes limits
        axes: plt.Axes = visualization.axes[0, 0]
        try:
            axes.set_xlim(0, max(self.__sample.max_delay, axes.get_xlim()[1]))
            axes.set_ylim(min(self.__sample.cluster_powers.min(), axes.get_ylim()[0]), 1.0)
        except ValueError: ...

    @property
    def title(self) -> str:
        return "Cluster Power Delay Profile"


class _AngleVisualization(VisualizableAttribute[ScatterVisualization]):
    """Visualization of the angles of arrival and departure."""

    __sample: ClusterDelayLineSample  # Sample to be visualized

    def __init__(self, sample: ClusterDelayLineSample) -> None:
        self.__sample = sample

    def _axes_dimensions(self, **kwargs) -> Tuple[int, int]:
        return (1, 2)

    def create_figure(self, **kwargs) -> Tuple[plt.FigureBase, VAT]:
        return plt.subplots(
            *self._axes_dimensions(**kwargs), subplot_kw={"projection": "polar"}, squeeze=False
        )

    def __generate_scatter_locations(self, center_los=True) -> np.ndarray:

        delta = self.__sample.transmitter_pose.translation - self.__sample.receiver_pose.translation
        abs_delta = np.linalg.norm(delta, 2)
        los_zenith = np.arccos((-delta[2] / abs_delta, delta[2] / abs_delta))
        los_azimuth = np.arctan((-delta[1] / delta[0], delta[1] / delta[0]))

        azimuth_offset = -los_azimuth if center_los else np.zeros(2)
        zenith_offset = -los_zenith if center_los else np.zeros(2)

        scatter_locations = np.empty(
            (self.__sample.num_clusters, 2, self.__sample.num_rays, 2), dtype=np.float64
        )

        for i, (azimuth_clusters, zenith_clusters) in enumerate(
            [
                [self.__sample.azimuth_of_arrival, self.__sample.zenith_of_arrival],
                [self.__sample.azimuth_of_departure, self.__sample.zenith_of_departure],
            ]
        ):
            azimuth_clusters = azimuth_clusters + azimuth_offset[0]
            zenith_clusters = abs(180 * (zenith_clusters + zenith_offset[0]) / pi) % 180

            # azimuth_points[zenith_points > 90] -= pi
            zenith_clusters[zenith_clusters > 90] = 90 - zenith_clusters[zenith_clusters > 90] % 90

            scatter_locations[:, i, :, 0] = azimuth_clusters
            scatter_locations[:, i, :, 1] = zenith_clusters

        return scatter_locations

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> ScatterVisualization:

        _axes = axes[0, :]
        colors = self._get_color_cycle()
        num_colors = len(colors)
        markers = ["x", "+", "o"]

        paths = np.empty((1, 2), dtype=np.object_)

        scatter_locations = self.__generate_scatter_locations()
        paths = np.empty((1, 2, 25), dtype=np.object_)
        c = 0
        for c, cluster in enumerate(scatter_locations):
            paths[0, 0, c] = _axes[0].scatter(
                cluster[0, :, 0],
                cluster[0, :, 1],
                marker=markers[int(c / num_colors)],
                color=colors[c % num_colors],
            )
            paths[0, 1, c] = _axes[1].scatter(
                cluster[1, :, 0],
                cluster[1, :, 1],
                marker=markers[int(c / num_colors)],
                color=colors[c % num_colors],
            )

        # Add additional invisible clusters in case the number of clusters changes during redraws
        for c in range(c + 1, 25):
            paths[0, 0, c] = _axes[0].scatter(
                [], [], marker=markers[int(c / num_colors)], color=colors[c % num_colors]
            )
            paths[0, 1, c] = _axes[1].scatter(
                [], [], marker=markers[int(c / num_colors)], color=colors[c % num_colors]
            )

        ax: PolarAxes
        for ax in _axes:
            ax.set_rmax(90)
            ax.grid(True)
            ax.set_xlabel("Azimuth")
            ax.set_ylabel("Zenith")
            ax.set_rlabel_position(22.5)

        _axes[0].set_title("Angles of Arrival")
        _axes[1].set_title("Angles of Departure")

        return ScatterVisualization(figure, _axes, paths)

    def _update_visualization(self, visualization: ScatterVisualization, **kwargs) -> None:
        scatter_locations = self.__generate_scatter_locations()

        c = 0
        for c, cluster in enumerate(scatter_locations):
            visualization.paths[0, 0, c].set_offsets(cluster[0])
            visualization.paths[0, 1, c].set_offsets(cluster[1])

        # Hide additional clusters
        for c in range(c + 1, 25):
            visualization.paths[0, 0, c].set_offsets(np.empty((0, 2), dtype=np.float64))
            visualization.paths[0, 1, c].set_offsets(np.empty((0, 2), dtype=np.float64))

    @property
    def title(self) -> str:
        return "Ray Angles"


class ClusterDelayLineSample(ChannelSample):
    """Sample of a 3GPP Cluster Delay Line channel model."""

    # Sub-cluster partitions for the three strongest clusters
    subcluster_indices: List[List[int]] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 18, 19],
        [8, 9, 10, 11, 16, 17],
        [12, 13, 14, 15],
    ]

    __line_of_sight: bool
    __rice_factor: float
    __azimuth_of_arrival: np.ndarray
    __zenith_of_arrival: np.ndarray
    __azimuth_of_departure: np.ndarray
    __zenith_of_departure: np.ndarray
    __delay_offset: float
    __cluster_delays: np.ndarray
    __cluster_delay_spread: float
    __cluster_powers: np.ndarray
    __polarization_transformations: np.ndarray
    __max_delay: float
    __power_delay_visualization: _PowerDelayVisualization
    __angle_visualization: _AngleVisualization

    def __init__(
        self,
        line_of_sight: bool,
        rice_factor: float,
        azimuth_of_arrival: np.ndarray,
        zenith_of_arrival: np.ndarray,
        azimuth_of_departure: np.ndarray,
        zenith_of_departure: np.ndarray,
        delay_offset: float,
        cluster_delays: np.ndarray,
        cluster_delay_spread: float,
        cluster_powers: np.ndarray,
        polarization_transformations: np.ndarray,
        state: LinkState,
    ) -> None:

        # Initialize base class
        ChannelSample.__init__(self, state)

        # Initialize attributes
        self.__line_of_sight = line_of_sight
        self.__rice_factor = rice_factor
        self.__azimuth_of_arrival = azimuth_of_arrival
        self.__zenith_of_arrival = zenith_of_arrival
        self.__azimuth_of_departure = azimuth_of_departure
        self.__zenith_of_departure = zenith_of_departure
        self.__delay_offset = delay_offset
        self.__cluster_delays = cluster_delays
        self.__cluster_delay_spread = cluster_delay_spread
        self.__cluster_powers = cluster_powers
        self.__polarization_transformations = polarization_transformations
        self.__power_delay_visualization = _PowerDelayVisualization(self)
        self.__angle_visualization = _AngleVisualization(self)

        # Infer additional parameters
        self.__num_clusters = cluster_delays.shape[0]
        self.__num_rays = azimuth_of_arrival.shape[1]
        self.__max_delay = (
            max(np.max(cluster_delays[:3] + cluster_delay_spread * 2.56), cluster_delays.max())
            + self.__delay_offset
        )

    @property
    def line_of_sight(self) -> bool:
        """Does the realization include direct line of sight between the two devices?"""

        return self.__line_of_sight

    @property
    def rice_factor(self) -> float:
        """Rice factor."""

        return self.__rice_factor

    @property
    def azimuth_of_arrival(self) -> np.ndarray:
        """Azimuth of arrival angles in radians."""

        return self.__azimuth_of_arrival

    @property
    def zenith_of_arrival(self) -> np.ndarray:
        """Zenith of arrival angles in radians."""

        return self.__zenith_of_arrival

    @property
    def azimuth_of_departure(self) -> np.ndarray:
        """Azimuth of departure angles in radians."""

        return self.__azimuth_of_departure

    @property
    def zenith_of_departure(self) -> np.ndarray:
        """Zenith of departure angles in radians."""

        return self.__zenith_of_departure

    @property
    def cluster_delays(self) -> np.ndarray:
        """Cluster delays in seconds."""

        return self.__cluster_delays

    @property
    def cluster_delay_spread(self) -> float:
        """Cluster delay spread in seconds."""

        return self.__cluster_delay_spread

    @property
    def cluster_powers(self) -> np.ndarray:
        """Cluster powers."""

        return self.__cluster_powers

    @property
    def polarization_transformations(self) -> np.ndarray:
        """Polarization transformations."""

        return self.__polarization_transformations

    @property
    def num_clusters(self) -> int:
        """Number of realized scatter clusters."""

        return self.__num_clusters

    @property
    def num_rays(self) -> int:
        """Number of rays within each cluster."""

        return self.__num_rays

    @property
    def max_delay(self) -> float:
        """Maximum expected delay in seconds."""

        return self.__max_delay

    @property
    def delay_offset(self) -> float:
        """Delay offset in seconds."""

        return self.__delay_offset

    @property
    def expected_energy_scale(self) -> float:
        return float(np.sum(self.cluster_powers))

    def __ray_impulse_generator(
        self, sampling_rate: float, num_samples: int, center_frequency: float
    ) -> Generator[Tuple[np.ndarray, np.ndarray, float], None, None]:
        """Sequential generation of individual ray impulse responses.

        Subroutine of :meth:`._propagate`.

        Args:

           sampling_rate:
                Sampling rate in Hertz.

           num_samples:
                Number of samples to generate.

           center_frequency:
                Center frequency in Hertz.
        """

        # First summand scaling of equation 7.5-30 in ETSI TR 138 901 v17.0.0
        rice_factor_lin = db2lin(self.rice_factor)
        nlos_scale = (1 + rice_factor_lin) ** -0.5 if self.line_of_sight else 1.0

        # Compute the number of clusters, considering the first two clusters get split into 3 partitions
        num_split_clusters = min(2, self.num_clusters)
        virtual_num_clusters = 3 * num_split_clusters + max(0, self.num_clusters - 2)

        # Prepare the cluster delays, equation 7.5-26
        subcluster_delays: np.ndarray = np.repeat(
            self.cluster_delays[:num_split_clusters, None], 3, axis=1
        ) + self.cluster_delay_spread * np.array([0.0, 1.28, 2.56])
        virtual_cluster_delays = np.concatenate(
            (subcluster_delays.flatten(), self.cluster_delays[num_split_clusters:])
        )

        # Wavelength factor
        wavelength_factor = center_frequency / speed_of_light
        relative_velocity = self.receiver_velocity - self.transmitter_velocity
        fast_fading = wavelength_factor * np.arange(num_samples) / sampling_rate

        # Compute the cluster propagation parameters
        for subcluster_idx in range(0, virtual_num_clusters):
            cluster_idx = int(subcluster_idx / 3) if subcluster_idx < 6 else subcluster_idx - 4
            ray_indices = (
                ClusterDelayLineSample.subcluster_indices[cluster_idx]
                if cluster_idx < num_split_clusters
                else range(self.num_rays)
            )
            delay: float = virtual_cluster_delays[subcluster_idx]

            for aoa, zoa, aod, zod, jones in zip(
                self.azimuth_of_arrival[cluster_idx, ray_indices],
                self.zenith_of_arrival[cluster_idx, ray_indices],
                self.azimuth_of_departure[cluster_idx, ray_indices],
                self.zenith_of_departure[cluster_idx, ray_indices],
                self.polarization_transformations[:, :, cluster_idx, ray_indices].transpose(
                    2, 0, 1
                ),
            ):
                # Compute directive unit vectors
                tx_direction = Direction.From_Spherical(aod, zod)
                rx_direction = Direction.From_Spherical(aoa, zoa)

                # Combination of Equation 7.5-23, 7.5.24 and 7.5.28
                tx_array_response = self.transmitter_antennas.cartesian_array_response(
                    center_frequency, tx_direction.view(np.ndarray), "global", AntennaMode.TX
                )
                rx_array_response = self.receiver_antennas.cartesian_array_response(
                    center_frequency, rx_direction.view(np.ndarray), "global", AntennaMode.RX
                )

                # Equation 7.5-22 and 7.5.28 in ETSI TR 138 901 v17.0.0
                channel: np.ndarray = (
                    rx_array_response
                    @ jones
                    @ tx_array_response.T
                    * (sqrt(self.cluster_powers[cluster_idx] / self.num_rays) * nlos_scale)
                )

                wave_vector = np.array(
                    [cos(aoa) * sin(zoa), sin(aoa) * sin(zoa), cos(zoa)], dtype=float
                )
                impulse: np.ndarray = np.exp(
                    np.inner(wave_vector, relative_velocity) * fast_fading * 2j * pi
                )

                yield channel, impulse, delay

        # If line of sight is enabled, yield an additional parameter touple for the line of sight component
        if self.__line_of_sight:
            device_vector = self.receiver_pose.translation - self.transmitter_pose.translation
            los_distance = np.linalg.norm(device_vector, 2)
            rx_wave_vector = device_vector / los_distance

            # Equation 7.5-29
            tx_array_response = self.transmitter_antennas.cartesian_array_response(
                center_frequency, self.receiver_pose.translation, "global", AntennaMode.TX
            )
            rx_array_response = self.receiver_antennas.cartesian_array_response(
                center_frequency, self.transmitter_pose.translation, "global", AntennaMode.RX
            )

            # Second summand of equation 7.5-30, including equation 7.5-29 of ETSI TR 138 901 v17.0.0
            los_delay: float = self.cluster_delays[0]
            los_channel: np.ndarray = (
                rx_array_response
                @ np.array([[1, 0], [-1, 0]])
                @ tx_array_response.T
                * (rice_factor_lin / (1 + rice_factor_lin)) ** 0.5
            )
            los_impulse: np.ndarray = np.exp(-2j * pi * los_distance * wavelength_factor) * np.exp(
                np.inner(rx_wave_vector, relative_velocity) * fast_fading * 2j * pi
            )

            yield los_channel, los_impulse, los_delay

    def _propagate(self, signal: SignalBlock, interpolation: InterpolationMode) -> SignalBlock:

        max_delay_in_samples = ceil((self.max_delay) * self.bandwidth)
        propagated_samples = np.zeros(
            (self.num_receive_antennas, signal.num_samples + max_delay_in_samples),
            dtype=np.complex128,
        )

        # Prepare the optimal einsum path ahead of time for faster execution
        channel = np.zeros(
            (self.num_receive_antennas, self.num_transmit_antennas), dtype=np.complex128
        )
        impulse = np.zeros(signal.num_samples, dtype=np.complex128)
        einsum_subscripts = "ij,jk,k->ik"
        einsum_path = np.einsum_path(
            einsum_subscripts, channel, signal, impulse, optimize="optimal"
        )[0]

        for channel, impulse, delay in self.__ray_impulse_generator(
            self.bandwidth, signal.num_samples, self.carrier_frequency
        ):
            if interpolation == InterpolationMode.NEAREST:
                delay_in_samples = int((delay + self.delay_offset) * self.bandwidth)
                propagated_samples[
                    :, delay_in_samples : delay_in_samples + signal.num_samples
                ] += np.einsum(einsum_subscripts, channel, signal, impulse, optimize=einsum_path)

        return SignalBlock(propagated_samples, signal._offset)

    def state(
        self,
        num_samples: int,
        max_num_taps: int,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> ChannelStateInformation:
        max_delay_in_samples = min(max_num_taps, ceil(self.max_delay * self.bandwidth))
        raw_state = np.zeros(
            (
                self.num_receive_antennas,
                self.num_transmit_antennas,
                num_samples,
                1 + max_delay_in_samples,
            ),
            dtype=np.complex128,
        )

        for channel, impulse, delay in self.__ray_impulse_generator(
            self.bandwidth, num_samples, self.carrier_frequency
        ):
            # Compute the delay in samples
            delay_tap_index = int((delay + self.delay_offset) * self.bandwidth)

            if delay_tap_index >= max_num_taps:
                continue  # pragma: no cover

            # Compute the resulting channel state
            raw_state[:, :, :, delay_tap_index] += np.einsum(
                "ij,k->ijk", channel, impulse, optimize=False
            )

        return ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, raw_state)

    @property
    def plot_power_delay(self) -> _PowerDelayVisualization:
        return self.__power_delay_visualization

    @property
    def plot_angles(self) -> _AngleVisualization:
        return self.__angle_visualization

    def plot_rays(self, title: str | None = None) -> plt.Figure:
        with Executable.style_context():
            axis: Axes3D
            figure, axis = plt.subplots(subplot_kw={"projection": "3d"})
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        magnitude = np.linalg.norm(
            self.transmitter_pose.translation - self.receiver_pose.translation
        )

        for i, (aoa, zoa, aod, zod) in enumerate(
            zip(
                self.azimuth_of_arrival,
                self.zenith_of_arrival,
                self.azimuth_of_departure,
                self.zenith_of_departure,
            )
        ):
            tx_vectors = np.array(
                [np.sin(zod) * np.cos(aod), np.sin(zod) * np.sin(aod), np.cos(zod)]
            )
            rx_vectors = -np.array(
                [np.sin(zoa) * np.cos(aoa), np.sin(zoa) * np.sin(aoa), np.cos(zoa)]
            )

            for vectors, origin in zip(
                (tx_vectors, rx_vectors),
                (self.transmitter_pose.translation, self.receiver_pose.translation),
            ):
                stops = vectors * magnitude + origin[:, np.newaxis]
                for stop in stops.T:
                    axis.plot(
                        [origin[0], stop[0]],
                        [origin[1], stop[1]],
                        [origin[2], stop[2]],
                        color=colors[i % len(colors)],
                    )

            """start_arrival = self.transmitter_position
            stop_arrival = self.transmitter_position + arrival
            start_departure = self.receiver_position
            stop_departure = self.receiver_position + departure"""

        axis.set_zlim(-magnitude, magnitude)
        axis.scatter(*self.transmitter_pose.translation, marker="o", color="w")
        axis.text(*self.transmitter_pose.translation, "Tx", zorder=1)
        axis.scatter(*self.receiver_pose.translation, marker="o", color="w")
        axis.text(*self.receiver_pose.translation, "Rx", zorder=1)

        figure.suptitle("Rays" if title is None else title)
        return figure

    @staticmethod
    def _angular_spread(angles: np.ndarray, powers: np.ndarray) -> float:
        """Compute the angular spread of a set of data.

        Implements equation (A-1) of :footcite:t:`3GPP:TR38901`.

        Args:

           angles:
                Numpy array of angles in radians for all paths of propagation.

           powers:
                Numpy angles of power for all paths of propagation.

        Returns: The angluar spread in radians.

        Raises:

            ValueError: If the dimensions of `angles` and `powers` don't match.
        """

        if angles.shape != powers.shape:
            raise ValueError("Dimensions of angles and powers don't match")

        summed_power = np.sum(powers, axis=None)
        weighted_angle_sum = np.sum(np.exp(1j * angles) * powers, axis=None)

        spread = np.sqrt(-2 * np.log(np.abs(weighted_angle_sum / summed_power)))
        return spread

    @cached_property
    def azimuth_arrival_spread(self) -> float:
        """Spread of azimuth of arrival angles.

        Returns: Angular spread in radians.
        """

        return self._angular_spread(
            self.azimuth_of_arrival,
            np.repeat(self.cluster_powers[:, None], self.azimuth_of_arrival.shape[1], axis=1),
        )

    @cached_property
    def azimuth_departure_spread(self) -> float:
        """Spread of azimuth of departure angles.

        Returns: Angular spread in radians.
        """

        return self._angular_spread(
            self.azimuth_of_departure,
            np.repeat(self.cluster_powers[:, None], self.azimuth_of_departure.shape[1], axis=1),
        )

    @cached_property
    def zenith_arrival_spread(self) -> float:
        """Spread of zenith of arrival angles.

        Returns: Angular spread in radians.
        """

        return self._angular_spread(
            self.zenith_of_arrival,
            np.repeat(self.cluster_powers[:, None], self.zenith_of_arrival.shape[1], axis=1),
        )

    @cached_property
    def zenith_departure_spread(self) -> float:
        """Spread of zenith of departure angles.

        Returns: Angular spread in radians.
        """

        return self._angular_spread(
            self.zenith_of_departure,
            np.repeat(self.cluster_powers[:, None], self.zenith_of_departure.shape[1], axis=1),
        )

    def reciprocal(self, state: LinkState) -> ClusterDelayLineSample:
        """Generate a reciprocal sample of the current sample.

        Args:

           state:
                State of the reciprocal sample.

        Returns: A reciprocal sample.
        """

        return ClusterDelayLineSample(
            self.line_of_sight,
            self.rice_factor,
            self.azimuth_of_departure,
            self.zenith_of_departure,
            self.azimuth_of_arrival,
            self.zenith_of_arrival,
            self.delay_offset,
            self.cluster_delays,
            self.cluster_delay_spread,
            self.cluster_powers,
            self.polarization_transformations,
            state,
        )


class LargeScaleState(SerializableEnum):
    """Large scale state of a 3GPP Cluster Delay Line channel model."""

    @property
    @abstractmethod
    def line_of_sight(self) -> bool:
        """Direct line of sight between two linked devices."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def value(self) -> int:
        """Integer value of the state."""
        ...  # pragma: no cover


class LOSState(LargeScaleState, IntEnum):
    """Line of sight state of the cluster delay line model."""

    LOS = 0
    """Direct line of sight connection without blockage between
    base station and user terminal.
    """

    NLOS = 1
    """Physical blockage between base station and user terminal."""

    @property
    def line_of_sight(self) -> bool:
        return self == LOSState.LOS

    @property
    def value(self) -> int:
        return int(self)


class O2IState(LargeScaleState, IntEnum):
    """O2I state of the cluster delay line model."""

    LOS = 0
    """Direct line of sight connection without blockage between
    base station and user terminal.
    """

    NLOS = 1
    """Physical blockage between base station and user terminal."""

    O2I = 2
    """User terminal loacated inside a building."""

    @property
    def line_of_sight(self) -> bool:
        return self == LOSState.LOS

    @property
    def value(self) -> int:
        return int(self)


LSST = TypeVar("LSST", bound=LargeScaleState)
"""Type variable for large scale state types."""


class ClusterDelayLineSampleParameters(object):
    """Data class for initialization parameters of a 3GPP Cluster Delay Line channel model sample."""

    __carrier_frequency: float
    __distance_3d: float
    __distance_2d: float
    __base_height: float
    __terminal_height: float

    def __init__(
        self,
        carrier_frequency: float,
        distance_3d: float,
        distance_2d: float,
        base_height: float,
        terminal_height: float,
    ) -> None:
        """
        Args:

           carrier_frequency:
                Carrier frequency in Hz.

           distance_3d:
                Distance between the base station and the user terminal in meters.

           distance_2d:
                Horizontal distance between the base station and the user terminal in meters.

           base_height:
                Height of the base station in meters.

           terminal_height:
                Height of the user terminal in meters.
        """

        self.__carrier_frequency = carrier_frequency
        self.__distance_3d = distance_3d
        self.__distance_2d = distance_2d
        self.__base_height = base_height
        self.__terminal_height = terminal_height

    @property
    def carrier_frequency(self) -> float:
        """Carrier frequency in Hz."""

        return self.__carrier_frequency

    @property
    def distance_3d(self) -> float:
        """Distance between the base station and the user terminal in meters."""

        return self.__distance_3d

    @property
    def distance_2d(self) -> float:
        """Horizontal distance between the base station and the user terminal in meters."""

        return self.__distance_2d

    @property
    def base_height(self) -> float:
        """Height of the base station in meters."""

        return self.__base_height

    @property
    def terminal_height(self) -> float:
        """Height of the user terminal in meters."""

        return self.__terminal_height


@dataclass
class ClusterDelayLineRealizationParameters(Serializable):
    """Data class for initialization parameters of a 3GPP Cluster Delay Line channel model realization."""

    state_variable: ConsistentUniform
    delay_normalization: DelayNormalization
    oxygen_absorption: bool
    large_scale_parameters: np.ndarray
    """Large scale parameters of the channel model.

    Numpy vector with the following elements with the elements containing
    samples of a standard normal distribution with specific cross-correlations.
    """

    cluster_shadowing_log_variable: ConsistentGaussian
    cluster_delays_variable: ConsistentUniform
    azimuth_angle_variation_variable: ConsistentGaussian
    azimuth_spread_sign: np.ndarray
    zenith_angle_variation_variable: ConsistentGaussian
    zenith_spread_sign: np.ndarray
    angle_coupling_indices: np.ndarray
    cross_polarization_power_variable: ConsistentGaussian
    cross_polarization_phase_variable: ConsistentUniform

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.state_variable, "state_variable")
        process.serialize_object(self.delay_normalization, "delay_normalization")
        process.serialize_integer(self.oxygen_absorption, "oxygen_absorption")
        process.serialize_array(self.large_scale_parameters, "large_scale_parameters")
        process.serialize_object(
            self.cluster_shadowing_log_variable, "cluster_shadowing_log_variable"
        )
        process.serialize_object(self.cluster_delays_variable, "cluster_delays_variable")
        process.serialize_object(
            self.azimuth_angle_variation_variable, "azimuth_angle_variation_variable"
        )
        process.serialize_array(self.azimuth_spread_sign, "azimuth_spread_sign")
        process.serialize_object(
            self.zenith_angle_variation_variable, "zenith_angle_variation_variable"
        )
        process.serialize_array(self.zenith_spread_sign, "zenith_spread_sign")
        process.serialize_array(self.angle_coupling_indices, "angle_coupling_indices")
        process.serialize_object(
            self.cross_polarization_power_variable, "cross_polarization_power_variable"
        )
        process.serialize_object(
            self.cross_polarization_phase_variable, "cross_polarization_phase_variable"
        )

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> ClusterDelayLineRealizationParameters:
        return ClusterDelayLineRealizationParameters(
            process.deserialize_object("state_variable", ConsistentUniform),
            process.deserialize_object("delay_normalization", DelayNormalization),
            bool(process.deserialize_integer("oxygen_absorption")),
            process.deserialize_array("large_scale_parameters", np.float64),
            process.deserialize_object("cluster_shadowing_log_variable", ConsistentGaussian),
            process.deserialize_object("cluster_delays_variable", ConsistentUniform),
            process.deserialize_object("azimuth_angle_variation_variable", ConsistentGaussian),
            process.deserialize_array("azimuth_spread_sign", np.int64),
            process.deserialize_object("zenith_angle_variation_variable", ConsistentGaussian),
            process.deserialize_array("zenith_spread_sign", np.int64),
            process.deserialize_array("angle_coupling_indices", np.int64),
            process.deserialize_object("cross_polarization_power_variable", ConsistentGaussian),
            process.deserialize_object("cross_polarization_phase_variable", ConsistentUniform),
        )


class ClusterDelayLineRealization(ChannelRealization[ClusterDelayLineSample], Generic[LSST]):
    """Realization of a 3GPP Cluster Delay Line channel model."""

    # Cluster scaling factors for azimuth angles
    # Table 7.5-2 in ETSI TR 138 901 v17.0.0
    __azimuth_scaling_factors = np.array(
        [
            [4, 0.779],
            [5, 0.86],
            [8, 1.018],
            [10, 1.090],
            [11, 1.123],
            [12, 1.146],
            [14, 1.19],
            [15, 1.211],
            [16, 1.226],
            [19, 1.273],
            [20, 1.289],
            [25, 1.358],
        ],
        dtype=np.float64,
    )

    # Cluster scaling factors for zenith angles
    # Table 7.5-4 in ETSI TR 138 901 v17.0.0
    __zenith_scaling_factors = np.array(
        [
            [8, 0.889],
            [10, 0.957],
            [11, 1.031],
            [12, 1.104],
            [15, 1.1088],
            [19, 1.184],
            [20, 1.178],
            [25, 1.282],
        ],
        dtype=np.float64,
    )

    # Ray offset angles
    # Table 7.5-3 in ETSI TR 138 901 v17.0.0
    _ray_offset_angles = np.array(
        [
            0.0447,
            -0.0447,
            0.1413,
            -0.1413,
            0.2492,
            -0.2492,
            0.3715,
            -0.3715,
            0.5129,
            -0.5129,
            0.6797,
            -0.6797,
            0.8844,
            -0.8844,
            1.1481,
            -1.1481,
            1.5195,
            -1.5195,
            2.1551,
            -2.1551,
        ],
        dtype=np.float64,
    )

    # Oxygen absorption factors
    # Table 7.6-1 in ETSI TR 138 901 v17.0.0
    # First column: Carrier frequency in GHz
    # Second column: Oxygen absorption factor in dB/km
    __oxygen_loss = np.array(
        [
            [52.0, 0.0],
            [53.0, 1.0],
            [54.0, 2.0],
            [55.0, 4.0],
            [65.0, 6.6],
            [75.0, 9.7],
            [58.0, 12.6],
            [59.0, 14.6],
            [60.0, 15.0],
            [61.0, 14.6],
            [62.0, 14.3],
            [63.0, 10.5],
            [64.0, 6.8],
            [65.0, 3.9],
            [66.0, 1.9],
            [67.0, 1.0],
            [68.0, 0.0],
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        expected_state: LSST | None,
        state_realization: ConsistentRealization,
        parameters: ClusterDelayLineRealizationParameters,
        sample_hooks: Set[ChannelSampleHook[ClusterDelayLineSample]],
        gain: float = 1.0,
    ) -> None:
        """
        Args:

            expected_state:
                Expected large-scale state of the channel.
                If not specified, the large-scale state is randomly generated.

            state_realization:
                Realization of the large scale state.

            parameters:
                General parameters of the cluster delay line realization.

            sample_hooks:
                Hooks for callback functions during the sample generation.

            gain:
                Linear amplitude scaling factor if signals propagated over the channel.
        """

        # Initialize base class
        ChannelRealization.__init__(self, sample_hooks, gain)

        # Initialize attributes
        self.__expected_state = expected_state
        self.__state_realization = state_realization
        self.__state_variable = parameters.state_variable
        self.__delay_normalization = parameters.delay_normalization
        self.__oxygen_absorption = parameters.oxygen_absorption
        self.__large_scale_parameters = parameters.large_scale_parameters
        self.__cluster_shadowing_log_variable = parameters.cluster_shadowing_log_variable
        self.__cluster_delays_variable = parameters.cluster_delays_variable
        self.__azimuth_angle_variation_variable = parameters.azimuth_angle_variation_variable
        self.__azimuth_spread_sign = parameters.azimuth_spread_sign
        self.__zenith_angle_variation_variable = parameters.zenith_angle_variation_variable
        self.__zenith_spread_sign = parameters.zenith_spread_sign
        self.__angle_coupling_indices = parameters.angle_coupling_indices
        self.__xpr_variable = parameters.cross_polarization_power_variable
        self.__xpr_phase = parameters.cross_polarization_phase_variable

    @property
    def expected_state(self) -> LSST | None:
        """Expected state of the channel realization.

        If None, the state will be randomly drawn from the state variable for each sample.
        """
        return self.__expected_state

    @abstractmethod
    def _pathloss_dB(self, state: LSST, parameters: ClusterDelayLineSampleParameters) -> float:
        """Power loss during signal propagation in dB.

        Args:

            state:
                Large scale state of the channel.

            parameters:
                Parameters of the channel realization.

        Returns: Pathloss in dB.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _small_scale_realization(self, state: LSST) -> ConsistentRealization:
        """Fetch the random realization of the small scale state.

        Args:

            state:
                Large scale state of the channel.

        Returns:
            ConsistentRealization: Realization of the small scale state.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _sample_large_scale_state(
        self, state_variable_realization: float, parameters: ClusterDelayLineSampleParameters
    ) -> LSST:
        """Sample the large scale state of the channel.

        Returns: Large scale state of the channel.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _delay_spread_mean(self, state: LSST, carrier_frequency: float) -> float:
        """Mean of the cluster delay spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{DS}` and :math:`\\mu_{\\mathrm{lgDS}}` within the the standard, respectively.

        Returns: Mean delay spread in seconds.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _delay_spread_std(self, state: LSST, carrier_frequency: float) -> float:
        """Standard deviation of the cluster delay spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{DS}` and :math:`\\sigma_{\\mathrm{lgDS}}` within the the standard, respectively.

        Returns: Delay spread standard deviation in seconds.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _aod_spread_mean(self, state: LSST, carrier_frequency: float) -> float:
        """Mean of the Azimuth Angle-of-Departure spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{ASD}` and :math:`\\mu_{\\mathrm{lgASD}}` within the the standard, respectively.

        Returns: Mean angle spread in seconds
        """
        ...  # pragma: no cover

    @abstractmethod
    def _aod_spread_std(self, state: LSST, carrier_frequency: float) -> float:
        """Standard deviation of the Azimuth Angle-of-Departure spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{ASD}` and :math:`\\sigma_{\\mathrm{lgASD}}` within the the standard, respectively.

        Returns: Angle spread standard deviation in seconds.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _aoa_spread_mean(self, state: LSST, carrier_frequency: float) -> float:
        """Mean of the Azimuth Angle-of-Arriaval spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{ASA}` and :math:`\\mu_{\\mathrm{lgASA}}` within the the standard, respectively.

        Returns: Mean angle spread in seconds
        """
        ...  # pragma: no cover

    @abstractmethod
    def _aoa_spread_std(self, state: LSST, carrier_frequency: float) -> float:
        """Standard deviation of the Azimuth Angle-of-Arrival spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{ASA}` and :math:`\\sigma_{\\mathrm{lgASA}}` within the the standard, respectively.

        Returns: Angle spread standard deviation in seconds.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _zoa_spread_mean(self, state: LSST, carrier_frequency: float) -> float:
        """Mean of the Zenith Angle-of-Arriaval spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{ZSA}` and :math:`\\mu_{\\mathrm{lgZSA}}` within the the standard, respectively.

        Returns: Mean angle spread in seconds
        """
        ...  # pragma: no cover

    @abstractmethod
    def _zoa_spread_std(self, state: LSST, carrier_frequency: float) -> float:
        """Standard deviation of the Zenith Angle-of-Arrival spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{ZSA}` and :math:`\\sigma_{\\mathrm{lgZSA}}` within the the standard, respectively.

        Returns: Angle spread standard deviation in seconds.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...  # pragma: no cover

    ###############################
    # ToDo: Shadow fading function
    ###############################

    @abstractmethod
    def _rice_factor_mean(self) -> float:
        """Mean of the rice factor distribution.

        The rice factor realization and its mean are referred to as
        :math:`K` and :math:`\\mu_K` within the the standard, respectively.

        Returns: Rice factor mean in dB.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _rice_factor_std(self) -> float:
        """Standard deviation of the rice factor distribution.

        The rice factor realization and its standard deviation are referred to as
        :math:`K` and :math:`\\sigma_K` within the the standard, respectively.

        Returns: Rice factor standard deviation in dB.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _delay_scaling(self, state: LSST) -> float:
        """Delay scaling proportionality factor

        Referred to as :math:`r_{\\tau}` within the standard.

        Returns: Scaling factor.

        Raises:
            ValueError:
                If scaling factor is smaller than one.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _cross_polarization_power_mean(self, state: LSST) -> float:
        """Mean of the cross-polarization power.

        The cross-polarization power and its mean are referred to as
        :math:`\\mathrm{XPR}` and :math:`\\mu_{\\mathrm{XPR}}` within the the standard, respectively.

        Returns: Mean power in dB.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _cross_polarization_power_std(self, state: LSST) -> float:
        """Standard deviation of the cross-polarization power.

        The cross-polarization power and its standard deviation are referred to as
        :math:`\\mathrm{XPR}` and :math:`\\sigma_{\\mathrm{XPR}}` within the the standard, respectively.

        Returns: Power standard deviation in dB.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _num_clusters(self, state: LSST) -> int:
        """Number of clusters in the channel realization."""
        ...  # pragma: no cover

    @abstractmethod
    def _cluster_aod_spread(self, state: LSST) -> float:
        """Azimuth Angle-of-Departure spread within an individual cluster.

        Referred to as :math:`c_{ASD}` within the standard.

        Returns: Angle spread in degrees.

        Raises:
            ValueError: If spread is smaller than zero.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _cluster_aoa_spread(self, state: LSST) -> float:
        """Azimuth Angle-of-Arrival spread within an individual cluster.

        Referred to as :math:`c_{ASA}` within the standard.

        Returns: Angle spread in degrees.

        Raises:
            ValueError: If spread is smaller than zero.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _cluster_zoa_spread(self, state: LSST) -> float:
        """Zenith Angle-of-Arrival spread within an individual cluster.

        Referred to as :math:`c_{ZSA}` within the standard.

        Returns: Angle spread in degrees.

        Raises:
            ValueError: If spread is smaller than zero.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _cluster_shadowing_std(self, state: LSST) -> float:
        """Standard deviation of the cluster shadowing.

        Referred to as :math:`\\zeta` within the the standard.

        Returns: Cluster shadowing standard deviation.

        Raises:
            ValueError: If the deviation is smaller than zero.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _zod_spread_mean(
        self, state: LSST, parameters: ClusterDelayLineSampleParameters
    ) -> float: ...  # pragma: no cover

    @abstractmethod
    def _zod_spread_std(
        self, state: LSST, parameters: ClusterDelayLineSampleParameters
    ) -> float: ...  # pragma: no cover

    @abstractmethod
    def _zod_offset(self, state: LSST, parameters: ClusterDelayLineSampleParameters) -> float:
        """Offset between Zenith Angle-of-Arrival and Angle-of-Departure.

        The offset is referred to as :math:`\\mu_{\\mathrm{offset,ZOD}}` within the standard.

        Returns: The offset in degrees.
        """
        ...  # pragma: no cover

    def _sample_large_scale_parameters(
        self, state: LSST, parameters: ClusterDelayLineSampleParameters
    ) -> Tuple[float, ...]:
        """Sample large-scale parameters.

        Representation of step 4 in ETSI TR 138.901 v17.0.0.

        Args:

            state: Large scale state of the channel.
            parameters: Parameters of the channel realization.

        Returns: Tuple of large-scale parameters.
        """

        mean = np.array(
            [
                self._delay_spread_mean(state, parameters.carrier_frequency),
                self._aod_spread_mean(state, parameters.carrier_frequency),
                self._aoa_spread_mean(state, parameters.carrier_frequency),
                self._zoa_spread_mean(state, parameters.carrier_frequency),
                self._zod_spread_mean(state, parameters),
                self._rice_factor_mean(),
                0.0,
            ],
            dtype=np.float64,
        )
        std = np.array(
            [
                self._delay_spread_std(state, parameters.carrier_frequency),
                self._aod_spread_std(state, parameters.carrier_frequency),
                self._aoa_spread_std(state, parameters.carrier_frequency),
                self._zoa_spread_std(state, parameters.carrier_frequency),
                self._zod_spread_std(state, parameters),
                self._rice_factor_std(),
                1.0,
            ],
            dtype=np.float64,
        )

        # Enforce the proper cross-correlations between the large-scale parameters
        # The routine is described in section 3.3.1 of the WINNERII final report
        S = mean + std * self.__large_scale_parameters[state.value, ...]

        # Transform to the proper distribution
        DS = 10 ** S[0]
        ASD = min(10 ** S[1], 104)
        ASA = min(10 ** S[2], 104)
        ZSA = min(10 ** S[3], 52)
        ZSD = min(10 ** S[4], 52)
        K = S[5]
        SF = S[6]
        return DS, ASD, ASA, ZSA, ZSD, K, SF

    def _cluster_delays(
        self,
        consistent_sample: ConsistentSample,
        state: LSST,
        delay_spread: float,
        rice_factor: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute a single sample set of normalized cluster delays.

        A single cluster delay is referred to as :math:`\\tau_n` within the the standard.
        Implementation of equations 7.5-1 to 7.5-4 in TR 138.901 v17.0.0.

        Args:

            consistent_sample:
                Sample of the consistent random process.

            state:
                Large scale state of the channel.

            line_of_sight:
                Is the realization line of sight?

            delay_spread:
                Delay spread in seconds.
                Denoted by :math:`\\mathrm{DS}` in the standard.

            rice_factor:
                Rice factor K in dB.
                Denoted by :math:`K` in the standard.

        Returns:

            Tuple of
                - Raw delay samples
                - True delays
                - Minimal delay
        """

        # Equation 7.5-1 in TR 138.901 v17.0.0
        r_tau = self._delay_scaling(state)
        X_n = self.__cluster_delays_variable.sample(consistent_sample)[: self._num_clusters(state)]
        unsorted_delays = -r_tau * delay_spread * np.log(X_n)

        # Equation 7.5-2 in TR 138.901 v17.0.0
        # Sort the delays in ascending order
        sorted_delays = np.sort(unsorted_delays)
        minimal_delay = sorted_delays[0]
        sorted_delays -= minimal_delay

        # In case of line of sight, scale the delays by the appropriate K-factor
        # Equations 7.5-3 and 7.5-4 in TR 138.901 v17.0.0
        scaled_delays = sorted_delays.copy()
        if state.line_of_sight:
            rice_scale = (
                0.775 - 0.0433 * rice_factor + 2e-4 * rice_factor**2 + 17e-5 * rice_factor**3
            )
            scaled_delays /= rice_scale
            minimal_delay = 0.0  # This is required for the oxygen absorption computation

        # Return the raw and scaled delays, since they are both required for further processing
        return sorted_delays, scaled_delays, minimal_delay

    def _cluster_powers(
        self,
        consistent_sample: ConsistentSample,
        state: LSST,
        delay_spread: float,
        delays: np.ndarray,
        rice_factor: float,
    ) -> np.ndarray:
        """Compute a single sample set of normalized cluster power factors from delays.

        A single cluster power factor is referred to as :math:`P_n` within the the standard.

        Args:

            consistent_sample:
                Sample of the consistent random process.

            state:
                Large scale state of the channel.

            delay_spread:
                Delay spread in seconds.
                Denoted by :math:`\\mathrm{DS}` in the standard.

            delays:
                Vector of cluster delays.
                Denoted by :math:`\\tau` in the standard.

            rice_factor:
                Rice factor in dB.
                Denoted by :math:`K` in the standard.

        Returns: Vector of cluster power scales.
        """

        delay_scaling = self._delay_scaling(state)
        shadowing = 10 ** (
            -0.1
            * self.__cluster_shadowing_log_variable.sample(
                consistent_sample, 0.0, self._cluster_shadowing_std(state)
            )[: self._num_clusters(state)]
        )

        # Equation 7.5-5 in TR 138.901 v17.0.0
        powers = np.exp(-delays * (delay_scaling - 1) / (delay_scaling * delay_spread)) * shadowing

        # In case of line of sight, add a specular component to the cluster delays
        if state.line_of_sight:
            # Equations 7.5-7 and 7.5-8 in TR 138.901 v17.0.0
            linear_rice_factor = db2lin(rice_factor)
            powers /= (1 + linear_rice_factor) * np.sum(powers.flat)
            powers[0] += linear_rice_factor / (1 + linear_rice_factor)

        else:
            # Equation 7.5-6 in TR 138.901 v17.0.0
            powers /= np.sum(powers.flat)

        # Eliminate clusters with powers less than -25 dB with respect to the strongest cluster
        cutoff_power = powers[0] * 10 ** (-25 / 10)
        powers = powers[np.where(powers >= cutoff_power)]

        return powers

    def _ray_azimuth_angles(
        self,
        consistent_sample: ConsistentSample,
        state: LSST,
        cluster_powers: np.ndarray,
        spread: float,
        rice_factor: float,
        los_azimuth: float,
        direction: Literal["arrival", "departure"],
    ) -> np.ndarray:
        """Compute cluster ray azimuth angles of arrival or departure.

        Args:

            consistent_sample:
                Sample of the consistent random process.

            state:
                Large scale state of the channel.

            cluster_powers:
                Vector of cluster powers. The length determines the number of clusters.

            spread:
                Root-mean squared Azimuth spread in degrees.
                Denoted by :math:`\\mathrm{ASD}` or :math:`\\mathrm{ASA}` in the standard.

            rice_factor
                Rice factor in dB.
                Denoted by :math:`K` in the standard.

            los_azimuth:
                Line of sight azimuth angle to the target in degrees.

            direction:
                Direction of the angles. Either "arrival" or "departure".

        Returns:
            Matrix of angles in degrees.
            The first dimension indicates the cluster index, the second dimension the ray index.
        """

        # Determine the closest scaling factor
        scale_index = np.argmin(np.abs(self.__azimuth_scaling_factors[:, 0] - len(cluster_powers)))
        angle_scale = self.__azimuth_scaling_factors[scale_index, 1]

        # Scale the scale (hehe) in the line of sight case
        if state.line_of_sight:
            angle_scale *= (
                1.1035 - 0.028 * rice_factor - 2e-3 * rice_factor**2 + 1e-4 * rice_factor**3
            )

        angles: np.ndarray = (
            2 * spread / 1.4 / angle_scale * np.sqrt(-np.log(cluster_powers / cluster_powers.max()))
        )

        # Assign positive / negative integers and add some noise, Equation 7.5-11 in TR 138.901 v17.0.0
        angle_variation = (spread / 7) * self.__azimuth_angle_variation_variable.sample(
            consistent_sample
        )
        spread_angles = (
            self.__azimuth_spread_sign[: cluster_powers.size] * angles
            + angle_variation[: cluster_powers.size]
        )

        # Add the actual line of sight term
        if state.line_of_sight:
            # The first angle within the list is exactly the line of sight component
            spread_angles += los_azimuth - spread_angles[0]

        else:
            spread_angles += los_azimuth

        # Spread the angles
        cluster_spread = (
            self._cluster_aoa_spread(state)
            if direction == "arrival"
            else self._cluster_aod_spread(state)
        )
        ray_offsets = cluster_spread * self._ray_offset_angles
        ray_angles = np.tile(spread_angles[:, None], len(ray_offsets)) + ray_offsets

        return ray_angles

    def _ray_zoa(
        self,
        consistent_sample: ConsistentSample,
        state: LSST,
        cluster_powers: np.ndarray,
        spread: float,
        rice_factor: float,
        los_zenith: float,
    ) -> np.ndarray:
        """Compute cluster ray zenith angles of arrival.

        Args:

            consistent_sample:
                Sample of the consistent random process.

            state:
                Large scale state of the channel.

            cluster_powers:
                Vector of cluster powers. The length determines the number of clusters.

            spread:
                Zenith of arrival angular spread in degrees.
                Denoted by :math:`\\mathrm{ZSA}` in the standard.

            rice_factor:
                Rice factor in dB.
                Denoted by :math:`K` in the standard.

            los_zenith:
                Line of sight zenith angle to the target in degrees.

        Returns:
            Matrix of angles in degrees.
            The first dimension indicates the cluster index, the second dimension the ray index.
        """

        # Equation 7.5-15 of TR 138.901 v17.0.0
        scaling_factor_index = np.argmin(
            np.abs(self.__zenith_scaling_factors[:, 0] - cluster_powers.size)
        )
        zenith_scaling_factor = self.__zenith_scaling_factors[scaling_factor_index, 1]
        if state.line_of_sight:
            zenith_scaling_factor *= (
                1.3086 + 0.0339 * rice_factor - 0.0077 * rice_factor**2 + 2e-4 * rice_factor**3
            )

        # Generate angle starting point
        # Equation 7.5-14 of TR 138.901 v17.0.0
        zenith_centroids: np.ndarray = (
            -spread * np.log(cluster_powers / cluster_powers.max()) / zenith_scaling_factor
        )

        # Equation 7.5-16 of TR 138.901 v17.0.0
        # ToDo: Treat the BST-UT case!!!! (los_zenith = 90°)
        cluster_variation = self.__zenith_angle_variation_variable.sample(
            consistent_sample, 0.0, (spread / 7)
        )
        cluster_zenith = (
            self.__zenith_spread_sign[: cluster_powers.size] * zenith_centroids
            + cluster_variation[: cluster_powers.size]
        )

        # Equation 7.5-17 of TR 138.901 v17.0.0
        if state == LOSState.LOS:
            cluster_zenith += los_zenith - cluster_zenith[0]

        else:
            cluster_zenith += los_zenith

        # Spread the angles
        # Equation 7.5 -18 of TR 138.901 v17.0.0
        ray_offsets = self._cluster_zoa_spread(state) * self._ray_offset_angles
        ray_zenith = np.tile(cluster_zenith[:, None], len(ray_offsets)) + ray_offsets

        return ray_zenith

    def _ray_zod(
        self,
        consistent_sample: ConsistentSample,
        state: LSST,
        parameters: ClusterDelayLineSampleParameters,
        cluster_powers: np.ndarray,
        spread: float,
        rice_factor: float,
        los_zenith: float,
    ) -> np.ndarray:
        """Compute cluster ray zenith angles of departure.

        Args:

            consistent_sample:
                Sample of the consistent random process.

            state:
                Large scale state of the channel.

            cluster_powers:
                Vector of cluster powers. The length determines the number of clusters.

            spread:
                Zenith of departure angular spread in degrees.
                Denoted by :math:`\\mathrm{ZSD}` in the standard.

            rice_factor:
                Rice factor in dB.

            los_zenith:
                Line of sight zenith angle to the target in degrees.

        Returns:
            Matrix of angles in degrees.
            The first dimension indicates the cluster index, the second dimension the ray index.
        """

        # Equation 7.5-15 of TR 138.901 v17.0.0
        scaling_factor_index = np.argmin(
            np.abs(self.__zenith_scaling_factors[:, 0] - cluster_powers.size)
        )
        zenith_scaling_factor = self.__zenith_scaling_factors[scaling_factor_index, 1]
        if state.line_of_sight:
            zenith_scaling_factor *= (
                1.3086 + 0.0339 * rice_factor - 0.0077 * rice_factor**2 + 2e-4 * rice_factor**3
            )

        # Generate angle starting point
        # Equation 7.5-14 of TR 138.901 v17.0.0
        zenith_centroids: np.ndarray = (
            -spread * np.log(cluster_powers / cluster_powers.max()) / zenith_scaling_factor
        )

        # Equation 7.5-19 of TR 138.901 v17.0.0
        # ToDo: Treat the BST-UT case!!!! (los_zenith = 90°)
        cluster_variation = self.__zenith_angle_variation_variable.sample(
            consistent_sample, 0.0, (spread / 7)
        )
        cluster_zenith = (
            self.__zenith_spread_sign[: cluster_powers.size] * zenith_centroids
            + cluster_variation[: cluster_powers.size]
            + self._zod_offset(
                state, parameters
            )  # This line is the only difference to equation 7.5-16
        )

        # Equation 7.5-17 of TR 138.901 v17.0.0
        if state == LOSState.LOS:
            cluster_zenith += los_zenith - cluster_zenith[0]

        else:
            cluster_zenith += los_zenith

        # Spread the angles
        # Equation 7.5-20 of TR 138.901 v17.0.0
        zod_spread_mean = self._zod_spread_mean(state, parameters)
        ray_offsets = 3 / 8 * 10**zod_spread_mean * self._ray_offset_angles
        ray_zenith = np.tile(cluster_zenith[:, None], len(ray_offsets)) + ray_offsets

        return ray_zenith

    def _sample(self, channel_state: LinkState) -> ClusterDelayLineSample:

        # Compute the line of sight distance between the devices, i.e. base station to user terminal
        distance_3d: float = np.linalg.norm(channel_state.transmitter.position - channel_state.receiver.position, 2)  # type: ignore[assignment]
        distance_2d: float = np.linalg.norm(  # type: ignore[assignment]
            channel_state.transmitter.position[:2] - channel_state.receiver.position[:2], 2
        )
        base_height: float = max(
            channel_state.transmitter.position[2], channel_state.receiver.position[2]
        )
        terminal_height: float = min(
            channel_state.transmitter.position[2], channel_state.receiver.position[2]
        )
        parameters = ClusterDelayLineSampleParameters(
            channel_state.carrier_frequency, distance_3d, distance_2d, base_height, terminal_height
        )

        # Ensure transmiter and receiver are not located at identical positions
        if distance_3d == 0.0:
            raise RuntimeError(
                "Identical device positions violate the far-field assumption of the 3GPP CDL channel model"
            )

        # Decide on the overall large-scale state of the channel
        if self.expected_state is None:
            state_variable_sample = float(
                self.__state_variable.sample(
                    self.__state_realization.sample(
                        channel_state.transmitter.position, channel_state.receiver.position
                    )
                )
            )
            state = self._sample_large_scale_state(state_variable_sample, parameters)
        else:
            state = self.expected_state
        small_scale_realization = self._small_scale_realization(state)
        parameters_sample = small_scale_realization.sample(
            channel_state.transmitter.position, channel_state.receiver.position
        )

        # Sample large-scale parameters
        DS, ASD, ASA, ZSA, ZSD, K, SF = self._sample_large_scale_parameters(state, parameters)

        # Compute line of sight directions in local coordinates for both devices
        tx_los_direction = channel_state.transmitter.pose.invert().transform_direction(
            channel_state.receiver.position - channel_state.transmitter.position, True
        )
        rx_los_direction = channel_state.receiver.pose.invert().transform_direction(
            channel_state.transmitter.position - channel_state.receiver.position, True
        )

        # Extract directive angles in spherical coordinates
        tx_los_angles = tx_los_direction.to_spherical()
        rx_los_angles = rx_los_direction.to_spherical()

        raw_cluster_delays, cluster_delays, minimal_delay = self._cluster_delays(
            parameters_sample, state, DS, K
        )
        cluster_powers = self._cluster_powers(parameters_sample, state, DS, raw_cluster_delays, K)
        raw_cluster_delays = raw_cluster_delays[: cluster_powers.size]
        cluster_delays = cluster_delays[: cluster_powers.size]

        # Compute cluster angles
        ray_aod = (
            pi
            / 180
            * self._ray_azimuth_angles(
                parameters_sample,
                state,
                cluster_powers,
                ASD,
                K,
                180 * tx_los_angles[0] / pi,
                "departure",
            )
        )
        ray_aoa = (
            pi
            / 180
            * self._ray_azimuth_angles(
                parameters_sample,
                state,
                cluster_powers,
                ASA,
                K,
                180 * rx_los_angles[0] / pi,
                "arrival",
            )
        )
        ray_zod = (
            pi
            / 180
            * self._ray_zod(
                parameters_sample,
                state,
                parameters,
                cluster_powers,
                ZSD,
                K,
                180 * tx_los_angles[1] / pi,
            )
        )
        ray_zoa = (
            pi
            / 180
            * self._ray_zoa(
                parameters_sample, state, cluster_powers, ZSA, K, 180 * rx_los_angles[1] / pi
            )
        )

        # Couple cluster angles randomly
        # Step 8 in the small-scale parameter generation of ETSI TR 138.901 v17.0.0
        # This is equivalent of shuffeling the cluster ray indices according to the coupling indices
        # ToDo: Re-check if this is correct
        shuffled_ray_aoa = np.take_along_axis(
            ray_aoa, self.__angle_coupling_indices[0, : cluster_powers.size, ...], axis=1
        )
        shuffled_ray_zoa = np.take_along_axis(
            ray_zoa, self.__angle_coupling_indices[1, : cluster_powers.size, ...], axis=1
        )
        shuffled_ray_aod = np.take_along_axis(
            ray_aod, self.__angle_coupling_indices[2, : cluster_powers.size, ...], axis=1
        )
        shuffled_ray_zod = np.take_along_axis(
            ray_zod, self.__angle_coupling_indices[3, : cluster_powers.size, ...], axis=1
        )

        # Generate cross-polarization power ratios (step 9)
        # Equation 7.5-21 in TR 138.901 v17.0.0
        cross_polarization_factor = 10 ** (
            -0.05
            * self.__xpr_variable.sample(
                parameters_sample,
                self._cross_polarization_power_mean(state),
                self._cross_polarization_power_std(state),
            )
        )

        # Draw initial random phases (step 10)
        # A single 2x2 slice represents the jones matrix transforming the polarization of a single ray
        polarization_transformations = np.exp(2j * pi * self.__xpr_phase.sample(parameters_sample))
        polarization_transformations[0, 1, ::] *= cross_polarization_factor
        polarization_transformations[1, 0, ::] *= cross_polarization_factor

        # Compute a delay offset accounting for the time of flight over the line of sight
        # Equations 7.6-43 and 7.6-44 in TR 138.901 v17.0.0
        if self.__delay_normalization == DelayNormalization.TOF:
            delay_offset = distance_3d / speed_of_light
        else:
            delay_offset = 0.0

        # Step 12
        # Model pathloss
        # Table 7.4.1-1 in TR 138.901 v17.0.0
        path_power_scale = db2lin(-self._pathloss_dB(state, parameters))

        # Model oxygen power loss
        # 7.6-1 in TR 138.901 v17.0.0
        if self.__oxygen_absorption:
            oxygen_loss_factor = np.interp(
                channel_state.carrier_frequency * 1e-9,
                self.__oxygen_loss[0, :],
                self.__oxygen_loss[1, :],
            )
            oxygen_loss = 10 ** (
                -1e-2
                * oxygen_loss_factor
                * (distance_3d + speed_of_light * (cluster_delays + minimal_delay))
            )
        else:
            oxygen_loss = 1.0

        # Compute the expected amplitude scale factor of the sample
        expected_loss = self.gain * oxygen_loss
        true_cluster_powers = cluster_powers * path_power_scale * expected_loss**2

        return ClusterDelayLineSample(
            state.line_of_sight,
            K,
            shuffled_ray_aoa,
            shuffled_ray_zoa,
            shuffled_ray_aod,
            shuffled_ray_zod,
            delay_offset,
            cluster_delays,
            DS,
            true_cluster_powers,
            polarization_transformations,
            channel_state,
        )

    def _reciprocal_sample(
        self, sample: ClusterDelayLineSample, state: LinkState
    ) -> ClusterDelayLineSample:

        # Check if an optimized resampling is possible, otherwise sample the realization again
        # Resample if the sample parameters don't match
        if (
            state.bandwidth != sample.bandwidth
            or state.carrier_frequency != sample.carrier_frequency
        ):
            return self._sample(state)

        if np.any(state.transmitter.position != sample.receiver_pose.translation) or np.any(
            state.receiver.position != sample.transmitter_pose.translation
        ):
            return self._sample(state)

        return sample.reciprocal(state)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        ChannelRealization.serialize(self, process)
        if self.expected_state is not None:
            process.serialize_object(self.expected_state, "expected_state")
        process.serialize_object(self.__state_realization, "state_realization")
        process.serialize_object(
            ClusterDelayLineRealizationParameters(
                state_variable=self.__state_variable,
                delay_normalization=self.__delay_normalization,
                oxygen_absorption=self.__oxygen_absorption,
                large_scale_parameters=self.__large_scale_parameters,
                cluster_shadowing_log_variable=self.__cluster_shadowing_log_variable,
                cluster_delays_variable=self.__cluster_delays_variable,
                azimuth_angle_variation_variable=self.__azimuth_angle_variation_variable,
                azimuth_spread_sign=self.__azimuth_spread_sign,
                zenith_angle_variation_variable=self.__zenith_angle_variation_variable,
                zenith_spread_sign=self.__zenith_spread_sign,
                angle_coupling_indices=self.__angle_coupling_indices,
                cross_polarization_power_variable=self.__xpr_variable,
                cross_polarization_phase_variable=self.__xpr_phase,
            ),
            "parameters",
        )

    @classmethod
    @override
    def _DeserializeParameters(cls, process: DeserializationProcess) -> dict[str, object]:
        base_parameters = ChannelRealization._DeserializeParameters(process)
        base_parameters.update(
            {
                "state_realization": process.deserialize_object(
                    "state_realization", ConsistentRealization
                ),
                "parameters": process.deserialize_object(
                    "parameters", ClusterDelayLineRealizationParameters
                ),
            }
        )
        return base_parameters


CDLRT = TypeVar("CDLRT", bound=ClusterDelayLineRealization)
"""Type of a 3GPP Cluster Delay Line channel realization."""


class ClusterDelayLineBase(Channel[CDLRT, ClusterDelayLineSample], Generic[CDLRT, LSST]):
    """Base class for all 3GPP Cluster Delay Line channel models."""

    _DEFAULT_GAIN = 1.0
    _DEFAULT_DELAY_NORMALIZATION = DelayNormalization.ZERO
    _DEFAULT_OXYGEN_ABSORPTION = True

    __delay_normalization: DelayNormalization
    __oxygen_absorption: bool
    __expected_state: LSST | None

    def __init__(
        self,
        delay_normalization: DelayNormalization = _DEFAULT_DELAY_NORMALIZATION,
        oxygen_absorption: bool = _DEFAULT_OXYGEN_ABSORPTION,
        expected_state: LSST | None = None,
        gain: float = _DEFAULT_GAIN,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            gain:
                Linear gain factor a signal amplitude experiences when being propagated over this realization.
                :math:`1.0` by default.

            delay_normalization:
                The delay normalization routine applied during channel sampling.

            oxygen_absorption:
                Model oxygen absorption in the channel.
                Enabled by default.

            expected_state:
                Expected large-scale state of the channel.
                If :py:obj:`None`, the state is randomly generated during each sample of the channel's realization.

            gain:
                Linear channel energy gain factor.
                Initializes the :meth:`gain<hermespy.channel.channel.Channel.gain>` property.
                :math:`1.0` by default.

            seed:
                Seed used to initialize the pseudo-random number generator.
        """

        # Initialize base class
        Channel.__init__(self, gain, seed)

        # Initialize class attributes
        self.delay_normalization = delay_normalization
        self.oxygen_absorption = oxygen_absorption
        self.expected_state = expected_state

        # Generate dual consistent random variables
        self.__state_generator = ConsistentGenerator(self)
        self.__parameter_generator = ConsistentGenerator(self)

        # Configure the random variable distributions
        self.__state_variable = self.__state_generator.uniform()
        num_clusters = self.max_num_clusters
        num_rays = self.max_num_rays
        self.__cluster_shadowing_log_variable = self.__parameter_generator.gaussian((num_clusters,))
        self.__cluster_delays_variable = self.__parameter_generator.uniform((num_clusters,))
        self.__azimuth_variation_variable = self.__parameter_generator.gaussian((num_clusters,))
        self.__zenith_variation_variable = self.__parameter_generator.gaussian((num_clusters,))
        self.__cross_polarization_power_variable = self.__parameter_generator.gaussian(
            (num_clusters, num_rays)
        )
        self.__cross_polarization_phase_variable = self.__parameter_generator.uniform(
            (2, 2, num_clusters, num_rays)
        )

    @property
    def delay_normalization(self) -> DelayNormalization:
        """The delay normalization routine applied during channel sampling."""

        return self.__delay_normalization

    @delay_normalization.setter
    def delay_normalization(self, normalization: DelayNormalization) -> None:
        self.__delay_normalization = normalization

    @property
    def oxygen_absorption(self) -> bool:
        """Model oxygen absorption in the channel."""

        return self.__oxygen_absorption

    @oxygen_absorption.setter
    def oxygen_absorption(self, absorption: bool) -> None:
        self.__oxygen_absorption = absorption

    @property
    def expected_state(self) -> LSST | None:
        """Expected large-scale state of the channel.

        If `None`, the state is randomly generated during each sample of the channel's realization.
        """

        return self.__expected_state

    @expected_state.setter
    def expected_state(self, state: LSST | None) -> None:
        self.__expected_state = state

    @property
    @abstractmethod
    def max_num_clusters(self) -> int:
        """Maximum number of clusters a realization will generate."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def max_num_rays(self) -> int:
        """Maximum number of rays a realization will generate per cluster."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def _large_scale_correlations(self) -> np.ndarray:
        """Correlation factors between the large-scale parameters."""
        ...  # pragma: no cover

    @abstractmethod
    def _initialize_realization(
        self,
        state_generator: ConsistentGenerator,
        parameter_generator: ConsistentGenerator,
        parameters: ClusterDelayLineRealizationParameters,
    ) -> CDLRT:
        """Initialize a new realization of the channel model."""
        ...  # pragma: no cover

    @cache
    def __C_MM(self) -> np.ndarray:
        """Subroutine to compute the cross-correlation matrix of the large-scale parameters.

        Args:

            state: Large scale state index of the channel.

        Returns: Lower-triangular root of the cross-correlation matrix.
        """

        C_MM = np.empty((3, 7, 7), dtype=np.float64)
        for c, C in enumerate(self._large_scale_correlations):
            C_MM_squared = np.array(
                [
                    #    DS,    ASD,   ASA,   ZSA,   ZSD,   K,     SF
                    [1.0, C[0], C[1], C[15], C[14], C[8], C[4]],  # DS
                    [C[0], 1.0, C[5], C[17], C[16], C[6], C[3]],  # ASD
                    [C[1], C[5], 1.0, C[19], C[18], C[7], C[2]],  # ASA
                    [C[15], C[17], C[19], 1.0, C[20], C[13], C[11]],  # ZSA
                    [C[14], C[16], C[18], C[20], 1.0, C[12], C[10]],  # ZSD
                    [C[8], C[6], C[7], C[13], C[12], 1.0, C[9]],  # K
                    [C[4], C[3], C[2], C[11], C[10], C[9], 1.0],  # SF
                ],
                dtype=np.float64,
            )

            # Section 4 of ETSI TR 138.901 v17.0.0 hints at using the cholesky decomposition
            # to enforce the expected cross-correlations between the large-scale parameters
            C_MM[c, ...] = np.linalg.cholesky(C_MM_squared)

        return C_MM

    def _realize(self) -> CDLRT:

        # Realize static random variables
        azimuth_spread_sign = self._rng.choice([-1.0, 1.0], size=(self.max_num_clusters,))
        zenith_spread_sign = self._rng.choice([-1.0, 1.0], size=(self.max_num_clusters,))

        # Random coupling indices for step 8 in ETSI TR 138 901 v17.0.0
        angle_candidate_indices = np.arange(self.max_num_rays)
        angle_coupling_indices = np.array(
            [
                [
                    self._rng.permutation(angle_candidate_indices)
                    for _ in range(self.max_num_clusters)
                ]
                for _ in range(4)
            ]
        )

        # Generate standard normal realizations of the large-scale parameters
        LSPs = self.__C_MM() @ self._rng.standard_normal(7)

        parameters = ClusterDelayLineRealizationParameters(
            self.__state_variable,
            self.delay_normalization,
            self.oxygen_absorption,
            LSPs,
            self.__cluster_shadowing_log_variable,
            self.__cluster_delays_variable,
            self.__azimuth_variation_variable,
            azimuth_spread_sign,
            self.__zenith_variation_variable,
            zenith_spread_sign,
            angle_coupling_indices,
            self.__cross_polarization_power_variable,
            self.__cross_polarization_phase_variable,
        )

        return self._initialize_realization(
            self.__state_generator, self.__parameter_generator, parameters
        )

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.gain, "gain")
        process.serialize_object(self.delay_normalization, "delay_normalization")
        process.serialize_integer(self.oxygen_absorption, "oxygen_absorption")
        if self.expected_state is not None:
            process.serialize_object(self.expected_state, "expected_state")

    @classmethod
    def _DeserializeParameters(cls, process: DeserializationProcess) -> dict[str, object]:
        return {
            "gain": process.deserialize_floating("gain", cls._DEFAULT_GAIN),
            "delay_normalization": process.deserialize_object(
                "delay_normalization", DelayNormalization, cls._DEFAULT_DELAY_NORMALIZATION
            ),
            "oxygen_absorption": bool(
                process.deserialize_integer("oxygen_absorption", cls._DEFAULT_OXYGEN_ABSORPTION)
            ),
            "expected_state": process.deserialize_object("expected_state", LargeScaleState, None),
        }
