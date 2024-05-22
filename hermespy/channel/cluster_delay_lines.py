# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from enum import Enum
from functools import cached_property
from math import ceil, sin, cos, sqrt
from typing import Any, Generator, Literal, List, Tuple, Type, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from h5py import Group
from scipy.constants import pi, speed_of_light

from hermespy.core import (
    AntennaMode,
    ChannelStateInformation,
    ChannelStateFormat,
    Device,
    Direction,
    Executable,
    HDFSerializable,
    Serializable,
    Signal,
)
from hermespy.tools import db2lin
from .channel import Channel, ChannelRealization, InterpolationMode

if TYPE_CHECKING:
    from hermespy.simulation import SimulatedDevice  # pragma: no cover

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DelayNormalization(Enum):
    """Normalization routine applied to a set of sampled delays.

    Configuration option to :class:.ClusterDelayLineBase models.
    """

    ZERO = 0
    """Normalize the delays, so that the minimal delay is zero"""

    TOF = 1
    """The minimal delay is the time of flight between two devices"""

    NONE = 2
    """No delay normalization is applied.

    Only relevant for debugging purposes.
    """


class ClusterDelayLineRealization(ChannelRealization):
    """Realization of a 3GPP Cluster Delay Line channel model."""

    __line_of_sight: bool
    __rice_factor: float
    __azimuth_of_arrival: np.ndarray
    __zenith_of_arrival: np.ndarray
    __azimuth_of_departure: np.ndarray
    __zenith_of_departure: np.ndarray
    __cluster_delays: np.ndarray
    __cluster_delay_spread: float
    __cluster_powers: np.ndarray
    __polarization_transformations: np.ndarray
    __max_delay: float

    def __init__(
        self,
        alpha_device: Device,
        beta_device: Device,
        gain: float,
        line_of_sight: bool,
        rice_factor: float,
        azimuth_of_arrival: np.ndarray,
        zenith_of_arrival: np.ndarray,
        azimuth_of_departure: np.ndarray,
        zenith_of_departure: np.ndarray,
        cluster_delays: np.ndarray,
        cluster_delay_spread: float,
        cluster_powers: np.ndarray,
        polarization_transformations: np.ndarray,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> None:
        # Initialize base class
        ChannelRealization.__init__(self, alpha_device, beta_device, gain, interpolation_mode)

        # Initialize class members
        self.__line_of_sight = line_of_sight
        self.__rice_factor = rice_factor
        self.__azimuth_of_arrival = azimuth_of_arrival
        self.__zenith_of_arrival = zenith_of_arrival
        self.__azimuth_of_departure = azimuth_of_departure
        self.__zenith_of_departure = zenith_of_departure
        self.__cluster_delays = cluster_delays
        self.__cluster_delay_spread = cluster_delay_spread
        self.__cluster_powers = cluster_powers
        self.__polarization_transformations = polarization_transformations

        # Infer additional parameters
        self.__num_clusters = cluster_delays.shape[0]
        self.__num_rays = azimuth_of_arrival.shape[1]
        self.__max_delay = max(
            np.max(cluster_delays[:3] + cluster_delay_spread * 2.56), cluster_delays.max()
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

    def __ray_impulse_generator(
        self,
        transmitter: Device,
        receiver: Device,
        sampling_rate: float,
        num_samples: int,
        center_frequency: float,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, float], None, None]:
        """Sequential generation of individual ray impulse responses.

        Subroutine of :meth:`._propagate`.

        Args:

            transmitter (Device):
                Transmitting device.

            receiver (Device):
                Receiving device.

            sampling_rate (float):
                Sampling rate in Hertz.

            num_samples (int):
                Number of samples to generate.

            center_frequency (float):
                Center frequency in Hertz.
        """

        # First summand scaling of equation 7.5-30
        rice_factor_lin = db2lin(self.rice_factor)
        nlos_scale = (1 + rice_factor_lin) ** -0.5 if self.line_of_sight else 1.0

        # Compute the number of clusters, considering the first two clusters get split into 3 partitions
        num_split_clusters = min(2, self.num_clusters)
        virtual_num_clusters = 3 * num_split_clusters + max(0, self.num_clusters - 2)

        # Prepare the cluster delays, equation 7.5-26
        subcluster_delays: np.ndarray = np.repeat(
            self.cluster_delays[:num_split_clusters, None], 3, axis=1
        ) + self.cluster_delay_spread * np.array([1.0, 1.28, 2.56])
        virtual_cluster_delays = np.concatenate(
            (subcluster_delays.flatten(), self.cluster_delays[num_split_clusters:])
        )

        # Wavelength factor
        wavelength_factor = center_frequency / speed_of_light
        relative_velocity = receiver.velocity - transmitter.velocity
        fast_fading = wavelength_factor * np.arange(num_samples) / sampling_rate

        # Compute the cluster propagation parameters
        for subcluster_idx in range(0, virtual_num_clusters):
            cluster_idx = int(subcluster_idx / 3) if subcluster_idx < 6 else subcluster_idx - 4
            ray_indices = (
                ClusterDelayLineBase.subcluster_indices[cluster_idx]
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
                tx_array_response = transmitter.antennas.cartesian_array_response(
                    center_frequency, tx_direction.view(np.ndarray), "global", AntennaMode.TX
                ).conj()
                rx_array_response = receiver.antennas.cartesian_array_response(
                    center_frequency, rx_direction.view(np.ndarray), "global", AntennaMode.RX
                )

                channel: np.ndarray = (
                    rx_array_response
                    @ jones
                    @ tx_array_response.T
                    * (sqrt(self.cluster_powers[cluster_idx] / self.num_clusters) * nlos_scale)
                )

                wave_vector = np.array(
                    [cos(aoa) * sin(zoa), sin(aoa) * sin(zoa), cos(zoa)], dtype=float
                )
                impulse: np.ndarray = np.exp(
                    np.inner(wave_vector, relative_velocity) * fast_fading * 2j * pi
                )

                yield channel, impulse, delay

        # If line of sight is enabled, yield an additional parameter touple for the line of sight component
        if self.line_of_sight:
            device_vector = receiver.position - transmitter.position
            los_distance = np.linalg.norm(device_vector, 2)
            rx_wave_vector = device_vector / los_distance

            # Equation 7.5-29
            tx_array_response = transmitter.antennas.cartesian_array_response(
                center_frequency, receiver.global_position, "global", AntennaMode.TX
            ).conj()
            rx_array_response = receiver.antennas.cartesian_array_response(
                center_frequency, transmitter.global_position, "global", AntennaMode.RX
            )

            # Second summand of equation 7.5-30
            los_delay: float = self.cluster_delays[0]
            los_channel: np.ndarray = (
                rx_array_response @ tx_array_response.T * (rice_factor_lin / 1 + rice_factor_lin)
            )
            los_impulse: np.ndarray = np.exp(-2j * pi * los_distance * wavelength_factor) * np.exp(
                np.inner(rx_wave_vector, relative_velocity) * fast_fading * 2j * pi
            )

            yield los_channel, los_impulse, los_delay

    def _propagate(
        self,
        signal: Signal,
        transmitter: Device,
        receiver: Device,
        interpolation: InterpolationMode,
    ) -> Signal:
        max_delay_in_samples = ceil(self.max_delay * signal.sampling_rate)
        num_resulting_streams = receiver.antennas.num_receive_antennas
        propagated_samples = np.zeros(
            (num_resulting_streams, signal.num_samples + max_delay_in_samples), dtype=np.complex_
        )

        # Prepare the optimal einsum path ahead of time for faster execution
        channel = np.zeros(
            (num_resulting_streams, transmitter.antennas.num_transmit_antennas), dtype=np.complex_
        )
        impulse = np.zeros(signal.num_samples, dtype=np.complex_)
        einsum_subscripts = "ij,jk,k->ik"
        einsum_path = np.einsum_path(
            einsum_subscripts, channel, signal[:, :], impulse, optimize="optimal"
        )[0]

        for channel, impulse, delay in self.__ray_impulse_generator(
            transmitter,
            receiver,
            signal.sampling_rate,
            signal.num_samples,
            signal.carrier_frequency,
        ):
            if interpolation == InterpolationMode.NEAREST:
                delay_in_samples = int(delay * signal.sampling_rate)
                propagated_samples[
                    :, delay_in_samples : delay_in_samples + signal.num_samples
                ] += np.einsum(
                    einsum_subscripts, channel, signal[:, :], impulse, optimize=einsum_path
                )

        return signal.from_ndarray(propagated_samples)

    def state(
        self,
        transmitter: Device,
        receiver: Device,
        delay_offset: float,
        sampling_rate: float,
        num_samples: int,
        max_num_taps: int,
    ) -> ChannelStateInformation:
        carrier_frequency = transmitter.carrier_frequency
        max_delay_in_samples = min(max_num_taps, ceil(self.max_delay * sampling_rate))
        raw_state = np.zeros(
            (
                receiver.antennas.num_receive_antennas,
                transmitter.antennas.num_transmit_antennas,
                num_samples,
                1 + max_delay_in_samples,
            ),
            dtype=np.complex_,
        )

        for channel, impulse, delay in self.__ray_impulse_generator(
            transmitter, receiver, sampling_rate, num_samples, carrier_frequency
        ):
            # Compute the delay in samples
            delay_tap_index = int(delay * sampling_rate)

            if delay_tap_index >= max_num_taps:
                continue

            # Compute the resulting channel state
            raw_state[:, :, :, delay_tap_index] += np.einsum(
                "ij,k->ijk", channel, impulse, optimize=False
            )

        raw_state *= self.gain**0.5
        return ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, raw_state)

    def plot_angles(self, title: str | None = None, center_los: bool = True) -> plt.Figure:
        delta = self.alpha_device.position - self.beta_device.position
        los_zenith = np.arccos((-delta[2], delta[2]))
        los_azimuth = np.arctan((-delta[1] / delta[0], delta[1] / delta[0]))

        azimuth_offset = -los_azimuth if center_los else np.zeros(2)
        zenith_offset = -los_zenith if center_los else np.zeros(2)

        with Executable.style_context():
            figure, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"})
            num_colors = len(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        markers = ["x", "+", "o"]

        for i, (azimuth_clusters, zenith_clusters) in enumerate(
            [
                [self.azimuth_of_arrival, self.zenith_of_arrival],
                [self.azimuth_of_departure, self.zenith_of_departure],
            ]
        ):
            for azimuth_rays, zenith_rays in zip(azimuth_clusters, zenith_clusters):
                azimuth_points = azimuth_rays + azimuth_offset[0]
                zenith_points = abs(180 * (zenith_rays + zenith_offset[0]) / pi) % 180

                # azimuth_points[zenith_points > 90] -= pi
                zenith_points[zenith_points > 90] = 90 - zenith_points[zenith_points > 90] % 90

                axes[i].scatter(azimuth_points, zenith_points, marker=markers[int(i / num_colors)])

        # Plot line of sight indicators

        for axis_index, (azimuth, zenith) in enumerate(
            zip(los_azimuth + azimuth_offset, los_zenith + zenith_offset)
        ):
            axes[axis_index].scatter(azimuth, abs(180 * zenith / pi), marker="o", color="b")

            axes[axis_index].set_rmax(90)
            axes[axis_index].grid(True)
            axes[axis_index].set_xlabel("Azimuth")
            axes[axis_index].set_ylabel("Zenith")
            axes[axis_index].set_rlabel_position(22.5)

        figure.suptitle("Angle Realizations" if title is None else title)
        axes[0].set_title("Angles of Arrival")
        axes[1].set_title("Angles of Departure")

        return figure

    def plot_power_delay(self, title: str | None = None) -> plt.Figure:
        with Executable.style_context():
            figure, axis = plt.subplots()
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for c, (delay, power) in enumerate(zip(self.cluster_delays, self.cluster_powers)):
            stem = axis.stem(delay, power, markerfmt="x")
            plt.setp(stem[0], "color", colors[c % len(colors)])
            plt.setp(stem[1], "color", colors[c % len(colors)])

        figure.suptitle("Power Delay Profile" if title is None else title)
        axis.set_yscale("log")
        axis.xaxis.set_label_text("Delay")
        axis.yaxis.set_label_text("Power")

        return figure

    def plot_rays(self, title: str | None = None) -> plt.Figure:
        with Executable.style_context():
            figure, axis = plt.subplots(subplot_kw={"projection": "3d"})
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        magnitude = np.linalg.norm(self.alpha_device.position - self.beta_device.position)

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
                (tx_vectors, rx_vectors), (self.alpha_device.position, self.beta_device.position)
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
        axis.scatter(*self.alpha_device.position, marker="o", color="w")
        axis.text(*self.alpha_device.position, "Tx", zorder=1)
        axis.scatter(*self.beta_device.position, marker="o", color="w")
        axis.text(*self.beta_device.position, "Rx", zorder=1)

        figure.suptitle("Rays" if title is None else title)
        return figure

    @staticmethod
    def _angular_spread(angles: np.ndarray, powers: np.ndarray) -> float:
        """Compute the angular spread of a set of data.

        Implements equation (A-1) of :footcite:t:`3GPP:TR38901`.

        Args:

            angles (np.ndarray):
                Numpy array of angles in radians for all paths of propagation.

            powers (np.ndarray):
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

    def to_HDF(self, group: Group) -> None:
        """Serialize the object state to HDF5.

        Dumps the object's state and additional information to a HDF5 group.

        Args:

            group (Group):
                The HDF5 group to which the object is serialized.
        """

        # Serialize base class
        ChannelRealization.to_HDF(self, group)

        # Serialize attributes
        group.attrs["line_of_sight"] = self.line_of_sight
        group.attrs["rice_factor"] = self.rice_factor
        group.attrs["cluster_delay_spread"] = self.cluster_delay_spread

        # Serialize datasets
        HDFSerializable._write_dataset(group, "azimuth_of_arrival", self.azimuth_of_arrival)
        HDFSerializable._write_dataset(group, "zenith_of_arrival", self.zenith_of_arrival)
        HDFSerializable._write_dataset(group, "azimuth_of_departure", self.azimuth_of_departure)
        HDFSerializable._write_dataset(group, "zenith_of_departure", self.zenith_of_departure)
        HDFSerializable._write_dataset(group, "cluster_delays", self.cluster_delays)
        HDFSerializable._write_dataset(group, "cluster_powers", self.cluster_powers)
        HDFSerializable._write_dataset(
            group, "polarization_transformations", self.polarization_transformations
        )

    @classmethod
    def From_HDF(
        cls: Type[ClusterDelayLineRealization],
        group: Group,
        alpha_device: Device,
        beta_device: Device,
    ) -> ClusterDelayLineRealization:
        # Deserialize base class parameters
        params = cls._parameters_from_HDF(group)

        # Deserialize attributes
        params["line_of_sight"] = group.attrs["line_of_sight"]
        params["rice_factor"] = group.attrs["rice_factor"]
        params["cluster_delay_spread"] = group.attrs["cluster_delay_spread"]

        # Deserialize datasets
        params["azimuth_of_arrival"] = np.array(group["azimuth_of_arrival"], dtype=np.float_)
        params["zenith_of_arrival"] = np.array(group["zenith_of_arrival"], dtype=np.float_)
        params["azimuth_of_departure"] = np.array(group["azimuth_of_departure"], dtype=np.float_)
        params["zenith_of_departure"] = np.array(group["zenith_of_departure"], dtype=np.float_)
        params["cluster_delays"] = np.array(group["cluster_delays"], dtype=np.float_)
        params["cluster_powers"] = np.array(group["cluster_powers"], dtype=np.float_)
        params["polarization_transformations"] = np.array(
            group["polarization_transformations"], dtype=np.complex_
        )

        # Initialize cdl realization instance
        return cls(alpha_device, beta_device, **params)


class ClusterDelayLineBase(Channel[ClusterDelayLineRealization]):
    delay_normalization: DelayNormalization
    """The delay normalization routine applied during channel sampling."""

    # Cluster scaling factors for the angle of arrival
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
        dtype=float,
    )

    __zenith_scaling_factors = np.array(
        [
            [8, 0.889],
            [10, 0.957],
            [11, 1.031],
            [12, 1.104],
            [15, 1.108],
            [19, 1.184],
            [20, 1.178],
            [25, 1.282],
        ],
        dtype=float,
    )

    # Ray offset angles
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
        ]
    )

    # Sub-cluster partitions for the three strongest clusters
    subcluster_indices: List[List[int]] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 18, 19],
        [8, 9, 10, 11, 16, 17],
        [12, 13, 14, 15],
    ]

    def __init__(
        self,
        alpha_device: SimulatedDevice | None = None,
        beta_device: SimulatedDevice | None = None,
        gain: float = 1.0,
        delay_normalization: DelayNormalization = DelayNormalization.ZERO,
        **kwargs,
    ) -> None:
        """
        Args:

            alpha_device (SimulatedDevice, optional):
                First device linked by the :class:`.ClusterDelayLineBase` instance that generated this realization.

            beta_device (SimulatedDevice, optional):
                Second device linked by the :class:`.ClusterDelayLineBase` instance that generated this realization.

            gain (float, optional):
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default.

            delay_normalization (DelayNormalization, optional):

                The delay normalization routine applied during channel sampling.
        """

        # Initialize base class
        Channel.__init__(self, alpha_device, beta_device, gain, **kwargs)

        # Initialize class attributes
        self.delay_normalization = delay_normalization

    @property
    @abstractmethod
    def line_of_sight(self) -> bool:
        """Does this model assume direct line of sight between the two devices?

        Referred to as :math:`LOS` within the standard.

        Returns:
            bool: Line of sight indicator.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def delay_spread_mean(self) -> float:
        """Mean of the cluster delay spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{DS}` and :math:`\\mu_{\\mathrm{lgDS}}` within the the standard, respectively.

        Returns:
            float: Mean delay spread in seconds.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def delay_spread_std(self) -> float:
        """Standard deviation of the cluster delay spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{DS}` and :math:`\\sigma_{\\mathrm{lgDS}}` within the the standard, respectively.

        Returns:
            float: Delay spread standard deviation in seconds.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def aod_spread_mean(self) -> float:
        """Mean of the Azimuth Angle-of-Departure spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{ASD}` and :math:`\\mu_{\\mathrm{lgASD}}` within the the standard, respectively.

        Returns:
            float: Mean angle spread in seconds
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def aod_spread_std(self) -> float:
        """Standard deviation of the Azimuth Angle-of-Departure spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{ASD}` and :math:`\\sigma_{\\mathrm{lgASD}}` within the the standard, respectively.

        Returns:
            float: Angle spread standard deviation in seconds.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def aoa_spread_mean(self) -> float:
        """Mean of the Azimuth Angle-of-Arriaval spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{ASA}` and :math:`\\mu_{\\mathrm{lgASA}}` within the the standard, respectively.

        Returns:
            float: Mean angle spread in seconds
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def aoa_spread_std(self) -> float:
        """Standard deviation of the Azimuth Angle-of-Arrival spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{ASA}` and :math:`\\sigma_{\\mathrm{lgASA}}` within the the standard, respectively.

        Returns:
            float: Angle spread standard deviation in seconds.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def zoa_spread_mean(self) -> float:
        """Mean of the Zenith Angle-of-Arriaval spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{ZSA}` and :math:`\\mu_{\\mathrm{lgZSA}}` within the the standard, respectively.

        Returns:
            float: Mean angle spread in seconds
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def zoa_spread_std(self) -> float:
        """Standard deviation of the Zenith Angle-of-Arrival spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{ZSA}` and :math:`\\sigma_{\\mathrm{lgZSA}}` within the the standard, respectively.

        Returns:
            float: Angle spread standard deviation in seconds.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def zod_spread_mean(self) -> float:
        """Mean of the Zenith Angle-of-Departure spread.

        The spread realization and its mean are referred to as
        :math:`\\mathrm{ZOD}` and :math:`\\mu_{\\mathrm{lgZSD}}` within the the standard, respectively.

        Returns:
            float: Mean angle spread in degrees
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def zod_spread_std(self) -> float:
        """Standard deviation of the Zenith Angle-of-Departure spread.

        The spread realization and its standard deviation are referred to as
        :math:`\\mathrm{ZOD}` and :math:`\\sigma_{\\mathrm{lgZOD}}` within the the standard, respectively.

        Returns:
            float: Angle spread standard deviation in degrees.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def zod_offset(self) -> float:
        """Offset between Zenith Angle-of-Arrival and Angle-of-Departure.

        The offset is referred to as :math:`\\mu_{\\mathrm{offset,ZOD}}` within the standard.

        Returns:
            float: The offset in degrees.
        """
        ...  # pragma: no cover

    ###############################
    # ToDo: Shadow fading function
    ###############################

    @property
    @abstractmethod
    def rice_factor_mean(self) -> float:
        """Mean of the rice factor distribution.

        The rice factor realization and its mean are referred to as
        :math:`K` and :math:`\\mu_K` within the the standard, respectively.

        Returns:
            float: Rice factor mean in dB.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def rice_factor_std(self) -> float:
        """Standard deviation of the rice factor distribution.

        The rice factor realization and its standard deviation are referred to as
        :math:`K` and :math:`\\sigma_K` within the the standard, respectively.

        Returns:
            float: Rice factor standard deviation in dB.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def delay_scaling(self) -> float:
        """Delay scaling proportionality factor

        Referred to as :math:`r_{\\tau}` within the standard.

        Returns:
            float: Scaling factor.

        Raises:
            ValueError:
                If scaling factor is smaller than one.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def cross_polarization_power_mean(self) -> float:
        """Mean of the cross-polarization power.

        The cross-polarization power and its mean are referred to as
        :math:`\\mathrm{XPR}` and :math:`\\mu_{\\mathrm{XPR}}` within the the standard, respectively.

        Returns:
            float: Mean power in dB.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def cross_polarization_power_std(self) -> float:
        """Standard deviation of the cross-polarization power.

        The cross-polarization power and its standard deviation are referred to as
        :math:`\\mathrm{XPR}` and :math:`\\sigma_{\\mathrm{XPR}}` within the the standard, respectively.

        Returns:
            float: Power standard deviation in dB.

        Raises:
            ValueError: If the standard deviation is smaller than zero.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def num_clusters(self) -> int:
        """Number of clusters.

        Referred to as :math:`M` within the standard.

        Returns:
            int: Number of clusters.

        Raises:
            ValueError: If the number of clusters is smaller than one.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def num_rays(self) -> int:
        """Number of rays per cluster.

        Referred to as :math:`N` within the standard.

        Returns:
            int: Number of rays.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def cluster_delay_spread(self) -> float:
        """Delay spread within an individual cluster.

        Referred to as :math:`c_{DS}` within the standard.

        Returns:
            float: Delay spread in seconds.

        Raises:
            ValueError: If spread is smaller than zero.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def cluster_aod_spread(self) -> float:
        """Azimuth Angle-of-Departure spread within an individual cluster.

        Referred to as :math:`c_{ASD}` within the standard.

        Returns:
            float: Angle spread in degrees.

        Raises:
            ValueError: If spread is smaller than zero.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def cluster_aoa_spread(self) -> float:
        """Azimuth Angle-of-Arrival spread within an individual cluster.

        Referred to as :math:`c_{ASA}` within the standard.

        Returns:
            float: Angle spread in degrees.

        Raises:
            ValueError: If spread is smaller than zero.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def cluster_zoa_spread(self) -> float:
        """Zenith Angle-of-Arrival spread within an individual cluster.

        Referred to as :math:`c_{ZSA}` within the standard.

        Returns:
            float: Angle spread in degrees.

        Raises:
            ValueError: If spread is smaller than zero.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def cluster_shadowing_std(self) -> float:
        """Standard deviation of the cluster shadowing.

        Referred to as :math:`\\zeta` within the the standard.

        Returns:
            float: Cluster shadowing standard deviation.

        Raises:
            ValueError: If the deviation is smaller than zero.
        """
        ...  # pragma: no cover

    def _cluster_delays(
        self, delay_spread: float, rice_factor: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a single sample set of normalized cluster delays.

        A single cluster delay is referred to as :math:`\\tau_n` within the the standard.

        Args:

            delay_spread (float):
                Delay spread in seconds.

            rice_factor (float):
                Rice factor K in dB.

        Returns:

            Tuple of
                - DRaw delay samples
                - True delays
        """

        # Generate delays according to the configured spread and scales
        raw_delays = (
            -self.delay_scaling * delay_spread * np.log(self._rng.uniform(size=self.num_clusters))
        )

        # Sort the delays in ascending order
        raw_delays.sort()

        # Normalize delays if the respective flag is enabled
        if (
            self.delay_normalization == DelayNormalization.ZERO
            or self.delay_normalization == DelayNormalization.TOF
        ):
            raw_delays -= raw_delays[0]

        # Scale delays, if required by the configuration
        scaled_delays = raw_delays.copy()

        # In case of line of sight, scale the delays by the appropriate K-factor
        if self.line_of_sight:
            rice_scale = (
                0.775 - 0.0433 * rice_factor + 2e-4 * rice_factor**2 + 17e-6 * rice_factor**3
            )
            scaled_delays /= rice_scale

        # Account for the time of flight over the line of sight, if required
        if self.delay_normalization == DelayNormalization.TOF:
            time_of_flight = (
                np.linalg.norm(self.alpha_device.position - self.beta_device.position, 2)
                / speed_of_light
            )
            scaled_delays += time_of_flight

        # Return the raw and scaled delays, since they are both required for further processing
        return raw_delays, scaled_delays

    def _cluster_powers(
        self, delay_spread: float, delays: np.ndarray, rice_factor: float
    ) -> np.ndarray:
        """Compute a single sample set of normalized cluster power factors from delays.

        A single cluster power factor is referred to as :math:`P_n` within the the standard.

        Args:

            delay_spread (float):
                Delay spread in seconds.

            delays (np.ndarray):
                Vector of cluster delays.

            rice_factor (float):
                Rice factor K in dB.

        Returns:

            np.ndarray:
                Vector of cluster power scales.
        """

        shadowing = 10 ** (
            -0.1 * self._rng.normal(scale=self.cluster_shadowing_std, size=delays.shape)
        )
        powers = (
            np.exp(-delays * (self.delay_scaling - 1) / (self.delay_scaling * delay_spread))
            * shadowing
        )

        # In case of line of sight, add a specular component to the cluster delays
        if self.line_of_sight:
            linear_rice_factor = db2lin(rice_factor)
            powers /= (1 + linear_rice_factor) * np.sum(powers.flat)
            powers[0] += linear_rice_factor / (1 + linear_rice_factor)

        else:
            powers /= np.sum(powers.flat)

        return powers

    def _ray_azimuth_angles(
        self,
        cluster_powers: np.ndarray,
        rice_factor: float,
        los_azimuth: float,
        direction: Literal["arrival", "departure"],
    ) -> np.ndarray:
        """Compute cluster ray azimuth angles of arrival or departure.

        Args:

            cluster_powers (np.ndarray):
                Vector of cluster powers. The length determines the number of clusters.

            rice_factor (float):
                Rice factor in dB.

            los_azimuth (float):
                Line of sight azimuth angle to the target in degrees.

        Returns:
            np.ndarray:
                Matrix of angles in degrees.
                The first dimension indicates the cluster index, the second dimension the ray index.
        """

        # Determine the closest scaling factor
        scale_index = np.argmin(np.abs(self.__azimuth_scaling_factors[:, 0] - len(cluster_powers)))
        angle_scale = self.__azimuth_scaling_factors[scale_index, 1]
        size = cluster_powers.shape

        # Scale the scale (hehe) in the line of sight case
        if self.line_of_sight:
            angle_scale *= (
                1.1035 - 0.028 * rice_factor - 2e-3 * rice_factor**2 + 1e-4 * rice_factor**3
            )

        # Draw azimuth angle spread from the distribution
        spread_mean = self.aoa_spread_mean if direction == "arrival" else self.aod_spread_mean
        spread_std = self.aoa_spread_std if direction == "arrival" else self.aod_spread_std
        spread = 10 ** self._rng.normal(spread_mean, spread_std, size=size)

        angles: np.ndarray = (
            2
            * (spread / 1.4)
            * np.sqrt(-np.log(cluster_powers / cluster_powers.max()))
            / angle_scale
        )

        # Assign positive / negative integers and add some noise
        angle_variation = self._rng.normal(0.0, (spread / 7) ** 2, size=size)
        angle_spread_sign = self._rng.choice([-1.0, 1.0], size=size)
        spread_angles = angle_spread_sign * angles + angle_variation

        # Add the actual line of sight term
        if self.line_of_sight:
            # The first angle within the list is exactly the line of sight component
            spread_angles += los_azimuth - spread_angles[0]

        else:
            spread_angles += los_azimuth

        # Spread the angles
        cluster_spread = (
            self.cluster_aoa_spread if direction == "arrival" else self.cluster_aod_spread
        )
        ray_offsets = cluster_spread * self._ray_offset_angles
        ray_angles = np.tile(spread_angles[:, None], len(ray_offsets)) + ray_offsets

        return ray_angles

    def _ray_zoa(
        self, cluster_powers: np.ndarray, rice_factor: float, los_zenith: float
    ) -> np.ndarray:
        """Compute cluster ray zenith angles of arrival.

        Args:

            cluster_powers (np.ndarray):
                Vector of cluster powers. The length determines the number of clusters.

            rice_factor (float):
                Rice factor in dB.

            los_zenith (float):
                Line of sight zenith angle to the target in degrees.

        Returns:
            np.ndarray:
                Matrix of angles in degrees.
                The first dimension indicates the cluster index, the second dimension the ray index.
        """

        size = cluster_powers.shape

        # Select the scaling factor
        scale_index = np.argmin(np.abs(self.__zenith_scaling_factors[:, 0] - len(cluster_powers)))
        zenith_scale = self.__zenith_scaling_factors[scale_index, 1]

        if self.line_of_sight:
            zenith_scale *= (
                1.3086 + 0.0339 * rice_factor - 0.0077 * rice_factor**2 + 2e-4 * rice_factor**3
            )

        # Draw zenith angle spread from the distribution
        zenith_spread = 10 ** self._rng.normal(self.zoa_spread_mean, self.zoa_spread_std)

        # Generate angle starting point
        zenith_centroids: np.ndarray = (
            -zenith_spread * np.log(cluster_powers / cluster_powers.max()) / zenith_scale
        )

        cluster_variation = self._rng.normal(0.0, (zenith_spread / 7) ** 2, size=size)
        cluster_sign = self._rng.choice([-1.0, 1.0], size=size)

        # ToDo: Treat the BST-UT case!!!! (los_zenith = 90°)
        cluster_zenith = cluster_sign * zenith_centroids + cluster_variation

        if self.line_of_sight:
            cluster_zenith += los_zenith - cluster_zenith[0]

        else:
            cluster_zenith += los_zenith

        # Spread the angles
        ray_offsets = self.cluster_zoa_spread * self._ray_offset_angles
        ray_zenith = np.tile(cluster_zenith[:, None], len(ray_offsets)) + ray_offsets

        return ray_zenith

    def _ray_zod(
        self, cluster_powers: np.ndarray, rice_factor: float, los_zenith: float
    ) -> np.ndarray:
        """Compute cluster ray zenith angles of departure.

        Args:

            cluster_powers (np.ndarray):
                Vector of cluster powers. The length determines the number of clusters.

            rice_factor (float):
                Rice factor in dB.

            los_zenith (float):
                Line of sight zenith angle to the target in degrees.

        Returns:
            np.ndarray:
                Matrix of angles in degrees.
                The first dimension indicates the cluster index, the second dimension the ray index.
        """

        size = cluster_powers.shape

        # Select the scaling factor
        scale_index = np.argmin(np.abs(self.__zenith_scaling_factors[:, 0] - len(cluster_powers)))
        zenith_scale = self.__zenith_scaling_factors[scale_index, 1]

        if self.line_of_sight:
            zenith_scale *= (
                1.3086 + 0.0339 * rice_factor - 0.0077 * rice_factor**2 + 2e-4 * rice_factor**3
            )

        # Draw zenith angle spread from the distribution
        zenith_spread = 10 ** self._rng.normal(self.zod_spread_mean, self.zod_spread_std, size=size)

        # Generate angle starting point
        zenith_centroids: np.ndarray = (
            -zenith_spread * np.log(cluster_powers / cluster_powers.max()) / zenith_scale
        )

        cluster_variation = self._rng.normal(0.0, (zenith_spread / 7) ** 2, size=size)
        cluster_sign = self._rng.choice([-1.0, 1.0], size=size)

        # ToDo: Treat the BST-UT case!!!! (los_zenith = 90°)
        # Equation 7.5-19
        cluster_zenith = cluster_sign * zenith_centroids + cluster_variation + self.zod_offset

        if self.line_of_sight:
            cluster_zenith += los_zenith - cluster_zenith[0]

        else:
            cluster_zenith += los_zenith

        # Spread the angles
        # Equation 7.5 -20
        ray_offsets = 3 / 8 * 10**self.zoa_spread_mean * self._ray_offset_angles
        ray_zenith = np.tile(cluster_zenith[:, None], len(ray_offsets)) + ray_offsets

        return ray_zenith

    def _realize(self) -> ClusterDelayLineRealization:
        delay_spread = 10 ** self._rng.normal(self.delay_spread_mean, self.delay_spread_std)
        rice_factor = self._rng.normal(loc=self.rice_factor_mean, scale=self.rice_factor_std)

        # Compute the line of sight distance
        los_distance = np.linalg.norm(
            self.alpha_device.global_position - self.beta_device.global_position, 2
        )

        # Ensure transmiter and receiver are not located at identical positions
        if los_distance == 0.0:
            raise RuntimeError(
                "Identical device positions violate the far-field assumption of the 3GPP CDL channel model"
            )

        # Compute line of sight directions in local coordinates for both devices
        tx_los_direction = self.alpha_device.backwards_transformation.transform_direction(
            self.beta_device.global_position, True
        )
        rx_los_direction = self.beta_device.backwards_transformation.transform_direction(
            self.alpha_device.global_position, True
        )

        # Extract directive angles in spherical coordinates
        tx_los_angles = tx_los_direction.to_spherical()
        rx_los_angles = rx_los_direction.to_spherical()

        # Compute cluster delays and powers
        num_clusters = self.num_clusters
        num_rays = self.num_rays

        raw_cluster_delays, cluster_delays = self._cluster_delays(delay_spread, rice_factor)
        cluster_powers = self._cluster_powers(delay_spread, raw_cluster_delays, rice_factor)

        # Compute cluster angles
        ray_aod = (
            pi
            / 180
            * self._ray_azimuth_angles(
                cluster_powers, rice_factor, 180 * tx_los_angles[0] / pi, "departure"
            )
        )
        ray_aoa = (
            pi
            / 180
            * self._ray_azimuth_angles(
                cluster_powers, rice_factor, 180 * rx_los_angles[0] / pi, "arrival"
            )
        )
        ray_zod = pi / 180 * self._ray_zod(cluster_powers, rice_factor, 180 * tx_los_angles[1] / pi)
        ray_zoa = pi / 180 * self._ray_zoa(cluster_powers, rice_factor, 180 * rx_los_angles[1] / pi)

        # Couple cluster angles randomly (step 8)
        # This is equivalent to shuffeling the angles within each cluster set
        for ray_angles in (ray_aod, ray_aoa, ray_zod, ray_zoa):
            for a in ray_angles:
                self._rng.shuffle(a)

        # Generate cross-polarization power ratios (step 9)
        cross_polarization = 10 ** (
            0.1
            * self._rng.normal(
                self.cross_polarization_power_mean,
                self.cross_polarization_power_std,
                size=(num_clusters, num_rays),
            )
        )

        # Draw initial random phases (step 10)
        # A single 2x2 slice represents the jones matrix transforming the polarization of a single ray
        polarization_transformations = np.exp(
            2j * pi * self._rng.uniform(size=(2, 2, num_clusters, num_rays))
        )
        polarization_transformations[0, 1, ::] *= cross_polarization**-0.5
        polarization_transformations[1, 0, ::] *= cross_polarization**-0.5

        return ClusterDelayLineRealization(
            self.alpha_device,
            self.beta_device,
            self.gain,
            self.line_of_sight,
            rice_factor,
            ray_aoa,
            ray_zoa,
            ray_aod,
            ray_zod,
            cluster_delays,
            self.cluster_delay_spread,
            cluster_powers,
            polarization_transformations,
        )

    @property
    def _center_frequency(self) -> float:
        return 0.5 * (self.alpha_device.carrier_frequency + self.beta_device.carrier_frequency)

    def recall_realization(self, group: Group) -> ClusterDelayLineRealization:
        return ClusterDelayLineRealization.From_HDF(group, self.alpha_device, self.beta_device)


class ClusterDelayLine(ClusterDelayLineBase, Serializable):
    """3GPP Cluster Delay Line Channel Model."""

    yaml_tag = "ClusterDelayLine"

    __line_of_sight: bool
    __delay_spread_mean: float
    __delay_spread_std: float
    __aod_spread_mean: float
    __aod_spread_std: float
    __aoa_spread_mean: float
    __aoa_spread_std: float
    __zoa_spread_mean: float
    __zoa_spread_std: float
    __zod_spread_mean: float
    __zod_spread_std: float
    __zod_offset: float
    __rice_factor_mean: float
    __rice_factor_std: float
    __delay_scaling: float
    __cross_polarization_power_mean: float
    __cross_polarization_power_std: float
    __num_clusters: int
    __num_rays: int
    __cluster_delay_spread: float
    __cluster_aod_spread: float
    __cluster_aoa_spread: float
    __cluster_zoa_spread: float
    __cluster_shadowing_std: float

    def __init__(
        self,
        alpha_device: SimulatedDevice | None = None,
        beta_device: SimulatedDevice | None = None,
        gain: float = 1.0,
        line_of_sight: bool = True,
        delay_spread_mean: float = 7.14,
        delay_spread_std: float = 0.38,
        aod_spread_mean: float = 1.21,
        aod_spread_std: float = 0.41,
        aoa_spread_mean: float = 1.73,
        aoa_spread_std: float = 0.28,
        zoa_spread_mean: float = 0.73,
        zoa_spread_std: float = 0.34,
        zod_spread_mean: float = 0.1,
        zod_spread_std: float = 0.0,
        zod_offset: float = 0.0,
        rice_factor_mean: float = 9.0,
        rice_factor_std: float = 5.0,
        delay_scaling: float = 1.0,
        cross_polarization_power_mean: float = 9.0,
        cross_polarization_power_std: float = 3.0,
        num_clusters: int = 12,
        num_rays: int = 20,
        cluster_delay_spread: float = 5e-9,
        cluster_aod_spread: float = 5.0,
        cluster_aoa_spread: float = 17.0,
        cluster_zoa_spread: float = 7.0,
        cluster_shadowing_std: float = 3.0,
        **kwargs: Any,
    ) -> None:
        """
        Args:

            alpha_device (SimulatedDevice, optional):
                First device linked by the :class:`.ClusterDelayLine` instance that generated this realization.

            beta_device (SiumulatedDevice, optional):
                Second device linked by the :class:`.ClusterDelayLine` instance that generated this realization.

            gain (float, optional):
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default.

            num_clusters (int, optional):
                Number of generated clusters per channel sample.

            delay_spread (float, optional):
                Root-Mean-Square spread of the cluster delay in seconds.

            delay_scaling (float, optional):
                Delay distribution proportionality factor.

            rice_factor_mean (float, optional):
                Mean of the rice factor K.

            rice_factor_std (float, optional):
                Standard deviation of the rice factor K.

            cluster_shadowing_std (float, optional):
                Cluster shadowing standard deviation in dB.

            line_of_sight (bool, optional):
                Is this model a line-of-sight model?

        """

        # Set initial parameters
        self.line_of_sight = line_of_sight
        self.delay_spread_mean = delay_spread_mean
        self.delay_spread_std = delay_spread_std
        self.aod_spread_mean = aod_spread_mean
        self.aod_spread_std = aod_spread_std
        self.aoa_spread_mean = aoa_spread_mean
        self.aoa_spread_std = aoa_spread_std
        self.zoa_spread_mean = zoa_spread_mean
        self.zoa_spread_std = zoa_spread_std
        self.zod_spread_mean = zod_spread_mean
        self.zod_spread_std = zod_spread_std
        self.zod_offset = zod_offset
        self.rice_factor_mean = rice_factor_mean
        self.rice_factor_std = rice_factor_std
        self.delay_scaling = delay_scaling
        self.cross_polarization_power_mean = cross_polarization_power_mean
        self.cross_polarization_power_std = cross_polarization_power_std
        self.num_clusters = num_clusters
        self.num_rays = num_rays
        self.cluster_delay_spread = cluster_delay_spread
        self.cluster_aod_spread = cluster_aod_spread
        self.cluster_aoa_spread = cluster_aoa_spread
        self.cluster_zoa_spread = cluster_zoa_spread
        self.cluster_shadowing_std = cluster_shadowing_std

        # Initialize base class
        ClusterDelayLineBase.__init__(self, alpha_device, beta_device, gain, **kwargs)

    @property
    def line_of_sight(self) -> bool:
        return self.__line_of_sight

    @line_of_sight.setter
    def line_of_sight(self, value: bool) -> None:
        self.__line_of_sight = value

    @property
    def delay_spread_mean(self) -> float:
        return self.__delay_spread_mean

    @delay_spread_mean.setter
    def delay_spread_mean(self, value: float) -> None:
        self.__delay_spread_mean = value

    @property
    def delay_spread_std(self) -> float:
        return self.__delay_spread_std

    @delay_spread_std.setter
    def delay_spread_std(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Delay spread standard deviation must be greater or equal to zero")

        self.__delay_spread_std = value

    @property
    def aod_spread_mean(self) -> float:
        return self.__aod_spread_mean

    @aod_spread_mean.setter
    def aod_spread_mean(self, value: float) -> None:
        self.__aod_spread_mean = value

    @property
    def aod_spread_std(self) -> float:
        return self.__aod_spread_std

    @aod_spread_std.setter
    def aod_spread_std(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Angle spread standard deviation must be greater or equal to zero")

        self.__aod_spread_std = value

    @property
    def aoa_spread_mean(self) -> float:
        return self.__aoa_spread_mean

    @aoa_spread_mean.setter
    def aoa_spread_mean(self, value: float) -> None:
        self.__aoa_spread_mean = value

    @property
    def aoa_spread_std(self) -> float:
        return self.__aoa_spread_std

    @aoa_spread_std.setter
    def aoa_spread_std(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Angle spread standard deviation must be greater or equal to zero")

        self.__aoa_spread_std = value

    @property
    def zoa_spread_mean(self) -> float:
        return self.__zoa_spread_mean

    @zoa_spread_mean.setter
    def zoa_spread_mean(self, value: float) -> None:
        self.__zoa_spread_mean = value

    @property
    def zoa_spread_std(self) -> float:
        return self.__zoa_spread_std

    @zoa_spread_std.setter
    def zoa_spread_std(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Angle spread standard deviation must be greater or equal to zero")

        self.__zoa_spread_std = value

    @property
    def zod_spread_mean(self) -> float:
        return self.__zod_spread_mean

    @zod_spread_mean.setter
    def zod_spread_mean(self, value: float) -> None:
        self.__zod_spread_mean = value

    @property
    def zod_spread_std(self) -> float:
        return self.__zod_spread_std

    @zod_spread_std.setter
    def zod_spread_std(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Zenith spread standard deviation must be greater or equal to zero")

        self.__zod_spread_std = value

    @property
    def zod_offset(self) -> float:
        return self.__zod_offset

    @zod_offset.setter
    def zod_offset(self, value: float) -> None:
        self.__zod_offset = value

    ###############################
    # ToDo: Shadow fading function
    ###############################

    @property
    def rice_factor_mean(self) -> float:
        return self.__rice_factor_mean

    @rice_factor_mean.setter
    def rice_factor_mean(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Rice factor must be greater or equal to zero")

        self.__rice_factor_mean = value

    @property
    def rice_factor_std(self) -> float:
        return self.__rice_factor_std

    @rice_factor_std.setter
    def rice_factor_std(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Rice factor standard deviation must be greater or equal to zero")

        self.__rice_factor_std = value

    @property
    def delay_scaling(self) -> float:
        return self.__delay_scaling

    @delay_scaling.setter
    def delay_scaling(self, value: float) -> None:
        if value < 1.0:
            raise ValueError("Delay scaling must be greater or equal to one")

        self.__delay_scaling = value

    @property
    def cross_polarization_power_mean(self) -> float:
        return self.__cross_polarization_power_mean

    @cross_polarization_power_mean.setter
    def cross_polarization_power_mean(self, value: float) -> None:
        self.__cross_polarization_power_mean = value

    @property
    def cross_polarization_power_std(self) -> float:
        return self.__cross_polarization_power_std

    @cross_polarization_power_std.setter
    def cross_polarization_power_std(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(
                "Cross-polarization power standard deviation must be greater or equal to zero"
            )

        self.__cross_polarization_power_std = value

    @property
    def num_clusters(self) -> int:
        return self.__num_clusters

    @num_clusters.setter
    def num_clusters(self, value: int) -> None:
        if value < 1:
            raise ValueError("Number of clusters must be greater or equal to one")

        self.__num_clusters = value

    @property
    def num_rays(self) -> int:
        return self.__num_rays

    @num_rays.setter
    def num_rays(self, value: int) -> None:
        if value < 1:
            raise ValueError("Number of rays per cluster must be greater or equal to one")

        self.__num_rays = value

    @property
    def cluster_delay_spread(self) -> float:
        return self.__cluster_delay_spread

    @cluster_delay_spread.setter
    def cluster_delay_spread(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Cluster delay spread must be greater or equal to zero")

        self.__cluster_delay_spread = value

    @property
    def cluster_aod_spread(self) -> float:
        return self.__cluster_aod_spread

    @cluster_aod_spread.setter
    def cluster_aod_spread(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Cluster angle spread must be greater or equal to zero")

        self.__cluster_aod_spread = value

    @property
    def cluster_aoa_spread(self) -> float:
        return self.__cluster_aoa_spread

    @cluster_aoa_spread.setter
    def cluster_aoa_spread(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Cluster angle spread must be greater or equal to zero")

        self.__cluster_aoa_spread = value

    @property
    def cluster_zoa_spread(self) -> float:
        return self.__cluster_zoa_spread

    @cluster_zoa_spread.setter
    def cluster_zoa_spread(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Cluster angle spread must be greater or equal to zero")

        self.__cluster_zoa_spread = value

    @property
    def cluster_shadowing_std(self) -> float:
        return self.__cluster_shadowing_std

    @cluster_shadowing_std.setter
    def cluster_shadowing_std(self, value: float) -> None:
        if value < 0.0:
            raise ValueError(
                "Cluster shadowing standard deviation must be greater or equal to zero"
            )

        self.__cluster_shadowing_std = value
