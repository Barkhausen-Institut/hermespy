# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC
from typing import Any, Generator, Set, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
from h5py import Group
from numpy import cos, exp
from scipy.constants import pi
from sparse import GCXS  # type: ignore

from hermespy.core import (
    AntennaArrayState,
    AntennaMode,
    ChannelStateInformation,
    ChannelStateFormat,
    HDFSerializable,
    Serializable,
    SignalBlock,
    VAT,
)
from ..channel import (
    Channel,
    ChannelSample,
    LinkState,
    ChannelSampleHook,
    ChannelRealization,
    InterpolationMode,
)
from ..consistent import ConsistentUniform, ConsistentGenerator, ConsistentRealization

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class AntennaCorrelation(ABC):
    """Base class for statistical modeling of antenna array correlations."""

    __channel: Channel | None

    def __init__(self, channel: Channel | None = None) -> None:
        """

        Args:

            channel (Channel, optional):
                Channel this correlation model configures.
                `None` if the model is currently considered floating.
        """

        self.channel = channel

    def sample_covariance(self, antennas: AntennaArrayState, mode: AntennaMode) -> np.ndarray:
        """Sample the covariance matrix of a given antenna array.

        Args:

            antennas (AntennaArrayState):
                State of the antenna array.

            mode (AntennaMode):
                Mode of the antenna array, i.e. transmit or receive.

        Returns: Two-dimensional numpy array representing the covariance matrix.
        """
        ...  # pragma: no cover

    @property
    def channel(self) -> Channel | None:
        """The channel this correlation model configures.

        Returns:
            Handle to the channel.
            `None` if the model is currently considered floating
        """

        return self.__channel

    @channel.setter
    def channel(self, value: Channel | None) -> None:
        self.__channel = value


class CustomAntennaCorrelation(Serializable, AntennaCorrelation):
    """Customizable antenna correlations."""

    yaml_tag = "CustomCorrelation"
    """YAML serialization tag"""

    __covariance_matrix: np.ndarray

    def __init__(self, covariance: np.ndarray) -> None:
        """
        Args:

            covariance (numpy.ndarray):
                Postive definte square antenna covariance matrix.
        """

        self.covariance = covariance

    def sample_covariance(self, antennas: AntennaArrayState, mode: AntennaMode) -> np.ndarray:
        num_antennas = (
            antennas.num_transmit_antennas
            if mode == AntennaMode.TX
            else antennas.num_receive_antennas
        )

        if self.__covariance_matrix.shape[0] < num_antennas:
            raise ValueError("Antenna correlation matrix does not match the number of antennas")

        return self.__covariance_matrix[:num_antennas, :num_antennas]

    @property
    def covariance(self) -> np.ndarray:
        """Postive definte square antenna covariance matrix."""

        return self.__covariance_matrix

    @covariance.setter
    def covariance(self, value: np.ndarray) -> None:
        if value.ndim != 2 or not np.allclose(value, value.T.conj()):
            raise ValueError("Antenna correlation must be a hermitian matrix")

        if np.any(np.linalg.eigvals(value) <= 0.0):
            raise ValueError("Antenna correlation matrix must be positive definite")

        self.__covariance_matrix = value


class MultipathFadingSample(ChannelSample):
    """Immutable sample of a statistical multipath fading channel.

    Generated by the :meth:`sample()<MultipathFadingRealization.sample>` routine of a :class:`MultipathFadingRealization<MultipathFadingRealization>`.
    """

    __spatial_response: np.ndarray
    __max_delay: float

    def __init__(
        self,
        power_profile: np.ndarray,
        delay_profile: np.ndarray,
        los_angles: np.ndarray,
        nlos_angles: np.ndarray,
        los_phases: np.ndarray,
        nlos_phases: np.ndarray,
        los_gains: np.ndarray,
        nlos_gains: np.ndarray,
        los_doppler: float,
        nlos_doppler: float,
        spatial_response: np.ndarray,
        gain: float,
        state: LinkState,
    ) -> None:
        """
        Args:

            transmitter_state (DeviceState):
                State of the transmitting device at the time of sampling.

            receiver_state (DeviceState):
                State of the receiving device at the time of sampling.

            carrier_frequency (float):
                Carrier frequency of the channel in Hz.

            bandwidth (float):
                Bandwidth of the propagated signal in Hz.

            path_realizations (Sequence[PathRealization]):
                Realizations of the individual propagation paths.

            spatial_response (numpy.ndarray):
                Spatial response matrix of the channel realization considering `alpha_device` is the transmitter and `beta_device` is the receiver.

            interpolation_mode (InterpolationMode, optional):
                Interpolation behaviour of the channel realization's delay components with respect to the proagated signal's sampling rate.
        """

        # Initialize base class
        ChannelSample.__init__(self, state)

        # Initialize class attributes
        self.__power_profile = power_profile
        self.__delay_profile = delay_profile
        self.__los_angles = los_angles
        self.__nlos_angles = nlos_angles
        self.__los_phases = los_phases
        self.__nlos_phases = nlos_phases
        self.__los_gains = los_gains
        self.__nlos_gains = nlos_gains
        self.__los_doppler = los_doppler
        self.__nlos_doppler = nlos_doppler
        self.__spatial_response = spatial_response
        self.__gain = gain

        # Infer additional parameters
        self.__max_delay = self.__delay_profile.max()

    @property
    def power_profile(self) -> np.ndarray:
        """Power loss factor of each individual multipath tap."""

        return self.__power_profile

    @property
    def delay_profile(self) -> np.ndarray:
        """Delay in seconds of each individual multipath tap."""

        return self.__delay_profile

    @property
    def los_angles(self) -> np.ndarray:
        """Angle of arrival of the line-of sight components."""

        return self.__los_angles

    @property
    def nlos_angles(self) -> np.ndarray:
        """Angle of arrival of the non-line-of sight components."""

        return self.__nlos_angles

    @property
    def los_phases(self) -> np.ndarray:
        """Phase of the line-of sight components."""

        return self.__los_phases

    @property
    def nlos_phases(self) -> np.ndarray:
        """Phase of the non-line-of sight components."""

        return self.__nlos_phases

    @property
    def los_gains(self) -> np.ndarray:
        """Gain factors of the line-of sight components."""

        return self.__los_gains

    @property
    def nlos_gains(self) -> np.ndarray:
        """Gain factors of the non-line-of sight components."""

        return self.__nlos_gains

    @property
    def los_doppler(self) -> float:
        """Doppler frequency shift of the line-of sight components."""

        return self.__los_doppler

    @property
    def nlos_doppler(self) -> float:
        """Doppler frequency shift of the non-line-of sight components."""

        return self.__nlos_doppler

    @property
    def spatial_response(self) -> np.ndarray:
        """Spatial response matrix of the channel realization considering `alpha_device` is the transmitter and `beta_device` is the receiver."""

        return self.__spatial_response

    @property
    def gain(self) -> float:
        """Linear energy gain factor a signal experiences when being propagated over this realization."""

        return self.__gain

    @property
    def expected_energy_scale(self) -> float:
        return self.gain * np.sum(self.power_profile)

    def __path_impulse_generator(
        self, num_samples: int
    ) -> Generator[Tuple[np.ndarray, int], None, None]:

        path_delay_samples = np.rint(self.delay_profile * self.bandwidth).astype(int)

        num_sinusoids = self.__nlos_angles.shape[1]
        n = 1 + np.arange(num_sinusoids)
        timestamps = np.arange(num_samples) / self.bandwidth

        nlos_time = self.nlos_doppler * timestamps
        los_time = self.los_doppler * timestamps

        for (
            delay,
            power,
            los_gain,
            nlos_gain,
            los_angle,
            nlos_angles,
            los_phase,
            nlos_phases,
        ) in zip(
            path_delay_samples,
            self.power_profile,
            self.los_gains,
            self.nlos_gains,
            self.los_angles,
            self.nlos_angles,
            self.los_phases,
            self.nlos_phases,
        ):

            impulse = nlos_gain * np.sum(
                np.exp(
                    1j
                    * (
                        np.outer(nlos_time, np.cos((2 * pi * n + nlos_angles) / num_sinusoids))
                        + nlos_phases[None, :]
                    )
                ),
                axis=1,
                keepdims=False,
            )

            # Add the specular component
            impulse += los_gain * exp(1j * (los_time * cos(los_angle) + los_phase))

            # Scale by the overall path power
            impulse *= (self.gain * power) ** 0.5
            yield impulse, delay

    def state(
        self,
        num_samples: int,
        max_num_taps: int,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> ChannelStateInformation:
        num_taps = min(1 + round(self.__max_delay * self.bandwidth), max_num_taps)
        siso_csi = np.zeros((num_samples, num_taps), dtype=np.complex_)
        for path_impulse, path_delay_samples in self.__path_impulse_generator(num_samples):
            # Skip paths with delays larger than the maximum delay required by the CSI request
            if path_delay_samples > num_taps:
                continue

            siso_csi[:, path_delay_samples] += path_impulse

        # For the multipath fading model, the MIMO CSI is the outer product of the SISO CSI with the spatial response
        # The resulting multidimensional array is sparse in its fourth dimension and converted to a GCXS array for memory efficiency
        mimo_csi = GCXS.from_numpy(
            np.einsum("ij,kl->ijkl", self.spatial_response, siso_csi), compressed_axes=(0, 1, 2)
        )

        state = ChannelStateInformation(
            ChannelStateFormat.IMPULSE_RESPONSE, mimo_csi, num_delay_taps=num_taps
        )
        return state

    def _propagate(self, signal: SignalBlock, interpolation: InterpolationMode) -> SignalBlock:
        max_delay_in_samples = round(self.__max_delay * self.bandwidth)
        num_propagated_samples = signal.num_samples + max_delay_in_samples

        # Propagate the transmitted samples
        propagated_samples = np.zeros(
            (self.spatial_response.shape[0], num_propagated_samples), dtype=np.complex_
        )
        for path_impulse, path_num_delay_samples in self.__path_impulse_generator(
            signal.num_samples
        ):
            propagated_samples[
                :, path_num_delay_samples : path_num_delay_samples + signal.num_samples
            ] += (signal * path_impulse[np.newaxis, :])

        # Apply the channel's spatial response
        propagated_samples = self.spatial_response @ propagated_samples

        # Return the result
        propagated_block = SignalBlock(propagated_samples, signal.offset)
        return propagated_block

    def plot_power_delay(self, axes: VAT | None = None) -> Tuple[plt.Figure, VAT]:
        if axes:
            _axes = axes
            figure = axes[0, 0].get_figure()
        else:
            figure, _axes = plt.subplots(1, 1, squeeze=False)
            figure.suptitle("Power Delay Profile")

        ax: plt.Axes = _axes.flat[0]
        ax.stem(self.__delay_profile, self.__power_profile)
        ax.set_xlabel("Delay [s]")
        ax.set_ylabel("Power [Watts]")
        ax.set_yscale("log")

        return figure, _axes


class MultipathFadingRealization(ChannelRealization[MultipathFadingSample]):
    """Realization of a statistical multipath fading channel.

    Generated by the :meth:`realize()<MultipathFadingChannel.realize>` routine of a :class:`MultipathFadingChannel<MultipathFadingChannel>`.
    """

    def __init__(
        self,
        random_realization: ConsistentRealization,
        antenna_correlation_variable: ConsistentUniform,
        los_angles_variable: float | ConsistentUniform,
        nlos_angles_variable: ConsistentUniform,
        los_phases_variable: ConsistentUniform,
        nlos_phases_variable: ConsistentUniform,
        power_profile: np.ndarray,
        delay_profile: np.ndarray,
        los_gains: np.ndarray,
        nlos_gains: np.ndarray,
        los_doppler: float,
        nlos_doppler: float,
        antenna_correlation: AntennaCorrelation | None,
        sample_hooks: Set[ChannelSampleHook[MultipathFadingSample]],
        gain: float,
    ) -> None:

        # Initialize base class
        ChannelRealization.__init__(self, sample_hooks, gain)

        # Initialize class attributes
        self.__random_realization = random_realization
        self.__antenna_correlation_variable = antenna_correlation_variable
        self.__los_angles_variable = los_angles_variable
        self.__path_angles_variable = nlos_angles_variable
        self.__los_phases_variable = los_phases_variable
        self.__path_phases_variable = nlos_phases_variable
        self.__power_profile = power_profile
        self.__delay_profile = delay_profile
        self.__los_gains = los_gains
        self.__nlos_gains = nlos_gains
        self.__los_doppler = los_doppler
        self.__nlos_doppler = nlos_doppler
        self.__antenna_correlation = antenna_correlation

    def _sample(self, state: LinkState) -> MultipathFadingSample:

        # Sample the spatiall consistent random realization
        consistent_sample = self.__random_realization.sample(
            state.transmitter.pose.translation, state.receiver.pose.translation
        )

        # Generate MIMO channel response
        spatial_response = np.exp(
            2j * np.pi * self.__antenna_correlation_variable.sample(consistent_sample)
        )[
            : state.receiver.antennas.num_receive_antennas,
            : state.transmitter.antennas.num_transmit_antennas,
        ]

        if self.__antenna_correlation is not None:
            spatial_response = (
                self.__antenna_correlation.sample_covariance(
                    state.receiver.antennas, AntennaMode.RX
                )
                @ spatial_response
                @ self.__antenna_correlation.sample_covariance(
                    state.transmitter.antennas, AntennaMode.TX
                )
            )

        # Sample multipath components
        los_angles = (
            self.__los_angles_variable * np.ones_like(self.__power_profile)
            if isinstance(self.__los_angles_variable, float)
            else 2 * np.pi * self.__los_angles_variable.sample(consistent_sample)
        )
        nlos_angles = -np.pi + 2 * np.pi * self.__path_angles_variable.sample(consistent_sample)
        los_phases = -np.pi + 2 * np.pi * self.__los_phases_variable.sample(consistent_sample)
        nlos_phases = -np.pi + 2 * np.pi * self.__path_phases_variable.sample(consistent_sample)

        return MultipathFadingSample(
            self.__power_profile,
            self.__delay_profile,
            los_angles,
            nlos_angles,
            los_phases,
            nlos_phases,
            self.__los_gains,
            self.__nlos_gains,
            self.__los_doppler,
            self.__nlos_doppler,
            spatial_response,
            self.gain,
            state,
        )

    def to_HDF(self, group: Group) -> None:
        # Serialize class attributes
        self.__random_realization.to_HDF(group.create_group("random_realization"))
        HDFSerializable._write_dataset(group, "power_profile", self.__power_profile)
        HDFSerializable._write_dataset(group, "delay_profile", self.__delay_profile)
        HDFSerializable._write_dataset(group, "los_gains", self.__los_gains)
        HDFSerializable._write_dataset(group, "nlos_gains", self.__nlos_gains)
        group.attrs["los_doppler"] = self.__los_doppler
        group.attrs["nlos_doppler"] = self.__nlos_doppler
        group.attrs["gain"] = self.gain

    @staticmethod
    def From_HDF(
        group: Group,
        random_realization: ConsistentRealization,
        antenna_correlation_variable: ConsistentUniform,
        los_angles_variable: float | ConsistentUniform,
        nlos_angles_variable: ConsistentUniform,
        los_phases_variable: ConsistentUniform,
        nlos_phases_variable: ConsistentUniform,
        antenna_correlation: AntennaCorrelation | None,
        sample_hooks: Set[ChannelSampleHook[MultipathFadingSample]],
    ) -> MultipathFadingRealization:

        # Deserialize base class
        random_realization = ConsistentRealization.from_HDF(group["random_realization"])
        power_profile = np.array(group["power_profile"], dtype=np.float_)
        delay_profile = np.array(group["delay_profile"], dtype=np.float_)
        los_gains = np.array(group["los_gains"], dtype=np.float_)
        nlos_gains = np.array(group["nlos_gains"], dtype=np.float_)
        los_doppler = group.attrs["los_doppler"]
        nlos_doppler = group.attrs["nlos_doppler"]
        gain = group.attrs["gain"]

        return MultipathFadingRealization(
            random_realization,
            antenna_correlation_variable,
            los_angles_variable,
            nlos_angles_variable,
            los_phases_variable,
            nlos_phases_variable,
            power_profile,
            delay_profile,
            los_gains,
            nlos_gains,
            los_doppler,
            nlos_doppler,
            antenna_correlation,
            sample_hooks,
            gain,
        )

    def _reciprocal_sample(
        self, sample: MultipathFadingSample, state: LinkState
    ) -> MultipathFadingSample:
        return MultipathFadingSample(
            sample.power_profile,
            sample.delay_profile,
            sample.los_angles,
            sample.nlos_angles,
            sample.los_phases,
            sample.nlos_phases,
            sample.los_gains,
            sample.nlos_gains,
            sample.los_doppler,
            sample.nlos_doppler,
            sample.spatial_response.T,
            sample.gain,
            state,
        )


class MultipathFadingChannel(
    Channel[MultipathFadingRealization, MultipathFadingSample], Serializable
):
    """Base class for the implementation of stochastic multipath fading channels."""

    yaml_tag = "MultipathFading"

    __delays: np.ndarray
    __power_profile: np.ndarray
    __rice_factors: np.ndarray
    __max_delay: float
    __num_resolvable_paths: int
    __num_sinusoids: int
    __los_angle: float | None
    __los_gains: np.ndarray
    __doppler_frequency: float
    __los_doppler_frequency: float | None
    __antenna_correlation: AntennaCorrelation | None

    def __init__(
        self,
        delays: np.ndarray | List[float],
        power_profile: np.ndarray | List[float],
        rice_factors: np.ndarray | List[float],
        correlation_distance: float = float("inf"),
        num_sinusoids: int | None = None,
        los_angle: float | None = None,
        doppler_frequency: float | None = None,
        los_doppler_frequency: float | None = None,
        antenna_correlation: AntennaCorrelation | None = None,
        gain: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """
        Args:

            delays (numpy.ndarray):
                Delay in seconds of each individual multipath tap.
                Denoted by :math:`\\tau_{\\ell}` within the respective equations.

            power_profile (numpy.ndarray):
                Power loss factor of each individual multipath tap.
                Denoted by :math:`g_{\\ell}` within the respective equations.

            rice_factors (numpy.ndarray):
                Rice factor balancing line of sight and multipath in each individual channel tap.
                Denoted by :math:`K_{\\ell}` within the respective equations.

            correlation_distance (float, optional):
                Distance at which channel samples are considered to be uncorrelated.
                :math:`\\infty` by default, i.e. the channel is considered to be fully correlated in space.

            num_sinusoids (int, optional):
                Number of sinusoids used to sample the statistical distribution.
                Denoted by :math:`N` within the respective equations.

            los_angle (float, optional):
                Angle phase of the line of sight component within the statistical distribution.

            doppler_frequency (float, optional):
                Doppler frequency shift of the statistical distribution.
                Denoted by :math:`\\omega_{\\ell}` within the respective equations.

            antenna_correlation (AntennaCorrelation, optional):
                Antenna correlation model.
                By default, the channel assumes ideal correlation, i.e. no cross correlations.

            gain (float, optional):
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default.

            \**kwargs (Any, optional):
                Channel base class initialization parameters.

        Raises:

            ValueError: If the length of `delays`, `power_profile` and `rice_factors` is not identical.
            ValueError: If delays are smaller than zero.
            ValueError: If power factors are smaller than zero.
            ValueError: If rice factors are smaller than zero.
        """

        # Convert delays, power profile and rice factors to numpy arrays if they were provided as lists
        self.__delays = np.array(delays) if isinstance(delays, list) else delays
        self.__power_profile = (
            np.array(power_profile) if isinstance(power_profile, list) else power_profile
        )
        self.__rice_factors = (
            np.array(rice_factors) if isinstance(rice_factors, list) else rice_factors
        )

        if (
            self.__delays.ndim != 1
            or self.__power_profile.ndim != 1
            or self.__rice_factors.ndim != 1
        ):
            raise ValueError("Delays, power profile and rice factors must be vectors")

        if len(delays) < 1:
            raise ValueError("Configuration must contain at least one delay tap")

        if len(delays) != len(power_profile) or len(power_profile) != len(rice_factors):
            raise ValueError(
                "Delays, power profile and rice factor vectors must be of equal length"
            )

        if np.any(self.__delays < 0.0):
            raise ValueError("Delays must be greater or equal to zero")

        if np.any(self.__power_profile < 0.0):
            raise ValueError("Power profile factors must be greater or equal to zero")

        if np.any(self.__rice_factors < 0.0):
            raise ValueError("Rice factors must be greater or equal to zero")

        # Initialize base class
        self.__antenna_correlation = None
        Channel.__init__(self, gain=gain, **kwargs)

        # Sort delays
        sorting = np.argsort(delays)

        self.__delays = self.__delays[sorting]
        self.__power_profile = self.__power_profile[sorting]
        self.__rice_factors = self.__rice_factors[sorting]
        self.__num_sinusoids = 20 if num_sinusoids is None else num_sinusoids
        self.los_angle = self._rng.uniform(-pi, pi) if los_angle is None else los_angle
        self.doppler_frequency = 0.0 if doppler_frequency is None else doppler_frequency
        self.__los_doppler_frequency = None

        if los_doppler_frequency is not None:
            self.los_doppler_frequency = los_doppler_frequency

        # Infer additional parameters
        self.__max_delay = max(self.__delays)
        self.__num_resolvable_paths = len(self.__delays)

        rice_inf_pos = np.isposinf(self.__rice_factors)
        rice_num_pos = np.invert(rice_inf_pos)
        self.__los_gains = np.empty(self.num_resolvable_paths, dtype=float)
        self.__non_los_gains = np.empty(self.num_resolvable_paths, dtype=float)

        self.__los_gains[rice_inf_pos] = 1.0
        self.__los_gains[rice_num_pos] = np.sqrt(
            self.__rice_factors[rice_num_pos] / (1 + self.__rice_factors[rice_num_pos])
        )
        self.__non_los_gains[rice_num_pos] = np.sqrt(
            1 / ((1 + self.__rice_factors[rice_num_pos]) * self.__num_sinusoids)
        )
        self.__non_los_gains[rice_inf_pos] = 0.0

        # Update correlations (required here to break dependency cycle during init)
        self.antenna_correlation = antenna_correlation
        self.correlation_distance = correlation_distance

        self.__rng = ConsistentGenerator(self)
        # self.__antenna_correlation_variable = self.__rng.uniform((self.beta_device.num_antennas, self.alpha_device.num_antennas))
        self.__antenna_correlation_variable = self.__rng.uniform((10, 10))
        self.__los_angles_variable = (
            self.__rng.uniform((self.__num_resolvable_paths,))
            if self.__los_angle is not None
            else self.__los_angle
        )
        self.__nlos_angles_variable = self.__rng.uniform(
            (self.num_resolvable_paths, self.__num_sinusoids)
        )
        self.__los_phases_variable = self.__rng.uniform((self.num_resolvable_paths,))
        self.__nlos_phases_variable = self.__rng.uniform(
            (self.num_resolvable_paths, self.__num_sinusoids)
        )

    @property
    def correlation_distance(self) -> float:
        """Correlation distance in meters.

        Represents the distance over which the antenna correlation is assumed to be constant.
        """

        return self.__correlation_distance

    @correlation_distance.setter
    def correlation_distance(self, distance: float) -> None:
        if distance < 0:
            raise ValueError("Correlation distance must be greater or equal to zero")
        self.__correlation_distance = distance

    @property
    def delays(self) -> np.ndarray:
        """Delays for each propagation path in seconds.

        Represented by the sequence

        .. math::

           \\left[\\tau_{1},\\, \\dotsc,\\, \\tau_{L} \\right]^{\\mathsf{T}} \\in \\mathbb{R}_{+}^{L}

        of :math:`L` propagtion delays within the respective equations.
        """

        return self.__delays

    @property
    def power_profile(self) -> np.ndarray:
        """Gain factors of each propagation path.

        Represented by the sequence

        .. math::

           \\left[g_{1},\\, \\dotsc,\\, g_{L} \\right]^{\\mathsf{T}} \\in \\mathbb{R}_{+}^{L}

        of :math:`L` propagtion factors within the respective equations.
        """

        return self.__power_profile

    @property
    def rice_factors(self) -> np.ndarray:
        """Rice factors balancing line of sight and non-line of sight power components for each propagation path.

        Represented by the sequence

        .. math::

           \\left[K_{1},\\, \\dotsc,\\, K_{L} \\right]^{\\mathsf{T}} \\in \\mathbb{R}_{+}^{L}

        of :math:`L` factors within the respective equations.
        """

        return self.__rice_factors

    @property
    def doppler_frequency(self) -> float:
        """Doppler frequency in :math:`Hz`.

        Represented by :math:`\\omega` within the respective equations.
        """

        return self.__doppler_frequency

    @doppler_frequency.setter
    def doppler_frequency(self, frequency: float) -> None:
        self.__doppler_frequency = frequency

    @property
    def los_doppler_frequency(self) -> float:
        """Line of sight Doppler frequency in :math:`Hz`.

        Represented by :math:`\\omega` within the respective equations.
        """

        if self.__los_doppler_frequency is None:
            return self.doppler_frequency

        return self.__los_doppler_frequency

    @los_doppler_frequency.setter
    def los_doppler_frequency(self, frequency: float | None) -> None:
        self.__los_doppler_frequency = frequency

    @property
    def max_delay(self) -> float:
        """Maximum propagation delay in seconds."""

        return self.__max_delay

    @property
    def num_resolvable_paths(self) -> int:
        """Number of dedicated propagation paths.

        Represented by :math:`L` within the respective equations.
        """

        return self.__num_resolvable_paths

    @property
    def num_sinusoids(self) -> int:
        """Number of sinusoids assumed to model the fading in time-domain.

        Represented by :math:`N` within the respective equations.

        Raises:

            ValueError: For values smaller than zero.
        """

        return self.__num_sinusoids

    @num_sinusoids.setter
    def num_sinusoids(self, num: int) -> None:
        if num < 0:
            raise ValueError("Number of sinusoids must be greater or equal to zero")

        self.__num_sinusoids = num

    @property
    def los_angle(self) -> float | None:
        """Line of sight doppler angle in radians.

        Represented by :math:`\\theta_{0}` within the respective equations.
        """

        return self.__los_angle

    @los_angle.setter
    def los_angle(self, angle: float | None) -> None:
        self.__los_angle = angle

    def _realize(self) -> MultipathFadingRealization:
        return MultipathFadingRealization(
            self.__rng.realize(self.correlation_distance),
            self.__antenna_correlation_variable,
            self.__los_angles_variable,
            self.__nlos_angles_variable,
            self.__los_phases_variable,
            self.__nlos_phases_variable,
            self.__power_profile,
            self.__delays,
            self.__los_gains,
            self.__non_los_gains,
            self.los_doppler_frequency,
            self.doppler_frequency,
            self.antenna_correlation,
            self.sample_hooks,
            self.gain,
        )

    @property
    def antenna_correlation(self) -> AntennaCorrelation | None:
        """Antenna correlations.

        Returns:
            Handle to the correlation model.
            :py:obj:`None`, if no model was configured and ideal correlation is assumed.
        """

        return self.__antenna_correlation

    @antenna_correlation.setter
    def antenna_correlation(self, value: AntennaCorrelation | None) -> None:
        if value is not None:
            value.channel = self

        self.__alpha_correlation = value

    def recall_realization(self, group: Group) -> MultipathFadingRealization:
        return MultipathFadingRealization.From_HDF(
            group,
            self.__rng.realize(self.correlation_distance),
            self.__antenna_correlation_variable,
            self.__los_angles_variable,
            self.__nlos_angles_variable,
            self.__los_phases_variable,
            self.__nlos_phases_variable,
            self.antenna_correlation,
            self.sample_hooks,
        )
