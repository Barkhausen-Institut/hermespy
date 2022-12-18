# -*- coding: utf-8 -*-
"""
=================
Multipath Fading
=================
"""

from __future__ import annotations
from abc import abstractmethod, ABC
from itertools import product
from typing import Any, Optional, TYPE_CHECKING, Union, List

import numpy as np
from numpy import cos, exp
from scipy.constants import pi

from hermespy.core import Serializable
from hermespy.core.device import Device
from hermespy.tools import delay_resampling_matrix
from .channel import Channel, ChannelRealization

if TYPE_CHECKING:
    from hermespy.simulation import SimulatedDevice  # pragma no cover

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class AntennaCorrelation(ABC):
    """Base class for statistical modeling of antenna array correlations."""

    __channel: Optional[MultipathFadingChannel]
    __device: Optional[SimulatedDevice]

    def __init__(self, channel: Optional[Channel] = None, device: Optional[Device] = None) -> None:

        self.channel = channel
        self.device = device

    @property
    @abstractmethod
    def covariance(self) -> np.ndarray:
        """Antenna covariance matrix.

        Returns: Two-dimensional numpy array representing the covariance matrix.
        """
        ...  # pragma no cover

    @property
    def channel(self) -> Optional[MultipathFadingChannel]:
        """The channel this correlation model configures.

        Returns:
            Handle to the channel.
            `None` if the model is currently considered floating
        """

        return self.__channel

    @channel.setter
    def channel(self, value: Optional[MultipathFadingChannel]) -> None:

        self.__channel = value

    @property
    def device(self) -> Optional[SimulatedDevice]:
        """The device this correlation model is based upon.

        Returns:
            Handle to the device.
            `None` if the device is currently unknown.
        """

        return self.__device

    @device.setter
    def device(self, value: Optional[SimulatedDevice]) -> None:

        self.__device = value


class CustomAntennaCorrelation(Serializable, AntennaCorrelation):
    """Customizable antenna correlations."""

    yaml_tag = "CustomCorrelation"
    """YAML serialization tag"""

    __covariance_matrix: np.ndarray

    def __init__(self, covariance: np.ndarray) -> None:
        """
        Args:

            covariance (np.ndarray):
                Postive definte square antenna covariance matrix.
        """

        self.covariance = covariance

    @property
    def covariance(self) -> np.ndarray:

        if self.device is not None and self.device.num_antennas != self.__covariance_matrix.shape[0]:
            raise RuntimeError(f"Device with {self.device.num_antennas} antennas does not match covariance matrix of magnitude {self.__covariance_matrix.shape[0]}")

        return self.__covariance_matrix

    @covariance.setter
    def covariance(self, value: np.ndarray) -> None:

        if value.ndim != 2 or not np.allclose(value, value.T.conj()):
            raise ValueError("Antenna correlation must be a hermitian matrix")

        if np.any(np.linalg.eigvals(value) <= 0.0):
            raise ValueError("Antenna correlation matrix must be positive definite")

        self.__covariance_matrix = value


class MultipathFadingChannel(Channel, Serializable):
    """Implements a stochastic fading multipath channel.

    For MIMO systems, the received signal is the addition of the signal transmitted
    at all antennas.
    The channel model is defined in the parameters, which should have the following fields:
    - param.delays - numpy.ndarray containing the delays of all significant paths (in s)
    - param.power_delay_profile - numpy.ndarray containing the average power of each path
    (in linear scale). It must have the same number of elements as 'param.delays'
    - param.k_factor_rice: numpy.ndarray containing the K-factor of the Ricean
    distribution for each path (in linear scale). It must have the same number of
    elements as 'param.delays'

    The channel is time-variant, with the auto-correlation depending on  its maximum
    'doppler_frequency' (in Hz).
    Realizations in different drops are independent.

    The model supports multiple antennas, and the correlation among different antennas can be
    specified in parameters 'tx_cov_matrix' and 'rx_cov_matrix', for the both transmitter and
    at the receiver side, respectively.
    They both must be Hermitian, positive definite square matrices.

    Rayleigh/Rice fading and uncorrelated scattering is considered. Fading follows Jakes'
    Doppler spectrum, using the simulation approach from :footcite:t:`2006:xiao`,
    which is based on the sum of sinusoids with random phases.

    If the delays are not multiple of the sampling interval, then sinc-based interpolation is
    considered.

    Antenna correlation considers the Kronecker model, as described in :footcite:t:`2002:yu`.

    The channel will provide 'number_rx_antennas' outputs to a signal
    consisting of 'number_tx_antennas' inputs. A random number generator,
    given by 'rnd' may be needed. The sampling rate is the same at both input and output
    of the channel, and is given by 'sampling_rate' samples/second.
    `tx_cov_matrix` and `rx_cov_matrix` are covariance matrices for transmitter
    and receiver.
    """

    yaml_tag = "MultipathFading"
    serialized_attributes = {"impulse_response_interpolation", "interpolate_signals"}

    delay_resolution_error: float = 0.4
    __delays: np.ndarray
    __power_profile: np.ndarray
    __rice_factors: np.ndarray
    __max_delay: float
    __num_resolvable_paths: int
    __num_sinusoids: int
    __los_angle: Optional[float]
    los_gains: np.ndarray
    __doppler_frequency: float
    __los_doppler_frequency: Optional[float]
    # Correlation of the first device
    __alpha_correlation: Optional[AntennaCorrelation]
    # Correlation of the second device
    __beta_correlation: Optional[AntennaCorrelation]
    interpolate_signals: bool

    def __init__(
        self,
        delays: Union[np.ndarray, List[float]],
        power_profile: Union[np.ndarray, List[float]],
        rice_factors: Union[np.ndarray, List[float]],
        num_sinusoids: Optional[float] = None,
        los_angle: Optional[float] = None,
        doppler_frequency: Optional[float] = None,
        los_doppler_frequency: Optional[float] = None,
        interpolate_signals: bool = None,
        alpha_correlation: Optional[AntennaCorrelation] = None,
        beta_correlation: Optional[AntennaCorrelation] = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            delays (np.ndarray):
                Delay in seconds of each individual multipath tap.

            power_profile (np.ndarray):
                Power loss factor of each individual multipath tap.

            rice_factors (np.ndarray):
                Rice factor balancing line of sight and multipath in each individual channel tap.

            transmitter (Transmitter, optional):
                The modem transmitting into this channel.

            receiver (Receiver, optional):
                The modem receiving from this channel.

            scenario (Scenario, optional):
                The scenario this channel is attached to.

            active (bool, optional):
                Channel activity flag.

            gain (float, optional):
                Channel power gain.

            random_generator (rnd.Generator, optional):
                Generator object for random number sequences.

            num_sinusoids (int, optional):
                Number of sinusoids used to sample the statistical distribution.

            los_angle (float, optional):
                Angle phase of the line of sight component within the statistical distribution.

            doppler_frequency (float, optional):
                Doppler frequency shift of the statistical distribution.

            alpha_correlation(AntennaCorrelation, optional):
                Antenna correlation model at the first device.
                By default, the channel assumes ideal correlation, i.e. no cross correlations.

            beta_correlation(AntennaCorrelation, optional):
                Antenna correlation model at the second device.
                By default, the channel assumes ideal correlation, i.e. no cross correlations.

            **kwargs (Any, optional):
                Channel base class initialization parameters.

        Raises:
            ValueError:
                If the length of `delays`, `power_profile` and `rice_factors` is not identical.
                If delays are smaller than zero.
                If power factors are smaller than zero.
                If rice factors are smaller than zero.
        """

        # Convert delays, power profile and rice factors to numpy arrays if they were provided as lists
        self.__delays = np.array(delays) if isinstance(delays, list) else delays
        self.__power_profile = np.array(power_profile) if isinstance(power_profile, list) else power_profile
        self.__rice_factors = np.array(rice_factors) if isinstance(rice_factors, list) else rice_factors

        if self.__delays.ndim != 1 or self.__power_profile.ndim != 1 or self.__rice_factors.ndim != 1:
            raise ValueError("Delays, power profile and rice factors must be vectors")

        if len(delays) < 1:
            raise ValueError("Configuration must contain at least one delay tap")

        if len(delays) != len(power_profile) or len(power_profile) != len(rice_factors):
            raise ValueError("Delays, power profile and rice factor vectors must be of equal length")

        if np.any(self.__delays < 0.0):
            raise ValueError("Delays must be greater or equal to zero")

        if np.any(self.__power_profile < 0.0):
            raise ValueError("Power profile factors must be greater or equal to zero")

        if np.any(self.__rice_factors < 0.0):
            raise ValueError("Rice factors must be greater or equal to zero")

        # Sort delays
        sorting = np.argsort(delays)

        self.__delays = self.__delays[sorting]
        self.__power_profile = self.__power_profile[sorting]
        self.__rice_factors = self.__rice_factors[sorting]
        self.__num_sinusoids = 20 if num_sinusoids is None else num_sinusoids
        self.los_angle = los_angle
        self.doppler_frequency = 0.0 if doppler_frequency is None else doppler_frequency
        self.__los_doppler_frequency = None
        self.interpolate_signals = interpolate_signals
        self.alpha_correlation = None
        self.beta_correlation = None

        if los_doppler_frequency is not None:
            self.los_doppler_frequency = los_doppler_frequency

        # Infer additional parameters
        self.__max_delay = max(self.__delays)
        self.__num_resolvable_paths = len(self.__delays)

        rice_inf_pos = np.isposinf(self.__rice_factors)
        rice_num_pos = np.invert(rice_inf_pos)
        self.los_gains = np.empty(self.num_resolvable_paths, dtype=float)
        self.non_los_gains = np.empty(self.num_resolvable_paths, dtype=float)

        self.los_gains[rice_inf_pos] = 1.0
        self.los_gains[rice_num_pos] = np.sqrt(self.__rice_factors[rice_num_pos] / (1 + self.__rice_factors[rice_num_pos]))
        self.non_los_gains[rice_num_pos] = 1 / np.sqrt(1 + self.__rice_factors[rice_num_pos])
        self.non_los_gains[rice_inf_pos] = 0.0

        # Initialize base class
        Channel.__init__(self, **kwargs)

        # Update correlations (required here to break dependency cycle during init)
        self.alpha_correlation = alpha_correlation
        self.beta_correlation = beta_correlation

    @property
    def delays(self) -> np.ndarray:
        """Access configured path delays.

        Returns:
            np.ndarray: Path delays.
        """

        return self.__delays

    @property
    def power_profile(self) -> np.ndarray:
        """Access configured power profile.

        Returns:
            np.ndarray: Power profile.
        """

        return self.__power_profile

    @property
    def rice_factors(self) -> np.ndarray:
        """Access configured rice factors.

        Returns:
            np.ndarray: Rice factors.
        """

        return self.__rice_factors

    @property
    def doppler_frequency(self) -> float:
        """Access doppler frequency shift.

        Returns:
            float: Doppler frequency shift in Hz.
        """

        return self.__doppler_frequency

    @doppler_frequency.setter
    def doppler_frequency(self, frequency: float) -> None:
        """Modify doppler frequency shift configuration.

        Args:
            frequency (float): New doppler frequency shift in Hz.
        """

        self.__doppler_frequency = frequency

    @property
    def los_doppler_frequency(self) -> float:
        """Access doppler frequency shift of the line of sight component.

        Returns:
            float: Doppler frequency shift in Hz.
        """

        if self.__los_doppler_frequency is None:
            return self.doppler_frequency

        return self.__los_doppler_frequency

    @los_doppler_frequency.setter
    def los_doppler_frequency(self, frequency: Optional[float]) -> None:
        """Modify doppler frequency shift configuration of the line of sigh component.

        Args:
            frequency (Optional[float]): New doppler frequency shift in Hz.
        """

        self.__los_doppler_frequency = frequency

    @property
    def max_delay(self) -> float:
        """Access the maximum multipath delay.

        Returns:
            float: The maximum delay.
        """

        return self.__max_delay

    @property
    def num_resolvable_paths(self) -> int:
        """Access the configured number of fading sequences generating a single impulse response.

        Returns:
            int: The number of sequences.
        """

        return self.__num_resolvable_paths

    @property
    def num_sinusoids(self) -> int:
        """Access the configured number of sinusoids within one fading sequence.

        Returns:
            int: The number of sinusoids.
        """

        return self.__num_sinusoids

    @num_sinusoids.setter
    def num_sinusoids(self, num: int) -> None:
        """Modify the configured number of sinusoids within one fading sequence.

        Args:
            num (int): The new number of sinusoids.

        Raises:
            ValueError: If `num` is smaller than zero.
        """

        if num < 0:
            raise ValueError("Number of sinusoids must be greater or equal to zero")

        self.__num_sinusoids = num

    @property
    def los_angle(self) -> Optional[float]:
        """Access configured angle of arrival of the specular model component.

        Returns:
            Optional[float]: The AoA in radians, `None` if it is not configured.
        """

        return self.__los_angle

    @los_angle.setter
    def los_angle(self, angle: Optional[float]) -> None:
        """Access configured angle of arrival of the specular model component.

        Args:
            angle (Optional[float]): The new angle of arrival in radians.
        """

        self.__los_angle = angle

    def realize(self,
                num_samples: int,
                sampling_rate: float) -> ChannelRealization:

        max_delay_in_samples = int(self.__delays[-1] * sampling_rate)
        timestamps = np.arange(num_samples) / sampling_rate

        impulse_response = np.zeros((num_samples, self.receiver.antennas.num_antennas, self.transmitter.antennas.num_antennas, max_delay_in_samples + 1), dtype=complex)

        interpolation_filter: Optional[np.ndarray] = None
        if self.impulse_response_interpolation:
            interpolation_filter = self.interpolation_filter(sampling_rate)

        for power, path_idx, los_gain, nlos_gain in zip(self.__power_profile, range(self.num_resolvable_paths), self.los_gains, self.non_los_gains):

            for rx_idx, tx_idx in product(range(self.transmitter.antennas.num_antennas), range(self.receiver.antennas.num_antennas)):
                signal_weights = power**0.5 * self.__tap(timestamps, los_gain, nlos_gain)

                if interpolation_filter is not None:
                    impulse_response[:, rx_idx, tx_idx, :] += np.outer(signal_weights, interpolation_filter[path_idx, :])

                else:
                    delay_idx = int(self.__delays[path_idx] * sampling_rate)
                    impulse_response[:, rx_idx, tx_idx, delay_idx] += signal_weights

        # Force a signal covariance at the transmitter if configured
        if self.alpha_correlation is not None:

            alpha_covariance = self.alpha_correlation.covariance
            impulse_response = np.tensordot(alpha_covariance, impulse_response, (0, 2)).transpose((1, 2, 0, 3))

        # Force a signal covariance at the receiver if configured
        if self.beta_correlation is not None:

            beta_covariance = self.beta_correlation.covariance
            impulse_response = np.tensordot(beta_covariance, impulse_response, (0, 1)).transpose((1, 0, 2, 3))

        return ChannelRealization(self, self.gain * impulse_response)

    def __tap(self, timestamps: np.ndarray, los_gain: complex, nlos_gain: complex) -> np.ndarray:
        """Generate a single fading sequence tap.

        Implements equation (18) of the underlying paper.

        Args:

            timestamps (np.ndarray):
                Time instances at which the channel should be sampled.

            los_gain (complex):
                Gain of the line-of-sight (specular) model component.

            nlos_gain (complex):
                Gain of the non-line-of-sight model components.

        Returns:

            np.ndarray:
                Channel gains at requested timestamps.
        """

        nlos_doppler = self.doppler_frequency
        nlos_angles = self._rng.uniform(0, 2 * pi, self.num_sinusoids)
        nlos_phases = self._rng.uniform(0, 2 * pi, self.num_sinusoids)

        nlos_component = np.zeros(len(timestamps), dtype=complex)
        for s in range(self.num_sinusoids):
            nlos_component += exp(1j * (nlos_doppler * timestamps * cos((2 * pi * s + nlos_angles[s]) / self.num_sinusoids) + nlos_phases[s]))

        nlos_component *= nlos_gain * (self.num_sinusoids**-0.5)

        if self.los_angle is not None:
            los_angle = self.los_angle

        else:
            los_angle = self._rng.uniform(0, 2 * pi)

        los_doppler = self.los_doppler_frequency
        los_phase = self._rng.uniform(0, 2 * pi)
        los_component = los_gain * exp(1j * (los_doppler * timestamps * cos(los_angle) + los_phase))
        return los_component + nlos_component

    def interpolation_filter(self, sampling_rate: float) -> np.ndarray:
        """Create an interpolation filter matrix.

        Args:
            sampling_rate: The sampling rate to which to interpolate.

        Returns:
            np.ndarray:
                Interpolation filter matrix containing filters for each configured resolvable path.
        """

        num_delay_samples = int(self.__delays[-1] * sampling_rate)
        filter_instances = np.empty((self.num_resolvable_paths, num_delay_samples + 1), float)

        for path_idx, delay in enumerate(self.__delays):

            resampling_matrix = delay_resampling_matrix(sampling_rate, 1, delay, num_delay_samples + 1)
            filter_instances[path_idx, :] = resampling_matrix[:, 0] / np.linalg.norm(resampling_matrix)

        return filter_instances

    @property
    def alpha_correlation(self) -> Optional[AntennaCorrelation]:
        """Antenna correlation at the first device.

        Returns:
            Handle to the correlation model.
            `None`, if not model was configured and ideal correlation is assumed.
        """

        return self.__alpha_correlation

    @alpha_correlation.setter
    def alpha_correlation(self, value: AntennaCorrelation) -> None:

        if value is not None:

            value.channel = self
            value.device = self.transmitter

        self.__alpha_correlation = value

    @property
    def beta_correlation(self) -> Optional[AntennaCorrelation]:
        """Antenna correlation at the second device.

        Returns:
            Handle to the correlation model.
            `None`, if not model was configured and ideal correlation is assumed.
        """

        return self.__beta_correlation

    @beta_correlation.setter
    def beta_correlation(self, value: AntennaCorrelation) -> None:

        if value is not None:

            value.channel = self
            value.device = self.receiver

        self.__beta_correlation = value

    @Channel.transmitter.setter
    def transmitter(self, value: SimulatedDevice) -> None:

        Channel.transmitter.fset(self, value)

        # Register new device at correlation model
        if self.alpha_correlation is not None:
            self.alpha_correlation.device = value

    @Channel.receiver.setter
    def receiver(self, value: SimulatedDevice) -> None:

        Channel.receiver.fset(self, value)

        # Register new device at correlation model
        if self.beta_correlation is not None:
            self.beta_correlation.device = value
