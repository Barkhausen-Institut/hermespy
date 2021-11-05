# -*- coding: utf-8 -*-
"""Multipath Fading Channel Model."""

from __future__ import annotations
from scipy.constants import pi
from numpy import cos, exp
from typing import TYPE_CHECKING, Optional, Type
from ruamel.yaml import SafeRepresenter, MappingNode
from itertools import product
import numpy as np

from channel.channel import Channel

if TYPE_CHECKING:
    from modem import Transmitter, Receiver

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class MultipathFadingChannel(Channel):
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
    Doppler spectrum, using the simulation approach from:

        `C. Xiao, Zheng, Y. and Beaulieu, N. "Novel sum-of-sinusoids simulation models for
        rayleigh and rician fading channels. IEEE Trans. Wireless Comm., 5(12), 2006`

    which is based on the sum of sinusoids with random phases.

    If the delays are not multiple of the sampling interval, then sinc-based interpolation is
    considered.

    Antenna correlation considers the Kronecker model, as described in:

        K. Yu, M. Bengtsson, B. Ottersten, D. McNamara, P. Karlsson, and M. Beach, “A wideband
        statistical model for NLOS indoor MIMO channels,” IEEE Veh. Technol. Conf.
        (VTC - Spring), 2002

    The channel will provide 'number_rx_antennas' outputs to a signal
    consisting of 'number_tx_antennas' inputs. A random number generator,
    given by 'rnd' may be needed. The sampling rate is the same at both input and output
    of the channel, and is given by 'sampling_rate' samples/second.
    `tx_cov_matrix` and `rx_cov_matrix` are covariance matrices for transmitter
    and receiver.

    Attributes:
        __delays (np.ndarray): Delay per propagation case in seconds.
        __power_profile (np.ndarray): Power per propagation case.
        __rice_factors (np.ndarray): Rice factor per propagation case.
        __max_delay (float): Maximum propagation delay in seconds.
        __num_resolvable paths (int): Number of resolvable paths within the multipath model.
        __num_sinusoids (int): Number of sinusoids components per sample sequence.
        __los_angle (Optional[float]): Line of sight angle of arrival.
        __los_gains (np.array): Path gains for line of sight component in sample sequence, derived from rice factor
        __non_los_gains (np.array): Path gains for non-line of sight in sample sequence, derived from rice factor
        __transmit_precoding (np.ndarray): Precoding matrix for antenna streams before propagation.
        __receive_postcoding (np.ndarray): Postcoding matrix for antenna streams after propagation.
        __doppler_frequency (float): Doppler frequency in Hz.
        __los_doppler_frequency (Optional[float]): Optional doppler frequency for the line of sight component.
        interpolate_signals (bool): Interpolate signals during time-delay modeling. Disabled by default.
    """

    yaml_tag = u'MultipathFading'
    yaml_matrix = True
    __delays: np.ndarray
    __power_profile: np.ndarray
    __rice_factors: np.ndarray
    __max_delay: float
    __num_resolvable_paths: int
    __num_sinusoids: int
    __los_angle: Optional[float]
    __los_gains: np.ndarray
    __transmit_precoding: Optional[np.ndarray]
    __receive_postcoding: Optional[np.ndarray]
    __doppler_frequency: float
    __los_doppler_frequency: Optional[float]
    interpolate_signals: bool

    def __init__(self,
                 delays: np.ndarray,
                 power_profile: np.ndarray,
                 rice_factors: np.ndarray,
                 transmitter: Optional[Transmitter] = None,
                 receiver: Optional[Receiver] = None,
                 active: Optional[bool] = None,
                 gain: Optional[float] = None,
                 num_sinusoids: Optional[float] = None,
                 los_angle: Optional[float] = None,
                 doppler_frequency: Optional[float] = None,
                 los_doppler_frequency: Optional[float] = None,
                 transmit_precoding: Optional[np.ndarray] = None,
                 receive_postcoding: Optional[np.ndarray] = None,
                 interpolate_signals: bool = None) -> None:
        """Object initialization.

        Args:
            delays (np.ndarray): Delay in seconds of each individual multipath tap.
            power_profile (np.ndarray): Power loss factor of each individual multipath tap.
            rice_factors (np.ndarray): Rice factor balancing line of sight and multipath in each individual channel tap.
            transmitter (Transmitter, optional): The modem transmitting into this channel.
            receiver (Receiver, optional): The modem receiving from this channel.
            active (bool, optional): Channel activity flag.
            gain (float, optional): Channel power gain.
            num_sinusoids (int, optional): Number of sinusoids used to sample the statistical distribution.
            los_angle (float, optional): Angle phase of the line of sight component within the statistical distribution.
            doppler_frequency (float, optional): Doppler frequency shift of the statistical distribution.
            transmit_precoding (np.ndarray): Transmit precoding matrix.
            receive_postcoding (np.ndarray): Receive postcoding matrix.

        Raises:
            ValueError:
                If the length of `delays`, `power_profile` and `rice_factors` is not identical.
                If delays are smaller than zero.
                If power factors are smaller than zero.
                If rice factors are smaller than zero.
        """

        if delays.ndim != 1 or power_profile.ndim != 1 or rice_factors.ndim != 1:
            raise ValueError("Delays, power profile and rice factors must be vectors")

        if len(delays) != len(power_profile) or len(power_profile) != len(rice_factors):
            raise ValueError("Delays, power profile and rice factor vectors must be of equal length")

        if len(delays) < 1:
            raise ValueError("Configuration must contain at least one delay tap")

        if np.any(delays < 0.0):
            raise ValueError("Delays must be greater or equal to zero")

        if np.any(power_profile < 0.0):
            raise ValueError("Power profile factors must be greater or equal to zero")

        if np.any(rice_factors < 0.0):
            raise ValueError("Rice factors must be greater or equal to zero")

        # Init base class
        Channel.__init__(self, transmitter, receiver, active, gain)

        self.__delays = delays
        self.__power_profile = power_profile
        self.__rice_factors = rice_factors
        self.__num_sinusoids = 10       # TODO: Old implementation was 20. WHY? Slightly more than 8 seems sufficient...
        self.los_angle = los_angle
        self.__transmit_precoding = None
        self.__receive_postcoding = None
        self.__doppler_frequency = 0.0
        self.__los_doppler_frequency = None
        self.interpolate_signals = interpolate_signals

        if num_sinusoids is not None:
            self.num_sinusoids = num_sinusoids

        if doppler_frequency is not None:
            self.doppler_frequency = doppler_frequency

        if los_doppler_frequency is not None:
            self.los_doppler_frequency = los_doppler_frequency

        if transmit_precoding is not None:
            self.transmit_precoding = transmit_precoding

        if receive_postcoding is not None:
            self.receive_postcoding = receive_postcoding

        # Infer additional parameters
        self.__max_delay = max(delays)
        self.__num_resolvable_paths = len(delays)

        rice_inf_pos = np.isposinf(rice_factors)
        rice_num_pos = np.invert(rice_inf_pos)
        self.__los_gains = np.empty(self.num_resolvable_paths, dtype=float)
        self.__non_los_gains = np.empty(self.num_resolvable_paths, dtype=float)

        self.__los_gains[rice_inf_pos] = 1.0
        self.__los_gains[rice_num_pos] = np.sqrt(rice_factors[rice_num_pos]) / np.sqrt(1 + rice_factors[rice_num_pos])

        self.__non_los_gains[rice_num_pos] = 1 / np.sqrt(1 + rice_factors[rice_num_pos])
        self.__non_los_gains[rice_inf_pos] = 0.0

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
    def transmit_precoding(self) -> Optional[np.ndarray]:
        """Access transmit precoding matrix.

        Returns:
            Optional[np.ndarray]: Transmit precoding matrix, None if no matrix ix configured.
        """

        return self.__transmit_precoding

    @transmit_precoding.setter
    def transmit_precoding(self, matrix: Optional[np.ndarray]) -> None:
        """Configure transmit precoding matrix.

        Args:
            matrix (Optional[np.ndarray]): New transmit precoding matrix.

        Raises:
            ValueError: If `matrix` is not a matrix (does not have two dimensions).
            ValueError: If `matrix` is not positive definite or hermitian.
        """

        if matrix is None:

            self.__transmit_precoding = None
            return

        if matrix.ndim != 2:
            raise ValueError("Transmit precoding must be a matrix (an array with two dimensions)")

        if not np.array_equal(matrix, matrix.conj().transpose()):
            raise ValueError("Transmit precoding matrix must be hermitian")

        if not np.all(np.linalg.eigvals(matrix) > 0):
            raise ValueError("Transmit precoding matrix must be positive definite")

        self.__transmit_precoding = matrix

    @property
    def receive_postcoding(self) -> Optional[np.ndarray]:
        """Access receive postcoding matrix.

        Returns:
            Optional[np.ndarray]: Receive postcoding matrix, None if no matrix ix configured.
        """

        return self.__receive_postcoding
    
    @receive_postcoding.setter
    def receive_postcoding(self, matrix: Optional[np.ndarray]) -> None:
        """Configure receive postcoding matrix.

        Args:
            matrix (Optional[np.ndarray]): New receive postcoding matrix.

        Raises:
            ValueError: If `matrix` is not a matrix (does not have two dimensions).
            ValueError: If `matrix` is not positive definite or hermitian.
        """

        if matrix is None:

            self.__receive_postcoding = None
            return

        if matrix.ndim != 2:
            raise ValueError("Receive postcoding must be a matrix (an array with two dimensions)")

        if not np.array_equal(matrix, matrix.conj().transpose()):
            raise ValueError("Receive postcoding matrix must be hermitian")

        if not np.all(np.linalg.eigvals(matrix) > 0):
            raise ValueError("Receive postcoding matrix must be positive definite")

        self.__receive_postcoding = matrix

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

    def impulse_response(self, timestamps: np.ndarray) -> np.ndarray:

        delays_in_samples = np.round(self.__delays * self.scenario.sampling_rate).astype(int)
        max_delay_in_samples = delays_in_samples.max()

        impulse_response = np.zeros((len(timestamps),
                                     self.receiver.num_antennas,
                                     self.transmitter.num_antennas,
                                     max_delay_in_samples + 1), dtype=complex)

        for power, delay_in_samples, los_gain, nlos_gain in zip(self.__power_profile, delays_in_samples,
                                                                self.__non_los_gains, self.__non_los_gains):

            power_factor = np.sqrt(power)

            for rx_idx, tx_idx in product(range(self.transmitter.num_antennas), range(self.receiver.num_antennas)):

                signal_weights = power_factor * self.__tap(timestamps, los_gain, nlos_gain)
                impulse_response[:, rx_idx, tx_idx, delay_in_samples] = signal_weights

        self.recent_response = impulse_response
        return impulse_response

    def __tap(self, timestamps: np.ndarray, los_gain: complex, nlos_gain: complex) -> np.ndarray:
        """Generate a single fading sequence tap.

        Implements equation (18) of the underlying paper.

        Args:
            timestamps (np.ndarray): Time instances at which the channel should be sampled.
            los_gain (complex): Gain of the line-of-sight (specular) model component.
            nlos_gain (complex): Gain of the non-line-of-sight model components.

        Returns:
            np.ndarray: Channel gains at requested timestamps.
        """

        nlos_doppler = self.doppler_frequency
        nlos_angles = np.random.uniform(0, 2*pi, self.num_sinusoids)
        nlos_phases = np.random.uniform(0, 2*pi, self.num_sinusoids)

        nlos_component = np.zeros(len(timestamps), dtype=complex)
        for s in range(self.num_sinusoids):

            nlos_component += exp(1j * (nlos_doppler * timestamps * cos((2*pi*s + nlos_angles[s]) / self.num_sinusoids) +
                                        nlos_phases[s]))

        nlos_component *= nlos_gain * (self.num_sinusoids ** -.5)

        if self.los_angle is not None:
            los_angle = self.los_angle

        else:
            los_angle = np.random.uniform(0, 2*pi)

        los_doppler = self.los_doppler_frequency
        los_phase = np.random.uniform(0, 2*pi)
        los_component = los_gain * exp(1j * (los_doppler * timestamps * cos(los_angle) + los_phase))

        return los_component + nlos_component

    @classmethod
    def to_yaml(cls: Type[MultipathFadingChannel], representer: SafeRepresenter,
                node: MultipathFadingChannel) -> MappingNode:
        """Serialize a channel object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (MultipathFadingChannel):
                The channel instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            'delays': node.delays.tolist(),
            'power_profile': node.power_profile.tolist(),
            'active': node.active,
            'gain': node.gain,
            'num_sinusoids': node.num_sinusoids,
            'los_angle': node.los_angle,
            'doppler_frequency': node.doppler_frequency,
            'los_doppler_frequency': node.los_doppler_frequency,
            'transmit_precoding': node.transmit_precoding,
            'receive_postcoding': node.receive_postcoding,
            'interpolate_signals': node.interpolate_signals
        }

        transmitter_index, receiver_index = node.indices

        yaml = representer.represent_mapping(u'{.yaml_tag} {} {}'.format(cls, transmitter_index, receiver_index), state)
        return yaml
