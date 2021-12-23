# -*- coding: utf-8 -*-
"""Radar Channel Model."""
from __future__ import annotations
from typing import Type

import numpy as np
from ruamel.yaml import SafeRepresenter, MappingNode
from scipy import constants

from hermespy.channel import Channel
from hermespy.tools import db2lin, lin2db, DbConversionType

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarChannel(Channel):
    """Implements a monostatic radar channel in base-band.

    The radar channel is currently implemented as a single-point reflector.
    The model also considers the presence of self-interference due to leakage from transmitter to receiver.

    Attenuation is considered constant and calculated according to the radar range equation. The received signal is
    considered to have the same power as the transmitted signal, and attenuation will be taken into account in the level
    of the self-interference.

    Moving targets are also taken into account, considering both Doppler and a change in the delay during a drop.

    Both the reflected signal and the self interference will have a random phase.

    Obs.:
    Currently only one transmit and receive antennas is supported.
    Only a radial velocity is considered.
    """

    yaml_tag = u'RadarChannel'
    yaml_matrix = True

    __target_range: float
    __radar_cross_section: float
    __carrier_frequency: float
    target_exists: bool
    __tx_rx_isolation_db: float
    __tx_antenna_gain_db: float
    __rx_antenna_gain_db: float
    __losses_db: float
    __velocity: float
    __filter_response_in_samples: int

    def __init__(self,
                 target_range: float,
                 radar_cross_section: float,
                 target_exists: bool = True,
                 tx_rx_isolation_db: float = float("inf"),
                 tx_antenna_gain_db: float = 0,
                 rx_antenna_gain_db: float = 0,
                 losses_db: float = 0,
                 velocity: float = 0,
                 filter_response_in_samples: int = 21,
                 **kwargs
                 ) -> None:
        """Object initialization.

        Args:
            target_range(float): distance from transmitter to target object
            radar_cross_section(float): in m**2
            target_exists(bool, optional): True if there is a target, False if there is only noise/clutter
            tx_rx_isolation_db(float, optional): isolation between transmitter and receiver (leakage) in dB
                                                 (default = inf)
            tx_antenna_gain_db(float, optional):
            rx_antenna_gain_db(float, optional): antenna gains in dBi (default = 0)
            losses_db(float, optional): any additional atmospheric and/or cable losses, in dB (default = 0)
            velocity(float, optional): radial velocity, in m/s (default = 0)
            filter_response_in_samples(int, optional): length of interpolation filter in samples (default = 7)

        Raises:
            ValueError:
                If target_range < 0.
                If radar_cross_section < 0.
                If carrier_frequency <= 0.
                If more than one antenna is considered.
        """

        # Init base class
        Channel.__init__(self, **kwargs)

        if self.num_inputs > 1 or self.num_outputs > 1:
            raise ValueError("Multiple antennas are not supported")

        self.target_range = target_range
        self.radar_cross_section = radar_cross_section
        self.target_exists = target_exists
        self.tx_rx_isolation_db = tx_rx_isolation_db
        self.__tx_antenna_gain_db = tx_antenna_gain_db
        self.__rx_antenna_gain_db = rx_antenna_gain_db
        self.__losses_db = losses_db
        self.velocity = velocity
        self.__filter_response_in_samples = filter_response_in_samples

        # random phases
        self._phase_self_interference = 0
        self._phase_echo = 0

    @property
    def target_range(self) -> float:
        """Access configured target range.

        Returns:
            float: range [m]
        """
        return self.__target_range

    @target_range.setter
    def target_range(self, value: float) -> None:
        """Modify the configured number of the target range

        Args:
            value (float): The new target range.

        Raises:
            ValueError: If `value` is less than zero.
        """

        if value < 0:
            raise ValueError("Target range must be greater than or equal to zero")

        self.__target_range = value

    @property
    def radar_cross_section(self) -> float:
        """Access configured radar cross section.

        Returns:
            float: radar cross section [m**2]
        """
        return self.__radar_cross_section

    @radar_cross_section.setter
    def radar_cross_section(self, value: float) -> None:
        """Modify the configured number of the radar cross section

        Args:
            value (float): The new RCS.

        Raises:
            ValueError: If `value` is less than zero.
        """

        if value < 0:
            raise ValueError("Target range must be greater than or equal to zero")

        self.__radar_cross_section = value

    @property
    def tx_rx_isolation_db(self) -> float:
        """Access configured TX/RX isolation

        Returns:
            float: TX/RX isolation [dB]
        """
        return self.__tx_rx_isolation_db

    @tx_rx_isolation_db.setter
    def tx_rx_isolation_db(self, value: bool) -> None:
        """Modify the configured tx/rx isolation

        Args:
            value (bool): The new tx/rx isolation
        """
        self.__tx_rx_isolation_db = value

    @property
    def tx_antenna_gain_db(self) -> float:
        """Access configured TX antenna gain

        Returns:
            float: TX antenna gain [dBi]
        """
        return self.__tx_antenna_gain_db

    @property
    def rx_antenna_gain_db(self) -> float:
        """Access configured RX antenna gain

        Returns:
            float: RX antenna gain [dBi]
        """
        return self.__rx_antenna_gain_db

    @property
    def losses_db(self) -> float:
        """Access configured (atmospheric and cable) losses

        Returns:
            float: losses [dB]
        """
        return self.__losses_db

    @property
    def velocity(self) -> float:
        """Access configured velocity

        Returns:
            float: velocity [m/s]
        """
        return self.__velocity

    @velocity.setter
    def velocity(self, value: float) -> None:
        """Modify the configured velocity

        Args:
            value (float): The new velocity
        """
        self.__velocity = value

    @property
    def filter_response_in_samples(self) -> int:
        """Access configured interpolation filter response length

        Returns:
            int: length of interpolation filter in samples
        """
        return self.__filter_response_in_samples

    @property
    def delay(self) -> float:
        """Get delay from target

        Returns:
            float: propagation delay [s]
        """
        return 2 * self.__target_range / constants.speed_of_light

    @property
    def attenuation(self) -> float:
        """Get attenuation of returned echo

        Returns:
            float: power attenuation in linear scale
        """
        wavelength = constants.speed_of_light / self.transmitter.carrier_frequency
        attenuation = (db2lin(self.__tx_antenna_gain_db + self.__rx_antenna_gain_db - self.__losses_db)
                       * wavelength ** 2 * self.__radar_cross_section / (4 * np.pi)**3 / self.__target_range**4)

        return attenuation

    @property
    def attenuation_db(self) -> float:
        return lin2db(self.attenuation)

    def init_drop(self) -> None:
        """Initializes random channel parameters for each drop, by selecting random phases"""
        self._phase_self_interference = self._rng.random() * 2 * np.pi - np.pi
        self._phase_echo = self._rng.random() * 2 * np.pi - np.pi

    def propagate(self, tx_signal: np.ndarray) -> np.ndarray:
        """Modifies the input signal and returns it after channel propagation.

        Currently only a single antenna is supported

        Args:
            tx_signal (np.ndarray):
                Input signal to channel with shape of `N_tx_antennas x n_samples`.

        Returns:
            np.ndarray:
                Signal after channel propagation, containing echo and self interference.
                If input is an array of size 'number_tx_antennas' X 'number_of_samples', then the output is of size
                'number_rx_antennas' X 'number_of_samples' + L, with L accounting for the propagation delay and filter
                overhead.
        """
        doppler_frequency = 2 * self.transmitter.carrier_frequency * self.__velocity / constants.speed_of_light

        samples_in_frame = tx_signal.shape[1]
        frame_length = samples_in_frame / self.transmitter.sampling_rate

        # minimum and maximum delay during whole drop
        max_delay = np.maximum(self.delay, self.delay + 2 * self.velocity * frame_length / constants.speed_of_light)
        min_delay = np.minimum(self.delay, self.delay + 2 * self.velocity * frame_length / constants.speed_of_light)

        # delay in samples considering filter overheads
        filter_overhead = int(self.__filter_response_in_samples / 2)
        min_delay_in_samples = int(np.max(np.floor(min_delay * self.transmitter.sampling_rate) - filter_overhead, 0))
        max_delay_in_samples = int(np.ceil(max_delay * self.transmitter.sampling_rate) + filter_overhead)

        delayed_signal = np.zeros((1, samples_in_frame + max_delay_in_samples), dtype=complex)

        if self.target_exists:

            time = np.arange(samples_in_frame + max_delay_in_samples) / self.transmitter.sampling_rate

            # variable echo delay during drop
            echo_delay = self.delay + 2 * self.velocity * time[:samples_in_frame] / constants.speed_of_light

            # time-variant convolution
            for idx, delay_in_samples in enumerate(range(min_delay_in_samples, max_delay_in_samples + 1)):
                delay = delay_in_samples / self.transmitter.sampling_rate
                delay_gain = np.sinc(self.transmitter.sampling_rate * (delay - echo_delay))
                delayed_signal[0, delay_in_samples: delay_in_samples+samples_in_frame] += \
                    tx_signal.flatten() * delay_gain

            # random phase and Doppler shift
            delayed_signal = delayed_signal * np.exp(-1j * (2 * np.pi * doppler_frequency * time
                                                            + self._phase_echo))

        # add self interference
        self_interference = (np.hstack((tx_signal, np.zeros((1, max_delay_in_samples))))
                             / np.sqrt(self.attenuation) * np.exp(1j * self._phase_self_interference)
                             / db2lin(self.tx_rx_isolation_db, conversion_type=DbConversionType.AMPLITUDE))
        rx_signal = delayed_signal + self_interference

        return rx_signal

    def get_impulse_response(self, timestamps: np.array) -> np.ndarray:
        """Calculates the channel impulse response.

        This method can be used for instance by the transceivers to obtain the
        channel state information.

        Args:
            timestamps (np.array): Time instants with length T to calculate the
                response for.

        Returns:
            np.ndarray:
                Impulse response in all 'number_rx_antennas' x 'number_tx_antennas' channels
                at the time instants given in vector 'timestamps'.
                `impulse_response` is a 4D-array, with the following dimensions:
                1- sampling instants, 2 - Rx antennas, 3 - Tx antennas, 4 - delays
                (in samples)
                The shape is T x number_rx_antennas x number_tx_antennas x (L+1)
        """
        doppler_frequency = 2 * self.transmitter.carrier_frequency * self.__velocity / constants.speed_of_light

        max_delay = np.maximum(self.delay,
                               self.delay + 2 * self.velocity * np.max(timestamps) / constants.speed_of_light)
        filter_overhead = int(self.__filter_response_in_samples / 2)

        max_delay_in_samples = int(np.ceil(max_delay * self.transmitter.sampling_rate)) + filter_overhead

        impulse_response = np.zeros((timestamps.size, self.num_outputs, self.num_inputs,
                                     max_delay_in_samples), dtype=complex)

        for idx, timestamp in enumerate(timestamps):
            delay = np.arange(max_delay_in_samples) / self.transmitter.sampling_rate

            if self.target_exists:
                echo_delay = self.delay + 2 * self.velocity * timestamp / constants.speed_of_light
                time = timestamp + np.arange(max_delay_in_samples) / self.transmitter.sampling_rate
                echo_phase = np.exp(-1j * (2 * np.pi * doppler_frequency * time + self._phase_echo))

                impulse_response[idx, 0, 0, :] = (np.sinc(self.transmitter.sampling_rate * (delay - echo_delay))
                                                  * echo_phase)

            impulse_response[idx, 0, 0, 0] += (np.exp(1j * self._phase_self_interference)
                                               / np.sqrt(self.attenuation)
                                               / db2lin(self.tx_rx_isolation_db,
                                                        conversion_type=DbConversionType.AMPLITUDE))

        return impulse_response

    @classmethod
    def to_yaml(cls: Type[RadarChannel], representer: SafeRepresenter,
                node: RadarChannel) -> MappingNode:
        """Serialize a radar channel object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (RadarChannel):
                The channel instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            'target_range': node.target_range,
            'radar_cross_section': node.radar_cross_section,
            'gain': node.gain,
            'tx_rx_isolation_db': node.tx_rx_isolation_db,
            'tx_antenna_gain_db': node.tx_antenna_gain_db,
            'rx_antenna_gain_db': node.rx_antenna_gain_db,
            'losses_db': node.losses_db,
            'velocity': node.velocity,
            'filter_response_in_samples': node.filter_response_in_samples,
        }

        transmitter_index, receiver_index = node.indices

        yaml = representer.represent_mapping(u'{.yaml_tag} {} {}'.format(cls, transmitter_index, receiver_index), state)
        return yaml
