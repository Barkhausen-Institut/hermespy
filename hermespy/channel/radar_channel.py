# -*- coding: utf-8 -*-
"""
====================================
Single-Target Radar Channel Modeling
====================================
"""

from __future__ import annotations
from typing import Type

import numpy as np
from ruamel.yaml import SafeRepresenter, MappingNode
from scipy.constants import pi, speed_of_light

from ..core.device import FloatingError
from ..tools import db2lin
from .channel import Channel

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarChannel(Channel):
    """Model of a monostatic radar channel in base-band.

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

    ToDo: Add literature references for this channel model.
    """

    yaml_tag = u'RadarChannel'
    yaml_matrix = True

    __target_range: float
    __radar_cross_section: float
    __target_azimuth: float
    __target_zenith: float
    target_exists: bool
    __losses_db: float
    __velocity: float

    def __init__(self,
                 target_range: float,
                 radar_cross_section: float,
                 target_azimuth: float = 0.,
                 target_zenith: float = 0.,
                 target_exists: bool = True,
                 losses_db: float = 0,
                 velocity: float = 0,
                 **kwargs) -> None:
        """
        Args:

            target_range (float):
                Distance from transmitter to target object

            radar_cross_section (float):
                Radar cross section (RCS) of the assumed single-point reflector in m**2

            target_azimuth (float, optional):
                Target location azimuth angle in radians, considering spherical coordinates.
                Zero by default.
                
            target_zenith (float, optional):
                Target location zenith angle in radians, considering spherical coordinates.
                Zero by default.

            target_exists (bool, optional):
                True if a target exists, False if there is only noise/clutter

            losses_db (float, optional):
                Any additional atmospheric and/or cable losses, in dB (default = 0)

            velocity (float, optional):
                Radial velocity, in m/s (default = 0)

        Raises:
            ValueError:
                If target_range < 0.
                If radar_cross_section < 0.
                If carrier_frequency <= 0.
                If more than one antenna is considered.
        """

        # Init base class
        Channel.__init__(self, **kwargs)

        self.target_range = target_range
        self.radar_cross_section = radar_cross_section
        self.target_azimuth = target_azimuth
        self.target_zenith = target_zenith
        self.target_exists = target_exists
        self.__losses_db = losses_db
        self.velocity = velocity

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
    def target_azimuth(self) -> float:
        """Target position azimuth in spherical coordiantes.
        
        Returns:
        
            Azimuth angle in radians.
        """
        
        return self.__target_azimuth
    
    @target_azimuth.setter
    def target_azimuth(self, value: float) -> None:
        
        self.__target_azimuth = value
        
        
    @property
    def target_zenith(self) -> float:
        """Target position zenith in spherical coordiantes.
        
        Returns:
        
            Zenith angle in radians.
        """
        
        return self.__target_zenith
    
    @target_zenith.setter
    def target_zenith(self, value: float) -> None:
        
        self.__target_zenith = value

    @property
    def losses_db(self) -> float:
        """Access configured (atmospheric and cable) losses

        Returns:
            float: losses [dB]
        """
        return self.__losses_db

    @property
    def delay(self) -> float:
        """Get delay from target

        Returns:
            float: propagation delay [s]
        """
        return 2 * self.__target_range / speed_of_light

    def impulse_response(self,
                         num_samples: int,
                         sampling_rate: float) -> np.ndarray:

        # For the radar channel, only channels linking the same device are currently feasible
        if self.transmitter is not self.receiver:
            raise RuntimeError("Radar channels may only link the same devices")

        if self.transmitter is None:
            raise FloatingError("Radar channel must be anchored to a transmitting device")

        if self.transmitter.carrier_frequency <= 0.:
            raise RuntimeError("Radar channel does not support base-band transmissions")

        # Impulse response sample timestamps
        timestamps = np.arange(num_samples) / sampling_rate

        velocity = self.velocity

        # Infer relevant parameters
        wavelength = speed_of_light / self.transmitter.carrier_frequency
        doppler_frequency = 2 * velocity / wavelength
        max_delay = self.delay + 2 * velocity * timestamps[-1] / speed_of_light
        max_delay_in_samples = int(np.ceil(max_delay * self.transmitter.sampling_rate))

        impulse_response = np.zeros((num_samples, self.num_outputs, self.num_inputs, max_delay_in_samples),
                                    dtype=complex)

        # If no target is present we may abort already
        if not self.target_exists:
            return impulse_response

        # The radar target's channel weight is essentially a mix of
        # 1. The phase shift during reflection (uniformly distributed)
        # 2. The power loss during reflection (semi-deterministic, depends on rcs, wavelength and target distance)
        reflection_phase = self._rng.uniform(0, 1)
        power_factor = (wavelength ** 2 * self.__radar_cross_section / (4 * pi) ** 3 / self.__target_range ** 4
                        * db2lin(self.__losses_db))

        delay_taps = np.arange(max_delay_in_samples) / sampling_rate
        
        array_response = self.transmitter.antennas.spherical_response(self.transmitter.carrier_frequency, self.target_azimuth, self.target_zenith)
        mimo_response = np.outer(array_response.conj(), array_response)

        for idx, timestamp in enumerate(timestamps):

            echo_delay = self.delay + 2 * self.velocity * timestamp / speed_of_light
            time = timestamp + np.arange(max_delay_in_samples) / sampling_rate
            echo_weights = power_factor * np.exp(2j * pi * (doppler_frequency * time + reflection_phase))

            interpolated_impulse_tap = np.sinc(sampling_rate * (delay_taps - echo_delay)) * echo_weights

            # Note that this impulse response selection is technically incorrect,
            # since it is only feasible for planar arrays
            impulse_response[idx, ::] = np.tensordot(mimo_response, interpolated_impulse_tap, axes=0)

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
            'losses_db': node.losses_db,
            'velocity': node.velocity,
        }

        return representer.represent_mapping(cls.yaml_tag, state)
