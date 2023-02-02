# -*- coding: utf-8 -*-
"""
====================================
Single-Target Radar Channel Modeling
====================================
"""

from __future__ import annotations
from typing import Optional, Tuple, Union

import numpy as np
from scipy.constants import pi, speed_of_light

from .channel import Channel, ChannelRealization
from hermespy.tools import db2lin
from hermespy.core import FloatingError, Serializable

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarChannelRealization(ChannelRealization):
    """Realization of a radar channel."""

    __ground_truth: np.ndarray

    def __init__(self,
                 channel: RadarChannel,
                 impulse_response: np.ndarray,
                 ground_truth: np.ndarray) -> None:

        self.__ground_truth = ground_truth
        ChannelRealization.__init__(self, channel, impulse_response)

    @property
    def ground_truth(self) -> np.ndarray:
        """Ground truth of the channel realization.

        Returns: A copy of the ground truth.
        """

        return self.__ground_truth.copy()


class RadarChannel(Channel[RadarChannelRealization], Serializable):
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
    Clutter not yet modelled.

    ToDo: Add literature references for this channel model.
    """

    yaml_tag = "RadarChannel"
    serialized_attributes = {"impulse_response_interpolation", "target_exists", "attenuate"}

    __target_range: float
    __radar_cross_section: float
    __target_azimuth: float
    __target_zenith: float
    target_exists: bool
    __losses_db: float
    __target_velocity: float
    attenuate: bool

    def __init__(self, target_range: Union[float, Tuple[float, float]], radar_cross_section: float, target_azimuth: float = 0.0, target_zenith: float = 0.0, target_exists: bool = True, losses_db: float = 0, velocity: Union[float, np.ndarray] = 0, attenuate: bool = True, **kwargs) -> None:
        """
        Args:

            target_range (Union[float, Tuple[float, float]]):
                Absolute distance of target and radar sensor in meters.
                Either a specific distance or a range of minimal and maximal target distance.

            radar_cross_section (float):
                Radar cross section (RCS) of the assumed single-point reflector in m**2

            target_azimuth (float, optional):
                Target location azimuth angle in radians, considering spherical coordinates.
                Zero by default.

            target_zenith (float, optional):
                Target location zenith angle in radians, considering spherical coordinates.
                Zero by default.

            target_exists (bool, optional):
                True if a target exists, False if there is only noise/clutter (default 0 True)

            losses_db (float, optional):
                Any additional atmospheric and/or cable losses, in dB (default = 0)

            velocity (Union[float, np.ndarray], optional):
                Velocity as a 3D vector (or as a float), in m/s (default = 0)

            attenuate (bool, optional):
                If True, then signal will be attenuated depending on the range, RCS and losses.
                If False, then received power is equal to transmit power.

        Raises:
            ValueError:
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
        self.target_velocity = velocity
        self.attenuate = attenuate

    @property
    def target_range(self) -> float:
        """Absolute distance of target and radar sensor.

        Returns: Target range in meters.

        Raises:

            ValueError: If the range is smaller than zero.
        """

        return self.__target_range

    @target_range.setter
    def target_range(self, value: Union[float, Tuple[float, float]]) -> None:

        if isinstance(value, (float, int)):

            if value < 0.0:
                raise ValueError("Target range must be greater or equal to zero")

        elif isinstance(value, (tuple, list)):

            if len(value) != 2:
                raise ValueError("Target range span must be a tuple of two")

            if value[1] < value[0]:
                raise ValueError("Target range span second value must be greater than first value")

            if value[1] < 0.0:
                raise ValueError("Target range span minimum must be greater or equal to zero")

        else:
            raise ValueError("Unknown targer range format")

        self.__target_range = value

    @property
    def target_velocity(self) -> np.ndarray:
        """Perceived target velocity.

        Returns: Velocity in m/s.
        """

        return self.__target_velocity

    @target_velocity.setter
    def target_velocity(self, value: float) -> None:

        self.__target_velocity = value

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
            raise ValueError("Radar cross section be greater than or equal to zero")

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

    def realize(self,
                num_samples: int,
                sampling_rate: float) -> RadarChannelRealization:

        if self.transmitter is None:
            raise FloatingError("Radar channel must be anchored to a transmitting device")

        if self.transmitter.carrier_frequency <= 0.0:
            raise RuntimeError("Radar channel does not support base-band transmissions")

        # For the radar channel, only channels linking the same device are currently feasible
        if self.transmitter is not self.receiver:
            raise RuntimeError("Radar channels may only link the same devices")

        # Impulse response sample timestamps
        timestamps = np.arange(num_samples) / sampling_rate

        # The overall perceived velocity is a sum of the target's radial velocity
        # and the transmitter's velocity component pointing towards the target
        # target_normal = np.array([cos(self.target_azimuth) * sin(self.target_zenith), sin(self.target_azimuth) * sin(self.target_zenith), cos(self.target_zenith)], dtype=float)
        # transmitter_velocity_abs = np.linalg.norm(self.transmitter.velocity, 2)
        # transmitter_target_velocity = transmitter_velocity_abs * np.dot(target_normal, self.transmitter.velocity / transmitter_velocity_abs) if transmitter_velocity_abs > 0. else 0.
        # velocity = self.target_velocity - transmitter_target_velocity

        # Infer relevant parameters
        wavelength = speed_of_light / self.transmitter.carrier_frequency
        doppler_frequency = 2 * self.target_velocity / ((1 - self.target_velocity / speed_of_light) * wavelength)
        target_range = self.target_range if isinstance(self.target_range, (float, int)) else self._rng.uniform(self.target_range[0], self.target_range[1])
        delay = 2 * target_range / speed_of_light
        max_delay = delay + 2 * self.target_velocity * timestamps[-1] / speed_of_light
        max_delay_in_samples = int(np.ceil(max_delay * self.transmitter.sampling_rate))

        impulse_response = np.zeros((self.num_outputs, self.num_inputs, num_samples, 1 + max_delay_in_samples), dtype=complex)

        # If no target is present we may abort already
        if not self.target_exists:
            return RadarChannelRealization(self, impulse_response, np.empty((0, 3), dtype=float))

        # The radar target's channel weight is essentially a mix of
        # 1. The phase shift during reflection (uniformly distributed)
        # 2. The power loss during reflection (semi-deterministic, depends on rcs, wavelength and target distance)
        reflection_phase = self._rng.uniform(0, 1)

        power_factor = 1.0
        if self.attenuate:
            power_factor = wavelength**2 * self.__radar_cross_section / (4 * pi) ** 3 / target_range**4 * db2lin(self.__losses_db)

        delay_taps = np.arange(1 + max_delay_in_samples) / sampling_rate

        array_response = self.transmitter.antennas.spherical_response(self.transmitter.carrier_frequency, self.target_azimuth, self.target_zenith)
        mimo_response = np.outer(array_response.conj(), array_response)

        for idx, timestamp in enumerate(timestamps):

            echo_delay = delay + 2 * self.target_velocity * timestamp / speed_of_light
            time = timestamp + delay_taps
            echo_weights = power_factor * np.exp(2j * pi * (doppler_frequency * time + reflection_phase))

            interpolated_impulse_tap = np.sinc(sampling_rate * (delay_taps - echo_delay)) * echo_weights

            # Note that this impulse response selection is technically incorrect,
            # since it is only feasible for planar arrays
            impulse_response[:, :, idx, :] = np.tensordot(mimo_response, interpolated_impulse_tap, axes=0)

        ground_truth = np.array([[0.0, 0.0, target_range]])
        return RadarChannelRealization(self, impulse_response, ground_truth)

    def null_hypothesis(self,
                        realization: Optional[RadarChannelRealization] = None) -> RadarChannelRealization:
        """Generate a channel realization missing the target to be estimated.

        Args:
    
            realization (RadarChannelRealization, optional):
                Channel realization for which to generated a null hypothesis.
                By default, the recent channel realization will be assumed.

        Returns: Null hypothesis channel realization.

        Raises:

            RuntimeError: If no `realization` was provided and the channel hasn't been propagated over yet.
        """

        # Assume the last channel propagation realization if the realization has not been specified
        if realization is None:

            realization = self.realization
            
            if realization is None:
                raise RuntimeError("Channel has not been propagated over yet")

        impulse_response = np.zeros(realization.state.shape, dtype=complex)
        return RadarChannelRealization(self, impulse_response, realization.ground_truth)
