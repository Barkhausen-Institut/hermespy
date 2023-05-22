# -*- coding: utf-8 -*-
"""
====================================
Single-Target Radar Channel Modeling
====================================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from itertools import product
from typing import Generic, List, Sequence, Set, Tuple, TypeVar

import numpy as np
from scipy.constants import pi, speed_of_light

from .channel import Channel, ChannelRealization
from hermespy.core import Device, Direction, FloatingError, Moveable, Serializable, Transformable, Transformation
from hermespy.tools import amplitude_path_loss

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarTarget(ABC):
    """Abstract base class of radar targets."""

    __static: bool

    def __init__(self, static: bool = False) -> None:
        """
        Args:

            static (bool, optional):
                Is the target visible during null hypothesis testing?
                Disabled by default.
        """

        self.__static = static

    @abstractmethod
    def get_cross_section(self, impinging_direction: Direction, emerging_direction: Direction) -> float:
        """Query the target's radar cross section.

        Args:

            impinging_direction (Direction):
                Direction from which a far-field source impinges onto the target model.

            emerging_direction (Direction):
                Direction in which the scatter wave leaves the target model.

        Returns: The assumed radar cross section in :math:`m^2`.
        """
        ...  # pragma: no cover

    @abstractmethod
    def get_velocity(self) -> np.ndarray:
        """Query the target's velocity.

        Returns: A cartesian velocity vector in m/s.
        """
        ...  # pragma: no cover

    @abstractmethod
    def get_forwards_transformation(self) -> Transformation:
        """Query the target's global forwards transformation.

        Returns: The forwards transformation matrix.
        """
        ...  # pragma: no cover

    @abstractmethod
    def get_backwards_transformation(self) -> Transformation:
        """Query the target's global backwards transformation.

        Returns: The backwards transformation matrix.
        """
        ...  # pragma: no cover

    @property
    def static(self) -> bool:
        """Is the target visible in the null hypothesis?"""

        return self.__static


class RadarCrossSectionModel(ABC):
    """Base class for spatial radar cross section models."""

    @abstractmethod
    def get_cross_section(self, impinging_direction: Direction, emerging_direction: Direction) -> float:
        """Query the model's cross section.

        Args:

            impinging_direction (Direction):
                Direction from which a far-field source impinges onto the cross section model.

            emerging_direction (Direction):
                Direction in which the scatter wave leaves the cross section model.

        Returns: The assumed cross section in :math:`m^2`.
        """
        ...  # pragma: no cover


class FixedCrossSection(RadarCrossSectionModel):
    """Model of a fixed cross section.

    Can be interpreted as a spherical target floating in space.
    """

    __cross_section: float

    def __init__(self, cross_section: float) -> None:
        """
        Args:

            cross_section (float):
                The cross section in m^2.
        """

        self.cross_section = cross_section

    @property
    def cross_section(self) -> float:
        """The assumed cross section.

        Returns: The cross section in m^2.

        Raises:

            ValueError: For cross sections smaller than zero.
        """

        return self.__cross_section

    @cross_section.setter
    def cross_section(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Radar cross section must be greater or equal to zero")

        self.__cross_section = value

    def get_cross_section(self, _: Direction, __: Direction) -> float:
        return self.__cross_section


class VirtualRadarTarget(Transformable, RadarTarget, Serializable):
    """Model of a spatial radar target only existing within a channe link."""

    yaml_tag = "VirtualTarget"

    __velocity: np.ndarray
    __cross_section: RadarCrossSectionModel

    def __init__(self, cross_section: RadarCrossSectionModel, velocity: np.ndarray | None = None, pose: Transformation | None = None, static: bool = False) -> None:
        """
        Args:

            cross_section (RadarCrossSectionModel):
                The assumed cross section model.

            velocity (np.ndarray, optional):
                The targets velocity. See :meth:`VirtualRadarTarget.velocity`.
                By default, a resting target is assumed.

            pose (Transformation | None, optional):
                The target's global pose.
                By default, the coordinate system origin is assumed.

            static (bool, optional):
                See :meth:`RadarTarget.static`.
                Disabled by default.
        """

        # Initialize base classes
        Transformable.__init__(self, pose)
        RadarTarget.__init__(self, static=static)
        Serializable.__init__(self)

        # Initialize class attributes
        self.cross_section = cross_section
        self.velocity = np.zeros(3, dtype=float) if velocity is None else velocity

    @property
    def cross_section(self) -> RadarCrossSectionModel:
        """The represented radar cross section model."""

        return self.__cross_section

    @cross_section.setter
    def cross_section(self, value: RadarCrossSectionModel) -> None:
        self.__cross_section = value

    def get_cross_section(self, impinging_direction: Direction, emerging_direction: Direction) -> float:
        return self.cross_section.get_cross_section(impinging_direction, emerging_direction)

    @property
    def velocity(self) -> np.ndarray:
        """The assumed velocity vector.

        Returns: Cartesian numpy vector describing the target's velocity in m/s.
        """

        return self.__velocity

    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        self.__velocity = value

    def get_velocity(self) -> np.ndarray:
        return self.velocity

    def get_forwards_transformation(self) -> Transformation:
        return self.forwards_transformation

    def get_backwards_transformation(self) -> Transformation:
        return self.backwards_transformation


class PhysicalRadarTarget(RadarTarget, Serializable):
    """Model of a spatial radar target representing a moveable object.

    The radar target will always be modeled at its moveable global position.
    """

    yaml_tag = "PhysicalTarget"

    __cross_section: RadarCrossSectionModel
    __moveable: Moveable

    def __init__(self, cross_section: RadarCrossSectionModel, moveable: Moveable, static: bool = False) -> None:
        """
        Args:

            cross_section (RadarCrossSectionModel):
                The assumed cross section model.

            moveable (Moveable):
                The moveable object this radar target represents.

            static (bool, optional):
                See :meth:`RadarTarget.static`.
                Disabled by default.
        """

        # Initialize base classes
        RadarTarget.__init__(self, static=static)
        Serializable.__init__(self)

        # Initialize properties
        self.cross_section = cross_section
        self.__moveable = moveable

    @property
    def cross_section(self) -> RadarCrossSectionModel:
        """The represented radar cross section model."""

        return self.__cross_section

    @cross_section.setter
    def cross_section(self, value: RadarCrossSectionModel) -> None:
        self.__cross_section = value

    @property
    def moveable(self) -> Moveable:
        """Moveble this radar model is attached to.

        Returns: Handle to the moveable object.
        """

        return self.__moveable

    def get_cross_section(self, impinging_direction: Direction, emerging_direction: Direction) -> float:
        return self.cross_section.get_cross_section(impinging_direction, emerging_direction)

    def get_velocity(self) -> np.ndarray:
        return self.__moveable.velocity

    def get_forwards_transformation(self) -> Transformation:
        return self.__moveable.forwards_transformation

    def get_backwards_transformation(self) -> Transformation:
        return self.__moveable.backwards_transformation


RCRT = TypeVar("RCRT", bound="RadarChannelRealization")
"""Type of radar channel realization."""


class RadarChannelRealization(ChannelRealization):
    """Realization of a radar channel."""

    @staticmethod
    def ImpulseResponse(path_realizations: Sequence[RadarPathRealization], gain: float, num_samples: int, sampling_rate: float, transmitter: Device, receiver: Device) -> np.ndarray:
        """Generate the sampled impulse response of a set of radar path realizations.

        Args:

            path_realizations (Sequence[RadarPathRealization]):
                Sequence of realized radar propagation paths to be sampled into the impulse response.

            gain (float):
                The linear applied channel gain factor.

            num_samples (int):
                Number of generated time-domain impulse response samples.

            sampling_rate (float):
                Sampling rate of the impulse response in Hz.

            transmitter (Device):
                The transmitting device feeding into the represented channel.

            receiver (Device):
                The device reiving from the represented channel.
        """

        # Impulse response sample timestamps
        timestamps = np.arange(num_samples) / sampling_rate

        # Infer parameter limits
        max_time = float(timestamps[-1]) if num_samples > 0 else 0.0
        max_propagation_delay = max(path_realizations, key=lambda t: t.delay).delay if len(path_realizations) > 0 else 0.0
        max_velocity = 0.0  # ToDo
        # This equation can be improved to compute the max delay for each target realization
        max_delay = max_propagation_delay + 2 * max_velocity * max_time / speed_of_light
        max_delay_in_samples = int(np.ceil(max_delay * sampling_rate))

        # Generate impulse response
        impulse_response = np.zeros((receiver.antennas.num_receive_antennas, transmitter.antennas.num_transmit_antennas, num_samples, 1 + max_delay_in_samples), dtype=complex)

        # Impulse response delay timestamps
        delay_taps = np.arange(1 + max_delay_in_samples) / sampling_rate

        for (tidx, timestamp), target in product(enumerate(timestamps), path_realizations):  # type: ignore
            target_velocity = 0.0
            echo_delay = target.delay + 2 * target_velocity * timestamp / speed_of_light
            time = timestamp + delay_taps
            echo_weights = target.power_factor * np.exp(2j * pi * (target.doppler_shift * time + target.phase_shift))

            interpolated_impulse_tap = np.sinc(sampling_rate * (delay_taps - echo_delay)) * echo_weights

            # Note that this impulse response selection is technically incorrect,
            # since it is only feasible for planar arrays
            impulse_response[:, :, tidx, :] += np.tensordot(target.mimo_response, interpolated_impulse_tap, axes=0)

        # Apply the channel gain
        impulse_response *= gain**0.5
        return impulse_response

    def __init__(self, channel: Channel, gain: float, impulse_response: np.ndarray) -> None:
        """
        Args:

            channel (Channel):
                Reference to the realized channel.

            gain (float):
                Linear propagation power gain factor.

            impulse_response (np.ndarray):
                Channel impulse response tensor.

        Raises:

            ValueError: If `gain` is negative.
        """

        if gain < 0.0:
            raise ValueError(f"Channel power gain must be greater or equal to zero (not {gain})")

        # Initialize the base class
        ChannelRealization.__init__(self, channel, impulse_response)

        # Initialize class attributes
        self.__gain = gain

    @property
    def gain(self) -> float:
        """Applied linear channel power gain factor.

        Returns: Channel power gain.
        """

        return self.__gain

    @abstractmethod
    def null_hypothesis(self: RCRT, num_samples: int, sampling_rate: float) -> RCRT:
        """Generate a null hypothesis realization. from a given channel realization.

        Null hypothesis realizations will remove non-static propagation components from the channel model.
        This function is, for example, accessed to evaluate a radar link's receiver operating characteristics.

        Args:

            num_samples (int):
                Number of generated time-domain impulse response samples.

            sampling_rate (float):
                Sampling rate of the impulse response in Hz.

        Returns: The null hypothesis radar channel realization.
        """
        ...  # pragma: no cover


class RadarChannelBase(Generic[RCRT], Channel[RCRT]):
    """Base class of all radar channel implementations."""

    __attenuate: bool  # Should signals be attenuated during propagation modeling?

    def __init__(self, attenuate: bool = True, *args, **kwargs) -> None:
        """
        Args:

            attenuate (bool, optional):
                Radar channel attenuation flag, see also :meth:`RadarChannelBase.attenuate`.
                Enabled by default.
        """

        # Initialize base class
        Channel.__init__(self, *args, **kwargs)

        # Initialize class attributes
        self.__attenuate = attenuate

    @property
    def attenuate(self) -> bool:
        """Radar channel attenuation flag.

        If enabled, losses such as free-space propagation and radar cross sections will be considered.
        """

        return self.__attenuate

    @attenuate.setter
    def attenuate(self, value: bool) -> None:
        self.__attenuate = value

    def _realize_target(self, carrier_frequency: float, target: RadarTarget) -> RadarTargetRealization:
        """Realize a single radar target's channel propagation path.

        Args:

            carrier_frequency (float):
                The assumed signal carrier frequency in Hz.

            target (RadarTarget):
                The radar target to be realized.

        Returns: The realized propagation path.

        Raises:

            ValueError: If `carrier_frequency` is smaller or equal to zero.
            FloatingError: If transmitter or receiver are not specified.
            RuntimeError: If `target` and the channel's linked devices are located at identical global positions
        """

        if carrier_frequency <= 0.0:
            raise ValueError("Radar channel linked device carrier frequencies must be greater than zero")

        if self.transmitter is None or self.receiver is None:
            raise FloatingError("Radar channel's linked devices not specified")

        # Query target global coordiante system transformations
        target_backwards_transform = target.get_backwards_transformation()
        target_forwards_transform = target.get_forwards_transformation()

        # Make sure the transmitter / receiver positions don't collide with target locations
        # This implicitly violates the far-field assumption and leads to numeric instabilities
        if np.array_equal(target_backwards_transform.translation, self.transmitter.global_position):
            raise RuntimeError("Radar channel transmitter position colliding with an assumed target location")

        if np.array_equal(target_backwards_transform.translation, self.receiver.global_position):
            raise RuntimeError("Radar channel receiver position colliding with an assumed target location")

        # Compute the impinging and emerging far-field wave direction from the target
        impinging_direction = target_backwards_transform.transform_direction(self.transmitter.global_position, normalize=True)
        emerging_direction = target_forwards_transform.transform_direction(self.receiver.global_position, normalize=True)

        # Query the radar cross section from the target's model given impinging and emerging directions
        cross_section = target.get_cross_section(impinging_direction, emerging_direction)

        # Query reflection phase shift
        reflection_phase = self._rng.uniform(0, 1)

        # Compute the wave's propagated distance and propagation delay
        distance = np.linalg.norm(self.transmitter.global_position - target_forwards_transform.translation) + np.linalg.norm(self.receiver.global_position - target_forwards_transform.translation)
        delay = distance / speed_of_light

        # Model the doppler-shift from transmitter to receiver
        # ToDo: Non-reciprocity in this case
        target_velocity = target.get_velocity()
        doppler_velocity = np.dot(self.receiver.velocity - target_velocity, impinging_direction) - np.dot(self.transmitter.velocity - target_velocity, emerging_direction)
        doppler_shift = doppler_velocity * carrier_frequency / speed_of_light

        # Compute the power factor given the radar range equation
        power_factor = 1.0
        if self.attenuate:
            wavelength = speed_of_light / carrier_frequency
            power_factor = wavelength * np.sqrt(cross_section) / (4 * pi) ** 1.5 / distance**2  # * db2lin(self.__losses_db)

        # Model the sensor arrays' spatial responses
        rx_response = self.receiver.antennas.cartesian_array_response(carrier_frequency, target_forwards_transform.translation, "global").conj()  # Is the conjugate here correct?
        tx_response = self.transmitter.antennas.cartesian_array_response(carrier_frequency, target_forwards_transform.translation, "global")
        mimo_response = np.inner(rx_response, tx_response)

        # Return realized information wrapped in a target realization dataclass
        return RadarTargetRealization(reflection_phase, delay, doppler_shift, power_factor, mimo_response, target.static)

    def null_hypothesis(self, num_samples: int, sampling_rate: float, realization: RCRT | None = None) -> RCRT:
        """Generate a channel realization missing the target to be estimated.

        Args:

            realization (RCRT, optional):
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

        return realization.null_hypothesis(num_samples, sampling_rate)


class RadarPathRealization(object):
    """Realization of a radar propagation path between transmitter and receiver"""

    __phase_shift: float
    __delay: float
    __doppler_shift: float
    __power_factor: float
    __mimo_response: np.ndarray
    __static: bool

    def __init__(self, phase_shift: float, delay: float, doppler_shift: float, power_factor: float, mimo_response: np.ndarray, static: bool = False) -> None:
        """
        Args:

            phase_shift (float):
                Phase shift of the propagation path in radians.

            delay (float):
                Propagation delay in seconds.

            doppler_shift (float):
                Doppler shift in Hz.

            power_factor (float):
                Linear factor a propagated signal gets scaled by in its amplitude.

            mimo_response (np.ndarray):
                Spatial sensor array response of the propagation.
                Must be numpy matrix.

            static (bool, optional):
                Is the path considered static?
                Static paths will remain during null hypothesis testing.
                Disabled by default.
        """

        self.__phase_shift = phase_shift
        self.__delay = delay
        self.__doppler_shift = doppler_shift
        self.__power_factor = power_factor
        self.__mimo_response = mimo_response
        self.__static = static

    @property
    def phase_shift(self) -> float:
        """Phase shift of the wave during reflection at the target.

        Returns: Phase shift in the interval :math:`[0, 1]`.
        """

        return self.__phase_shift

    @property
    def delay(self) -> float:
        """Propagation delay of the wave from transmitter over target to receiver.

        Returns: Propagation delay in seconds.
        """

        return self.__delay

    @property
    def doppler_shift(self) -> float:
        """Frequency shift perceived by the receiver with respected to the transmitted center frequency.

        Returns: Shift in Hz.
        """

        return self.__doppler_shift

    @property
    def power_factor(self) -> float:
        """Power loss factor the wave during free space propagation and reflection

        Returns: Linear power factor.
        """

        return self.__power_factor

    @property
    def mimo_response(self) -> np.ndarray:
        """Multipath sensor array response matrix from transmitter to receiver.

        Includes polarization losses.

        Returns: Numpy matrix of antenna phase shift factors.
        """

        return self.__mimo_response

    @property
    def static(self) -> bool:
        return self.__static


class RadarTargetRealization(RadarPathRealization):
    """Realization of a radar propagation path resulting from a target scattering"""

    ...  # pragma: no cover


class RadarInterferenceRealization(RadarPathRealization):
    """Realization of a line of sight interference propgation path between a radar transmitter and receiver"""

    ...  # pragma: no cover


class SingleTargetRadarChannelRealization(RadarChannelRealization):
    """Realization of a single target radar channel."""

    __target_realization: RadarTargetRealization | None

    def __init__(self, channel: Channel, gain: float, target_realization: RadarTargetRealization | None, num_samples: int, sampling_rate: float) -> None:
        """
        Args:
            channel (Channel):
                The represented channel instance.

            gain (float):
                Linear propagation power gain factor.

            target_realization (RadarTargetRealization | None):
                Single target realization.
                `None` if no target should be present.


            num_samples (int):
                Number of generated time-domain impulse response samples.

            sampling_rate (float):
                Sampling rate of the impulse response in Hz.
        """

        # Generate the realization's sampled impulse response
        impulse_response = self.ImpulseResponse([target_realization] if target_realization is not None else [], channel.gain, num_samples, sampling_rate, channel.transmitter, channel.receiver)

        # Initialize the base class
        RadarChannelRealization.__init__(self, channel, gain, impulse_response)

        # Initialize class attributes
        self.__target_realization = target_realization

    @property
    def target_realization(self) -> RadarTargetRealization | None:
        """Realized radar target.

        Returns:
            Handle to the realized target.
            `None` if no target was considered.
        """

        return self.__target_realization

    def null_hypothesis(self, num_samples: int, sampling_rate: float) -> SingleTargetRadarChannelRealization:
        return SingleTargetRadarChannelRealization(self.channel, self.gain, None, num_samples, sampling_rate)


class MultiTargetRadarChannelRealization(RadarChannelRealization):
    """Realization of a spatial multi target radar channel."""

    __interference_realization: RadarInterferenceRealization | None
    __target_realizations: Sequence[RadarTargetRealization]

    def __init__(self, channel: Channel, gain: float, interference_realization: RadarInterferenceRealization | None, target_realizations: Sequence[RadarTargetRealization], num_samples: int, sampling_rate: float) -> None:
        """
        Args:
            channel (Channel):
                The represented channel instance.

            gain (float):
                Linear propagation power gain factor.

            interference_realizations (RadarInterferenceRealization | None):
                Realization of the line of sight interference.
                `None` if no interference should be considered.

            target_realizations (Sequence[RadarTargetRealization]):
                Sequence of radar target realizations considered within the radar channel.

            num_samples (int):
                Number of generated time-domain impulse response samples.

            sampling_rate (float):
                Sampling rate of the impulse response in Hz.
        """

        # Generate the realization's sampled impulse response
        impulse_response_paths: List[RadarPathRealization] = list(target_realizations)

        if interference_realization is not None:
            impulse_response_paths.append(interference_realization)

        impulse_response = self.ImpulseResponse(target_realizations, gain, num_samples, sampling_rate, channel.transmitter, channel.receiver)

        # Initialize the base class
        RadarChannelRealization.__init__(self, channel, gain, impulse_response)

        # Initialize class attributes
        self.__interference_realization = interference_realization
        self.__target_realizations = target_realizations

    @property
    def interference_realization(self) -> RadarInterferenceRealization | None:
        """Realization of the line of sight interference between both linked radars.

        Returns:
            The interference realization.
            `None` if no interference is considered.
        """

        return self.__interference_realization

    @property
    def target_realizations(self) -> Sequence[RadarTargetRealization]:
        """Realized radar targets.

        Returns: Sequence of radar target realizations.
        """

        return self.__target_realizations

    def null_hypothesis(self, num_samples: int, sampling_rate: float) -> MultiTargetRadarChannelRealization:
        null_hypothesis_target_realizations = []
        for target_realization in self.target_realizations:
            if target_realization.static:
                null_hypothesis_target_realizations.append(target_realization)

        null_hypothesis_interference = self.interference_realization
        if null_hypothesis_interference is not None and not null_hypothesis_interference.static:
            null_hypothesis_interference = None

        return MultiTargetRadarChannelRealization(self.channel, self.gain, null_hypothesis_interference, null_hypothesis_target_realizations, num_samples, sampling_rate)


class MultiTargetRadarChannel(RadarChannelBase[MultiTargetRadarChannelRealization], Serializable):
    yaml_tag = "SpatialRadarChannel"

    interfernce: bool
    """Consider interference between linked devices.

    Only applies in the bistatic case, where transmitter and receiver are two dedicated device instances.
    """

    __targets: Set[RadarTarget]

    def __init__(self, attenuate: bool = True, interference: bool = True, *args, **kwargs) -> None:
        # Initialize base classes
        RadarChannelBase.__init__(self, attenuate, *args, **kwargs)
        Serializable.__init__(self)

        # Initialize attributes
        self.interference = interference
        self.__targets = set()

    @property
    def targets(self) -> Set[RadarTarget]:
        return self.__targets

    def add_target(self, target: RadarTarget):
        if target not in self.targets:
            self.__targets.add(target)

    def make_target(self, moveable: Moveable, cross_section: RadarCrossSectionModel, *args, **kwargs) -> PhysicalRadarTarget:
        target = PhysicalRadarTarget(cross_section, moveable, *args, **kwargs)
        self.add_target(target)

        return target

    def _realize_interference(self, carrier_frequency: float) -> RadarInterferenceRealization | None:
        """Realize the channel model's line of sight interference.

        Args:

            carrier_frequency (float):
                The assumed signal carrier frequency in Hz.

        Returns:
            The realized propagation path.
            `None` if :attr:`.interference` is disabled or the channel models a monostatic radar

        Raises:

            ValueError: If `carrier_frequency` is smaller or equal to zero.
            FloatingError: If transmitter or receiver are not specified.
            RuntimeError: If transmitter and receiver are at the same global location.
        """

        if carrier_frequency <= 0.0:
            raise ValueError("Radar channel linked device carrier frequencies must be greater than zero")

        if self.transmitter is None or self.receiver is None:
            raise FloatingError("Radar channel's linked devices not specified")

        # Return None if interference modeling is disabled
        if not self.interference or self.transmitter is self.receiver:
            return None

        if np.array_equal(self.transmitter.global_position, self.receiver.global_position):
            raise RuntimeError("Linked devices may not be located at identical global positions")

        # Generate a random phase shift between transmitter and receiver
        phase = self._rng.uniform(0, 1)

        # Compute a vector pointing from transmitter to receiver
        connection = self.receiver.global_position - self.transmitter.global_position

        # Compute the wave's propagated distance and propagation delay
        distance = float(np.linalg.norm(connection))
        delay = distance / speed_of_light

        # Model the doppler-shift from transmitter to receiver
        # ToDo: Non-reciprocity in this case
        relative_velocity = np.dot(self.transmitter.velocity - self.receiver.velocity, connection / distance)
        doppler_shift = relative_velocity * carrier_frequency / speed_of_light

        # Compute the power factor given the radar range equation
        power_factor = 1.0
        if self.attenuate:
            power_factor = amplitude_path_loss(carrier_frequency, distance)

        # Model the sensor arrays' spatial responses
        rx_response = self.receiver.antennas.cartesian_array_response(carrier_frequency, self.transmitter.global_position, "global").conj()  # Is the conjugate here correct?
        tx_response = self.transmitter.antennas.cartesian_array_response(carrier_frequency, self.receiver.global_position, "global")
        mimo_response = np.inner(rx_response, tx_response)

        # Return realized information wrapped in a target realization dataclass
        return RadarInterferenceRealization(phase, delay, doppler_shift, power_factor, mimo_response, True)

    def realize(self, num_samples: int, sampling_rate: float) -> MultiTargetRadarChannelRealization:
        if self.transmitter is None or self.receiver is None:
            raise FloatingError("Radar channel's linked devices not specified")

        # Realize radar channel parameters
        carrier_frequency = 0.5 * (self.transmitter.carrier_frequency + self.receiver.carrier_frequency)
        interference_realization = self._realize_interference(carrier_frequency)
        target_realizations = [self._realize_target(carrier_frequency, target) for target in self.targets]

        # Realize channel
        channel_realization = MultiTargetRadarChannelRealization(self, self.gain, interference_realization, target_realizations, num_samples, sampling_rate)
        return channel_realization


class SingleTargetRadarChannel(RadarChannelBase[SingleTargetRadarChannelRealization], Serializable):
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

    __target: VirtualRadarTarget
    __target_range: float | Tuple[float, float]
    __radar_cross_section: float
    __target_azimuth: float
    __target_zenith: float
    target_exists: bool
    __target_velocity: float | np.ndarray

    def __init__(self, target_range: float | Tuple[float, float], radar_cross_section: float, target_azimuth: float = 0.0, target_zenith: float = 0.0, target_exists: bool = True, velocity: float | np.ndarray = 0, attenuate: bool = True, **kwargs) -> None:
        """
        Args:

            target_range (float | Tuple[float, float]):
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

            velocity (float | np.ndarray , optional):
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

        # Initialize base class
        RadarChannelBase.__init__(self, attenuate=attenuate, **kwargs)

        # Initialize class properties
        self.__cross_section = FixedCrossSection(radar_cross_section)
        self.__target = VirtualRadarTarget(self.__cross_section)

        self.target_range = target_range
        self.radar_cross_section = radar_cross_section
        self.target_azimuth = target_azimuth
        self.target_zenith = target_zenith
        self.target_exists = target_exists
        self.target_velocity = velocity

    @property
    def target_range(self) -> float | Tuple[float, float]:
        """Absolute distance of target and radar sensor.

        Returns: Target range in meters.

        Raises:

            ValueError: If the range is smaller than zero.
        """

        return self.__target_range

    @target_range.setter
    def target_range(self, value: float | Tuple[float, float]) -> None:
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
    def target_velocity(self) -> float | np.ndarray:
        """Perceived target velocity.

        Returns: Velocity in m/s.
        """

        return self.__target_velocity

    @target_velocity.setter
    def target_velocity(self, value: float | np.ndarray) -> None:
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

    def realize(self, num_samples: int, sampling_rate: float) -> SingleTargetRadarChannelRealization:
        if self.transmitter is None or self.receiver is None:
            raise FloatingError("Radar channel's linked devices not specified")

        # Realize targets
        carrier_frequency = 0.5 * (self.transmitter.carrier_frequency + self.receiver.carrier_frequency)
        target_range = self.target_range if isinstance(self.target_range, (np.int_, np.float_, int, float)) else self._rng.uniform(*self.target_range)  # type: ignore

        # Update the internal target model, kinda hacky
        unit_direction = Direction.From_Spherical(self.target_azimuth, self.target_zenith)
        self.__target.position = unit_direction * target_range
        self.__target.velocity = unit_direction * self.target_velocity
        self.__cross_section.cross_section = self.radar_cross_section

        target_realization = self._realize_target(carrier_frequency, self.__target) if self.target_exists else None

        # Realize channel
        channel_realization = SingleTargetRadarChannelRealization(self, self.gain, target_realization, num_samples, sampling_rate)
        return channel_realization
