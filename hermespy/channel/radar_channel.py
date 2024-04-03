# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from math import ceil
from typing import Any, Generic, List, Mapping, Sequence, Set, Tuple, Type, TypeVar

import numpy as np
from h5py import Group
from scipy.constants import pi, speed_of_light
from sparse import GCXS  # type: ignore

from hermespy.channel.channel import InterpolationMode

from .channel import Channel, ChannelRealization
from hermespy.core import (
    ChannelStateInformation,
    ChannelStateFormat,
    Device,
    Direction,
    FloatingError,
    HDFSerializable,
    Moveable,
    Serializable,
    Signal,
    Transformable,
    Transformation,
)

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarTarget(ABC):
    """Abstract base class of radar targets.

    Radar targets represent reflectors of electromagnetic waves within :class:`RadarChannel` instances.
    """

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
    def get_cross_section(
        self, impinging_direction: Direction, emerging_direction: Direction
    ) -> float:
        """Query the target's radar cross section.


        The target radr cross section is denoted by the vector :math:`\\sigma_{\\ell}`
        within the respective equations.

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

        The target velocity is denoted by the vector :math:`\\mathbf{v}^{(\\ell)}`
        within the respective equations.

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
    def get_cross_section(
        self, impinging_direction: Direction, emerging_direction: Direction
    ) -> float:
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
                The cross section in :math:`\\mathrm{m}^2`.
        """

        self.cross_section = cross_section

    @property
    def cross_section(self) -> float:
        """The assumed cross section.

        Returns: The cross section in :math:`\\mathrm{m}^2`.

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

    def __init__(
        self,
        cross_section: RadarCrossSectionModel,
        velocity: np.ndarray | None = None,
        pose: Transformation | None = None,
        static: bool = False,
    ) -> None:
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

    def get_cross_section(
        self, impinging_direction: Direction, emerging_direction: Direction
    ) -> float:
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

    def __init__(
        self, cross_section: RadarCrossSectionModel, moveable: Moveable, static: bool = False
    ) -> None:
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

    def get_cross_section(
        self, impinging_direction: Direction, emerging_direction: Direction
    ) -> float:
        return self.cross_section.get_cross_section(impinging_direction, emerging_direction)

    def get_velocity(self) -> np.ndarray:
        return self.moveable.velocity

    def get_forwards_transformation(self) -> Transformation:
        return self.moveable.forwards_transformation

    def get_backwards_transformation(self) -> Transformation:
        return self.moveable.backwards_transformation


RCRT = TypeVar("RCRT", bound="RadarChannelRealization")
"""Type of radar channel realization."""


class RadarChannelRealization(ChannelRealization):
    """Realization of a radar channel.

    Generated by :meth:`RadarChannelBase.realize` and :meth:`RadarChannelBase.realize_interference`.
    """

    @property
    @abstractmethod
    def _path_realizations(self) -> Sequence[RadarPathRealization]:
        """Sequence of realized radar propagation paths."""
        ...  # pragma: no cover

    def _propagate(
        self,
        signal: Signal,
        transmitter: Device,
        receiver: Device,
        interpolation: InterpolationMode,
    ) -> Signal:
        delays = np.array(
            [
                path_realization.propagation_delay(transmitter, receiver)
                for path_realization in self._path_realizations
            ]
        )
        velocities = np.array(
            [
                path_realization.relative_velocity(transmitter, receiver)
                for path_realization in self._path_realizations
            ]
        )

        # Compute the expected sample overhead of the propagated sample resultin from propagtion delays
        if delays.size > 0:
            max_delay_in_samples = ceil(
                delays.max() * signal.sampling_rate
                + 2
                * velocities.max()
                * signal.num_samples
                / (signal.sampling_rate * speed_of_light)
            )
        else:
            max_delay_in_samples = 0

        propagated_samples = np.zeros(
            (receiver.antennas.num_receive_antennas, signal.num_samples + max_delay_in_samples),
            dtype=np.complex_,
        )

        # Compute the signal propagated along each respective path realization
        for path_realization, delay, velocity in zip(self._path_realizations, delays, velocities):
            path_realization.add_propagation(
                transmitter, receiver, signal, propagated_samples, delay, velocity
            )

        # Apply the channel gain
        propagated_samples *= self.gain**0.5

        return Signal(propagated_samples, signal.sampling_rate, signal.carrier_frequency)

    def state(
        self,
        transmitter: Device,
        receiver: Device,
        delay: float,
        sampling_rate: float,
        num_samples: int,
        max_num_taps: int,
    ) -> ChannelStateInformation:
        raw_state = np.zeros(
            (
                receiver.antennas.num_receive_antennas,
                transmitter.antennas.num_transmit_antennas,
                num_samples,
                max_num_taps,
            ),
            dtype=np.complex_,
        )
        for path_realization in self._path_realizations:
            path_realization.add_state(transmitter, receiver, delay, sampling_rate, raw_state)

        # Apply the channel gain
        raw_state *= self.gain**0.5

        return ChannelStateInformation(
            ChannelStateFormat.IMPULSE_RESPONSE, GCXS.from_numpy(raw_state)
        )

    @abstractmethod
    def null_hypothesis(self: RCRT) -> RCRT:
        """Generate a null hypothesis realization. from a given channel realization.

        Null hypothesis realizations will remove non-static propagation components from the channel model.
        This function is, for example, accessed to evaluate a radar link's receiver operating characteristics.

        Returns: The null hypothesis radar channel realization.
        """
        ...  # pragma: no cover

    @abstractmethod
    def ground_truth(self) -> np.ndarray:
        """Generate a ground truth realization from a given channel realization.

        Returns:
            The ground truth radar channel realization.
            A :math:`P\\times 3` matrix, where :math:`P` is the number of targets
            and each row contains the target's position in global coordinates.
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

    def _realize_target(self, target: RadarTarget) -> RadarTargetRealization:
        """Realize a single radar target's channel propagation path.

        Args:

            target (RadarTarget):
                The radar target to be realized.

        Returns: The realized propagation path.

        Raises:

            ValueError: If `carrier_frequency` is smaller or equal to zero.
            FloatingError: If transmitter or receiver are not specified.
            RuntimeError: If `target` and the channel's linked devices are located at identical global positions
        """

        # Query target global coordiante system transformations
        target_backwards_transform = target.get_backwards_transformation()
        target_forwards_transform = target.get_forwards_transformation()

        # Make sure the transmitter / receiver positions don't collide with target locations
        # This implicitly violates the far-field assumption and leads to numeric instabilities
        if np.array_equal(target_forwards_transform.translation, self.alpha_device.global_position):
            raise RuntimeError(
                "Radar channel transmitter position colliding with an assumed target location"
            )

        if np.array_equal(target_forwards_transform.translation, self.beta_device.global_position):
            raise RuntimeError(
                "Radar channel receiver position colliding with an assumed target location"
            )

        # Compute the impinging and emerging far-field wave direction from the target in local target coordinates
        target_impinging_direction = target_backwards_transform.transform_direction(
            self.alpha_device.global_position, normalize=True
        )
        target_emerging_direction = target_backwards_transform.transform_direction(
            self.beta_device.global_position, normalize=True
        )

        # Query the radar cross section from the target's model given impinging and emerging directions
        cross_section = target.get_cross_section(
            target_impinging_direction, target_emerging_direction
        )

        # Query reflection phase shift
        reflection_phase = self._rng.uniform(0, 2 * pi)

        # Return realized information wrapped in a target realization dataclass
        return RadarTargetRealization(
            target_forwards_transform.translation,
            target.get_velocity(),
            cross_section,
            reflection_phase,
            self.attenuate,
            target.static,
        )

    def null_hypothesis(self, realization: RCRT | None = None) -> RCRT:
        """Generate a channel realization missing the target to be estimated.

        Returns: Null hypothesis channel realization.

        Raises:

            RuntimeError: If no `realization` was provided and the channel hasn't been propagated over yet.
        """

        # Assume the last channel propagation realization if the realization has not been specified
        if realization is None:
            realization = self.realization

            if realization is None:
                raise RuntimeError("Channel has not been propagated over yet")

        return realization.null_hypothesis()


class RadarPathRealization(HDFSerializable):
    """Realization of a radar propagation path between transmitter and receiver"""

    __attenuate: bool
    __static: bool

    def __init__(self, attenuate: bool = True, static: bool = False) -> None:
        """
        Args:

            attenuate (bool, optional):
                Should the propagated signal be attenuated during propagation modeling?
                Enabled by default.

            static (bool, optional):
                Is the path considered static?
                Static paths will remain during null hypothesis testing.
                Disabled by default.
        """

        # Initialize class attributes
        self.__attenuate = attenuate
        self.__static = static

    @property
    def attenuate(self) -> bool:
        """Should a propagated signal be attenuated during propagation modeling?"""

        return self.__attenuate

    @attenuate.setter
    def attenuate(self, value: bool) -> None:
        self.__attenuate = value

    @property
    def static(self) -> bool:
        """Is the path considered static?"""

        return self.__static

    @static.setter
    def static(self, value: bool) -> None:
        self.__static = value

    @abstractmethod
    def propagation_delay(self, transmitter: Device, receiver: Device) -> float:
        """Propagation delay of the wave from transmitter over target to receiver.

        Denoted by :math:`\\tau_{\\ast}` within the respective equations.

        Args:

            transmitter (Device):
                Transmitting device.

            receiver (Device):
                Receiving device.

        Returns: Propagation delay in seconds.
        """
        ...  # pragma: no cover

    @abstractmethod
    def relative_velocity(self, transmitter: Device, receiver: Device) -> float:
        """Relative velocity between transmitter and receiver.

        Denoted by :math:`v_{\\ast}` within the respective equations.

        Args:

            transmitter (Device):
                Transmitting device.

            receiver (Device):
                Receiving device.

        Returns: Relative velocity in m/s.
        """
        ...  # pragma: no cover

    @abstractmethod
    def propagation_response(
        self, transmitter: Device, receiver: Device, carrier_frequency: float
    ) -> np.ndarray:
        """Multipath sensor array response matrix from transmitter to receiver.

        Includes polarization losses.

        Args:

            transmitter (Device):
                Transmitting device.

            receiver (Device):
                Receiving device.

            carrier_frequency (float):
                Carrier frequency of the propagated signal in Hz.
                Denoted by :math:`f_{\\mathrm{c}}^{(\\alpha)}` within the respective equations.

        Returns: Numpy matrix of antenna response weights.
        """
        ...  # pragma: no cover

    def add_propagation(
        self,
        transmitter: Device,
        receiver: Device,
        signal: Signal,
        propagated_samples: np.ndarray,
        propagation_delay: float | None = None,
        relative_velocity: float | None = None,
    ) -> None:
        """Add propagation of a signal over this path realization to a given sample buffer.

        Args:

            transmitter (Device):
                Transmitting device.

            receiver (Device):
                Receiving device.

            signal (Signal):
                Signal to be propagated.

            propagated_samples (np.ndarray):
                Sample buffer to be written to.

            propagation_delay (float, optional):
                Propagation delay of the wave from transmitter over target to receiver.
                If not specified, the delay will be queried from :meth:`propagation_delay`.

            relative_velocity (float, optional):
                Relative velocity between transmitter and receiver.
                If not specified, the velocity will be queried from :meth:`relative_velocity`.
        """

        # Query the required parameters
        propagation_delay = (
            self.propagation_delay(transmitter, receiver)
            if propagation_delay is None
            else propagation_delay
        )
        relative_velocity = (
            self.relative_velocity(transmitter, receiver)
            if relative_velocity is None
            else relative_velocity
        )
        propagation_response = self.propagation_response(
            transmitter, receiver, signal.carrier_frequency
        )

        delay_sample_offset = int(propagation_delay * signal.sampling_rate)
        doppler_shift = relative_velocity * signal.carrier_frequency / speed_of_light

        # ToDo: Exact time of flight resampling
        # echo_timestamps = propagation_delay + 2 * relative_velocity * signal.timestamps / speed_of_light
        # echo_weights = np.exp(2j * pi * (doppler_shift * echo_timestamps))
        echo_weights = np.exp(2j * pi * (doppler_shift * signal.timestamps))

        propagated_samples[
            :, delay_sample_offset : delay_sample_offset + signal.num_samples
        ] += np.einsum("ij,jk,k->ik", propagation_response, signal.samples, echo_weights)

    def add_state(
        self,
        transmitter: Device,
        receiver: Device,
        delay: float,
        sampling_rate: float,
        state: np.ndarray,
    ) -> None:
        """Add propagation of a signal over this path realization to a given channel state information sample buffer.

        Args:

            transmitter (Device):
                Transmitting device.

            receiver (Device):
                Receiving device.

            delay (float):
                Delay of the channel state information in seconds.

            sampling_rate (float):
                Sampling rate of the channel state information in Hz.

            state (np.ndarray):
                Sample buffer to be written to.
        """

        # Query the required parameters
        propagation_delay = self.propagation_delay(transmitter, receiver)
        relative_velocity = self.relative_velocity(transmitter, receiver)
        propagation_response = self.propagation_response(
            transmitter, receiver, transmitter.carrier_frequency
        )

        delay_sample_offset = int(propagation_delay * sampling_rate - delay)
        if delay_sample_offset < 0 or delay_sample_offset >= state.shape[3]:
            return

        doppler_shift = relative_velocity * transmitter.carrier_frequency / speed_of_light

        # echo_timestamps = delay + 2 * relative_velocity * np.arange(state.shape[2]) / speed_of_light
        # echo_weights = np.exp(2j * pi * (doppler_shift * echo_timestamps))
        echo_weights = np.exp(2j * pi * (doppler_shift / sampling_rate * np.arange(state.shape[2])))

        state[:, :, :, delay_sample_offset] += np.einsum(
            "ij,k->ijk", propagation_response, echo_weights
        )

    def to_HDF(self, group: Group) -> None:
        # Serialize the class attributes
        group.attrs["attenuate"] = self.attenuate
        group.attrs["static"] = self.static

    @classmethod
    def _parameters_from_HDF(cls: Type[RadarPathRealization], group: Group) -> Mapping[str, Any]:
        """Deserialize the object's parameters from HDF5.

        Intended to be used as a subroutine of :meth:`From_HDF`.

        Returns: The object's parmeters as a keyword argument dictionary.
        """

        return {"attenuate": group.attrs["attenuate"], "static": group.attrs["static"]}


class RadarTargetRealization(RadarPathRealization):
    """Realization of a radar propagation path resulting from a target scattering"""

    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        cross_section: float,
        reflection_phase: float,
        attenuate: bool = True,
        static: bool = False,
    ) -> None:
        """
        Args:

            position (np.ndarray):
                Global position of the path's target.

            velocity (np.ndarray):
                Global velocity of the path's target.

            cross_section (float):
                Radar cross section of the path's target in :math:`\\mathrm{m}^2`.

            reflection_phase (float):
                Reflection phase of the path's target in radians.

            attenuate (bool, optional):
                Should the propagated signal be attenuated during propagation modeling?
                Enabled by default.

            static (bool, optional):
                Is the path considered static?
                Static paths will remain during null hypothesis testing.
                Disabled by default.
        """

        # Initialize the base class
        RadarPathRealization.__init__(self, attenuate, static)

        # Initialize class attributes
        self.__global_position = position
        self.__global_velocity = velocity
        self.__cross_section = cross_section
        self.__reflection_phase = reflection_phase

    @property
    def position(self) -> np.ndarray:
        """Global position of the path's target.

        Denoted by :math:`\\mathbf{p}^{(\\ell)}` within the respective equations.
        """

        return self.__global_position

    @property
    def velocity(self) -> np.ndarray:
        """Global velocity of the path's target in m/s as a cartesian vector.

        Denoted by :math:`\\mathbf{v}^{(\\ell)}` within the respective equations.
        """

        return self.__global_velocity

    @property
    def cross_section(self) -> float:
        """Radar cross section of the path's target in :math:`\\mathrm{m}^2`.

        Denoted by :math:`\\sigma_{\\ell}` within the respective equations.
        """

        return self.__cross_section

    @property
    def reflection_phase(self) -> float:
        """Reflection phase of the path's target in radians.

        Represented by :math:`\\phi_{\\mathrm{Target}}^{(\\ell)}` within the respective equations.
        """

        return self.__reflection_phase

    def propagation_delay(self, transmitter: Device, receiver: Device) -> float:
        emerging_vector = self.position - transmitter.global_position
        impinging_vector = receiver.global_position - self.position

        delay = (
            np.linalg.norm(emerging_vector) + np.linalg.norm(impinging_vector)
        ) / speed_of_light
        return delay

    def relative_velocity(self, transmitter: Device, receiver: Device) -> float:
        target_position = self.position
        emerging_vector = target_position - transmitter.global_position
        impinging_vector = receiver.global_position - target_position

        # Model the doppler-shift from transmitter to receiver
        target_velocity = self.velocity
        relative_transmitter_velocity = np.dot(
            Direction.From_Cartesian(emerging_vector, normalize=True),
            target_velocity - transmitter.velocity,
        )
        relative_receiver_velocity = np.dot(
            Direction.From_Cartesian(impinging_vector, normalize=True),
            receiver.velocity - target_velocity,
        )

        return relative_transmitter_velocity + relative_receiver_velocity

    def propagation_response(
        self, transmitter: Device, receiver: Device, carrier_frequency: float
    ) -> np.ndarray:
        # Query the sensor array responses
        rx_response = receiver.antennas.cartesian_array_response(
            carrier_frequency, self.position, "global"
        ).conj()
        tx_response = transmitter.antennas.cartesian_array_response(
            carrier_frequency, self.position, "global"
        )

        if self.attenuate:
            # Compute propagation distances
            tx_distance = np.linalg.norm(self.position - transmitter.global_position)
            rx_distance = np.linalg.norm(receiver.global_position - self.position)

            wavelength = speed_of_light / carrier_frequency
            amplitude_factor = (
                wavelength * self.cross_section**0.5 / ((4 * pi) ** 1.5 * tx_distance * rx_distance)
            )

        else:
            amplitude_factor = 1.0

        # Compute the MIMO response
        return (
            amplitude_factor
            * np.exp(1j * self.reflection_phase)
            * np.inner(rx_response, tx_response)
        )

    def to_HDF(self, group: Group) -> None:
        # Serialize base class
        RadarPathRealization.to_HDF(self, group)

        # Serialize class attributes
        self._write_dataset(group, "position", self.position)
        self._write_dataset(group, "velocity", self.velocity)
        group.attrs["cross_section"] = self.cross_section
        group.attrs["reflection_phase"] = self.reflection_phase

    @classmethod
    def from_HDF(cls: Type[RadarTargetRealization], group: Group) -> RadarTargetRealization:
        # Deserialize base class
        parameters = RadarPathRealization._parameters_from_HDF(group)

        # Deserialize class attributes
        position = np.array(group["position"], dtype=np.float_)
        velocity = np.array(group["velocity"], dtype=np.float_)
        cross_section = group.attrs["cross_section"]
        reflection_phase = group.attrs["reflection_phase"]

        return RadarTargetRealization(
            position, velocity, cross_section, reflection_phase, **parameters
        )


class RadarInterferenceRealization(RadarPathRealization):
    """Realization of a line of sight interference propgation path between a radar transmitter and receiver"""

    def propagation_delay(self, transmitter: Device, receiver: Device) -> float:
        delay = (
            np.linalg.norm(receiver.global_position - transmitter.global_position) / speed_of_light
        )
        return delay

    def relative_velocity(self, transmitter: Device, receiver: Device) -> float:
        connection = Direction.From_Cartesian(
            receiver.global_position - transmitter.global_position, normalize=True
        ).view(np.ndarray)
        return np.dot(transmitter.velocity - receiver.velocity, connection)

    def propagation_response(
        self, transmitter: Device, receiver: Device, carrier_frequency: float
    ) -> np.ndarray:
        # Model the sensor arrays' spatial responses
        rx_response = receiver.antennas.cartesian_array_response(
            carrier_frequency, transmitter.global_position, "global"
        ).conj()
        tx_response = transmitter.antennas.cartesian_array_response(
            carrier_frequency, receiver.global_position, "global"
        )

        if self.attenuate:
            # Compute propagation distance
            distance = np.linalg.norm(receiver.global_position - transmitter.global_position)

            wavelength = speed_of_light / carrier_frequency
            amplitude_factor = wavelength / (4 * pi * distance)

        else:
            amplitude_factor = 1.0

        # Compute the MIMO response
        return amplitude_factor * np.inner(rx_response, tx_response)

    @classmethod
    def from_HDF(
        cls: Type[RadarInterferenceRealization], group: Group
    ) -> RadarInterferenceRealization:
        # Deserialize base class
        parameters = RadarPathRealization._parameters_from_HDF(group)
        return RadarInterferenceRealization(**parameters)


class SingleTargetRadarChannelRealization(RadarChannelRealization):
    """Realization of a single target radar channel.

    Generated by the :meth:`realize<SingleTargetRadarChannel.realize>` method of :class:`SingleTargetRadarChannel`.
    """

    __target_realization: RadarTargetRealization | None

    def __init__(
        self,
        alpha_device: Device,
        beta_device: Device,
        gain: float,
        target_realization: RadarTargetRealization | None,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> None:
        """
        Args:

            alpha_device (Device):
                First device linked by the :class:`.SingleTargetRadarChannelRealization` instance that generated this realization.

            beta_device (Device):
                Second device linked by the :class:`.SingleTargetRadarChannelRealization` instance that generated this realization.

            gain (float):
                Linear power gain factor a signal experiences when being propagated over this realization.

            target_realization (RadarTargetRealization | None):
                Single target realization.
                `None` if no target should be present.

            interpolation_mode (InterpolationMode, optional):
                Interpolation behaviour of the channel realization's delay components with respect to the proagated signal's sampling rate.
        """

        # Initialize the base class
        RadarChannelRealization.__init__(self, alpha_device, beta_device, gain, interpolation_mode)

        # Initialize class attributes
        self.__target_realization = target_realization

    @property
    def _path_realizations(self) -> Sequence[RadarPathRealization]:
        return [self.target_realization] if self.target_realization is not None else []

    @property
    def target_realization(self) -> RadarTargetRealization | None:
        """Realized radar target.

        Returns:
            Handle to the realized target.
            :py:obj:`None` if no target was considered.
        """

        return self.__target_realization

    def null_hypothesis(self) -> SingleTargetRadarChannelRealization:
        return SingleTargetRadarChannelRealization(
            self.alpha_device, self.beta_device, self.gain, None, self.interpolation_mode
        )

    def ground_truth(self) -> np.ndarray:
        return (
            np.empty((0, 3), dtype=float)
            if self.target_realization is None
            else self.target_realization.position[None, :]
        )

    def to_HDF(self, group: Group) -> None:
        # Serialize the base class
        RadarChannelRealization.to_HDF(self, group)

        # Serialize the target realization
        if self.target_realization is not None:
            target_group = group.create_group("target_realization")
            self.target_realization.to_HDF(target_group)

    @classmethod
    def From_HDF(
        cls: Type[SingleTargetRadarChannelRealization],
        group: Group,
        alpha_device: Device,
        beta_device: Device,
    ) -> SingleTargetRadarChannelRealization:
        parameters = cls._parameters_from_HDF(group)
        parameters["target_realization"] = (
            RadarTargetRealization.from_HDF(group["target_realization"])
            if "target_realization" in group
            else None
        )

        return SingleTargetRadarChannelRealization(alpha_device, beta_device, **parameters)


class MultiTargetRadarChannelRealization(RadarChannelRealization):
    """Realization of a spatial multi target radar channel.

    Generated by the :meth:`realize<MultiTargetRadarChannel.realize>` method of :class:`MultiTargetRadarChannel`.
    """

    __interference_realization: RadarInterferenceRealization | None
    __target_realizations: Sequence[RadarTargetRealization]

    def __init__(
        self,
        alpha_device: Device,
        beta_device: Device,
        gain: float,
        interference_realization: RadarInterferenceRealization | None,
        target_realizations: Sequence[RadarTargetRealization],
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> None:
        """
        Args:

            alpha_device (Device):
                First device linked by the :class:`.Channel` instance that generated this realization.

            beta_device (Device):
                Second device linked by the :class:`.Channel` instance that generated this realization.

            gain (float):
                Linear power gain factor a signal experiences when being propagated over this realization.

            interference_realizations (RadarInterferenceRealization | None):
                Realization of the line of sight interference.
                `None` if no interference should be considered.

            target_realizations (Sequence[RadarTargetRealization]):
                Sequence of radar target realizations considered within the radar channel.

            interpolation_mode (InterpolationMode, optional):
                Interpolation behaviour of the channel realization's delay components with respect to the proagated signal's sampling rate.
        """

        # Initialize the base class
        RadarChannelRealization.__init__(self, alpha_device, beta_device, gain, interpolation_mode)

        # Initialize class attributes
        self.__interference_realization = interference_realization
        self.__target_realizations = target_realizations

    @property
    def _path_realizations(self) -> Sequence[RadarPathRealization]:
        path_realizations: List[RadarPathRealization] = list(self.target_realizations)
        if self.interference_realization is not None:
            path_realizations.append(self.interference_realization)

        return path_realizations

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

    @property
    def num_targets(self) -> int:
        """Number of realized targets."""

        return len(self.target_realizations)

    def null_hypothesis(self) -> MultiTargetRadarChannelRealization:
        null_hypothesis_target_realizations = []
        for target_realization in self.target_realizations:
            if target_realization.static:
                null_hypothesis_target_realizations.append(target_realization)

        null_hypothesis_interference = self.interference_realization
        if null_hypothesis_interference is not None and not null_hypothesis_interference.static:
            null_hypothesis_interference = None

        return MultiTargetRadarChannelRealization(
            self.alpha_device,
            self.beta_device,
            self.gain,
            null_hypothesis_interference,
            null_hypothesis_target_realizations,
            self.interpolation_mode,
        )

    def ground_truth(self) -> np.ndarray:
        truth = np.empty((self.num_targets, 3), dtype=np.float_)
        for t, target in enumerate(self.target_realizations):
            truth[t, :] = target.position

        return truth

    def to_HDF(self, group: Group) -> None:
        # Serialize the base class
        RadarChannelRealization.to_HDF(self, group)

        # Serialize the interference realization
        if self.interference_realization is not None:
            target_group = group.create_group("interference_realization")
            self.interference_realization.to_HDF(target_group)

        # Serialize target realizations
        group.attrs["num_target_realizations"] = self.num_targets
        for t, target_realization in enumerate(self.target_realizations):
            target_realization.to_HDF(
                HDFSerializable._create_group(group, f"target_realization{t:02d}")
            )

    @classmethod
    def From_HDF(
        cls: Type[MultiTargetRadarChannelRealization],
        group: Group,
        alpha_device: Device,
        beta_device: Device,
    ) -> MultiTargetRadarChannelRealization:
        # Deserialize base class parameters
        parameters = RadarChannelRealization._parameters_from_HDF(group)
        parameters["interference_realization"] = (
            RadarInterferenceRealization.from_HDF(group["interference_realization"])
            if "interference_realization" in group
            else None
        )

        # Deserialize target realizations
        num_target_realizations = group.attrs.get("num_target_realizations", 0)
        target_realizations = [
            RadarTargetRealization.from_HDF(group[f"target_realization{t:02d}"])
            for t in range(num_target_realizations)
        ]

        return MultiTargetRadarChannelRealization(
            alpha_device, beta_device, target_realizations=target_realizations, **parameters
        )


class SingleTargetRadarChannel(RadarChannelBase[SingleTargetRadarChannelRealization], Serializable):
    """Model of a radar channel featuring a single reflecting target.

    The following minimal example outlines how to configure the channel model
    within the context of a :doc:`simulation.simulation.Simulation`:

    .. literalinclude:: ../scripts/examples/channel_SingleTargetRadarChannel.py
       :language: python
       :linenos:
       :lines: 11-30
    """

    yaml_tag = "RadarChannel"

    __target: VirtualRadarTarget
    __target_range: float | Tuple[float, float]
    __radar_cross_section: float
    __target_azimuth: float
    __target_zenith: float
    __target_exists: bool
    __target_velocity: float | np.ndarray

    def __init__(
        self,
        target_range: float | Tuple[float, float],
        radar_cross_section: float,
        target_azimuth: float = 0.0,
        target_zenith: float = 0.0,
        target_exists: bool = True,
        velocity: float | np.ndarray = 0,
        attenuate: bool = True,
        **kwargs,
    ) -> None:
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

            if value[0] < 0.0:
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

    @property
    def target_exists(self) -> bool:
        """Does an illuminated target exist?"""

        return self.__target_exists

    @target_exists.setter
    def target_exists(self, value: bool) -> None:
        self.__target_exists = value

    def _realize(self) -> SingleTargetRadarChannelRealization:
        # Realize targets
        target_range = self.target_range if isinstance(self.target_range, (np.int_, np.float_, int, float)) else self._rng.uniform(*self.target_range)  # type: ignore

        # Update the internal target model, kinda hacky
        unit_direction = Direction.From_Spherical(self.target_azimuth, self.target_zenith).view(
            np.ndarray
        )
        self.__target.position = unit_direction * target_range
        self.__target.velocity = unit_direction * self.target_velocity
        self.__cross_section.cross_section = self.radar_cross_section

        target_realization = self._realize_target(self.__target) if self.target_exists else None

        # Realize channel
        channel_realization = SingleTargetRadarChannelRealization(
            self.alpha_device,
            self.beta_device,
            self.gain,
            target_realization,
            self.interpolation_mode,
        )
        return channel_realization

    def recall_realization(self, group: Group) -> SingleTargetRadarChannelRealization:
        return SingleTargetRadarChannelRealization.From_HDF(
            group, self.alpha_device, self.beta_device
        )


class MultiTargetRadarChannel(RadarChannelBase[MultiTargetRadarChannelRealization], Serializable):
    """Model of a spatial radar channel featuring multiple reflecting targets.

    The following minimal example outlines how to configure the channel model
    within the context of a :doc:`simulation.simulation.Simulation`:

    .. literalinclude:: ../scripts/examples/channel_MultiTargetRadarChannel.py
       :language: python
       :linenos:
       :lines: 11-43
    """

    yaml_tag = "SpatialRadarChannel"

    interfernce: bool
    """Consider interference between linked devices.

    Only applies in the bistatic case, where transmitter and receiver are two dedicated device instances.
    """

    __targets: Set[RadarTarget]

    def __init__(self, attenuate: bool = True, interference: bool = True, *args, **kwargs) -> None:
        """
        Args:

            attenuate (bool, optional):
                Should the propagated signal be attenuated during propagation modeling?
                Enabled by default.

            interference (bool, optional):
                Should the channel model consider interference between the linked devices?
                Enabled by default.
        """

        # Initialize base classes
        RadarChannelBase.__init__(self, attenuate, *args, **kwargs)
        Serializable.__init__(self)

        # Initialize attributes
        self.interference = interference
        self.__targets = set()

    @property
    def targets(self) -> Set[RadarTarget]:
        """Set of targets considered within the radar channel."""

        return self.__targets

    def add_target(self, target: RadarTarget) -> None:
        """Add a new target to the radar channel.

        Args:

            target (RadarTarget):
                Target to be added.
        """

        if target not in self.targets:
            self.__targets.add(target)

    def make_target(
        self, moveable: Moveable, cross_section: RadarCrossSectionModel, *args, **kwargs
    ) -> PhysicalRadarTarget:
        """Declare a moveable to be a target within the radar channel.

        Args:

            moveable (Moveable):
                Moveable to be declared as a target.

            cross_section (RadarCrossSectionModel):
                Radar cross section model of the target.

            *args:
                Additional positional arguments passed to the target's constructor.

            **kwargs:
                Additional keyword arguments passed to the target's constructor.

        Returns:

            PhysicalRadarTarget: The newly created target.
        """

        target = PhysicalRadarTarget(cross_section, moveable, *args, **kwargs)
        self.add_target(target)

        return target

    def _realize_interference(self) -> RadarInterferenceRealization | None:
        """Realize the channel model's line of sight interference.

        Returns:
            The realized propagation path.
            `None` if :attr:`.interference` is disabled or the channel models a monostatic radar.
        """

        return (
            RadarInterferenceRealization(self.attenuate, True)
            if self.alpha_device is not self.beta_device
            else None
        )

    def _realize(self) -> MultiTargetRadarChannelRealization:
        if self.alpha_device is None or self.beta_device is None:
            raise FloatingError("Radar channel's linked devices not specified")

        # Realize radar channel parameters
        interference_realization = self._realize_interference()
        target_realizations = [self._realize_target(target) for target in self.targets]

        # Realize channel
        channel_realization = MultiTargetRadarChannelRealization(
            self.alpha_device,
            self.beta_device,
            self.gain,
            interference_realization,
            target_realizations,
            self.interpolation_mode,
        )
        return channel_realization

    def recall_realization(self, group: Group) -> MultiTargetRadarChannelRealization:
        return MultiTargetRadarChannelRealization.From_HDF(
            group, self.alpha_device, self.beta_device
        )
