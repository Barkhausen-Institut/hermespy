# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from math import ceil
from typing import Any, Generic, Mapping, Set, Sequence, Tuple, Type, TYPE_CHECKING, TypeVar

import numpy as np
from h5py import Group
from scipy.constants import pi, speed_of_light
from sparse import GCXS  # type: ignore

from hermespy.core import (
    ChannelStateInformation,
    ChannelStateFormat,
    Direction,
    HDFSerializable,
    SignalBlock,
)
from ..channel import (
    Channel,
    ChannelSample,
    LinkState,
    ChannelSampleHook,
    ChannelRealization,
    InterpolationMode,
)

if TYPE_CHECKING:
    from hermespy.simulation import DeviceState  # pragma: no cover

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


RCST = TypeVar("RCST", bound="RadarChannelSample")
"""Type of radar channel sample."""


class RadarChannelSample(ChannelSample):
    """Realization of a radar channel.

    Generated by :meth:`RadarChannelBase.realize` and :meth:`RadarChannelBase.realize_interference`.
    """

    __paths: Sequence[RadarPath]
    __gain: float

    def __init__(self, paths: Sequence[RadarPath], gain: float, state: LinkState) -> None:
        """
        Args:

            paths (Sequence[RadarPath]):
                Sequence of realized radar propagation paths.

            gain (float):
                Channel gain in linear scale.

            state (ChannelState):
                State of the channel at the time of sampling.
        """

        # Initialize base class
        ChannelSample.__init__(self, state)

        # Initialize class attributes
        self.__paths = paths
        self.__gain = gain

    @property
    def paths(self) -> Sequence[RadarPath]:
        """Sequence of realized radar propagation paths."""

        return self.__paths

    @property
    def gain(self) -> float:
        """Channel gain in linear scale."""

        return self.__gain

    def _propagate(self, signal: SignalBlock, interpolation: InterpolationMode) -> SignalBlock:
        delays = np.array(
            [
                path.propagation_delay(self.transmitter_state, self.receiver_state)
                for path in self.paths
            ]
        )
        velocities = np.array(
            [
                path.relative_velocity(self.transmitter_state, self.receiver_state)
                for path in self.paths
            ]
        )

        # Compute the expected sample overhead of the propagated sample resultin from propagtion delays
        if delays.size > 0:
            max_delay_in_samples = ceil(
                delays.max() * self.bandwidth
                + 2 * velocities.max() * signal.num_samples / (self.bandwidth * speed_of_light)
            )
        else:
            max_delay_in_samples = 0

        propagated_samples = np.zeros(
            (self.num_receive_antennas, signal.num_samples + max_delay_in_samples),
            dtype=np.complex_,
        )

        # Compute the signal propagated along each respective path realization
        for path, delay, velocity in zip(self.paths, delays, velocities):
            path.add_propagation(
                self.transmitter_state,
                self.receiver_state,
                signal,
                self.bandwidth,
                self.carrier_frequency,
                propagated_samples,
                delay,
                velocity,
            )

        # Apply the channel gain
        propagated_samples *= self.gain**0.5

        return SignalBlock(propagated_samples, signal._offset)

    def state(
        self,
        num_samples: int,
        max_num_taps: int,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> ChannelStateInformation:
        raw_state = np.zeros(
            (
                self.receiver_state.antennas.num_receive_antennas,
                self.transmitter_state.antennas.num_transmit_antennas,
                num_samples,
                max_num_taps,
            ),
            dtype=np.complex_,
        )
        for path in self.paths:
            path.add_state(
                self.transmitter_state,
                self.receiver_state,
                self.bandwidth,
                self.carrier_frequency,
                0.0,
                raw_state,
            )

        # Apply the channel gain
        raw_state *= self.gain**0.5

        return ChannelStateInformation(
            ChannelStateFormat.IMPULSE_RESPONSE, GCXS.from_numpy(raw_state)
        )

    def null_hypothesis(self) -> RadarChannelSample:
        """Generate a null hypothesis channel sample rom a given channel sample.

        Null hypothesis sample will remove non-static propagation components from the channel model.
        This function is, for example, accessed to evaluate a radar link's receiver operating characteristics.

        Returns: The null hypothesis radar channel realization.
        """

        # Remove non-static paths
        static_paths = [path for path in self.paths if path.static]

        return RadarChannelSample(
            static_paths,
            self.gain,
            LinkState(
                self.transmitter_state,
                self.receiver_state,
                self.carrier_frequency,
                self.bandwidth,
                0.0,
            ),
        )


class RadarChannelRealization(ChannelRealization[RadarChannelSample]):
    """Realization of a radar channel."""

    def __init__(
        self, sample_hooks: Set[ChannelSampleHook[RadarChannelSample]], gain: float
    ) -> None:
        """_summary_

        Args:

            sample_hooks (Set[ChannelSampleHook[CST]], optional):
                Hooks to be called after the channel is sampled.

            gain (float):
                Linear power gain factor a signal experiences when being propagated over this realization.
        """

        # Initialize base class
        ChannelRealization.__init__(self, sample_hooks, gain)

    @abstractmethod
    def _generate_paths(self, state: LinkState) -> Sequence[RadarPath]:
        """Generate the sequence of realized radar propagation paths.

        Subroutine of :meth:`RadarChannelRealization._sample`.

        Args:

            state (ChannelState):
                State of the channel at the time of sampling.

        Returns: Sequence of realized radar propagation paths.
        """
        ...  # pragma: no cover

    def _sample(self, state: LinkState) -> RadarChannelSample:
        return RadarChannelSample(self._generate_paths(state), self.gain, state)

    def _reciprocal_sample(
        self, sample: RadarChannelSample, state: LinkState
    ) -> RadarChannelSample:
        return RadarChannelSample(sample.paths, sample.gain, state)


RCRT = TypeVar("RCRT", bound=RadarChannelRealization)
"""Type of radar channel realization."""


class RadarPath(HDFSerializable):
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

    @property
    @abstractmethod
    def ground_truth(self) -> Tuple[np.ndarray, np.ndarray] | None:
        """Consolidate the true target information represented by this path.

        Either a tuple of the target's position and velocity or `None` if the path
        represents only interference / clutter.
        """
        ...  # pragma: no cover

    @abstractmethod
    def propagation_delay(self, transmitter: DeviceState, receiver: DeviceState) -> float:
        """Propagation delay of the wave from transmitter over target to receiver.

        Denoted by :math:`\\tau_{\\ast}` within the respective equations.

        Args:

            transmitter (DeviceState):
                Transmitting device.

            receiver (Device):
                Receiving device.

        Returns: Propagation delay in seconds.
        """
        ...  # pragma: no cover

    @abstractmethod
    def relative_velocity(self, transmitter: DeviceState, receiver: DeviceState) -> float:
        """Relative velocity between transmitter and receiver.

        Denoted by :math:`v_{\\ast}` within the respective equations.

        Args:

            transmitter (DeviceState):
                Transmitting device.

            receiver (DeviceState):
                Receiving device.

        Returns: Relative velocity in m/s.
        """
        ...  # pragma: no cover

    @abstractmethod
    def propagation_response(
        self, transmitter: DeviceState, receiver: DeviceState, carrier_frequency: float
    ) -> np.ndarray:
        """Multipath sensor array response matrix from transmitter to receiver.

        Includes polarization losses.

        Args:

            transmitter (DeviceState):
                Transmitting device.

            receiver (DeviceState):
                Receiving device.

            carrier_frequency (float):
                Carrier frequency of the propagated signal in Hz.
                Denoted by :math:`f_{\\mathrm{c}}^{(\\alpha)}` within the respective equations.

        Returns: Numpy matrix of antenna response weights.
        """
        ...  # pragma: no cover

    def add_propagation(
        self,
        transmitter_state: DeviceState,
        receiver_state: DeviceState,
        signal: np.ndarray,
        bandwidth: float,
        carrier_frequency: float,
        propagated_samples: np.ndarray,
        propagation_delay: float | None = None,
        relative_velocity: float | None = None,
    ) -> None:
        """Add propagation of a signal over this path realization to a given sample buffer.

        Args:

            transmitter (DeviceState):
                Transmitting device.

            receiver (DeviceState):
                Receiving device.

            signal (np.ndarray):
                Signal samples to be propagated.

            bandwidth (float):
                Sampling rate of the the propagated signal model in Hz.

            carrier_frequency (float):
                Central carrier frequency of the propagated signal in Hz.

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
            self.propagation_delay(transmitter_state, receiver_state)
            if propagation_delay is None
            else propagation_delay
        )
        relative_velocity = (
            self.relative_velocity(transmitter_state, receiver_state)
            if relative_velocity is None
            else relative_velocity
        )
        propagation_response = self.propagation_response(
            transmitter_state, receiver_state, carrier_frequency
        )

        delay_sample_offset = int(propagation_delay * bandwidth)
        doppler_shift = relative_velocity * carrier_frequency / speed_of_light

        # ToDo: Exact time of flight resampling
        # echo_timestamps = propagation_delay + 2 * relative_velocity * signal.timestamps / speed_of_light
        # echo_weights = np.exp(2j * pi * (doppler_shift * echo_timestamps))
        echo_weights = np.exp(2j * pi * (doppler_shift / bandwidth * np.arange(signal.shape[1])))

        propagated_samples[
            :, delay_sample_offset : delay_sample_offset + signal.shape[1]
        ] += np.einsum("ij,jk,k->ik", propagation_response, signal, echo_weights)

    def add_state(
        self,
        transmitter: DeviceState,
        receiver: DeviceState,
        bandwidth: float,
        carrier_frequency: float,
        delay: float,
        state: np.ndarray,
    ) -> None:
        """Add propagation of a signal over this path realization to a given channel state information sample buffer.

        Args:

            transmitter (DeviceState):
                Transmitting device.

            receiver (DeviceState):
                Receiving device.

            bandwidth (float):
                Sampling rate of the the propagated signal model in Hz.

            carrier_frequency (float):
                Central carrier frequency of the propagated signal in Hz.

            delay (float):
                Delay of the channel state information in seconds.

            state (np.ndarray):
                Sample buffer to be written to.
        """

        # Query the required parameters
        propagation_delay = self.propagation_delay(transmitter, receiver)
        relative_velocity = self.relative_velocity(transmitter, receiver)
        propagation_response = self.propagation_response(transmitter, receiver, carrier_frequency)

        delay_sample_offset = int(propagation_delay * bandwidth - delay)
        if delay_sample_offset < 0 or delay_sample_offset >= state.shape[3]:
            return

        doppler_shift = relative_velocity * carrier_frequency / speed_of_light

        # echo_timestamps = delay + 2 * relative_velocity * np.arange(state.shape[2]) / speed_of_light
        # echo_weights = np.exp(2j * pi * (doppler_shift * echo_timestamps))
        echo_weights = np.exp(2j * pi * (doppler_shift / bandwidth * np.arange(state.shape[2])))

        state[:, :, :, delay_sample_offset] += np.einsum(
            "ij,k->ijk", propagation_response, echo_weights
        )

    def to_HDF(self, group: Group) -> None:
        # Serialize the class attributes
        group.attrs["attenuate"] = self.attenuate
        group.attrs["static"] = self.static

    @classmethod
    def _parameters_from_HDF(cls: Type[RadarPath], group: Group) -> Mapping[str, Any]:
        """Deserialize the object's parameters from HDF5.

        Intended to be used as a subroutine of :meth:`From_HDF`.

        Returns: The object's parmeters as a keyword argument dictionary.
        """

        return {"attenuate": group.attrs["attenuate"], "static": group.attrs["static"]}


class RadarTargetPath(RadarPath):
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
        RadarPath.__init__(self, attenuate, static)

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

    @property
    def ground_truth(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.position, self.velocity

    def propagation_delay(self, transmitter: DeviceState, receiver: DeviceState) -> float:
        emerging_vector = self.position - transmitter.position
        impinging_vector = receiver.position - self.position

        delay = (
            np.linalg.norm(emerging_vector) + np.linalg.norm(impinging_vector)
        ) / speed_of_light
        return delay

    def relative_velocity(self, transmitter: DeviceState, receiver: DeviceState) -> float:
        target_position = self.position
        emerging_vector = target_position - transmitter.position
        impinging_vector = receiver.position - target_position

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
        self, transmitter: DeviceState, receiver: DeviceState, carrier_frequency: float
    ) -> np.ndarray:
        # Query the sensor array responses
        rx_response = receiver.antennas.cartesian_array_response(
            carrier_frequency, self.position, "global"
        )
        tx_response = transmitter.antennas.cartesian_array_response(
            carrier_frequency, self.position, "global"
        ).conj()

        if self.attenuate:
            # Compute propagation distances
            tx_distance = np.linalg.norm(self.position - transmitter.position)
            rx_distance = np.linalg.norm(receiver.position - self.position)

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
        RadarPath.to_HDF(self, group)

        # Serialize class attributes
        self._write_dataset(group, "position", self.position)
        self._write_dataset(group, "velocity", self.velocity)
        group.attrs["cross_section"] = self.cross_section
        group.attrs["reflection_phase"] = self.reflection_phase

    @classmethod
    def from_HDF(cls: Type[RadarTargetPath], group: Group) -> RadarTargetPath:
        # Deserialize base class
        parameters = RadarPath._parameters_from_HDF(group)

        # Deserialize class attributes
        position = np.array(group["position"], dtype=np.float_)
        velocity = np.array(group["velocity"], dtype=np.float_)
        cross_section = group.attrs["cross_section"]
        reflection_phase = group.attrs["reflection_phase"]

        return RadarTargetPath(position, velocity, cross_section, reflection_phase, **parameters)


class RadarInterferencePath(RadarPath):
    """Realization of a line of sight interference propgation path between a radar transmitter and receiver"""

    @property
    def ground_truth(self) -> None:
        #  Always None because the LOS interference does not represent a real target
        return None  # pragma: no cover

    def propagation_delay(self, transmitter: DeviceState, receiver: DeviceState) -> float:
        delay = np.linalg.norm(receiver.position - transmitter.position) / speed_of_light
        return delay

    def relative_velocity(self, transmitter: DeviceState, receiver: DeviceState) -> float:
        connection = Direction.From_Cartesian(
            receiver.position - transmitter.position, normalize=True
        ).view(np.ndarray)
        return np.dot(transmitter.velocity - receiver.velocity, connection)

    def propagation_response(
        self, transmitter: DeviceState, receiver: DeviceState, carrier_frequency: float
    ) -> np.ndarray:
        # Model the sensor arrays' spatial responses
        rx_response = receiver.antennas.cartesian_array_response(
            carrier_frequency, transmitter.position, "global"
        ).conj()
        tx_response = transmitter.antennas.cartesian_array_response(
            carrier_frequency, receiver.position, "global"
        )

        if self.attenuate:
            # Compute propagation distance
            distance = np.linalg.norm(receiver.position - transmitter.position)

            wavelength = speed_of_light / carrier_frequency
            amplitude_factor = wavelength / (4 * pi * distance)

        else:
            amplitude_factor = 1.0

        # Compute the MIMO response
        return amplitude_factor * np.inner(rx_response, tx_response)

    @classmethod
    def from_HDF(cls: Type[RadarInterferencePath], group: Group) -> RadarInterferencePath:
        # Deserialize base class
        parameters = RadarPath._parameters_from_HDF(group)
        return RadarInterferencePath(**parameters)


class RadarChannelBase(Generic[RCRT], Channel[RCRT, RadarChannelSample]):
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
