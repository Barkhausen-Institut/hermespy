# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod, ABC
from typing import Any, Sequence, Tuple, Type, TYPE_CHECKING, List

import matplotlib.pyplot as plt
import numpy as np
from h5py import Group
from numpy import cos, exp
from scipy.constants import pi
from sparse import GCXS  # type: ignore

from hermespy.core import (
    ChannelStateInformation,
    ChannelStateFormat,
    Device,
    HDFSerializable,
    Serializable,
    Signal,
    VAT,
)
from .channel import Channel, ChannelRealization, InterpolationMode

if TYPE_CHECKING:
    from hermespy.simulation import SimulatedDevice  # pragma: no cover

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class AntennaCorrelation(ABC):
    """Base class for statistical modeling of antenna array correlations."""

    __channel: Channel | None
    __device: SimulatedDevice | None

    def __init__(
        self, channel: Channel | None = None, device: SimulatedDevice | None = None
    ) -> None:
        self.channel = channel
        self.device = device

    @property
    @abstractmethod
    def covariance(self) -> np.ndarray:
        """Antenna covariance matrix.

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

    @property
    def device(self) -> SimulatedDevice | None:
        """The device this correlation model is based upon.

        Returns:
            Handle to the device.
            `None` if the device is currently unknown.
        """

        return self.__device

    @device.setter
    def device(self, value: SimulatedDevice | None) -> None:
        self.__device = value


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

    @property
    def covariance(self) -> np.ndarray:
        if (
            self.device is not None
            and self.device.num_antennas != self.__covariance_matrix.shape[0]
        ):
            raise RuntimeError(
                f"Device with {self.device.num_antennas} antennas does not match covariance matrix of magnitude {self.__covariance_matrix.shape[0]}"
            )

        return self.__covariance_matrix

    @covariance.setter
    def covariance(self, value: np.ndarray) -> None:
        if value.ndim != 2 or not np.allclose(value, value.T.conj()):
            raise ValueError("Antenna correlation must be a hermitian matrix")

        if np.any(np.linalg.eigvals(value) <= 0.0):
            raise ValueError("Antenna correlation matrix must be positive definite")

        self.__covariance_matrix = value


class PathRealization(HDFSerializable):
    """A single delay path of a Multipath Fading channel realization.

    Represents the single propagation path equation

    .. math::

       h_{\\ell}(t) =
          \\sqrt{\\frac{K_{\\ell}}{1 + K_{\\ell}}} \\mathrm{e}^{\\mathrm{j} t \\omega_{\\ell} \\cos(\\theta_{\\ell,0}) + \\mathrm{j} \\phi_{\\ell,0} }
          + \\sqrt{\\frac{1}{N(1 + K_{\\ell})}} \\sum_{n=1}^{N} \\mathrm{e}^{\\mathrm{j} t \\omega_{\\ell} \\cos\\left( \\frac{2\\pi n + \\theta_{\\ell,n}}{N} \\right) + \\mathrm{j} \\phi_{\\ell,n}}
    """

    __los_gain: float
    __los_angle: float
    __los_phase: float
    __los_doppler: float
    __nlos_gain: float
    __nlos_angles: np.ndarray
    __nlos_phases: np.ndarray
    __nlos_doppler: float

    def __init__(
        self,
        power: float,
        delay: float,
        los_gain: float,
        los_angle: float,
        los_phase: float,
        los_doppler: float,
        nlos_gain: float,
        nlos_angles: np.ndarray,
        nlos_phases: np.ndarray,
        nlos_doppler: float,
    ) -> None:
        """
        Args:

            power (float):
                Power of the represented path in Watts.
                Initializes the :attr:`.power` attribute.

            delay (float):
                Delay of the represented path in seconds.
                Initializes the :attr:`.delay` attribute.

            los_gain (float):
                Line of sight power of the represented path.
                Initializes the :attr:`.los_gain` attribute.

            los_angle (float):
                Line of sight doppler angle in radians.
                Initializes the :attr:`.los_angle` attribute.

            los_phase (float):
                Line of sight components phase in radians.
                Initializes the :attr:`.los_phase` attribute.

            los_doppler (float):
                Line of sight doppler frequency in :math:`\\mathrm{Hz}`.
                Initializes the :attr:`.los_doppler` attribute.

            nlos_gain (float):
                Non line of sight power of the represented path.
                Initializes the :attr:`.nlos_gain` attribute.

            nlos_angles (float):
                Non line of sight doppler angles in radians.
                Initializes the :attr:`.nlos_angles` attribute.

            nlos_phases (float):
                Non line of sight components phases in radians.
                Initializes the :attr:`.nlos_phases` attribute.

            nlos_doppler (float):
                Non line of sight doppler frequency in :math:`\\mathrm{Hz}`.
                Initializes the :attr:`.nlos_doppler` attribute.
        """

        # Initialize class attributes
        self.__power = power
        self.__delay = delay
        self.__los_gain = los_gain
        self.__los_angle = los_angle
        self.__los_phase = los_phase
        self.__los_doppler = los_doppler
        self.__nlos_gain = nlos_gain
        self.__nlos_angles = nlos_angles
        self.__nlos_phases = nlos_phases
        self.__nlos_doppler = nlos_doppler

    @classmethod
    def Realize(
        cls: Type[PathRealization],
        power: float,
        delay: float,
        los_gain: float,
        nlos_gain,
        los_doppler: float,
        nlos_doppler: float,
        los_angle: float | None = None,
        num_sinusoids: int = 20,
        rng: np.random.Generator | None = None,
    ) -> PathRealization:
        """Realize the path's random variables.

        Args:
            power (float): Power of the represented path in Watts.
            delay (float): Delay of the represented path in seconds.
            los_gain (float): Line of sight power component of the represented path.
            nlos_gain (_type_): Non line of sight power component of the represented path.
            los_doppler (float): Line of sight doppler frequency of the represented path.
            nlos_doppler (float): None line of sight doppler frequencs of the represented path.
            los_angle (float, optional): Line of sight doppler angle in radians.
            num_sinusoids (int, optional): Number of sinusoids. :math:`20` by default.
            rng (np.random.Generator, optional): Random generator used to realize the random variables.

        Returns: The realized path realization.
        """

        # Initialize a new random generator if none was provided
        _rng = np.random.default_rng() if rng is None else rng

        # Draw random realizations for the path
        los_angle = _rng.uniform(0, 2 * pi) if los_angle is None else los_angle
        los_phase = _rng.uniform(0, 2 * pi)
        nlos_angles = _rng.uniform(0, 2 * pi, num_sinusoids)
        nlos_phases = _rng.uniform(0, 2 * pi, num_sinusoids)

        # Intialize object from random realizations
        return cls(
            power,
            delay,
            los_gain,
            los_angle,
            los_phase,
            los_doppler,
            nlos_gain,
            nlos_angles,
            nlos_phases,
            nlos_doppler,
        )

    @property
    def power(self) -> float:
        """Power of the propagation path in Watts.

        Referred to as :math:`g_{\\ell}` within the respective equations.
        """

        return self.__power

    @property
    def delay(self) -> float:
        """Delay of the propagation path in seconds.

        Referred to as :math:`\\tau_{\ell}` within the respective equations.
        """

        return self.__delay

    @property
    def los_gain(self) -> float:
        """Gain of the path's specular line of sight component.

        Represented by

        .. math::

           \\sqrt{\\frac{K_{\ell}}{1 + K_{\ell}}}

        within the respective equations.
        """

        return self.__los_gain

    @property
    def los_angle(self) -> float:
        """Angle of the path's specular line of sight component in radians.

        Represented by :math:`\\theta_{\\ell}` within the respective equations.
        """

        return self.__los_angle

    @property
    def los_phase(self) -> float:
        """Phase of the path's specular line of sight component in radians.

        Represented by :math:`\\phi_{\\ell}` within the respective equations.
        """

        return self.__los_phase

    @property
    def los_doppler(self) -> float:
        """Doppler frequency of the path's specular line of sight component in Hz.

        Represented by :math:`\\omega_{\\ell}` within the respective equations.
        """

        return self.__los_doppler

    @property
    def nlos_gain(self) -> float:
        """Gain of the path's non-specular components.

        Represented by

        .. math::

           \\sqrt{\\frac{1}{1 + K_{\ell}}}

        within the respective equations.
        """

        return self.__nlos_gain

    @property
    def nlos_angles(self) -> np.ndarray:
        """Angles of the path's non-specular components in radians.

        Represented by the sequence

        .. math::

           \\left[\\theta_{\\ell,1},\\, \\dotsc,\\, \\theta_{\\ell,N} \\right]^{\\mathsf{T}} \\in [0, 2\\pi)^{N}

        of :math:`N` angles in radians within the respective equations.
        """

        return self.__nlos_angles

    @property
    def nlos_phases(self) -> np.ndarray:
        """Phases of the path's non-specular components in radians.

        Represented by the sequence

        .. math::

           \\left[\\phi_{\\ell,1},\\, \\dotsc,\\, \\phi_{\\ell,N} \\right]^{\\mathsf{T}} \\in [0, 2\\pi)^{N}

        of :math:`N` angles in radians within the respective equations.
        """

        return self.__nlos_phases

    @property
    def nlos_doppler(self) -> float:
        """Doppler frequency of the path's non-specular components in Hz.

        Represented by :math:`\\omega_{\\ell}` within the respective equations.
        """

        return self.__nlos_doppler

    def _impulse_response(self, timestamps: np.ndarray) -> np.ndarray:
        """Compute the impulse response of the represented multipath component.

        Args:

            timestamps (numpy.ndarray): Timestamps in seconds at which to sample the impulse response.

        Returns: The sampled impulse response.
        """

        num_sinusoids = len(self.__nlos_angles)

        # Initialize empty impulse response
        impulse_response = np.zeros(len(timestamps), dtype=np.complex_)

        # Sum up and normalize all non-specular components
        for s, (nlos_angle, nlos_phase) in enumerate(zip(self.nlos_angles, self.nlos_phases)):
            impulse_response += exp(
                1j
                * (
                    self.nlos_doppler * timestamps * cos((2 * pi * s + nlos_angle) / num_sinusoids)
                    + nlos_phase
                )
            )
        impulse_response *= self.nlos_gain * (num_sinusoids**-0.5)

        # Add the specular component
        impulse_response += self.los_gain * exp(
            1j * (self.los_doppler * timestamps * cos(self.los_angle) + self.los_phase)
        )

        # Scale by the overall path power
        impulse_response *= self.power**0.5

        return impulse_response

    def propagate(self, signal: Signal) -> np.ndarray:
        """Propagate a signal along the represented multipath component.

        Args:

            signal (Signal): The signal to be propagated.


        Returns: The propagated samples.
        """

        # Generate the path's impule response
        impulse_response = self._impulse_response(signal.timestamps)

        # Propagate the transmitted samples
        propagated_samples = signal[:, :] * impulse_response[np.newaxis, :]
        return propagated_samples

    def to_HDF(self, group: Group) -> None:
        group.attrs["power"] = self.__power
        group.attrs["delay"] = self.__delay
        group.attrs["los_gain"] = self.__los_gain
        group.attrs["los_angle"] = self.__los_angle
        group.attrs["los_phase"] = self.__los_phase
        group.attrs["los_doppler"] = self.los_doppler
        group.attrs["nlos_gain"] = self.__nlos_gain
        group.attrs["nlos_doppler"] = self.nlos_doppler
        HDFSerializable._write_dataset(group, "nlos_angles", self.__nlos_angles)
        HDFSerializable._write_dataset(group, "nlos_phases", self.__nlos_phases)

    @classmethod
    def from_HDF(cls: Type[PathRealization], group: Group) -> PathRealization:
        power = group.attrs["power"]
        delay = group.attrs["delay"]
        los_gain = group.attrs["los_gain"]
        los_angle = group.attrs["los_angle"]
        los_phase = group.attrs["los_phase"]
        los_doppler = group.attrs["los_doppler"]
        nlos_gain = group.attrs["nlos_gain"]
        nlos_doppler = group.attrs["nlos_doppler"]
        nlos_angles = np.array(group["nlos_angles"], dtype=np.float_)
        nlos_phases = np.array(group["nlos_phases"], dtype=np.float_)

        return cls(
            power,
            delay,
            los_gain,
            los_angle,
            los_phase,
            los_doppler,
            nlos_gain,
            nlos_angles,
            nlos_phases,
            nlos_doppler,
        )


class MultipathFadingRealization(ChannelRealization):
    """Realization of a multipath fading channel.

    Generated by the :meth:`realize()<MultipathFadingChannel.realize>` routine of :class:`MultipathFadingChannels<MultipathFadingChannel>`.
    """

    __path_realizations: Sequence[PathRealization]
    __spatial_response: np.ndarray
    __max_delay: float

    def __init__(
        self,
        alpha_device: Device,
        beta_device: Device,
        gain: float,
        path_realizations: Sequence[PathRealization],
        spatial_response: np.ndarray,
        max_delay: float,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> None:
        """
        Args:

            alpha_device (Device):
                First device linked by the :class:`.MultipathFadingChannel` instance that generated this realization.

            beta_device (Device):
                Second device linked by the :class:`.MultipathFadingChannel` instance that generated this realization.

            gain (float):
                Linear power gain factor a signal experiences when being propagated over this realization.

            path_realizations (Sequence[PathRealization]):
                Realizations of the individual propagation paths.

            spatial_response (numpy.ndarray):
                Spatial response matrix of the channel realization considering `alpha_device` is the transmitter and `beta_device` is the receiver.

            interpolation_mode (InterpolationMode, optional):
                Interpolation behaviour of the channel realization's delay components with respect to the proagated signal's sampling rate.
        """

        # Initialize base class
        ChannelRealization.__init__(self, alpha_device, beta_device, gain, interpolation_mode)

        # Initialize class attributes
        self.__path_realizations = path_realizations
        self.__spatial_response = spatial_response

        # Infer additional parameters
        self.__max_delay = (
            max(path.delay for path in path_realizations) if max_delay is None else max_delay
        )

    @classmethod
    def Realize(
        cls: Type[MultipathFadingRealization],
        alpha_device: SimulatedDevice,
        beta_device: SimulatedDevice,
        gain: float,
        power_profile: np.ndarray,
        delays: np.ndarray,
        los_gains: np.ndarray,
        nlos_gains: np.ndarray,
        los_doppler: float,
        nlos_doppler: float,
        alpha_correlation: AntennaCorrelation | None = None,
        beta_correlation: AntennaCorrelation | None = None,
        los_angle: float | None = None,
        num_sinusoids: int = 20,
        rng: np.random.Generator | None = None,
    ) -> MultipathFadingRealization:
        """Realize the random variables of a multipath fading channel.

        Args:
            alpha_device (SimulatedDevice): First device linked by the channel.
            beta_device (SimulatedDevice): Second device linked by the channel.
            gain (float): Overall channel gain factor.
            power_profile (numpy.ndarray): Powers of each propagation path.
            delays (numpy.ndarray): Delays of each propgation path.
            los_gains (numpy.ndarray): Line of sight powers of each propagation path.
            nlos_gains (numpy.ndarray): Non line lof sight powers of each proapgation path.
            los_doppler (float): Line of sight doppler frequency of each propagation path.
            nlos_doppler (float): Non line of sight dopller frequency of each propagation path.
            alpha_correlation (AntennaCorrelation, optional): Antenna correlations at `alpha_device`.
            beta_correlation (AntennaCorrelation, optional): Antennna correlations at `beta_device`.
            los_angle (float, optional): Line of sight doppler angle in radians.
            num_sinusoids (int, optional): Number of model sinusoids. Defaults to 20.
            rng (numpy.random.Generator, optional): Random generator used to realize the random variables.

        Returns: The realized realization.
        """

        # Initialize a new random generator if none was provided
        _rng = np.random.default_rng() if rng is None else rng

        # Generate MIMO channel response
        spatial_response = np.exp(
            1j * _rng.uniform(0, 2 * pi, (beta_device.num_antennas, alpha_device.num_antennas))
        )

        # Apply antenna array correlation models
        if alpha_correlation is not None:
            spatial_response = spatial_response @ alpha_correlation.covariance
        if beta_correlation is not None:
            spatial_response = beta_correlation.covariance @ spatial_response

        # Generate path realizations
        path_realizations: List[PathRealization] = []
        for power, delay, los_gain, nlos_gain in zip(power_profile, delays, los_gains, nlos_gains):
            path_realizations.append(
                PathRealization.Realize(
                    power,
                    delay,
                    los_gain,
                    nlos_gain,
                    los_doppler,
                    nlos_doppler,
                    los_angle,
                    num_sinusoids,
                    _rng,
                )
            )

        max_delay = delays.max()
        return cls(alpha_device, beta_device, gain, path_realizations, spatial_response, max_delay)

    @property
    def path_realizations(self) -> Sequence[PathRealization]:
        """Realizations of the individual propagation paths."""

        return self.__path_realizations

    def __directive_spatial_response(self, transmitter: Device, receiver: Device) -> np.ndarray:
        """Infer the spatial response for the given transmitter and receiver.

        Subroutine of :meth:`state<MultipathFadingRealization.state>` and :meth:`_propagate<MultipathFadingRealization.propagate>`.

        Args:
            transmitter (Device):
                The transmitter device.

            receiver (Device):
                The receiver device.

        Returns: The spatial channel response matrix.

        Raises:

            ValueError: If the provided transmitter and receiver do not match the devices the channel was realized for.
        """

        if transmitter == self.alpha_device and receiver == self.beta_device:
            return self.__spatial_response

        if transmitter == self.beta_device and receiver == self.alpha_device:
            return self.__spatial_response.T

        raise ValueError(
            "The provided transmitter and receiver do not match the devices the channel was realized for"
        )

    def state(
        self,
        transmitter: Device,
        receiver: Device,
        delay: float,
        sampling_rate: float,
        num_samples: int,
        max_num_taps: int,
    ) -> ChannelStateInformation:
        spatial_response = self.__directive_spatial_response(transmitter, receiver)
        num_taps = min(1 + int(self.__max_delay * sampling_rate), max_num_taps)
        timestamps = np.arange(num_samples) / sampling_rate + delay

        siso_csi = np.zeros((num_samples, num_taps), dtype=np.complex_)
        for path_realization in self.path_realizations:
            tap_index = int(path_realization.delay * sampling_rate)

            # Skip paths with delays larger than the maximum delay required by the CSI request
            if tap_index > num_taps:
                continue

            siso_csi[:, tap_index] = siso_csi[:, tap_index] + path_realization._impulse_response(
                timestamps
            )

        # For the multipath fading model, the MIMO CSI is the outer product of the SISO CSI with the spatial response
        # The resulting multidimensional array is sparse in its fourth dimension and converted to a GCXS array for memory efficiency
        mimo_csi = GCXS.from_numpy(
            np.einsum("ij,kl->ijkl", spatial_response * self.gain**0.5, siso_csi),
            compressed_axes=(0, 1, 2),
        )

        state = ChannelStateInformation(
            ChannelStateFormat.IMPULSE_RESPONSE, mimo_csi, num_delay_taps=num_taps
        )
        return state

    def _propagate(
        self,
        signal: Signal,
        transmitter: Device,
        receiver: Device,
        interpolation: InterpolationMode,
    ) -> Signal:
        # Infer propagation direction and transmpose spatial response if necessary
        spatial_response = self.__directive_spatial_response(transmitter, receiver)
        sampling_rate = signal.sampling_rate
        max_delay_in_samples = int(self.__max_delay * sampling_rate)
        num_transmitted_samples = signal.num_samples + max_delay_in_samples

        # Propagate the transmitted samples
        propagated_samples = np.zeros(
            (spatial_response.shape[0], num_transmitted_samples), dtype=np.complex_
        )
        for path_realization in self.path_realizations:
            num_delay_samples = int(path_realization.delay * sampling_rate)
            propagated_samples[
                :, num_delay_samples : num_delay_samples + signal.num_samples
            ] += path_realization.propagate(signal)

        # Apply the channel's spatial response
        propagated_samples = spatial_response @ propagated_samples

        # Return the result
        return signal.from_ndarray(propagated_samples)

    def plot_power_delay(self, axes: VAT | None = None) -> Tuple[plt.Figure, VAT]:
        if axes:
            _axes = axes
            figure = axes[0, 0].get_figure()
        else:
            figure, _axes = plt.subplots(1, 1, squeeze=False)
            figure.suptitle("Power Delay Profile")

        delays = np.array([path.delay for path in self.path_realizations])
        powers = np.array([path.power for path in self.path_realizations])

        ax: plt.Axes = _axes.flat[0]
        ax.stem(delays, powers)
        ax.set_xlabel("Delay [s]")
        ax.set_ylabel("Power [Watts]")
        ax.set_yscale("log")

        return figure, _axes

    def to_HDF(self, group: Group) -> None:
        ChannelRealization.to_HDF(self, group)

        group.attrs["num_path_realizations"] = len(self.__path_realizations)
        group.attrs["max_delay"] = self.__max_delay

        HDFSerializable._write_dataset(group, "spatial_response", self.__spatial_response)

        for r, path_realization in enumerate(self.__path_realizations):
            path_realization.to_HDF(
                HDFSerializable._create_group(group, f"path_realization_{r:02d}")
            )

    @classmethod
    def From_HDF(
        cls: Type[MultipathFadingRealization],
        group: Group,
        alpha_device: Device,
        beta_device: Device,
    ) -> MultipathFadingRealization:
        initialization_parameters = cls._parameters_from_HDF(group)
        num_path_realizations = group.attrs["num_path_realizations"]
        initialization_parameters["max_delay"] = group.attrs["max_delay"]
        initialization_parameters["path_realizations"] = [
            PathRealization.from_HDF(group[f"path_realization_{r:02d}"])
            for r in range(num_path_realizations)
        ]
        initialization_parameters["spatial_response"] = np.array(
            group["spatial_response"], dtype=np.complex_
        )

        return cls(alpha_device, beta_device, **initialization_parameters)


class MultipathFadingChannel(Channel[MultipathFadingRealization], Serializable):
    """Base class for the implementation of stochastic multipath fading channels.

    Allows for the direct configuration of the Multipath Fading Channel's parameters

    .. math::

       \\mathbf{g} &= \\left[ g_{1}, g_{2}, \\,\\dotsc,\\, g_{L}  \\right]^\mathsf{T} \\in \\mathbb{C}^{L} \\\\
       \\mathbf{k} &= \\left[ K_{1}, K_{2}, \\,\\dotsc,\\, K_{L}  \\right]^\mathsf{T} \\in \\mathbb{R}^{L} \\\\
       \\mathbf{\\tau} &= \\left[ \\tau_{1}, \\tau_{2}, \\,\\dotsc,\\, \\tau_{L}  \\right]^\mathsf{T} \\in \\mathbb{R}^{L} \\\\

    directly.
    Refer to :doc:`/api/channel.multipath_fading_channel` for a detailed description of the channel model.

    The following minimal example outlines how to configure the channel model
    within the context of a :doc:`simulation.simulation.Simulation`:

    .. literalinclude:: ../scripts/examples/channel_MultipathFadingChannel.py
       :language: python
       :linenos:
       :lines: 12-40

    """

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
    __alpha_correlation: AntennaCorrelation | None
    __beta_correlation: AntennaCorrelation | None

    def __init__(
        self,
        delays: np.ndarray | List[float],
        power_profile: np.ndarray | List[float],
        rice_factors: np.ndarray | List[float],
        alpha_device: SimulatedDevice | None = None,
        beta_device: SimulatedDevice | None = None,
        gain: float = 1.0,
        num_sinusoids: int | None = None,
        los_angle: float | None = None,
        doppler_frequency: float | None = None,
        los_doppler_frequency: float | None = None,
        alpha_correlation: AntennaCorrelation | None = None,
        beta_correlation: AntennaCorrelation | None = None,
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

            alpha_device (Device, optional):
                First device linked by the :class:`.MultipathFadingChannel` instance that generated this realization.

            beta_device (Device, otional):
                Second device linked by the :class:`.MultipathFadingChannel` instance that generated this realization.

            gain (float, optional):
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default.

            num_sinusoids (int, optional):
                Number of sinusoids used to sample the statistical distribution.
                Denoted by :math:`N` within the respective equations.

            los_angle (float, optional):
                Angle phase of the line of sight component within the statistical distribution.

            doppler_frequency (float, optional):
                Doppler frequency shift of the statistical distribution.
                Denoted by :math:`\\omega_{\\ell}` within the respective equations.

            alpha_correlation(AntennaCorrelation, optional):
                Antenna correlation model at the first device.
                By default, the channel assumes ideal correlation, i.e. no cross correlations.

            beta_correlation(AntennaCorrelation, optional):
                Antenna correlation model at the second device.
                By default, the channel assumes ideal correlation, i.e. no cross correlations.

            **kwargs (Any, optional):
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

        # Sort delays
        sorting = np.argsort(delays)

        self.__delays = self.__delays[sorting]
        self.__power_profile = self.__power_profile[sorting]
        self.__rice_factors = self.__rice_factors[sorting]
        self.__num_sinusoids = 20 if num_sinusoids is None else num_sinusoids
        self.los_angle = los_angle
        self.doppler_frequency = 0.0 if doppler_frequency is None else doppler_frequency
        self.__los_doppler_frequency = None
        self.alpha_correlation = None
        self.beta_correlation = None

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
        self.__non_los_gains[rice_num_pos] = 1 / np.sqrt(1 + self.__rice_factors[rice_num_pos])
        self.__non_los_gains[rice_inf_pos] = 0.0

        # Initialize base class
        Channel.__init__(self, alpha_device, beta_device, gain, **kwargs)

        # Update correlations (required here to break dependency cycle during init)
        self.alpha_correlation = alpha_correlation
        self.beta_correlation = beta_correlation

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
        return MultipathFadingRealization.Realize(
            self.alpha_device,
            self.beta_device,
            self.gain,
            self.__power_profile,
            self.__delays,
            self.__los_gains,
            self.__non_los_gains,
            self.los_doppler_frequency,
            self.doppler_frequency,
            self.alpha_correlation,
            self.beta_correlation,
            self.los_angle,
            self.__num_sinusoids,
            self._rng,
        )

    @property
    def alpha_correlation(self) -> AntennaCorrelation | None:
        """Antenna correlation at the first device.

        Returns:
            Handle to the correlation model.
            :py:obj:`None`, if no model was configured and ideal correlation is assumed.
        """

        return self.__alpha_correlation

    @alpha_correlation.setter
    def alpha_correlation(self, value: AntennaCorrelation | None) -> None:
        if value is not None:
            value.channel = self
            value.device = self.alpha_device

        self.__alpha_correlation = value

    @property
    def beta_correlation(self) -> AntennaCorrelation | None:
        """Antenna correlation at the second device.

        Returns:
            Handle to the correlation model.
            :py:obj:`None`, if no model was configured and ideal correlation is assumed.
        """

        return self.__beta_correlation

    @beta_correlation.setter
    def beta_correlation(self, value: AntennaCorrelation | None) -> None:
        if value is not None:
            value.channel = self
            value.device = self.beta_device

        self.__beta_correlation = value

    @Channel.alpha_device.setter  # type: ignore
    def alpha_device(self, value: SimulatedDevice) -> None:
        Channel.alpha_device.fset(self, value)  # type: ignore

        # Register new device at correlation model
        if self.alpha_correlation is not None:
            self.alpha_correlation.device = value

    @Channel.beta_device.setter  # type: ignore
    def beta_device(self, value: SimulatedDevice) -> None:
        Channel.beta_device.fset(self, value)  # type: ignore

        # Register new device at correlation model
        if self.beta_correlation is not None:
            self.beta_correlation.device = value

    def recall_realization(self, group: Group) -> MultipathFadingRealization:
        return MultipathFadingRealization.From_HDF(group, self.alpha_device, self.beta_device)
