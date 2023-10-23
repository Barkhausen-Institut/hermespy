# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, Generic, Tuple, Type, TypeVar, TYPE_CHECKING

import numpy as np
from h5py import Group
from scipy.constants import speed_of_light

from hermespy.core import ChannelStateInformation, ChannelStateFormat, Device, Signal
from hermespy.tools import amplitude_path_loss
from .channel import Channel, ChannelRealization, InterpolationMode

if TYPE_CHECKING:
    from hermespy.simulation import SimulatedDevice  # pragma: no cover

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


DCRT = TypeVar("DCRT", bound="DelayChannelRealization")
"""Type of delay channel realization"""


class DelayChannelRealization(ChannelRealization):
    """Realization of a delay channel model.

    Generated from :class:`DelayChannel's<DelayChannelBase>` :meth:`realize<DelayChannelBase.realize>` routine.
    """

    __delay: float
    __model_propagation_loss: bool

    def __init__(self, alpha_device: Device, beta_device: Device, gain: float, delay: float, model_propagation_loss: bool, interpolation_mode: InterpolationMode) -> None:
        """
        Args:

            alpha_device (Device):
                First device linked by the :class:`.DelayChannel` instance that generated this realization.

            beta_device (Device):
                Second device linked by the :class:`.DelayChannel` instance that generated this realization.

            gain (float):
                Linear power gain factor a signal experiences when being propagated over this realization.

            delay (float):
                Propagation delay in seconds.

            model_propagation_loss (bool):
                Should free space propagation loss be modeled?
        """

        # Initialize base class
        ChannelRealization.__init__(self, alpha_device, beta_device, gain, interpolation_mode)

        # Initialize class attributes
        self.__delay = delay
        self.__model_propagation_loss = model_propagation_loss

    @property
    def delay(self) -> float:
        """Propagation delay in seconds."""

        return self.__delay

    @property
    def model_propagation_loss(self) -> bool:
        """Should free-space propagation losses be modeled?"""

        return self.__model_propagation_loss

    @abstractmethod
    def _spatial_response(self, transmitter: Device, receiver: Device, carrier_frequency: float) -> np.ndarray:
        """Compute the realization's spatial response for a specific propagation direction.

        Args:

            transmitter (Device):
                Device transmitting into the realized channel.

            receiver (Device):
                Device receiving from the realized channel.

            carrier_frequency (float):
                Central frequency of the propagation in :math:`\\mathrm{Hz}`.

        Returns: Spatial channel response as a numpy matrix (two-dimensional vector).
        """
        ...  # pragma: no cover

    def __path_gain(self, carrier_frequency: float) -> float:
        """Compute the realization's linear path gain factor.

            carrier_frequency (float):
                Central frequency of the propagation in :math:`\\mathrm{Hz}`.

        Returns: Linear path gain scaling the amplitude of a propagated signal.
        """

        gain_factor = 1.0

        if self.model_propagation_loss:
            if carrier_frequency == 0.0:
                raise RuntimeError("Transmitting device's carrier frequency may not be zero, disable propagation path loss modeling")

            gain_factor *= amplitude_path_loss(carrier_frequency, self.delay * speed_of_light)

        return gain_factor

    def _propagate(self, signal: Signal, transmitter: Device, receiver: Device, interpolation: InterpolationMode) -> Signal:
        delay_samples = int(self.delay * signal.sampling_rate)
        spatial_response = self._spatial_response(transmitter, receiver, signal.carrier_frequency)
        gain_factor = self.__path_gain(signal.carrier_frequency)

        propagated_samples = np.append(np.zeros((receiver.antennas.num_receive_antennas, delay_samples), np.complex_), gain_factor * spatial_response @ signal.samples)

        propagated_signal = Signal(propagated_samples, signal.sampling_rate, signal.carrier_frequency)
        return propagated_signal

    def state(self, transmitter: Device, receiver: Device, delay_offset: float, sampling_rate: float, num_samples: int, max_num_taps: int) -> ChannelStateInformation:
        delay_samples = int(self.delay * sampling_rate)
        spatial_response = self._spatial_response(transmitter, receiver, transmitter.carrier_frequency)
        gain_factor = self.__path_gain(transmitter.carrier_frequency) * self.gain**0.5

        cir = np.zeros((receiver.antennas.num_receive_antennas, transmitter.antennas.num_transmit_antennas, num_samples, min(1 + delay_samples, max_num_taps)), dtype=np.complex_)
        if delay_samples < max_num_taps:
            cir[:, :, :, delay_samples] = gain_factor * spatial_response[:, :, np.newaxis]

        state = ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE, cir)
        return state

    def to_HDF(self, group: Group) -> None:
        # Serialize base class
        ChannelRealization.to_HDF(self, group)

        # Serialize attributes
        group.attrs["delay"] = self.delay
        group.attrs["model_propagation_loss"] = self.model_propagation_loss
        group.attrs["interpolation_mode"] = self.interpolation_mode.value

    @classmethod
    def _parameters_from_HDF(cls: type[ChannelRealization], group: Group) -> Dict[str, Any]:
        # Deserialize base class parameters
        parameters = ChannelRealization._parameters_from_HDF(group)

        # Deserialize attributes
        parameters["delay"] = float(group.attrs["delay"])
        parameters["model_propagation_loss"] = bool(group.attrs["model_propagation_loss"])
        parameters["interpolation_mode"] = InterpolationMode(group.attrs["interpolation_mode"])

        return parameters


class SpatialDelayChannelRealization(DelayChannelRealization):
    """Realization of a spatial delay channel.

    Generated from :class:`SpatialDelayChannel's<SpatialDelayChannel>` :meth:`realize<SpatialDelayChannel.realize>` routine.
    """

    def _spatial_response(self, transmitter: Device, receiver: Device, carrier_frequency: float) -> np.ndarray:
        transmit_response = transmitter.antennas.cartesian_array_response(carrier_frequency, receiver.global_position, "global")
        receive_response = receiver.antennas.cartesian_array_response(carrier_frequency, transmitter.global_position, "global")

        return receive_response @ transmit_response.T

    @classmethod
    def From_HDF(cls: Type[SpatialDelayChannelRealization], group: Group, alpha_device: Device, beta_device: Device) -> SpatialDelayChannelRealization:
        return cls(alpha_device, beta_device, **DelayChannelRealization._parameters_from_HDF(group))


class RandomDelayChannelRealization(DelayChannelRealization):
    """Realization of a random delay channel.

    Generated from :class:`RandomDelayChannel's<RandomDelayChannel>` :meth:`realize<RandomDelayChannel.realize>` routine.
    """

    def _spatial_response(self, transmitter: Device, receiver: Device, carrier_frequency: float) -> np.ndarray:
        return np.eye(receiver.antennas.num_receive_antennas, transmitter.antennas.num_transmit_antennas, dtype=np.complex_)

    @classmethod
    def From_HDF(cls: Type[RandomDelayChannelRealization], group: Group, alpha_device: Device, beta_device: Device) -> RandomDelayChannelRealization:
        return cls(alpha_device, beta_device, **DelayChannelRealization._parameters_from_HDF(group))


class DelayChannelBase(Generic[DCRT], Channel[DCRT]):
    """Base of delay channel models."""

    __model_propagation_loss: bool

    def __init__(self, alpha_device: SimulatedDevice | None = None, beta_device: SimulatedDevice | None = None, gain: float = 1.0, model_propagation_loss: bool = True, **kwargs) -> None:
        """
        Args:

            alpha_device (SimulatedDevice, optional):
                First device linked by the :class:`.DelayChannelBase` instance that generated this realization.

            beta_device (SimulatedDevice, optional):
                Second device linked by the :class:`.DelayChannelBase` instance that generated this realization.

            gain (float, optional):
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default.

            model_propagation_loss (bool, optional):
                Should free space propagation loss be modeled?
                Enabled by default.

            **kawrgs:
                :class:`Channel` base class initialization arguments.
        """

        # Initialize base class
        Channel.__init__(self, alpha_device, beta_device, gain, **kwargs)

        # Initialize class attributes
        self.__model_propagation_loss = model_propagation_loss

    @abstractmethod
    def _realize_delay(self) -> float:
        """Generate a delay realization.

        Returns: The delay in seconds.
        """
        ...  # pragma: no cover

    @property
    def model_propagation_loss(self) -> bool:
        """Should free space propagation loss be modeled?

        Returns: Enabled flag.
        """

        return self.__model_propagation_loss

    @model_propagation_loss.setter
    def model_propagation_loss(self, value: bool) -> None:
        self.__model_propagation_loss = value

    @abstractmethod
    def _init_realization(self, *args, **kwargs) -> DCRT:
        ...  # pragma: no cover

    def _realize(self) -> DCRT:
        delay = self._realize_delay()
        return self._init_realization(self.alpha_device, self.beta_device, self.gain, delay=delay, model_propagation_loss=self.model_propagation_loss, interpolation_mode=self.interpolation_mode)


class SpatialDelayChannel(DelayChannelBase[SpatialDelayChannelRealization]):
    """Delay channel based on spatial relations between the linked devices.

    The spatial delay channel requires both linked devices to specify their assumed positions.
    Its impulse response between two devices :math:`\\alpha` and :math:`\\beta` featuring :math:`N^{(\\alpha)}` and :math:`N^{(\\beta)}` antennas, respectively, is given by

    .. math::

       \\mathbf{H}(t,\\tau) = \\frac{1}{4\\pi f_\\mathrm{c}^{(\\alpha)}\\overline{\\tau}} \\mathbf{A}^{(\\alpha,\\beta)} \\delta(\\tau - \\overline{\\tau})\\ \\text{.}

    The assumed propagation delay between the two devices is given by

    .. math::

       \\overline{\\tau} = \\frac{\\|\\mathbf{p}^{(\\alpha)} - \\mathbf{p}^{(\\beta)}\\|_2}{c_0}

    and depends on the distance between the two devices located at positions :math:`\\mathbf{p}^{(\\alpha)}` and :math:`\\mathbf{p}^{(\\beta)}`.
    The sensor array response :math:`\\mathbf{A}^{(\\alpha,\\beta)}` depends on the device's relative orientation towards each other.

    The following minimal example outlines how to configure the channel model
    within the context of a :doc:`simulation.simulation.Simulation`:

    .. literalinclude:: ../scripts/examples/channel_RandomDelayChannel.py
       :language: python
       :linenos:
       :lines: 11-38

    """

    yaml_tag: str = "SpatialDelay"

    def _realize_delay(self) -> float:
        if self.alpha_device is None or self.beta_device is None:
            raise RuntimeError("The spatial delay channel requires the linked devices positions to be specified")

        distance = float(np.linalg.norm(self.alpha_device.global_position - self.beta_device.global_position))
        delay = distance / speed_of_light

        return delay

    def _init_realization(self, *args, **kwargs) -> SpatialDelayChannelRealization:
        return SpatialDelayChannelRealization(*args, **kwargs)

    def recall_realization(self, group: Group) -> SpatialDelayChannelRealization:
        return SpatialDelayChannelRealization.From_HDF(group, self.alpha_device, self.beta_device)


class RandomDelayChannel(DelayChannelBase[RandomDelayChannelRealization]):
    """Delay channel assuming a uniformly distributed random propagation between the linked devices.

    Its impulse response between two devices :math:`\\alpha` and :math:`\\beta` featuring :math:`N^{(\\alpha)}` and :math:`N^{(\\beta)}` antennas, respectively, is given by

    .. math::

       \\mathbf{H}(t,\\tau) = \\frac{1}{4\\pi f_\\mathrm{c}^{(\\alpha)}\\overline{\\tau}} \\mathbf{A}^{(\\alpha,\\beta)} \\delta(\\tau - \\overline{\\tau})\\ \\text{.}

    The assumed propagation delay is drawn from the uniform distribution

    .. math::

       \\overline{\\tau} \\sim \\mathcal{U}(\\tau_{\\mathrm{Min}}, \\tau_{\\mathrm{Max}})

    and lies in the interval between :math:`\\tau_\\mathrm{Min}` and :math:`\\tau_\\mathrm{Max}`.
    The sensor array response :math:`\\mathbf{A}^{(\\alpha,\\beta)}` is always assumed to be the identity matrix.

    The following minimal example outlines how to configure the channel model
    within the context of a :doc:`simulation.simulation.Simulation`:

    .. literalinclude:: ../scripts/examples/channel_RandomDelayChannel.py
       :language: python
       :linenos:
       :lines: 11-38

    """

    yaml_tag: str = "RandomDelay"

    __delay: float | Tuple[float, float]

    def __init__(self, delay: float | Tuple[float, float], *args, **kwargs) -> None:
        """
        Args:

            delay (float | Tuple[float, float]):
                Assumed propagation delay in seconds.
                If a scalar floating point, the delay is assumed to be constant.
                If a tuple of two floats, the tuple values indicate the mininum and maxium values of a uniform distribution, respectively.

            *args:
                :class:`.Channel` base class initialization parameters.

            **kwargs:
                :class:`.Channel` base class initialization parameters.
        """

        self.delay = delay
        DelayChannelBase.__init__(self, *args, **kwargs)

    @property
    def delay(self) -> float | Tuple[float, float]:
        """Assumed propagation delay in seconds.

        If set to a scalar floating point, the delay is assumed to be constant.
        If set to a tuple of two floats, the tuple values indicate the mininum and maxium values of a uniform distribution, respectively.

        Raises:

            ValueError: If the delay is set to a negative value.
            ValueError: If the delay is set to a tuple of two values where the first value is greater than the second value.
        """

        return self.__delay

    @delay.setter
    def delay(self, value: float | Tuple[float, float]) -> None:
        if isinstance(value, float):
            if value < 0.0:
                raise ValueError(f"Delay must be greater or equal to zero (not {value})")

        elif isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Delay limit tuple must contain two entries")

            if any(v < 0.0 for v in value):
                raise ValueError(f"Delay must be greater or equal to zero (not {value[0]} and {value[1]})")

            if value[0] > value[1]:
                raise ValueError("First tuple entry must be smaller or equal to second tuple entry")

        else:
            raise ValueError("Unsupported value type")

        self.__delay = value

    def _realize_delay(self) -> float:
        if isinstance(self.delay, float):
            return self.delay

        if isinstance(self.delay, tuple):
            return self._rng.uniform(self.delay[0], self.delay[1])

        raise RuntimeError("Unsupported type of delay")

    def _init_realization(self, *args, **kwargs) -> RandomDelayChannelRealization:
        return RandomDelayChannelRealization(*args, **kwargs)

    def recall_realization(self, group: Group) -> RandomDelayChannelRealization:
        return RandomDelayChannelRealization.From_HDF(group, self.alpha_device, self.beta_device)
