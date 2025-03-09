# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Generic, Set, TypeVar, TYPE_CHECKING
from typing_extensions import override, overload

import numpy as np

from hermespy.core import (
    AntennaArrayState,
    SignalBlock,
    DeserializationProcess,
    DeviceOutput,
    RandomNode,
    SerializableEnum,
    SerializationProcess,
    Signal,
    Transformation,
    ChannelStateInformation,
    Serializable,
)

if TYPE_CHECKING:
    from hermespy.simulation import (
        SimulatedDeviceState,
        SimulatedDevice,
        SimulationScenario,
    )  # pragma: no cover

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class InterpolationMode(SerializableEnum):
    """Interpolation behaviour for sampling and resampling routines.

    Considering a complex time series

    .. math::

       \\mathbf{s} = \\left[s_{0}, s_{1},\\,\\dotsc, \\, s_{M-1} \\right]^{\\mathsf{T}} \\in \\mathbb{C}^{M} \\quad \\text{with} \\quad s_{m} = s(\\frac{m}{f_{\\mathrm{s}}})

    sampled at rate :math:`f_{\\mathrm{s}}`, so that each sample
    represents a discrete sample of a time-continuous underlying function :math:`s(t)`.

    Given only the time-discrete sample vector :math:`\\mathbf{s}`,
    resampling refers to

    .. math::

       \\hat{s}(\\tau) = \\mathscr{F} \\left\\lbrace \\mathbf{s}, \\tau \\right\\rbrace

    estimating a sample of the original time-continuous function at time :math:`\\tau` given only the discrete-time sample vector :math:`\\mathbf{s}`.
    """

    NEAREST = 0
    """Interpolate to the nearest sampling instance.

    .. math::

       \\hat{s}(\\tau) = s_{\\lfloor \\tau f_{\\mathrm{s}} \\rfloor}

    Very fast, but not very accurate.
    """

    SINC = 1
    """Interpolate using sinc kernels.

    Also known as the Whittaker-Kotel'nikov-Shannon interpolation formula :footcite:p:`2002:meijering`.

    .. math::

       \\hat{s}(\\tau) = \\sum_{m=0}^{M-1} s_{m} \\operatorname{sinc} \\left( \\tau f_{\\mathrm{s}} - m \\right)

    Perfect for bandlimited signals, not very fast.
    """


CST = TypeVar("CST", bound="ChannelSample")
"""Type of channel sample."""

CRT = TypeVar("CRT", bound="ChannelRealization")
"""Type of channel realization."""

CT = TypeVar("CT", bound="Channel")
"""Type of channel."""


class ChannelSampleHook(Generic[CST]):
    """Hook for a callback to be called after a specific channel sample is generated."""

    __callback: Callable[[CST], None]
    __transmitter: SimulatedDevice | None
    __receiver: SimulatedDevice | None

    def __init__(
        self,
        callback: Callable[[CST], None],
        transmitter: SimulatedDevice | None,
        receiver: SimulatedDevice | None,
    ) -> None:
        """
        Args:

            callback (Callable[[CST], None]):
                Function to be called after the channel is sampled.

            transmitter (SimulatedDevice, optional):
                Transmitter device the hook is associated with.

            receiver (SimulatedDevice, optional):
                Receiver device the hook is associated with.
        """

        # Initialize class attributes
        self.__callback = callback
        self.__transmitter = transmitter
        self.__receiver = receiver

    def __call__(
        self, sample: CST, transmitter: SimulatedDevice | int, receiver: SimulatedDevice | int
    ) -> None:
        """Call the hook with the given sample.

        Args:

            sample (CST): The channel sample to be processed.
            transmitter (SimulatedDevice | int): The transmitter device the sample is associated with.
            receiver (SimulatedDevice | int): The receiver device the sample is associated with.
        """

        _transmitter = transmitter if isinstance(transmitter, int) else id(transmitter)
        _receiver = receiver if isinstance(receiver, int) else id(receiver)

        # Abort if the hook is associated with a specific device and the sample does not match
        if self.__transmitter is not None and _transmitter != id(self.__transmitter):
            return
        if self.__receiver is not None and _receiver != id(self.__receiver):
            return

        # Call the hook
        self.__callback(sample)


class LinkState(object):
    """Physical paramters of wireless channel link in time and space."""

    __transmitter: SimulatedDeviceState
    __receiver: SimulatedDeviceState
    __carrier_frequency: float
    __bandwidth: float
    __time: float

    def __init__(
        self,
        transmitter: SimulatedDeviceState,
        receiver: SimulatedDeviceState,
        carrier_frequency: float,
        bandwidth: float,
        time: float,
    ) -> None:
        """
        Args:

            transmitter (DeviceState):
                State of the transmitting device at the time of sampling.

            receiver (DeviceState):
                State of the receiving device at the time of sampling.

            carrier_frequency (float):
                Carrier frequency of the channel in Hz.

            bandwidth (float):
                Bandwidth of the propagated signal in Hz.

            time (float):
                Time of the channel state information in seconds.
        """

        # Initialize class attributes
        self.__transmitter = transmitter
        self.__receiver = receiver
        self.__carrier_frequency = carrier_frequency
        self.__bandwidth = bandwidth
        self.__time = time

    @property
    def transmitter(self) -> SimulatedDeviceState:
        """State of the transmitting device at the time of sampling."""

        return self.__transmitter

    @property
    def receiver(self) -> SimulatedDeviceState:
        """State of the receiving device at the time of sampling."""

        return self.__receiver

    @property
    def carrier_frequency(self) -> float:
        """Carrier frequency of the channel in Hz."""

        return self.__carrier_frequency

    @property
    def bandwidth(self) -> float:
        """Bandwidth of the propagated signal in Hz."""

        return self.__bandwidth

    @property
    def time(self) -> float:
        """Time of the channel state information in seconds."""

        return self.__time


class ChannelSample(object):
    """Immutable sample of a wireless channel model in time and space.

    Channel samples represent the channel state at a given point in time and
    two distinct observation points at transmitter and receiver in three-dimensional space.
    """

    __state: LinkState

    def __init__(self, state: LinkState) -> None:
        """
        Args:

            state (ChannelState):
                State of the channel at the time of sampling.
        """

        # Initialize class attributes
        self.__state = state

    @property
    def transmitter_state(self) -> SimulatedDeviceState:
        """State of the transmitting device at the time of sampling."""

        return self.__state.transmitter

    @property
    def receiver_state(self) -> SimulatedDeviceState:
        """State of the receiving device at the time of sampling."""

        return self.__state.receiver

    @property
    def transmitter_pose(self) -> Transformation:
        """Global position and orientation of the transmitter."""

        return self.__state.transmitter.pose

    @property
    def receiver_pose(self) -> Transformation:
        """Global position and orientation of the receiver."""

        return self.__state.receiver.pose

    @property
    def transmitter_velocity(self) -> np.ndarray:
        """Velocity of the transmitter in m/s."""

        return self.__state.transmitter.velocity

    @property
    def receiver_velocity(self) -> np.ndarray:
        """Velocity of the receiver in m/s."""

        return self.__state.receiver.velocity

    @property
    def transmitter_antennas(self) -> AntennaArrayState:
        """Antenna array model of the transmitter."""

        return self.__state.transmitter.antennas

    @property
    def receiver_antennas(self) -> AntennaArrayState:
        """Antenna array model of the receiver."""

        return self.__state.receiver.antennas

    @property
    def num_transmit_antennas(self) -> int:
        """Number of antennas at the transmitter."""

        return self.__state.transmitter.antennas.num_transmit_antennas

    @property
    def num_receive_antennas(self) -> int:
        """Number of antennas at the receiver."""

        return self.__state.receiver.antennas.num_receive_antennas

    @property
    def carrier_frequency(self) -> float:
        """Carrier frequency of the channel in Hz."""

        return self.__state.carrier_frequency

    @property
    def bandwidth(self) -> float:
        """Bandwidth of the propagated signal in Hz."""

        return self.__state.bandwidth

    @property
    def time(self) -> float:
        """Time of the channel state information in seconds."""

        return self.__state.time

    @property
    @abstractmethod
    def expected_energy_scale(self) -> float:
        """Expected linear scaling of a propagated signal's energy at each receiving antenna.

        Required to compute the expected energy of a signal after propagation,
        and therfore signal-to-noise ratios (SNRs) and signal-to-interference-plus-noise ratios (SINRs).
        """
        ...  # pragma: no cover

    @abstractmethod
    def _propagate(self, signal: SignalBlock, interpolation: InterpolationMode) -> SignalBlock:
        """Propagate radio-frequency band signals over a channel instance.

        Abstract subroutine of :meth:`propagate<ChannelSample.propagate>`.

        Args:

            signal (SignalBlock):
                The signal block to be propagated.

            interpolation (InterpolationMode):
                Interpolation behaviour of the channel realization's delay components with respect to the proagated signal's sampling rate.

        Returns: The propagated signal.
        """
        ...  # pragma: no cover

    def propagate(
        self: CST,
        signal: DeviceOutput | Signal,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> Signal:
        """Propagate a signal model over this realization.

        Let

        .. math::

           \\mathbf{X} = \\left[ \\mathbf{x}^{(0)}, \\mathbf{x}^{(1)},\\, \\dots,\\, \\mathbf{x}^{(M_\\mathrm{Tx} - 1)} \\right] \\in \\mathbb{C}^{N_\\mathrm{Tx} \\times M_\\mathrm{Tx}}

        be the `signal` transmitted by `transmitter` and

        .. math::

           \\mathbf{Y} = \\left[ \\mathbf{y}^{(0)}, \\mathbf{y}^{(1)},\\, \\dots,\\, \\mathbf{x}^{(M_\\mathrm{Rx} - 1)} \\right] \\in \\mathbb{C}^{N_\\mathrm{Rx} \\times M_\\mathrm{Rx}}

        the reception of `receiver`, this method implements the channel propagation equation

        .. math::

           \\mathbf{y}^{(m)} = \\sum_{\\tau = 0}^{m} \\mathbf{H}^{(m, \\tau)} \mathbf{x}^{(m-\\tau)} \\ \\text{.}

        It wraps :meth:`._propagate`, applies the channel :attr:`.gain` and returns a :class:`ChannelPropagation<hermespy.channel.channel.ChannelPropagation>` instance.
        If not specified, the transmitter and receiver are assumed to be the devices linked by the channel instance that generated this realization,
        meaning the transmitter is :attr:`alpha_device<.alpha_device>` and receiver is :attr:`beta_device<.beta_device>`.

        Args:

            signal (DeviceOutput | Signal):
                Signal model to be propagated.

            interpolation_mode (InterpolationMode, optional):
                Interpolation behaviour of the channel realization's delay components with respect to the proagated signal's sampling rate.
                If not specified, an integer rounding to the nearest sampling instance will be assumed.

        Returns: All information generated by the propagation.
        """

        # Convert signal argument to signal model
        if isinstance(signal, DeviceOutput):
            _signal = signal.mixed_signal

        elif isinstance(signal, Signal):
            _signal = signal

        else:
            raise ValueError("Signal is of unsupported type")

        # Assert that the signal's number of streams matches the number of antennas of the transmitter
        if _signal.num_streams != self.num_transmit_antennas:
            raise ValueError(
                f"Number of signal streams to be propagated does not match the number of transmitter antennas ({_signal.num_streams} != {self.num_transmit_antennas}))"
            )

        # Propagate each signal block
        signal_blocks_propagated = [self._propagate(b, interpolation_mode) for b in _signal]
        return _signal.Create(signal_blocks_propagated, self.bandwidth, self.carrier_frequency)

    @abstractmethod
    def state(
        self,
        num_samples: int,
        max_num_taps: int,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> ChannelStateInformation:
        """Generate the discrete channel state information from this channel realization.

        Denoted by

        .. math::

           \\mathbf{H}^{(m, \\tau)} \\in \\mathbb{C}^{N_{\\mathrm{Rx}} \\times N_{\\mathrm{Tx}}}

        within the respective equations.

        Args:

            num_samples (int):
                Number of discrete time-domain samples of the chanel state information.

            max_num_taps (int):
                Maximum number of delay taps considered per discrete time-domain sample.

            interpolation_mode (InterpolationMode, optional):
                Interpolation behaviour of the channel realization's delay components with respect to the proagated signal's sampling rate.
                If not specified, an integer rounding to the nearest sampling instance will be assumed.

        Returns: The channel state information representing this channel realization.
        """
        ...  # pragma: no cover


class ChannelRealization(Serializable, Generic[CST]):
    """Realization of a wireless channel channel model.

    Channel realizations represent a realization of all random processes of a wireless channel model.
    They are generated by the :meth:`realize()<Channel.realize>` method of :class:`.Channel` instances.
    """

    _DEFAULT_GAIN = 1.0

    __sample_hooks: Set[ChannelSampleHook[CST]]

    def __init__(
        self, sample_hooks: Set[ChannelSampleHook[CST]] | None = None, gain: float = _DEFAULT_GAIN
    ) -> None:
        """
        Args:

            sample_hooks (Set[ChannelSampleHook[CST]], optional):
                Hooks to be called after the channel is sampled.

            gain (float, optional):
                Linear power gain factor a signal experiences when being propagated over this realization.
                :math:`1.0` by default, meaning no gain or loss.
        """

        # Initialize class attributes
        self.__sample_hooks = set() if sample_hooks is None else sample_hooks
        self.__gain = gain

    @property
    def sample_hooks(self) -> Set[ChannelSampleHook[CST]]:
        """Hooks to be called after the channel is sampled."""

        return self.__sample_hooks.copy()

    @property
    def gain(self) -> float:
        """Linear power gain factor a signal experiences when being propagated over this realization."""

        return self.__gain

    @overload
    def sample(
        self,
        transmitter: SimulatedDeviceState,
        receiver: SimulatedDeviceState,
        carrier_frequency: float | None = None,
        bandwidth: float | None = None,
    ) -> CST:
        """Sample the channel realization at a given point in time and space.

        Wrapper around :meth:`._sample` that converts the input arguments to the correct type.

        Args:

            transmitter (DeviceState):
                State of the transmitting device at the time of sampling.

            receiver (DeviceState):
                State of the receiving device at the time of sampling.

            carrier_frequency (float, optional):
                Carrier frequency of the channel in Hz.
                If not specified, the transmitting device's carrier frequency will be assumed.

            bandwidth (float, optional):
                Bandwidth of the propagated signal in Hz.
                If not specified, the transmitting device's sampling rate will be assumed.

        Returns: The channel sample for the given configuration.
        """
        ...  # pragma: no cover

    @overload
    def sample(
        self,
        transmitter: SimulatedDevice,
        receiver: SimulatedDevice,
        timestamp: float = 0.0,
        carrier_frequency: float | None = None,
        bandwidth: float | None = None,
    ) -> CST:
        """Sample the channel realization at a given point in time and space.

        Wrapper around :meth:`._sample` that converts the input arguments to the correct type.

        Args:

            transmitter (SimulatedDevice):
                Transmitting device feeding into the channel model to be sampled.

            receiver (SimulatedDevice):
                Receiving device observing the channel model to be sampled.

            timestamp (float, optional):
                Time at which the channel is sampled in seconds.

            carrier_frequency (float, optional):
                Carrier frequency of the channel in Hz.
                If not specified, the transmitting device's carrier frequency will be assumed.

            bandwidth (float, optional):
                Bandwidth of the propagated signal in Hz.
                If not specified, the transmitting device's sampling rate will be assumed.

        Returns: The channel sample at the given point in time.
        """
        ...  # pragma: no cover

    def sample(
        self,
        transmitter: SimulatedDevice | SimulatedDeviceState,
        receiver: SimulatedDevice | SimulatedDeviceState,
        *args,
        **kwargs,
    ) -> CST:
        from hermespy.simulation import SimulatedDevice, SimulatedDeviceState

        if isinstance(transmitter, SimulatedDevice) and isinstance(receiver, SimulatedDevice):
            timestamp = float(args[0]) if len(args) > 0 else 0.0
            carrier_frequency = float(args[1]) if len(args) > 1 else None
            bandwidth = float(args[2]) if len(args) > 2 else None

            transmitter_device = id(transmitter)
            receiver_device = id(receiver)
            transmitter_state = transmitter.state(timestamp)
            receiver_state = receiver.state(timestamp)

        elif isinstance(transmitter, SimulatedDeviceState) and isinstance(
            receiver, SimulatedDeviceState
        ):
            timestamp = 0.0
            carrier_frequency = float(args[0]) if len(args) > 0 else None
            bandwidth = float(args[1]) if len(args) > 1 else None

            transmitter_device = transmitter.device_id
            receiver_device = receiver.device_id
            transmitter_state = transmitter
            receiver_state = receiver

        else:
            raise ValueError("Invalid input argument types for channel sampling.")

        _carrier_frequency = (
            carrier_frequency
            if carrier_frequency is not None
            else transmitter_state.carrier_frequency
        )
        _bandwidth = bandwidth if bandwidth is not None else transmitter_state.sampling_rate
        state = LinkState(
            transmitter_state, receiver_state, _carrier_frequency, _bandwidth, timestamp
        )

        # Generate a new sample
        sample = self._sample(state)

        # Notify the registered hooks
        for hook in self.sample_hooks:
            hook(sample, transmitter_device, receiver_device)

        return sample

    @abstractmethod
    def _sample(self, state: LinkState) -> CST:
        """Sample the channel realization at a given point in time and space.

        Abstract subroutine of :meth:`sample<hermespy.channel.channel.ChannelRealization.sample>`.

        Args:

            state (LinkState):
                State of the channel at the time of sampling.

        Returns: The channel sample at the given point in time.
        """
        ...  # pragma: no cover

    @overload
    def reciprocal_sample(
        self,
        sample: CST,
        transmitter: SimulatedDeviceState,
        receiver: SimulatedDeviceState,
        carrier_frequency: float | None = None,
        bandwidth: float | None = None,
    ) -> CST: ...  # pragma: no cover

    @overload
    def reciprocal_sample(
        self,
        sample: CST,
        transmitter: SimulatedDevice,
        receiver: SimulatedDevice,
        timestamp: float = 0.0,
        carrier_frequency: float | None = None,
        bandwidth: float | None = None,
    ) -> CST:
        """Sample the reciprocal channel realization at a given point in time and space.

        Wrapper around :meth:`._reciprocal_sample` that converts the input arguments to the correct type.

        Args:

            sample (CST):
                Channel sample to be reciprocally sampled.

            transmitter (SimulatedDevice):
                Transmitting device in the reciprocal channel at the time of sampling.

            receiver (SimulatedDevice):
                Receiving device in the reciprocal channel  at the time of sampling.

            timestamp (float, optional):
                Time at which the channel is sampled in seconds.
                Zero by default.

            carrier_frequency (float, optional):
                Carrier frequency of the channel in Hz.
                If not specified, the transmitting device's carrier frequency will be assumed.

            bandwidth (float, optional):
                Bandwidth of the propagated signal in Hz.
                If not specified, the transmitting device's sampling rate will be assumed.

        Returns: The channel sample at the given point in time.
        """
        ...  # pragma: no cover

    def reciprocal_sample(
        self,
        sample: CST,
        transmitter: SimulatedDevice | SimulatedDeviceState,
        receiver: SimulatedDevice | SimulatedDeviceState,
        *args,
        **kwargs,
    ) -> CST:
        from hermespy.simulation import SimulatedDevice, SimulatedDeviceState

        # Type hinting, required for proper type checking
        transmitter_device: SimulatedDevice | int
        receiver_device: SimulatedDevice | int

        if isinstance(transmitter, SimulatedDevice) and isinstance(receiver, SimulatedDevice):
            timestamp = float(args[0]) if len(args) > 0 else 0.0
            carrier_frequency = float(args[1]) if len(args) > 1 else None
            bandwidth = float(args[2]) if len(args) > 2 else None

            transmitter_device = transmitter
            receiver_device = receiver
            transmitter_state = transmitter.state(timestamp)
            receiver_state = receiver.state(timestamp)

        elif isinstance(transmitter, SimulatedDeviceState) and isinstance(
            receiver, SimulatedDeviceState
        ):
            timestamp = 0.0
            carrier_frequency = float(args[0]) if len(args) > 0 else None
            bandwidth = float(args[1]) if len(args) > 1 else None

            transmitter_device = transmitter.device_id
            receiver_device = receiver.device_id
            transmitter_state = transmitter
            receiver_state = receiver

        else:
            raise ValueError("Invalid input argument types for channel sampling.")

        _carrier_frequency = (
            carrier_frequency
            if carrier_frequency is not None
            else transmitter_state.carrier_frequency
        )
        _bandwidth = bandwidth if bandwidth is not None else transmitter_state.sampling_rate

        state = LinkState(
            transmitter_state, receiver_state, _carrier_frequency, _bandwidth, timestamp
        )

        # Generate a new sample
        reciprocal_sample = self._reciprocal_sample(sample, state)

        # Notify the registered hooks
        for hook in self.sample_hooks:
            hook(reciprocal_sample, transmitter_device, receiver_device)

        return reciprocal_sample

    @abstractmethod
    def _reciprocal_sample(self, sample: CST, state: LinkState) -> CST:
        """Sample the reciprocal channel realization at a given point in time and space.

        Abstract subroutine of :meth:`reciprocal_sample()<ChannelRealization.reciprocal_sample>`.

        Args:

            sample (CST):
                Channel sample to be reciprocally sampled.

            state (LinkState):
                Physical state of the channel at the time of sampling.

        Returns: The channel sample at the given point in time.
        """
        ...  # pragma: no cover

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.__gain, "gain")

    @classmethod
    def _DeserializeParameters(cls, process: DeserializationProcess) -> dict[str, object]:
        return {"gain": process.deserialize_floating("gain", cls._DEFAULT_GAIN)}


class Channel(ABC, RandomNode, Serializable, Generic[CRT, CST]):
    """Abstract base class of all channel models.

    Channel models represent the basic physical properties of a elemtromagnetic waves propagating through space in between devices.
    """

    _DEFAULT_GAIN = 1.0

    __scenario: SimulationScenario
    __gain: float
    __sample_hooks: Set[ChannelSampleHook[CST]]

    def __init__(self, gain: float = _DEFAULT_GAIN, seed: int | None = None) -> None:
        """
        Args:

            gain (float, optional):
                Linear channel energy gain factor.
                Initializes the :meth:`gain<gain>` property.
                :math:`1.0` by default.

            seed (int, optional):
                Seed used to initialize the pseudo-random number generator.
        """

        # Initialize base classes
        # Must be first in order for correct diamond resolve
        Serializable.__init__(self)
        RandomNode.__init__(self, seed=seed)

        # Default parameters
        self.gain = gain
        self.__scenario = None
        self.__sample_hooks = set()

    @property
    def scenario(self) -> SimulationScenario | None:
        """Simulation scenario the channel belongs to.

        Handle to the :class:`Scenario <hermespy.simulation.simulation.SimulationScenario>` this channel is asigned to.
        :py:obj:`None` if the channel is not part of any specific :class:`Scenario <hermespy.simulation.simulation.SimulationScenario>`.

        The recommended way to set the scenario is by calling the :meth:`set_channel<hermespy.simulation.simulation.    SimulationScenario.set_channel>` method:

        .. code-block:: python

           from hermespy.simulation import SimulationScenario

           scenario = SimulationScenario()
           alpha_device = scenario.new_device()
           beta_device = scenario.new_device()

           channel = Channel()
           scenario.set_channel(alpha_device, beta_device, channel)
        """

        return self.__scenario

    @scenario.setter
    def scenario(self, value: SimulationScenario) -> None:
        self.__scenario = value
        self.random_mother = value

    @property
    def gain(self) -> float:
        """Linear channel power gain factor.

        The default channel gain is 1.
        Realistic physical channels should have a gain less than one.

        For configuring logarithmic gains, set the attribute using the dB shorthand:

        .. code-block:: python

           from hermespy.core import dB

           # Configure a 10 dB gain
           channel.gain = dB(10)

        Raises:

            ValueError: For gains smaller than zero.
        """

        return self.__gain

    @gain.setter
    def gain(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Channel gain must be greater or equal to zero")

        self.__gain = value

    @property
    def sample_hooks(self) -> Set[ChannelSampleHook[CST]]:
        """Hooks to be called after a channel sample is generated."""

        return self.__sample_hooks.copy()

    def add_sample_hook(
        self,
        callback: Callable[[CST], None],
        transmitter: SimulatedDevice | None = None,
        receiver: SimulatedDevice | None = None,
    ) -> ChannelSampleHook[CST]:
        """Add a hook to be called after a channel sample is generated.

        Args:

            callback (Callable[[CST], None]):
                Function to be called after the channel is sampled.

            transmitter (SimulatedDevice, optional):
                Transmitter device the hook is associated with.
                If not specified the hook will be called for all transmitters.

            receiver (SimulatedDevice, optional):
                Receiver device the hook is associated with.
                If not specified the hook will be called for all receivers.
        """

        hook = ChannelSampleHook(callback, transmitter, receiver)
        self.__sample_hooks.add(hook)
        return hook

    def remove_sample_hook(self, hook: ChannelSampleHook[CST]) -> None:
        """Remove a hook from the list of hooks to be called after a channel sample is generated.

        Args:

            hook (ChannelSampleHook[CST], None]):
                Hook to be removed from the list of hooks.
        """

        self.__sample_hooks.discard(hook)

    @abstractmethod
    def _realize(self) -> CRT:
        """Generate a new channel realzation.

        Abstract subroutine of :meth:`realize<hermespy.channel.channel.Channel.realize>`.
        Each :class:`Channel<hermespy.channel.channel.Channel>` is required to implement their own
        :meth:`._realize` method.

        Returns: A new channel realization.
        """
        ...  # pragma: no cover

    def realize(self, cache: bool = True) -> CRT:
        """Generate a new channel realization.

        If `cache` is enabled, :attr:`.realization` will be updated
        to the newly generated :class:`ChannelRealization<hermespy.channel.channel.ChannelRealization>`.

        Args:

            cache (bool, optional):
                Cache the realization. Enabled by default.

        Returns: A new channel realization.
        """

        # Generate a new realization
        realization = self._realize()

        # Cache if the respective flag is enabled
        if cache:
            self.__last_realization = realization

        return realization

    @property
    def realization(self) -> CRT | None:
        """The last realization used for channel propagation.

        Updated every time :meth:`.propagate` or :meth:`.realize` are called and `cache` is enabled.
        :py:obj:`None` if :meth:`.realize` has not been called yet.
        """

        return self.__last_realization

    def propagate(
        self,
        signal: DeviceOutput | Signal,
        transmitter: SimulatedDevice,
        receiver: SimulatedDevice,
        timestamp: float = 0.0,
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> Signal:
        """Propagate radio-frequency band signals over this channel.

        Generates a new channel realization by calling :meth:`realize<.realize>` and propagates the provided signal over it.

        Let

        .. math::

           \\mathbf{X} = \\left[ \\mathbf{x}^{(0)}, \\mathbf{x}^{(1)},\\, \\dots,\\, \\mathbf{x}^{(M_\\mathrm{Tx} - 1)} \\right] \\in \\mathbb{C}^{N_\\mathrm{Tx} \\times M_\\mathrm{Tx}}

        be the `signal` transmitted by `transmitter` and

        .. math::

           \\mathbf{Y} = \\left[ \\mathbf{y}^{(0)}, \\mathbf{y}^{(1)},\\, \\dots,\\, \\mathbf{x}^{(M_\\mathrm{Rx} - 1)} \\right] \\in \\mathbb{C}^{N_\\mathrm{Rx} \\times M_\\mathrm{Rx}}

        the reception of `receiver`, this method implements the channel propagation equation

        .. math::

           \\mathbf{y}^{(m)} = \\sum_{\\tau = 0}^{m} \\mathbf{H}^{(m, \\tau)} \mathbf{x}^{(m-\\tau)} \\ \\text{.}

        Args:

            signal (DeviceOutput | Signal):
                Signal models emitted by `transmitter` associated with this wireless channel model.

            transmitter (SimulatedDevice):
                Device transmitting the `signal` to be propagated over this realization.

            receiver (SimulatedDevice):
                Device receiving the propagated `signal` after propagation.

            timestamp (float, optional):
                Time at which the signal is propagated in seconds.
                Defaults to 0.0.

            interpolation_mode (InterpolationMode, optional):
                Interpolation behaviour of the channel realization's delay components with respect to the proagated signal's sampling rate.

        Returns: The channel propagation resulting from the signal propagation.
        """

        # Generate a new realization
        realization = self.realize()

        # Sample the channel realization
        sample: ChannelSample = realization.sample(
            transmitter, receiver, timestamp, signal.carrier_frequency, signal.sampling_rate
        )

        # Propagate the provided signal
        propagation = sample.propagate(signal, interpolation_mode)

        # Return resulting propagation
        return propagation

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.__gain, "gain")
        if self.seed is not None:
            process.serialize_integer(self.seed, "seed")

    @classmethod
    def _DeserializeParameters(cls, process: DeserializationProcess) -> dict[str, object]:
        return {
            "gain": process.deserialize_floating("gain", cls._DEFAULT_GAIN),
            "seed": process.deserialize_integer("seed", None),
        }
