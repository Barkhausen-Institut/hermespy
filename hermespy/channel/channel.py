# -*- coding: utf-8 -*-
"""
================
Channel Modeling
================
"""

from __future__ import annotations
from enum import Enum
from typing import Generic, List, Optional, Tuple, TypeVar, Union, TYPE_CHECKING
from itertools import chain, product

import numpy as np

from hermespy.core import DeviceOutput, RandomNode, Signal, ChannelStateInformation
from hermespy.core.factory import Serializable
from hermespy.core.channel_state_information import ChannelStateFormat

if TYPE_CHECKING:
    from hermespy.simulation import SimulatedDevice, SimulationScenario

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PropagationDirection(Enum):
    """Channel signal propagation direction"""

    FORWARDS = 0
    BACKWARDS = 1


class ChannelRealization(ChannelStateInformation):
    """Realization of a wireless channel channel model."""

    __channel: Channel

    def __init__(self,
                 channel: Channel,
                 impulse_response: np.ndarray) -> None:
        """
        Args:

            channel (Channel): Channel from which the impulse response was generated.
            impulse_response (np.ndarray): The channel impulse response.
        """

        if impulse_response.ndim != 4:
            raise ValueError("Channel impulse response must be four-dimensional numpy tensor")

        self.__channel = channel
        ChannelStateInformation.__init__(self, ChannelStateFormat.IMPULSE_RESPONSE, impulse_response)

    @property
    def channel(self) -> Channel:
        """The channel from which the impulse response was generated.

        Returns: Handle to the channel instance.
        """

        return self.__channel
    
    def reciprocal(self) -> ChannelRealization:
        
        return ChannelRealization(self.channel, ChannelStateInformation.reciprocal(self).state)


ChannelRealizationType = TypeVar('ChannelRealizationType', bound=ChannelRealization)
"""Type of channel realization"""


class Channel(RandomNode, Serializable, Generic[ChannelRealizationType]):
    """An ideal distortion-less channel.

    It also serves as a base class for all other channel models.

    For MIMO systems, the received signal is the addition of the signal transmitted at all
    antennas.
    The channel will provide `number_rx_antennas` outputs to a signal
    consisting of `number_tx_antennas` inputs. Depending on the channel model,
    a random number generator, given by `rnd` may be needed. The sampling rate is
    the same at both input and output of the channel, and is given by `sampling_rate`
    samples/second.
    """

    yaml_tag: str = "Channel"
    serialized_attributes = {"impulse_response_interpolation"}

    __active: bool
    __transmitter: Optional[SimulatedDevice]
    __receiver: Optional[SimulatedDevice]
    __scenario: SimulationScenario
    __gain: float
    __sync_offset_low: float
    __sync_offset_high: float
    __last_realization: Optional[ChannelRealizationType]
    impulse_response_interpolation: bool

    def __init__(self, transmitter: Optional[SimulatedDevice] = None, receiver: Optional[SimulatedDevice] = None, devices: Optional[Tuple[SimulatedDevice, SimulatedDevice]] = None, active: Optional[bool] = None, gain: Optional[float] = None, sync_offset_low: float = 0.0, sync_offset_high: float = 0.0, impulse_response_interpolation: bool = True, seed: Optional[int] = None) -> None:
        """
        Args:

            transmitter (Transmitter, optional):
                The device transmitting into this channel.

            receiver (Receiver, optional):
                The device receiving from this channel.

            devices (Tuple[SimulatedDevice, SimulatedDevice], optional):
                Tuple of devices connected by this channel model.

            active (bool, optional):
                Channel activity flag.
                Activated by default.

            gain (float, optional):
                Channel power gain.
                1.0 by default.

            sync_offset_low (float, optional):
                Minimum synchronization error in seconds.

            sync_offset_high (float, optional):
                Maximum synchronization error in seconds.

            impulse_response_interpolation (bool, optional):
                Allow the impulse response to be interpolated during sampling.

            seed (int, optional):
                Seed used to initialize the pseudo-random number generator.
        """

        # Initialize base classes
        # Must be first in order for correct diamond resolve
        Serializable.__init__(self)
        RandomNode.__init__(self, seed=seed)

        # Default parameters
        self.__active = True
        self.__transmitter = None
        self.__receiver = None
        self.__gain = 1.0
        self.__scenario = None
        self.sync_offset_low = sync_offset_low
        self.sync_offset_high = sync_offset_high
        self.__last_realization = None
        self.impulse_response_interpolation = impulse_response_interpolation

        if transmitter is not None:
            self.transmitter = transmitter

        if receiver is not None:
            self.receiver = receiver

        if devices is not None:

            if self.receiver is not None or self.transmitter is not None:
                raise ValueError("Can't use 'devices' initialization argument in combination with specifying a transmitter / receiver")

            self.transmitter = devices[0]
            self.receiver = devices[1]

        if active is not None:
            self.active = active

        if gain is not None:
            self.gain = gain

    @property
    def active(self) -> bool:
        """Access channel activity flag.

        Returns:
            bool:
                Is the channel currently activated?
        """

        return self.__active

    @active.setter
    def active(self, active: bool) -> None:
        """Modify the channel activity flag.

        Args:
            active (bool):
                Is the channel currently activated?
        """

        self.__active = active

    @property
    def transmitter(self) -> SimulatedDevice:
        """SimulatedDevice transmitting into this channel.

        Returns:
            Transmitter: A handle to the modem transmitting into this channel.

        Raises:
            RuntimeError: If a transmitter is already configured.
        """

        return self.__transmitter

    @transmitter.setter
    def transmitter(self, value: SimulatedDevice) -> None:
        """Set the device transmitting into this channel."""

        # if self.__transmitter is not None:
        #    raise RuntimeError("Overwriting a transmitter configuration is not supported")

        self.__transmitter = value

    @property
    def receiver(self) -> SimulatedDevice:
        """SimulatedDevice receiving from this channel.

        Returns:
            Receiver: A handle to the device receiving from this channel.

        Raises:
            RuntimeError: If a receiver is already configured.
        """

        return self.__receiver

    @receiver.setter
    def receiver(self, value: SimulatedDevice) -> None:
        """Set the device receiving from this channel."""

        # if self.__receiver is not None:
        #    raise RuntimeError("Overwriting a receiver configuration is not supported")

        self.__receiver = value

    @property
    def scenario(self) -> Optional[SimulationScenario]:
        """Simulation scenario the channel belongs to.

        Returns:
            Handle to the :class:`Scenario <hermespy.simulation.simulation.SimulationScenario>`.
            `None` if the channel is considered floating.
        """

        return self.__scenario

    @scenario.setter
    def scenario(self, value: SimulationScenario) -> None:

        self.__scenario = value
        self.random_mother = value

    @property
    def sync_offset_low(self) -> float:
        """Synchronization error minimum.

        Returns:
            float: Minimum synchronization error in seconds.
        """

        return self.__sync_offset_low

    @sync_offset_low.setter
    def sync_offset_low(self, value: float) -> None:
        """Configure the synchronization error minimum.

        Args:
            value (float): Minimum synchronization error in seconds.

        Raises:
            ValueError: If `value` is smaller than zero.
        """

        if value < 0:
            raise ValueError("Synchronization offset lower bound must be greater or equal to zero")

        self.__sync_offset_low = value

    @property
    def sync_offset_high(self) -> float:
        """Synchronization error maximum.

        Returns:
            float: Maximum synchronization error in seconds.
        """

        return self.__sync_offset_high

    @sync_offset_high.setter
    def sync_offset_high(self, value: float) -> None:
        """Configure the synchronization error maximum.

        Args:
            value (float): Maximum synchronization error in seconds.

        Raises:
            ValueError: If `value` is smaller than zero.
        """

        if value < 0:
            raise ValueError("Synchronization offset upper bound must be greater or equal to zero")

        self.__sync_offset_high = value

    @property
    def gain(self) -> float:
        """Access the channel gain.

        The default channel gain is 1.
        Realistic physical channels should have a gain less than one.

        Returns:
            float:
                The channel gain.
        """

        return self.__gain

    @gain.setter
    def gain(self, value: float) -> None:
        """Modify the channel gain.

        Args:
            value (float):
                The new channel gain.

        Raises:
            ValueError: If gain is smaller than zero.
        """

        if value < 0.0:
            raise ValueError("Channel gain must be greater or equal to zero")

        self.__gain = value

    @property
    def num_inputs(self) -> int:
        """The number of streams feeding into this channel.

        Actually shadows the `num_streams` property of his channel's transmitter.

        Returns:
            int:
                The number of input streams.

        Raises:
            RuntimeError: If the channel is currently floating.
        """

        if self.__transmitter is None:
            raise RuntimeError("Error trying to access the number of inputs of a floating channel")

        return self.__transmitter.antennas.num_antennas

    @property
    def num_outputs(self) -> int:
        """The number of streams emerging from this channel.

        Actually shadows the `num_streams` property of his channel's receiver.

        Returns:
            int:
                The number of output streams.

        Raises:
            RuntimeError: If the channel is currently floating.
        """

        if self.__receiver is None:
            raise RuntimeError("Error trying to access the number outputs of a floating channel")

        return self.__receiver.antennas.num_antennas

    @staticmethod
    def __ensure_list(propagated_signal: Union[DeviceOutput, Signal, List[Signal], None]) -> List[Signal]:
        """Subroutine of :meth:`Channel.propagate`."""

        if isinstance(propagated_signal, DeviceOutput):
            return propagated_signal.emerging_signals
        
        if isinstance(propagated_signal, Signal):
            return [propagated_signal]
            
        if isinstance(propagated_signal, list):
            return propagated_signal
            
        if propagated_signal is None:
            return list()

        raise ValueError("Propagated signal is of unsupported type")

    def propagate(self,
                  forwards: Union[DeviceOutput, Signal, List[Signal], None] = None,
                  backwards: Union[DeviceOutput, Signal, List[Signal], None] = None,
                  realization: Optional[ChannelRealizationType] = None) -> Tuple[List[Signal], List[Signal], ChannelRealizationType]:
        """Propagate radio-frequency band signals over a channel instance.

        For the ideal channel in the base class, the MIMO channel is modeled as a matrix of ones.
        The routine samples a new impulse response, which will be converted to ChannelStateInformation.

        Args:

            forwards (Union[DeviceOutput, Signal, List[Signal]], optional):
                Signal models emitted by `device_alpha` associated with this wireless channel model.

            backwards (Union[Signal, List[Signal]], optional):
                Signal models emitted by `device_beta` associated with this wireless channel model.

            realization (ChannelRealizationType, optional):
                Channel realization over which to propagate the signals.
                If not specified, a new channel realization will be generated.

        Returns:

            Tuple[List[Signal], List[Signal], ChannelStateInformation]:

                forwards_receptions (List[Signal]):
                    Signal models impinging onto `device_beta` after channel propagation.

                backwards_receptions (List[Signal]):
                    Signal models impinging onto `device_alpha` after channel propagation.

                csi (ChannelStateInformation):
                    State of the channel during signal propagation.

        Raises:

            ValueError:
                If the number of streams in `forwards` is not one
                or the number of antennas in `device_alpha`.
                If the number of streams in `backwards` is not one
                or the number of antennas in `device_beta`.


            RuntimeError:
                If the scenario configuration is not supported by the default channel model.

            RuntimeError:
                If the channel is currently floating.
        """

        # Convert forwards and backwards transmissions to lists if required
        forwards = self.__ensure_list(forwards)
        backwards = self.__ensure_list(backwards)

        # Abort if the channel is considered floating, since physical device properties are required for
        # channel modeling
        if self.transmitter is None or self.receiver is None:
            raise RuntimeError("Channel is floating, making propagation simulation impossible")

        # Validate that the signal models contain the correct number of streams
        for signal in forwards:
            if signal.num_streams != self.transmitter.antennas.num_antennas:
                raise ValueError("Number of transmitted signal streams does not match number of transmit antennas")

        for signal in backwards:
            if signal.num_streams != self.receiver.antennas.num_antennas:
                raise ValueError("Number of transmitted signal streams does not match number of transmit antennas")

        # Determine the sampling rate and sample count of the CSI samples
        # For now, the sampling rate and sample count is the maximum over all provided signal models
        csi_sampling_rate = 0.0
        csi_num_samples = 0
        for signal in chain(forwards, backwards):

            csi_sampling_rate = max(csi_sampling_rate, signal.sampling_rate)
            csi_num_samples = max(csi_num_samples, signal.num_samples)

        # If the channel is inactive, propagation will result in signal loss
        # This is modeled by returning an zero-length signal and impulse-response (in time-domain) after propagation
        if not self.active:
            return [Signal.empty(csi_sampling_rate, self.receiver.num_antennas)], [Signal.empty(csi_sampling_rate, self.transmitter.num_antennas)], ChannelStateInformation.Ideal(self.num_outputs, self.num_inputs, 0)

        # Generate the channel's impulse response realization
        realization = self.realize(csi_num_samples, csi_sampling_rate) if realization is None else realization

        # Consider the a random synchronization offset between transmitter and receiver
        sync_offset: float = self._rng.uniform(low=self.__sync_offset_low, high=self.__sync_offset_high)

        # Compute the propgated signal samples for both channel directions
        forwards_receptions = [self.Propagate(signal.resample(csi_sampling_rate), realization, PropagationDirection.FORWARDS, sync_offset) for signal in forwards]
        backwards_receptions = [self.Propagate(signal.resample(csi_sampling_rate), realization, PropagationDirection.BACKWARDS, sync_offset) for signal in backwards]

        # Cache the realization and return results
        self.__last_realization = realization
        return forwards_receptions, backwards_receptions, realization

    @staticmethod
    def Propagate(signal: Signal,
                  realization: ChannelRealization,
                  direction: PropagationDirection = PropagationDirection.FORWARDS,
                  delay: float = 0.) -> Signal:
        """Propagate a single signal model given a specific channel realzation.

        Args:

            signal (Signal):
                Signal model to be propagated.

            realization (ChannelRealization):
                Channel realization over which the signal model should be propagated.

            direction (PropagationDirection, optional):
                Direction in which the propagation should be assumed.
                :class:`PropagationDirection.FORWARDS` by default.

            delay (float, optional):
                Additional delays, for example synchronization offsets.
                Zero by default.

        Returns: Propagated signal model.
        """

        realization = realization if direction is PropagationDirection.FORWARDS else realization.reciprocal()

        # Propagate the signal
        propagated_samples = np.zeros((realization.num_receive_streams, signal.num_samples + realization.num_delay_taps - 1), dtype=complex)

        for delay_index in range(realization.num_delay_taps):
            for tx_idx, rx_idx in product(range(realization.num_transmit_streams), range(realization.num_receive_streams)):

                delayed_signal = realization.state[rx_idx, tx_idx, :signal.num_samples, delay_index] * signal.samples[tx_idx, :]
                propagated_samples[rx_idx, delay_index : delay_index + signal.num_samples] += delayed_signal

        return Signal(propagated_samples, sampling_rate=signal.sampling_rate, carrier_frequency=signal.carrier_frequency, delay=signal.delay + delay)

    def realize(self,
                num_samples: int,
                sampling_rate: float) -> ChannelRealization:
        """Generate a new channel impulse response.

        Note that this is the core routine from which :meth:`.propagate` will create the channel state.

        Args:

            num_samples (int):
                Number of samples :math:`N` within the impulse response.

            sampling_rate (float):
                The rate at which the delay taps will be sampled, i.e. the delay resolution.

        Returns:
        
                Numpy aray representing the impulse response for all propagation paths between antennas.
                4-dimensional tensor of size :math:`M_\\mathrm{Rx} \\times M_\\mathrm{Tx} \\times N \\times (L+1)` where
                :math:`M_\\mathrm{Rx}` is the number of receiving antennas,
                :math:`M_\\mathrm{Tx}` is the number of transmitting antennas,
                :math:`N` is the number of propagated samples and
                :math:`L` is the maximum path delay (in samples).
                For the ideal  channel in the base class :math:`L = 0`.
        """

        if self.transmitter is None or self.receiver is None:
            raise RuntimeError("Channel is floating, making impulse response simulation impossible")

        # MISO case
        if self.receiver.antennas.num_antennas == 1:
            spatial_response = np.ones((1, self.transmitter.antennas.num_antennas), dtype=complex)

        # SIMO case
        elif self.transmitter.antennas.num_antennas == 1:
            spatial_response = np.ones((self.receiver.antennas.num_antennas, 1), dtype=complex)

        # MIMO case
        else:
            spatial_response = np.eye(self.receiver.antennas.num_antennas, self.transmitter.antennas.num_antennas, dtype=complex)

        # Scale by channel gain and add dimension for delay response
        impulse_response = self.gain * np.expand_dims(np.repeat(spatial_response[:, :, np.newaxis], num_samples, 2), axis=3)

        # Save newly generated response as most recent impulse response
        self.recent_response = impulse_response

        # Return resulting impulse response
        return ChannelRealization(self, impulse_response)

    @property
    def realization(self) -> Optional[ChannelRealizationType]:
        """The last realization used for channel propagation.

        Updated every time :meth:`.propagate` is called.

        Returns:
            The channel realization.
            `None` if :meth:`.propagate` has not been called yet.
        """

        return self.__last_realization

    @property
    def min_sampling_rate(self) -> float:
        """Minimal sampling rate required to adequately model the channel.

        Returns:
            float: The minimal sampling rate in Hz.
        """

        # Since the default channel is time-invariant, there are no sampling rate requirements
        return 0.0
