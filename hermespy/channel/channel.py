# -*- coding: utf-8 -*-
"""Channel model for wireless transmission links."""

from __future__ import annotations
from typing import Optional, Tuple, Type
from itertools import product

import numpy as np
from ruamel.yaml import SafeRepresenter, SafeConstructor, ScalarNode, MappingNode

from hermespy.core import Device, RandomNode
from hermespy.core.signal_model import Signal
from hermespy.core.channel_state_information import ChannelStateFormat, ChannelStateInformation


__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Channel(RandomNode):
    """Implements an ideal distortion-less channel.

    It also serves as a base class for all other channel models.

    For MIMO systems, the received signal is the addition of the signal transmitted at all
    antennas.
    The channel will provide `number_rx_antennas` outputs to a signal
    consisting of `number_tx_antennas` inputs. Depending on the channel model,
    a random number generator, given by `rnd` may be needed. The sampling rate is
    the same at both input and output of the channel, and is given by `sampling_rate`
    samples/second.
    """

    yaml_tag = u'Channel'
    yaml_matrix = True
    __active: bool
    __transmitter: Optional[Device]
    __receiver: Optional[Device]
    __gain: float
    __sync_offset_low: float
    __sync_offset_high: float
    impulse_response_interpolation: bool

    def __init__(self,
                 transmitter: Optional[Device] = None,
                 receiver: Optional[Device] = None,
                 active: Optional[bool] = None,
                 gain: Optional[float] = None,
                 sync_offset_low: float = 0.,
                 sync_offset_high: float = 0.,
                 impulse_response_interpolation: bool = True,
                 seed: Optional[int] = None) -> None:
        """Channel model initialization.

        Args:

            transmitter (Transmitter, optional):
                The modem transmitting into this channel.

            receiver (Receiver, optional):
                The modem receiving from this channel.

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

        # Initialize base class
        RandomNode.__init__(self, seed=seed)

        # Default parameters
        self.__active = True
        self.__transmitter = None
        self.__receiver = None
        self.__gain = 1.0
        self.__scenario = None
        self.sync_offset_low = sync_offset_low
        self.sync_offset_high = sync_offset_high
        self.recent_response = None
        self.impulse_response_interpolation = impulse_response_interpolation

        if transmitter is not None:
            self.transmitter = transmitter

        if receiver is not None:
            self.receiver = receiver

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
    def transmitter(self) -> Device:
        """Device transmitting into this channel.

        Returns:
            Transmitter: A handle to the modem transmitting into this channel.

        Raises:
            RuntimeError: If a transmitter is already configured.
        """

        return self.__transmitter

    @transmitter.setter
    def transmitter(self, value: Device) -> None:
        """Set the device transmitting into this channel."""

        if self.__transmitter is not None:
            raise RuntimeError("Overwriting a transmitter configuration is not supported")

        self.__transmitter = value

    @property
    def receiver(self) -> Device:
        """Device receiving from this channel.

        Returns:
            Receiver: A handle to the device receiving from this channel.

        Raises:
            RuntimeError: If a receiver is already configured.
        """

        return self.__receiver

    @receiver.setter
    def receiver(self, value: Device) -> None:
        """Set the device receiving from this channel."""

        if self.__receiver is not None:
            raise RuntimeError("Overwriting a receiver configuration is not supported")

        self.__receiver = value

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

        return self.__transmitter.num_antennas

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

        return self.__receiver.num_antennas

    def propagate(self, transmitted_signal: Signal) -> Tuple[Signal, ChannelStateInformation]:
        """Modifies the input signal and returns it after channel propagation.

        For the ideal channel in the base class, the MIMO channel is modeled as a matrix of ones.
        The routine samples a new impulse response, which will be converted to ChannelStateInformation.

        Args:

            transmitted_signal (Signal):
                Radio-Frequency band antenna signals to be propagated of this channel instance.

        Returns:
            (Signal, ChannelStateInformation):
                Tuple of the distorted signal after propagation and the respective channel state.
                Note that the channel may append samples to the propagated signal.
        Raises:

            ValueError:
                If the number of streams in `transmitted_signal` is not one or the number of transmitting antennas.

            RuntimeError:
                If the scenario configuration is not supported by the default channel model.

            RuntimeError:
                If the channel is currently floating.
        """

        if self.transmitter is None or self.receiver is None:
            raise RuntimeError("Channel is floating, making propagation simulation impossible")

        if transmitted_signal.num_streams != self.transmitter.num_antennas:
            raise ValueError("Number of transmitted signal streams does not match number of transmit antennas")

        # If the channel is inactive, propagation will result in signal loss
        # This is modeled by returning an zero-length signal and impulse-response (in time-domain) after propagation
        if not self.active:
            return (Signal.empty(transmitted_signal.sampling_rate),
                    ChannelStateInformation.Ideal(self.num_outputs, self.num_inputs, 0))

        # Generate the channel's impulse response
        impulse_response = self.impulse_response(transmitted_signal.timestamps, transmitted_signal.sampling_rate)

        # Consider the a random synchronization offset between transmitter and receiver
        sync_offset: float = self._rng.uniform(low=self.__sync_offset_low, high=self.__sync_offset_high)

        # The maximum delay in samples is modeled by the last impulse response dimension
        num_delay_samples = impulse_response.shape[3] - 1

        # Propagate the signal
        received_samples = np.zeros((self.receiver.num_antennas,
                                    transmitted_signal.num_samples + num_delay_samples), dtype=complex)

        for delay_index in range(num_delay_samples+1):
            for tx_idx, rx_idx in product(range(self.transmitter.num_antennas), range(self.receiver.num_antennas)):

                delayed_signal = impulse_response[:, rx_idx, tx_idx, delay_index] *\
                                 transmitted_signal.samples[tx_idx, :]
                received_samples[rx_idx, delay_index:delay_index+transmitted_signal.num_samples] += delayed_signal

        propagated_signal = Signal(received_samples,
                                   transmitted_signal.sampling_rate,
                                   carrier_frequency=transmitted_signal.carrier_frequency,
                                   delay=transmitted_signal.delay+sync_offset)
        channel_state = ChannelStateInformation(ChannelStateFormat.IMPULSE_RESPONSE,
                                                impulse_response.transpose((1, 2, 0, 3)))

        return propagated_signal, channel_state

    def impulse_response(self,
                         timestamps: np.ndarray,
                         sampling_rate: float) -> np.ndarray:
        """Sample a new channel impulse response.

        Note that this is the core routine from which `propagate` will create the channel state.

        Args:

            timestamps (np.ndarray):
                Time instants with length `T` to calculate the response for.

            sampling_rate (float):
                The rate at which the delay taps will be sampled, i.e. the delay resolution.

        Returns:
            np.ndarray:
                Impulse response in all `number_rx_antennas` x `number_tx_antennas`.
                4-dimensional array of size `T x number_rx_antennas x number_tx_antennas x (L+1)`
                where `L` is the maximum path delay (in samples). For the ideal
                channel in the base class, `L = 0`.
        """

        if self.transmitter is None or self.receiver is None:
            raise RuntimeError("Channel is floating, making impulse response simulation impossible")

        # MISO case
        if self.receiver.num_antennas == 1:
            impulse_responses = np.tile(np.ones((1, self.transmitter.num_antennas), dtype=complex),
                                        (timestamps.size, 1, 1))

        # SIMO case
        elif self.transmitter.num_antennas == 1:
            impulse_responses = np.tile(np.ones((self.receiver.num_antennas, 1), dtype=complex),
                                        (timestamps.size, 1, 1))

        # MIMO case
        else:
            impulse_responses = np.tile(np.eye(self.receiver.num_antennas, self.transmitter.num_antennas,
                                               dtype=complex), (timestamps.size, 1, 1))

        # Scale by channel gain and add dimension for delay response
        impulse_responses = self.gain * np.expand_dims(impulse_responses, axis=3)

        # Save newly generated response as most recent impulse response
        self.recent_response = impulse_responses

        # Return resulting impulse response
        return impulse_responses

    @property
    def min_sampling_rate(self) -> float:
        """Minimal sampling rate required to adequately model the channel.

        Returns:
            float: The minimal sampling rate in Hz.
        """

        # Since the default channel is time-invariant, there are no sampling rate requirements
        return 0.0

    @classmethod
    def to_yaml(cls: Type[Channel], representer: SafeRepresenter, node: Channel) -> MappingNode:
        """Serialize a channel object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Channel):
                The channel instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            'gain': node.__gain,
            'active': node.__active,
            'sync_offset_low': node.__sync_offset_low,
            'sync_offset_high': node.__sync_offset_high
        }

        transmitter_index, receiver_index = node.indices

        yaml = representer.represent_mapping(u'{.yaml_tag} {} {}'.format(cls, transmitter_index, receiver_index), state)
        return yaml

    @classmethod
    def from_yaml(cls: Type[Channel], constructor: SafeConstructor,  node: MappingNode) -> Channel:
        """Recall a new `Channel` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Channel` serialization.

        Returns:
            Channel:
                Newly created `Channel` instance. The internal references to modems will be `None` and need to be
                initialized by the `scenario` YAML constructor.

        """

        # Handle empty yaml nodes
        if isinstance(node, ScalarNode):
            return cls()

        return constructor.construct_mapping(node)
