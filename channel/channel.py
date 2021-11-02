# -*- coding: utf-8 -*-
"""Channel model for wireless transmission links."""

from __future__ import annotations
from typing import Type, Tuple, TYPE_CHECKING, Optional
import numpy as np
from ruamel.yaml import SafeRepresenter, SafeConstructor, ScalarNode, MappingNode

if TYPE_CHECKING:
    from modem import Transmitter, Receiver
    from scenario.scenario import Scenario

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class Channel:
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
    __transmitter: Optional[Transmitter]
    __receiver: Optional[Receiver]
    __gain: float

    def __init__(self,
                 transmitter: Optional[Transmitter] = None,
                 receiver: Optional[Receiver] = None,
                 active: Optional[bool] = None,
                 gain: Optional[float] = None,
                 sync_offset_low: Optional[int] = None,
                 sync_offset_high: Optional[int] = None) -> None:
        """Class constructor.

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
        """

        # Default parameters
        self.__active = True
        self.__transmitter = None
        self.__receiver = None
        self.__gain = 1.0
        self.__scenario = None
        self.__sync_offset_low = 0
        self.__sync_offset_high = 0

        if transmitter is not None:
            self.transmitter = transmitter

        if receiver is not None:
            self.receiver = receiver

        if active is not None:
            self.active = active

        if gain is not None:
            self.gain = gain

        if sync_offset_low is not None:
            if sync_offset_low < 0:
                raise ValueError("Lower bound must be >= 0.")
            self.__sync_offset_low = sync_offset_low

        if sync_offset_high is not None:
            if sync_offset_high < 0:
                raise ValueError("Higher bound must be >= 0.")
            self.__sync_offset_high = sync_offset_high

        self._verify_sync_offsets()

    def _verify_sync_offsets(self):
        if not (self.sync_offset_low <= self.sync_offset_high):
            raise ValueError("Lower bound of uniform distribution must be smaller than higher bound.")

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
    def transmitter(self) -> Transmitter:
        """Access the modem transmitting into this channel.

        Returns:
            Transmitter: A handle to the modem transmitting into this channel.
        """

        return self.__transmitter

    @transmitter.setter
    def transmitter(self, new_transmitter: Transmitter) -> None:
        """Configure the modem transmitting into this channel.

        Args:
            new_transmitter (Transmitter): The transmitter to be configured.

        Raises:
            RuntimeError: If a transmitter is already configured.
        """

        if self.__transmitter is not None:
            raise RuntimeError("Overwriting a transmitter configuration is not supported")

        self.__transmitter = new_transmitter

    @property
    def receiver(self) -> Receiver:
        """Access the modem receiving from this channel.

        Returns:
            Receiver: A handle to the modem receiving from this channel.
        """

        return self.__receiver

    @receiver.setter
    def receiver(self, new_receiver: Receiver) -> None:
        """Configure the modem receiving from this channel.

        Args:
            new_receiver (Receiver): The receiver to be configured.

        Raises:
            RuntimeError: If a receiver is already configured.
        """

        if self.__receiver is not None:
            raise RuntimeError("Overwriting a receiver configuration is not supported")

        self.__receiver = new_receiver

    @property
    def sync_offset_low(self) -> float:
        return self.__sync_offset_low

    @property
    def sync_offset_high(self) -> float:
        return self.__sync_offset_high

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
            raise RuntimeError("Error trying to access the number of inputs property of a floating channel")

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
            raise RuntimeError("Error trying to access the number of outputs property of a floating channel")

        return self.__receiver.num_antennas

    @property
    def indices(self) -> Tuple[int, int]:
        """The indices of this channel within the scenarios channel matrix.

        Returns:
            int:
                Transmitter index.
            int:
                Receiver index.
        """

        return self.__transmitter.index, self.__receiver.index

#    @abstractmethod
    def init_drop(self) -> None:
        """Initializes random channel parameters for each drop, if required by model."""
        pass

#    @abstractmethod
    def propagate(self, transmitted_signal: np.ndarray) -> np.ndarray:
        """Modifies the input signal and returns it after channel propagation.

        For the ideal channel in the base class, the MIMO channel is modeled as a matrix of one's.

        Args:

            transmitted_signal (np.ndarray):
                Input signal antenna signals to be propagated of this channel instance.
                The array is expected to be two-dimensional with shape `num_transmit_antennas`x`num_samples`.

        Returns:
            np.ndarray:
                The distorted signal after propagation.
                Two-dimensional array with shape `num_receive_antennas`x`num_propagated_samples`.
                Note that the channel may append samples to the propagated signal,
                so that `num_propagated_samples` is generally not equal to `num_samples`.

        Raises:

            ValueError:
                If the first dimension of `transmitted_signal` is not one or the number of transmitting antennas.

            RuntimeError:
                If the scenario configuration is not supported by the default channel model.

            RuntimeError:
                If the channel is currently floating.
        """

        if transmitted_signal.ndim != 2:
            raise ValueError("Transmitted signal must be a matrix (an array of two dimensions)")

        if self.transmitter is None or self.receiver is None:
            raise RuntimeError("Channel is floating, making propagation simulation impossible")

        # If just on stream feeds into the channel, the output shall be the repeated stream by default.
        # Note: This might not be accurate physical behaviour for some sensor array topologies!
        if transmitted_signal.shape[0] == 1:
            return self.gain * transmitted_signal.repeat(self.receiver.num_antennas, axis=0)

        # MISO case, results in a superposition at the receiver
        if self.receiver.num_antennas == 1:
            return self.gain * np.sum(transmitted_signal, axis=0, keepdims=True)

        if transmitted_signal.shape[0] != self.transmitter.num_antennas:
            raise ValueError("Number of transmitted signal streams does not match number of transmit antennas")

        if self.transmitter.num_antennas != self.receiver.num_antennas:
            raise ValueError("The default channel only supports links between modems with identical antenna count")

        return self.gain * transmitted_signal

#    @abstractmethod
    def impulse_response(self, timestamps: np.ndarray) -> np.ndarray:
        """Calculate the channel impulse responses.

        This method can be used for instance by the transceivers to obtain the channel state
        information.

        TODO: This does not actually seem to be model impulse responses!!!!!

        Args:
            timestamps (np.ndarray):
                Time instants with length `T` to calculate the response for.

        Returns:
            np.ndarray:
                Impulse response in all `number_rx_antennas` x `number_tx_antennas`.
                4-dimensional array of size `T x number_rx_antennas x number_tx_antennas x (L+1)`
                where `L` is the maximum path delay (in samples). For the ideal
                channel in the base class, `L = 0`.
        """

        if self.transmitter is None or self.receiver is None:
            raise RuntimeError("Channel is floating, making impulse response simulation impossible")

        impulse_responses = np.tile(np.ones((self.receiver.num_antennas, self.transmitter.num_antennas), dtype=complex),
                                    (timestamps.size, 1, 1))
        impulse_responses = np.expand_dims(impulse_responses, axis=3)

        return self.gain * impulse_responses

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

        state = constructor.construct_mapping(node)
        return cls(**state)

    def estimate(self, num_samples: int = 1) -> np.ndarray:
        """Returns estimated channel responses.

        Args:
            num_samples (int, optional): Number of discrete time samples.

        Unlike the impulse_response routine, errors may occur during channel estimation.
        """

        estimate = np.eye(self.transmitter.num_antennas, self.receiver.num_antennas, dtype=complex)
        bloated_estimate = estimate[np.newaxis, :, :, np.newaxis].repeat(num_samples, axis=0)
        return bloated_estimate

    def add_time_offset(self, signal: np.ndarray) -> np.ndarray:
        """Introduces a time delay to the signal."""
        sampling_rate = self.transmitter.scenario.sampling_rate
        time_delay_samples = int(sampling_rate * self.scenario.channel_time_offset)

        delay_samples = np.zeros(
            (signal.shape[0],
             time_delay_samples)
        )
        delayed_signal = np.hstack((delay_samples, signal))
        return delayed_signal