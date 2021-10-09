from __future__ import annotations
from typing import Type, List, Tuple, TYPE_CHECKING, Optional
from abc import abstractmethod
import numpy as np
from ruamel.yaml import RoundTripRepresenter, RoundTripConstructor, ScalarNode, MappingNode
from ruamel.yaml.comments import CommentedOrderedMap

if TYPE_CHECKING:
    from modem import Transmitter, Receiver


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

    yaml_tag = 'Channel'
    __active: bool
    __transmitter: Optional[Transmitter]
    __receiver: Optional[Receiver]
    __gain: float

    def __init__(self,
                 transmitter: Transmitter = None,
                 receiver: Receiver = None,
                 active: bool = None,
                 gain: float = None) -> None:
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

        if transmitter is not None:
            self.transmitter = transmitter

        if receiver is not None:
            self.receiver = receiver

        if active is not None:
            self.active = active

        if gain is not None:
            self.gain = gain

    @classmethod
    def to_yaml(cls: Type[Channel], representer: RoundTripRepresenter, node: Channel) -> ScalarNode:
        """Serialize a channel object to YAML.

        Args:
            representer (RoundTripRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Channel):
                The channel instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            'active': node.__active,
            'gain': node.__gain
        }

        transmitter_index, receiver_index = node.indices

        yaml = representer.represent_mapping(u'{.yaml_tag} {} {}'.format(cls, transmitter_index, receiver_index), state)
        return yaml

    @classmethod
    def from_yaml(cls: Type[Channel], constructor: RoundTripConstructor, tag_suffix: str, node: Node)\
            -> Tuple[Channel, int, int]:
        """Recall a new `Channel` instance from YAML.

        Args:
            constructor (RoundTripConstructor):
                A handle to the constructor extracting the YAML information.

            tag_suffix (str):
                Optional tag suffix in the YAML config describing the channel position within the channel matrix.
                Syntax is Channel_`(transmitter index)`_`(receiver_index)`.

            node (Node):
                YAML node representing the `Channel` serialization.

        Returns:
            Channel:
                Newly created `Channel` instance. The internal references to modems will be `None` and need to be
                initialized by the `scenario` YAML constructor.

            int:
                Transmitter index of modem transmitting into this channel.

            int:
                Receiver index of modem receiving from this channel.
            """

        indices = tag_suffix.split(' ')
        if indices[0] == '':
            indices.pop(0)

        state = constructor.construct_mapping(node, CommentedOrderedMap)
        return Channel(**state), int(indices[0]), int(indices[1])

    def move_to(self, transmitter: Transmitter, receiver: Receiver) -> None:
        """Move the channel to a new matrix position.

        transmitter (Transmitter):
            New transmitting modem.

        receiver (Receiver):
            New receiving modem.
        """

        self.__transmitter = transmitter
        self.__receiver = receiver

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
        """

        self.__gain = value

    @property
    def num_inputs(self) -> int:
        """The number of streams feeding into this channel.

        Actually shadows the `num_streams` property of his channel's transmitter.

        Returns:
            int:
                The number of input streams.
        """

        return self.__transmitter.num_streams

    @property
    def num_outputs(self) -> int:
        """The number of streams emerging from this channel.

        Actually shadows the `num_streams` property of his channel's receiver.

        Returns:
            int:
                The number of output streams.
        """

        return self.__receiver.num_streams

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

    @abstractmethod
    def init_drop(self) -> None:
        """Initializes random channel parameters for each drop, if required by model."""
        pass

    @abstractmethod
    def propagate(self, transmitted_signal: np.ndarray) -> np.ndarray:
        """Modifies the input signal and returns it after channel propagation.

        For the ideal channel in the base class, the MIMO channel is modeled as a matrix of one's.

        Args:
            transmitted_signal (np.ndarray): Input signal antenna signals to be propagated of this channel instance.

        Returns:
            np.ndarray:
                The distorted signal after propagation.
                The output depends on the channel model employed.

        Raises:
            ValueError: If the first dimension of `transmitted_signal` is not one or the number of transmitting antennas.
            RuntimeError: If the scenario configuration is not supported by the default channel model.
        """

        if transmitted_signal.ndim != 2:
            raise ValueError("Transmitted signal must be a matrix (an array of two dimensions)")

        # If just on stream feeds into the channel, the output shall be the repeated stream by default.
        # Note: This might not be accurate physical behaviour for some sensor array topologies!
        if transmitted_signal.shape[0] == 1:
            return self.gain * transmitted_signal.repeat(self.receiver.num_antennas, axis=0)

        if transmitted_signal.shape[0] != self.transmitter.num_antennas:
            raise ValueError("Number of transmitted signal streams does not match number of transmit antennas")

        if self.transmitter.num_antennas != self.receiver.num_antennas:
            raise ValueError("The default channel only supports links between modems with identical antenna count")

        return self.gain * transmitted_signal

    @abstractmethod
    def get_impulse_response(self, timestamps: np.array) -> np.ndarray:
        """Calculate the channel impulse responses.

        This method can be used for instance by the transceivers to obtain the channel state
        information.

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
        impulse_responses = np.tile(
            np.ones((self.receiver.num_antennas, self.transmitter.num_antennas), dtype=complex),
            (timestamps.size, 1, 1))

        impulse_responses = np.expand_dims(impulse_responses, axis=3)
        return impulse_responses * self.gain
