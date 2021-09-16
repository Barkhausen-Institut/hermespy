from __future__ import annotations
from typing import Type, TYPE_CHECKING
from abc import abstractmethod
import numpy as np
from ruamel.yaml import RoundTripRepresenter, RoundTripConstructor, Node

if TYPE_CHECKING:
    from modem import Modem


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
    __active: bool = False
    __transmitter: Modem = None
    __receiver: Modem = None
    __gain: float = 1.0

    def __init__(self,
                 transmitter: Modem,
                 receiver: Modem,
                 active: bool = None,
                 gain: float = None) -> None:
        """Class constructor.

        Args:
            transmitter (Modem):
                The modem transmitting into this channel.

            receiver (Modem):
                The modem receiving from this channel.

            active (bool, optional):
                Channel activity flag.

            gain (float, optional):
                Channel gain.
        """

        self.__transmitter = transmitter
        self.__receiver = receiver

        if active is not None:
            self.active = active

        if gain is not None:
            self.gain = gain

    @classmethod
    def to_yaml(cls: Type[Channel], representer: RoundTripRepresenter, node: Channel) -> Node:
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
        return representer.represent_mapping(cls.yaml_tag + "_{}_{}".format(transmitter_index, receiver_index), state)

    @classmethod
    def from_yaml(cls: Type[Channel], constructor: RoundTripConstructor, tag_suffix: str, node: Node) -> Channel:

        channel = cls.__new__(cls)
        yield channel

        scenario = [object for node, object in constructor.constructed_objects.items() if node.tag == 'Scenario'][0]

        indices = tag_suffix.split('_')
        if indices[0] == '':
            indices.pop(0)

        transmitter_index = int(indices[0])
        receiver_index = int(indices[1])
        scenario.channels[transmitter_index, receiver_index] = channel

        state = constructor.construct_mapping(node)
        channel.__init__(scenario.transmitters[transmitter_index], scenario.receivers[receiver_index], **state)


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
    def transmitter(self) -> Modem:
        """Access the modem transmitting into this channel.

        Returns:
            Modem: A handle to the modem transmitting into this channel.
        """

        return self.__transmitter

    @property
    def receiver(self) -> Modem:
        """Access the modem receiving from this channel.

        Returns:
            Modem: A handle to the modem receiving from this channel.
        """

        return self.__receiver

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
    def indices(self) -> (int, int):
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
    def propagate(self, tx_signal: np.ndarray) -> np.ndarray:
        """Modifies the input signal and returns it after channel propagation.

        For the ideal channel in the base class, the MIMO channel is modeled as a matrix of one's.

        If 'tx_signal' is an array of size `number_tx_antennas` X `number_of_samples`,
        then the output `rx_signal` will be an array of size
        `number_rx_antennas` X `number_of_samples`.

        Args:
            tx_signal (np.ndarray): Input signal.

        Returns:
            np.ndarray:
                The distorted signal after propagation. The output depends
                on the channel model employed.
        """

        # Convert input arrays to 1D-matrices
        if tx_signal.ndim == 1:
            tx_signal = np.reshape(tx_signal, (1, -1))

        if tx_signal.ndim != 2 or tx_signal.shape[0] != self.transmitter.num_streams:
            raise ValueError(
                'tx_signal must be an array with {:d} rows'.format(
                    self.transmitter.num_streams))

        # By default, we assume an ideal MIMO response
        rx_signal = np.ones((self.receiver.num_antennas, self.transmitter.num_antennas), dtype=complex) @ tx_signal
        return rx_signal * self.gain

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
        return impulse_responses * self.param.gain
