from __future__ import annotations
import numpy as np
from typing import List, Tuple, Type, TYPE_CHECKING, Optional
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node
from collections.abc import Iterable

import simulator_core.tools.constants as constants


if TYPE_CHECKING:

    from source.bits_source import BitsSource
    from modem import Modem, Transmitter, Receiver
    from channel import Channel
    from channel.multipath_fading_channel import MultipathFadingChannel
    from channel.noise import Noise
    from channel.rx_sampler import RxSampler


class Scenario:
    """Implements the simulation scenario.

    The scenario contains objects for all the different elements in a given simulation,
    such as modems, channel models, bit sources, etc.


    Attributes:
        __transmitters (List[Transmitter]):
            Modems transmitting electromagnetic waves within this scenario.

        __receivers (List[Receiver]);
            Modems receiving electromagnetic waves within this scenario.

        __channels (np.ndarray):
            MxN Matrix containing the channel configuration between all M transmitters and N receivers.

        __drop_duration (float):
            The physical simulation time of one scenario run in seconds.

    """

    yaml_tag = u'Scenario'
    __transmitters: List[Transmitter]
    __receivers: List[Receiver]
    __channels: np.ndarray
    __drop_duration: float

    def __init__(self,
                 drop_duration: float = 0.0) -> None:
        """Object initialization.

        Args:
            drop_duration (float, optional): The default drop duration in seconds.
        """

        self.__transmitters = []
        self.__receivers = []
        self.__channels = np.ndarray((0, 0), dtype=object)
        self.drop_duration = drop_duration

        self.sources: List[BitsSource] = []
        self.rx_samplers: List[RxSampler] = []

        self.noise: List[Noise] = []

    @classmethod
    def to_yaml(cls: Type[Scenario], representer: SafeRepresenter, node: Scenario) -> Node:
        """Serialize a scenario object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Scenario):
                The scenario instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        serialization = {
            'Modems': [*node.__transmitters, *node.__receivers],
            'Channels': node.__channels.flatten().tolist(),
            'drop_duration': node.__drop_duration
        }

        return representer.represent_mapping(cls.yaml_tag, serialization)

    @classmethod
    def from_yaml(cls: Type[Scenario], constructor: SafeConstructor, node: Node) -> Scenario:
        """Recall a new `Scenario` instance from YAML.

        Args:
            constructor (RoundTripConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Scenario` serialization.

        Returns:
            Scenario:
                Newly created `Scenario` instance.
            """

        state_scenario = constructor.construct_mapping(node, deep=True)

        modems = state_scenario.pop('Modems', None)
        channels = state_scenario.pop('Channels', None)

        scenario = cls(**state_scenario)

        # Integrate modems
        if isinstance(modems, Iterable):
            for modem in modems:

                # Integrate modem into scenario
                if isinstance(modem, Transmitter):
                    scenario.__transmitters.append(modem)

                elif isinstance(modem, Receiver):
                    scenario.__receivers.append(modem)

                else:
                    raise RuntimeWarning("Unknown modem type encountered")

                # Register scenario instance to the modems
                modem.scenario = scenario

        # Add default channel matrix
        scenario.__channels = np.empty((len(scenario.__transmitters), len(scenario.__receivers)), dtype=object)
        for t, transmitter in enumerate(scenario.__transmitters):
            for r, receiver in enumerate(scenario.__receivers):
                scenario.__channels[t, r] = Channel(transmitter, receiver)

        # Integrate configured channels into the default matrix
        if isinstance(channels, Iterable):
            for channel, transmitter_index, receiver_index in channels:

                channel.transmitter = scenario.transmitters[transmitter_index]
                channel.receiver = scenario.receivers[receiver_index]
                scenario.__channels[transmitter_index, receiver_index] = channel

        return scenario

    def generate_data_bits(self) -> List[np.array]:
        """Generate a set of data bits required to generate a single drop within this scenario.

        Returns:
            List[np.array]: Data bits required to generate a single drop.
        """

        data_bits = [np.random.randint(0, 2, transmitter.num_data_bits_per_frame) for transmitter in self.__transmitters]
        return data_bits

    @property
    def receivers(self) -> List[Receiver]:
        """Access receiving modems within this scenario.

        Returns:
            List[Modem]:
                A list of modems configured as receivers.
        """

        return self.__receivers

    @property
    def transmitters(self) -> List[Transmitter]:
        """Access transmitting modems within this scenario.

        Returns:
            List[Modem]:
                A list of modems configured as transmitters.
        """

        return self.__transmitters

    @property
    def channels(self) -> np.ndarray:
        """Access full channel matrix.

        Returns:
            np.ndarray:
                A numpy array containing channels between sender and receiver modems.
        """

        return self.__channels

    def channel(self, transmitter: Transmitter, receiver: Receiver) -> Channel:
        """Access a specific channel between two modems.

        Args:
            transmitter (Modem):
                The transmitting modem.

            receiver (Modem):
                The receiving modem.

        Returns:
            Channel:
                A handle to the transmission channel between `transmitter` and `receiver`.

        Raises:
            ValueError:
                Should `transmitter` or `receiver` not be registered with this scenario.
        """

        if transmitter not in self.transmitters:
            raise ValueError("Provided transmitter is not registered with this scenario")

        if receiver not in self.__receivers:
            raise ValueError("Provided receiver is not registered with this scenario")

        index_transmitter = self.__transmitters.index(transmitter)
        index_receiver = self.__receivers.index(receiver)

        return self.__channels[index_transmitter, index_receiver]

    def departing_channels(self, transmitter: Modem, active_only: bool = False) -> List[Channel]:
        """Collect all channels departing from a `transmitter`.

        Args:
            transmitter (Modem):
                The transmitting modem.

            active_only (bool, optional):
                Consider only active channels.

        Returns:
            List[Modem]:
                A list of departing channels.

        Raises:
            ValueError:
                Should `transmitter` not be registered with this scenario.
        """

        if transmitter not in self.__transmitters:
            raise ValueError("The provided transmitter is not registered with this scenario.")

        transmitter_index = self.__transmitters.index(transmitter)
        channels = self.__channels[transmitter_index, :].tolist()

        if active_only:
            channels = [channel for channel in channels if channel.active]

        return channels

    def arriving_channels(self, receiver: Modem, active_only: bool = False) -> List[Channel]:
        """Collect all channels arriving at a `receiver`.

        Args:
            receiver (Modem):
                The receiving modem.

            active_only (bool, optional):
                Consider only active channels.

        Returns:
            List[Modem]:
                A list of arriving channels.

        Raises:
            ValueError:
                Should `receiver` not be registered with this scenario.
        """

        if receiver not in self.__receivers:
            raise ValueError("The provided transmitter is not registered with this scenario.")

        receiver_index = self.__receivers.index(receiver)
        channels = self.__channels[receiver_index, :].tolist()

        if active_only:
            channels = [channel for channel in channels if channel.active]

        return channels

    def set_channel(self, transmitter_index: int, receiver_index: int, channel: Channel) -> None:
        """Specify a channel within the channel matrix.

        Warning:
            This function will expand the channel matrix dimension if insufficient.

        Args:
            transmitter_index (int):
                Index of the transmitter within the channel matrix.

            receiver_index (int):
                Index of the receiver within the channel matrix.

            channel (Channel):
                The channel instance to be set at position (`transmitter_index`, `receiver_index`).
        """

        self.__channels[transmitter_index, receiver_index] = channel
        channel.transmitter = self.transmitters[transmitter_index]
        channel.receiver = self.receivers[receiver_index]

    def add_receiver(self, **kwargs) -> Receiver:
        """Add a new receiving modem to the simulated scenario.

        Args:
            **kwargs:
                Modem configuration arguments.

        Returns:
            Modem:
                A handle to the newly created modem instance.
        """

        receiver_index = len(self.__receivers)
        kwargs['scenario'] = self
        receiver = Receiver(**kwargs)

        self.__receivers.append(receiver)

        if self.__channels.shape[0] == 0:

            self.__channels = np.empty((0, receiver_index + 1), dtype=object)

        elif self.__channels.shape[1] == 0:

            self.__channels = np.ndarray(
                [[Channel(transmitter, receiver)] for transmitter in self.transmitters], dtype=object)

        else:

            self.__channels = np.append(
                self.__channels, [[Channel(transmitter, receiver)] for transmitter in self.transmitters], axis=1)

        return receiver

    def add_transmitter(self, **kwargs) -> Transmitter:
        """Add a new transmitting modem to the simulated scenario.

        Args:
            **kwargs:
                Modem configuration arguments.

        Returns:
            Modem:
                A handle to the newly created modem instance.
        """

        transmitter_index = len(self.__transmitters)
        kwargs['scenario'] = self
        transmitter = Transmitter(**kwargs)

        self.__transmitters.append(transmitter)

        if self.__channels.shape[1] == 0:

            self.__channels = np.empty((transmitter_index + 1, 0), dtype=object)

        elif self.__channels.shape[0] == 0:

            self.__channels = np.array(
                [[Channel(transmitter, receiver) for receiver in self.receivers]], dtype=object)

        else:

            np.insert(self.__channels, transmitter_index,
                      [[Channel(transmitter, receiver) for receiver in self.receivers]], axis=0)

        return transmitter

    def remove_modem(self, modem: Modem) -> None:
        """Remove a modem from the scenario.

        Args:
            modem (Modem):
                The `modem` instance to be removed.

        Raises:
            ValueError:
                If the provided `modem` is not registered with this scenario.
        """

        if modem in self.__transmitters:

            index = self.__transmitters.index(modem)

            del self.__transmitters[index]          # Remove the actual modem
            np.delete(self.__channels, index, 0)    # Remove its departing channels

        elif modem in self.__receivers:

            index = self.__receivers.index(modem)

            del self.__receivers[index]             # Remove the actual modem
            np.delete(self.__channels, index, 1)    # Remove its arriving channels

        else:

            raise ValueError("The provided modem handle was not registered with this scenario")

    def transmit(self,
                 drop_duration: Optional[float] = None,
                 data_bits: Optional[np.array] = None) -> List[np.ndarray]:
        """Generate signals emitted by all transmitters registered with this scenario.

        Args:
            drop_duration (float, optional): Length of simulated transmission in seconds.
            data_bits (List[np.array], optional): The data bits to be sent by each transmitting modem.

        Returns:
            List[np.ndarray]: A list containing the the signals emitted by each transmitting modem.

        Raises:
            ValueError: On invalid `drop_duration`s.
            ValueError: If `data_bits` does not contain data for each transmitting modem.
        """

        if drop_duration is None:
            drop_duration = self.drop_duration

        if drop_duration <= 0.0:
            raise ValueError("Drop duration must be greater or equal to zero")

        transmitted_signals = []

        if data_bits is None:

            for transmitter in self.transmitters:
                transmitted_signals.append(transmitter.send(drop_duration))

        else:

            if len(data_bits) != len(self.__transmitters):
                raise ValueError("Data bits to be transmitted have insufficient streams for each configured transmitter")

            for transmitter, data in zip(self.transmitters, data_bits):
                transmitted_signals.append(transmitter.send(drop_duration, data))

        return transmitted_signals

    def propagate(self, transmitted_signals: List[np.ndarray]) -> List[np.ndarray]:
        """Propagate the signals generated by registered transmitters over the channel model.

        Signals receiving at each receive modem are a superposition of all transmit signals impinging
        onto the receive modem over activated channels.

        The signal stream matrices contain the number of antennas on the first dimension and the number of
        signal samples on the second dimension

        Args:
            transmitted_signals (List[np.ndarray]):
                List of signal streams emerging from each registered transmit modem.

        Returns:
            List[np.ndarray]:
                List of propagated signal streams impinging onto the registered receive modems.

        Raises:
            ValueError: If the number of `transmitted_signals` does not equal the number of registered transmit modems.
        """

        if len(transmitted_signals) != len(self.__transmitters):
            raise ValueError("Number of transmit signals {} does not match the number of registered transmit "
                             "modems {}".format(len(transmitted_signals), len(self.__transmitters)))

        # Initialize the propagated signals
        arriving_signals = [np.empty((receiver.num_antennas, 0), dtype=complex) for receiver in self.__receivers]

        # Loop over each channel within the channel matrix and propagate the signals over the respective channel model
        for transmitter_id, transmitted_signal in enumerate(transmitted_signals):
            for receiver_id, receiver in enumerate(self.__receivers):

                # Select responsible channel between respective transmitter and receiver
                channel: Channel = self.__channels[transmitter_id, receiver_id]

                # Skip propagation over channels flagged as inactive
                if not channel.active:
                    continue

                # Propagate signal emerging from transmitter over the channel
                propagated_signal = channel.propagate(transmitted_signal)

                # Extend the propagated signals matrix to hold more samples (if required
                sample_difference = propagated_signal.shape[1] - arriving_signals[receiver_id].shape[1]
                if sample_difference > 0:

                    arriving_signals[receiver_id] = np.append(arriving_signals[receiver_id],
                                                              np.zeros((receiver.num_antennas, sample_difference),
                                                                       dtype=complex), axis=1)
                    arriving_signals[receiver_id] += propagated_signal

                elif sample_difference < 0:
                    arriving_signals[receiver_id][:, :sample_difference] += propagated_signal

                else:
                    arriving_signals[receiver_id] += propagated_signal

        return arriving_signals

    def receive(self, arriving_signals: List[np.ndarray]) -> List[np.ndarray]:
        """Generate signals received by all receivers registered with this scenario.

        Args:
            arriving_signals (List[np.ndarray]):
                List of signal streams arriving at each receiving modem.

        Returns:
            List[np.ndarray]: A list containing the the signals emitted by each transmitting modem.

        Raises:
            ValueError:
                If the number of `arriving_signals` does not equal the number of registered receive modems.
        """

        if len(arriving_signals) != len(self.__receivers):
            raise ValueError("Number of arriving signals {} does not match the number of registered receive "
                             "modems {}".format(len(arriving_signals), len(self.__receivers)))

        data_bits: List[np.ndarray] = []

        for receiver_index, receiver in enumerate(self.__receivers):

            noise_variance = 0.0
            data = receiver.receive(arriving_signals[receiver_index], noise_variance)
            data_bits.append(data)

        return data_bits

    @property
    def drop_duration(self) -> float:
        """The scenario's default drop duration in seconds.

        Returns:
            float: The default drop duration.
        """

        # Return the largest frame length as default drop duration
        if self.__drop_duration == 0.0:

            min_drop_duration = 0.0
            for transmitter in self.__transmitters:

                max_frame_duration = transmitter.waveform_generator.max_frame_duration
                if max_frame_duration > min_drop_duration:
                    min_drop_duration = max_frame_duration

            return min_drop_duration

        else:
            return self.__drop_duration

    @drop_duration.setter
    def drop_duration(self, duration: float) -> None:
        """Modify the scenario's default drop duration.

        Args:
            duration (float): New drop duration in seconds.

        Raises:
            ValueError: If `duration` is less than zero.
        """

        if duration < 0.0:
            raise ValueError("Drop duration must be greater or equal to zero")

        self.__drop_duration = duration


from modem import Modem, Transmitter, Receiver
from channel import Channel
