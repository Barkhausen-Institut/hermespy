# -*- coding: utf-8 -*-
"""HermesPy scenario configuration."""

from __future__ import annotations
import numpy as np
import numpy.random as rnd
from typing import Generic, List, Type, Optional
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node
from collections.abc import Iterable

from .device import DeviceType, Transmitter, Receiver, Operator
from .random_node import RandomNode


__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Scenario(RandomNode, Generic[DeviceType]):
    """A simulation scenario.

    Scenarios consist of several devices transmitting and receiving electromagnetic signals.
    Each device can be operated by multiple operators simultaneously.
    """

    __slots__ = ['__devices']

    __devices: List[DeviceType]         # Registered devices within this scenario.

    def __init__(self,
                 seed: Optional[int] = None) -> None:
        """
        Args:

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.
        """

        RandomNode.__init__(self, seed=seed)
        self.__devices = []

    def add_device(self, device: DeviceType) -> None:
        """Add new device to scenario.

        Args:

            device (DeviceType):
                New device to be added to the scenario.

        Raises `ValueError` if the device already exists.
        """

        if self.device_registered(device):
            raise ValueError("Error trying to add an already registered device to a scenario")

        self.__devices.append(device)
        device.random_mother = self

    def device_registered(self, device: DeviceType) -> bool:
        """Check if an device is registered in this scenario.

        Args:
            device (DeviceType): The device to be checked.

        Returns:
            bool: The device's registration status.
        """

        return device in self.__devices

    @property
    def devices(self) -> List[DeviceType]:
        """Devices registered in this scenario.

        Returns:
            List[DeviceType]: List of devices.
        """

        return self.__devices.copy()

    @property
    def num_devices(self) -> int:
        """Number of devices in this scenario.

        Returns:
            int: Number of devices
        """

        return len(self.__devices)

    @property
    def transmitters(self) -> List[Transmitter]:
        """All transmitting operators within this scenario.

        Returns:
            List[Transmitter]: List of all transmitting operators.
        """

        transmitters: List[Transmitter] = []

        for device in self.__devices:
            transmitters.extend(device.transmitters)

        return transmitters
    
    @property
    def receivers(self) -> List[Receiver]:
        """All receiving operators within this scenario.

        Returns:
            List[Receiver]: List of all transmitting operators.
        """

        receivers: List[Receiver] = []

        for device in self.__devices:
            receivers.extend(device.receivers)

        return receivers

    @property
    def num_receivers(self) -> int:
        """Number of receiving operators within this scenario.

        Returns:
            int: The number of receivers.
        """

        num = 0
        for device in self.__devices:
            num += device.receivers.num_operators

        return num

    @property
    def num_transmitters(self) -> int:
        """Number of transmitting operators within this scenario.

        Returns:
            int: The number of transmitters.
        """

        num = 0
        for device in self.__devices:
            num += device.transmitters.num_operators

        return num

    @property
    def operators(self) -> List[Operator]:
        """All operators within this scenario.
        
        Returns:
            List[Operator]: List of all operators.
        """

        operators: List[Receiver] = []

        for device in self.__devices:

            operators.extend(device.receivers)
            operators.extend(device.transmitters)

        return operators

    @property
    def num_operators(self) -> int:
        """Number of operators within this scenario.

        Returns:
            int: The number of operators.
        """

        num = 0
        for device in self.__devices:
            num += device.transmitters.num_operators + device.receivers.num_operators

        return num


class ScenarioOld:
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

        random_generator (rnd.Generator):
            Generator used to create random sequences.

    """

    yaml_tag = u'Scenario'

    __channels: np.ndarray
    __drop_duration: float
    __sampling_rate: float
    random_generator: rnd.Generator

    def __init__(self,
                 drop_duration: float = 0.0,
                 random_generator: Optional[rnd.Generator] = None) -> None:
        """Object initialization.

        Args:
            drop_duration (float, optional):
                The default drop duration in seconds.

            random_generator (rnd.Generator, optional):
                The generator object used to create pseudo-random number sequences.
        """

        self.__devices = []
        self.__channels = np.ndarray((0, 0), dtype=object)
        self.drop_duration = drop_duration
        self.random_generator = rnd.default_rng(random_generator)

        self.sources: List[BitsSource] = []
        self.noise: List[Noise] = []

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
    def num_receivers(self) -> int:
        """Count the number of registered receivers within this scenario.

        Returns:
            int: The number of receivers.
        """

        return len(self.__receivers)

    @property
    def num_transmitters(self) -> int:
        """Count the number of registered transmitters within this scenario.

        Returns:
            int: The number of transmitters
        """

        return len(self.__transmitters)

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

    def departing_channels(self, transmitter: Transmitter, active_only: bool = False) -> List[Channel]:
        """Collect all channels departing from a `transmitter`.

        Args:
            transmitter (Transmitter):
                The transmitting modem.

            active_only (bool, optional):
                Consider only active channels.

        Returns:
            List[Channel]:
                A list of departing channels.

        Raises:
            ValueError:
                Should `transmitter` not be registered with this scenario.
        """

        if transmitter not in self.__transmitters:
            raise ValueError("The provided transmitter is not registered with this scenario.")

        transmitter_index = self.__transmitters.index(transmitter)
        channels: List[Channel] = self.__channels[transmitter_index, :].tolist()

        if active_only:
            channels = [channel for channel in channels if channel.active]

        return channels

    def arriving_channels(self, receiver: Receiver, active_only: bool = False) -> List[Channel]:
        """Collect all channels arriving at a `receiver`.

        Args:
            receiver (Receiver):
                The receiving modem.

            active_only (bool, optional):
                Consider only active channels.

        Returns:
            List[Channel]:
                A list of arriving channels.

        Raises:
            ValueError:
                Should `receiver` not be registered with this scenario.
        """

        if receiver not in self.__receivers:
            raise ValueError("The provided transmitter is not registered with this scenario.")

        receiver_index = self.__receivers.index(receiver)
        channels: List[Channel] = self.__channels[:, receiver_index].tolist()

        if active_only:
            channels = [channel for channel in channels if channel.active]

        return channels

    def set_channel(self, transmitter_index: int, receiver_index: int, channel: Channel) -> None:
        """Specify a channel within the channel matrix.

        Args:
            transmitter_index (int):
                Index of the transmitter within the channel matrix.

            receiver_index (int):
                Index of the receiver within the channel matrix.

            channel (Channel):
                The channel instance to be set at position (`transmitter_index`, `receiver_index`).

        Raises:
            ValueError:
                If `transmitter_index` or `receiver_index` are greater than the channel matrix dimensions.
        """

        if self.__channels.shape[0] <= transmitter_index or 0 > transmitter_index:
            raise ValueError("Transmitter index greater than channel matrix dimension")

        if self.__channels.shape[1] <= receiver_index or 0 > receiver_index:
            raise ValueError("Receiver index greater than channel matrix dimension")

        # Update channel field within the matrix
        self.__channels[transmitter_index, receiver_index] = channel

        # Set proper receiver and transmitter fields
        channel.transmitter = self.transmitters[transmitter_index]
        channel.receiver = self.receivers[receiver_index]
        channel.scenario = self

    def add_receiver(self, receiver: Receiver) -> None:
        """Add a new receiving modem to the simulated scenario.

        Args:
            receiver (Receiver): The receiver modem to be attached to this scenario.
        """

        # Register scenario to this transmit modem
        receiver.scenario = self

        # Store transmitter within internal transmit modem list
        receiver_index = len(self.__receivers)
        self.__receivers.append(receiver)

        # Adapt internal channel matrix
        if self.__channels.shape[0] == 0:

            self.__channels = np.empty((0, receiver_index + 1), dtype=np.object_)

        elif self.__channels.shape[1] == 0:

            self.__channels = np.array([[Channel(transmitter, receiver, self)] for transmitter in self.__transmitters])

        else:

            self.__channels = np.append(
                self.__channels,
                np.array([[Channel(transmitter, receiver, self)] for transmitter in self.transmitters]),
                axis=1
            )

    def add_transmitter(self, transmitter: Transmitter) -> None:
        """Add a new transmitting modem to the simulated scenario.

        Args:
            transmitter (Transmitter): The transmitter modem to be attached to this scenario.

        """

        # Register scenario to this transmit modem
        transmitter.scenario = self

        # Store transmitter within internal transmit modem list
        transmitter_index = len(self.__transmitters)
        self.__transmitters.append(transmitter)

        # Adapt internal channel matrix
        if self.__channels.shape[1] == 0:

            self.__channels = np.empty((transmitter_index + 1, 0), dtype=np.object_)

        elif self.__channels.shape[0] == 0:

            self.__channels = np.array([[Channel(transmitter, receiver, self) for receiver in self.__receivers]])

        else:

            self.__channels = np.append(self.__channels,
                                        np.array([[Channel(transmitter, receiver, self) for receiver in self.receivers]]),
                                        axis=0)

    def remove_modem(self, modem: Modem) -> None:
        """Remove a modem from the scenario.

        Args:
            modem (Modem):
                The `modem` instance to be removed.

        Raises:
            ValueError:
                If the provided `modem` is not registered with this scenario.
        """

        modem_deleted = False

        if modem in self.__transmitters:

            index = self.__transmitters.index(modem)

            del self.__transmitters[index]                            # Remove the actual modem
            self.__channels = np.delete(self.__channels, index, 0)    # Remove its departing channels
            modem_deleted = True

        if modem in self.__receivers:

            index = self.__receivers.index(modem)

            del self.__receivers[index]                               # Remove the actual modem
            self.__channels = np.delete(self.__channels, index, 1)    # Remove its arriving channels
            modem_deleted = True

        if not modem_deleted:
            raise ValueError("The provided modem handle was not registered with this scenario")


    def generate_data_bits(self) -> List[np.ndarray]:
        """Generate a set of data bits required to generate a single drop within this scenario.

        Returns:
            List[np.ndarray]: Data bits required to generate a single drop.
        """

        return [transmitter.generate_data_bits() for transmitter in self.__transmitters]

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
            'drop_duration': node.__drop_duration,
            "sampling_rate": node.__sampling_rate,
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
        sampling_rate = state_scenario.pop('sampling_rate', None)

        # Convert the random seed to a new random generator object if its specified within the config
        random_seed = state_scenario.pop('random_seed', None)
        if random_seed is not None:
            state_scenario['random_generator'] = rnd.default_rng(random_seed)

        # Create new scenario object
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
                scenario.__channels[t, r] = Channel(transmitter, receiver, scenario)

        # Integrate configured channels into the default matrix
        if isinstance(channels, Iterable):
            for channel, transmitter_index, receiver_index in channels:

                channel.transmitter = scenario.transmitters[transmitter_index]
                channel.receiver = scenario.receivers[receiver_index]
                channel.scenario = scenario
                scenario.__channels[transmitter_index, receiver_index] = channel

        # A configured scenario emerges from the depths
        return scenario

    @property
    def transmit_block_sizes(self) -> List[int]:
        """Bit block sizes required by registered transmitting modems.

        Returns:
            List[int]: Block size for each modem.
        """

        block_sizes: List[int] = []
        for transmitter in self.__transmitters:
            block_sizes.append(transmitter.encoder_manager.bit_block_size)

        return block_sizes

    @property
    def receive_block_sizes(self) -> List[int]:
        """Bit block sizes required by registered receiving modems.

        Returns:
            List[int]: Block size for each modem.
        """

        block_sizes: List[int] = []
        for receiver in self.__receivers:
            block_sizes.append(receiver.encoder_manager.bit_block_size)

        return block_sizes
