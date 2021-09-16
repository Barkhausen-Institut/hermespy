from __future__ import annotations
import numpy as np
from typing import List, Tuple, Type, TYPE_CHECKING
from ruamel.yaml import RoundTripRepresenter, RoundTripConstructor, Node

from parameters_parser.parameters_scenario import ParametersScenario
from parameters_parser.parameters_general import ParametersGeneral
from parameters_parser.parameters_channel import ParametersChannel
from simulator_core.random_streams import RandomStreams
import simulator_core.tools.constants as constants
from source.bits_source import BitsSource

from channel.multipath_fading_channel import MultipathFadingChannel
from channel.quadriga_channel import QuadrigaChannel
from channel.quadriga_interface import QuadrigaInterface
from channel.noise import Noise
from channel.rx_sampler import RxSampler

if TYPE_CHECKING:

    from modem import Modem, Transmitter, Receiver
    from channel import Channel


class Scenario:
    """Implements the simulation scenario.

    The scenario contains objects for all the different elements in a given simulation,
    such as modems, channel models, bit sources, etc.


    Attributes:
        tx_modems (list(Modem)): list of all modems to be used for transmission with
            'number_of_tx_modems' elements.
        rx_modems (list(Modem)): list of all modems to be used for reception with
            'number_of_rx_modems' elements.
        sources (list(BitsSource)): list of all bits sources. Each bit source will
            be associated to one transmit modem.
        rx_samplers (list(RxSampler)): list of all receive re-samplers. Each resampler
            will be associated to one receive modem
        channels (list(list(Channel))): list of all propagation channels.
            channels[i][j] contains the channel from transmit modem 'i' to receive modem 'j'
        noise (list(Noise)): list of all noise sources. Each noise source is
            associated to one receive modem

    """

    yaml_tag = 'Scenario'
    __transmitters: List[Transmitter] = []
    __receivers: List[Receiver] = []
    __channels: np.ndarray = np.empty((0, 0), dtype=np.object)

    def __init__(self, parameters: ParametersScenario = None, param_general: ParametersGeneral = None,
                 rnd: RandomStreams = None) -> None:
        self.sources: List[BitsSource] = []
        self.rx_samplers: List[RxSampler] = []

        self.noise: List[Noise] = []
        self.params: ParametersScenario
        self.param_general: ParametersGeneral

        if parameters is None and param_general is None and rnd is None:  # used to facilitate unit testing
            pass
        else:
            self.random = rnd
            self.params = parameters
            self.param_general = param_general
            if self.params.channel_model_params[0][0].multipath_model == "QUADRIGA":
                self._quadriga_interface = QuadrigaInterface(
                    self.params.channel_model_params[0][0])

            self.sources, self.tx_modems = self._create_transmit_modems()

            self.rx_samplers, self.rx_modems, self.noise = self._create_receiver_modems()

            self.channels = self._create_channels()

    @classmethod
    def to_yaml(cls: Type[Scenario], representer: RoundTripRepresenter, node: Scenario) -> Node:
        """Serialize a scenario object to YAML.

        Args:
            representer (BaseRepresenter):
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
            'Channels': node.__channels.flatten().tolist()
        }

        return representer.represent_mapping("Scenario", serialization)

    @classmethod
    def from_yaml(cls: Type[Scenario], constructor: RoundTripConstructor, node: Node) -> Scenario:
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

        scenario = cls.__new__(cls)
        yield scenario

        constructor.add_multi_constructor("Channel", Channel.from_yaml)
        state_scenario = constructor.construct_mapping(node, deep=True)

        state_scenario.pop('Modems', None)
        state_scenario.pop('Channels', None)
        scenario.__init__(**state_scenario)

    def _create_channels(self) -> List[List[Channel]]:
        """Creates channels according to parameters specification.

        Returns:
            channels (List[List[Channel]]):
                List of channels between the modems. Nested list:
                `[ tx1, .. txN] x no_rx_modems`.
        """
        channels: List[List[Channel]] = []
        for modem_rx, rx_channel_params in zip(
                self.rx_modems, self.params.channel_model_params):
            channels_rx: List[Channel] = []

            for modem_tx, channel_params in zip(
                    self.tx_modems, rx_channel_params):

                if channel_params.multipath_model == 'NONE':
                    channel = Channel(channel_params, self.random.get_rng('channel'),
                                      modem_tx.waveform_generator.param.sampling_rate)

                elif channel_params.multipath_model == 'STOCHASTIC':
                    relative_speed = np.linalg.norm(
                        modem_rx.param.velocity - modem_tx.param.velocity)
                    doppler_freq = relative_speed * \
                        modem_tx.param.carrier_frequency / constants.speed_of_light
                    channel = MultipathFadingChannel(channel_params, self.random.get_rng('channel'),
                                                     modem_tx.waveform_generator.param.sampling_rate, doppler_freq)

                elif channel_params.multipath_model == 'QUADRIGA':
                    channel = QuadrigaChannel(
                        modem_tx, modem_rx,
                        modem_tx.waveform_generator.param.sampling_rate,
                        self.random.get_rng('channel'), self._quadriga_interface)

                elif channel_params.multipath_model == '5G_TDL':
                    relative_speed = np.linalg.norm(
                        modem_rx.param.velocity - modem_tx.param.velocity)
                    doppler_freq = relative_speed * \
                        modem_tx.param.carrier_frequency / constants.speed_of_light
                    channel = MultipathFadingChannel(channel_params, self.random.get_rng('channel'),
                                                     modem_tx.waveform_generator.param.sampling_rate, doppler_freq)

                else:
                    raise ValueError(
                        'channel "' +
                        channel_params.multipath_model +
                        '" not supported')

                channels_rx.append(channel)
                modem_tx.set_channel(channel)
            modem_rx.set_channel(channels_rx[modem_rx.param.tx_modem])

            channels.append(channels_rx)
        return channels

    def _create_transmit_modems(self) -> Tuple[List[BitsSource], List[Modem]]:
        """Creates Tx Modems.

        Returns:
            (List[BitsSource], List[Modem]):
                `List[BitsSource]`: list of bitssources for each transmit modem.
                `List[Modem]`: List of transmit modems.
        """
        sources = []
        tx_modems = []

        for modem_count in range(self.params.number_of_tx_modems):
            modem_parameters = self.params.tx_modem_params[modem_count]
            sources.append(BitsSource(self.random.get_rng("source")))
            tx_modems.append(
                Modem(
                    modem_parameters,
                    sources[modem_count],
                    self.random.get_rng("hardware")))

        return sources, tx_modems

    def _create_receiver_modems(
            self) -> Tuple[List[RxSampler], List[Modem], List[Noise]]:
        """Creates receiver modems.

        Returns:
            (List[RxSampler], List[Modem], List[Noise]):
                `list(RxSampler)`: List of RxSamplers
                `list(Modem)`: List of created rx modems.
                `list(Noise)`:
        """

        noise = []
        rx_modems = []
        rx_samplers = []

        for modem_count in range(self.params.number_of_rx_modems):
            modem_parameters = self.params.rx_modem_params[modem_count]
            rx_modems.append(Modem(modem_parameters, self.sources[modem_parameters.tx_modem],
                                   self.random.get_rng("hardware"), self.tx_modems[modem_parameters.tx_modem]))

            noise.append(
                Noise(
                    self.param_general.snr_type,
                    self.random.get_rng("noise")))
            rx_samplers.append(RxSampler(
                modem_parameters.technology.sampling_rate,
                modem_parameters.carrier_frequency))
            tx_sampling_rates = [
                tx_modem.param.technology.sampling_rate for tx_modem in self.tx_modems]
            tx_center_frequencies = [
                tx_modem.param.carrier_frequency for tx_modem in self.tx_modems]
            rx_samplers[modem_count].set_tx_sampling_rate(np.asarray(tx_sampling_rates),
                                                          np.asarray(tx_center_frequencies))

        return rx_samplers, rx_modems, noise

    def init_drop(self) -> None:
        """Initializes variables for each drop or creates new random numbers.
        """

        for source in self.sources:
            source.init_drop()

        for channel in self.channels.ravel():
            channel.init_drop()

    def __get_channel_instance(
            self, channel_params: ParametersChannel, modem_tx: Modem, modem_rx: Modem) -> Channel:
        channel = None
        if channel_params.multipath_model == 'NONE':
            channel = Channel(channel_params, self.random.get_rng("channel"),
                              modem_rx.waveform_generator.param.sampling_rate)
        elif channel_params.multipath_model == 'STOCHASTIC':
            channel = MultipathFadingChannel(channel_params, self.random.get_rng("channel"),
                                             modem_rx.waveform_generator.param.sampling_rate)
        elif channel_params.multipath_model == 'QUADRIGA':
            pass

        return channel

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

    def channel(self, transmitter: Modem, receiver: Modem) -> Channel:
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
        receiver = Receiver(self, **kwargs)

        self.__receivers.append(receiver)

        if self.__channels.shape[0] == 0:

            self.__channels = np.empty((0, receiver_index + 1), dtype=object)

        elif self.__channels.shape[1] == 0:

            self.__channels = np.array(
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
        transmitter = Transmitter(self, **kwargs)

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

    def transmit(self, drop_duration: float) -> List[np.ndarray]:
        """Simulated signals emitted by transmitters.

        Args:
            drop_duration:
                Length of simulated transmission in seconds.

        Returns:
            A list containing the the signals emitted by each transmitting modem.

        Raises:
            ValueError:
                On invalid drop lengths.
        """

        if drop_duration <= 0.0:
            raise ValueError("Drop duration must be greater or equal to zero")

        transmitted_signals = []
        for transmitter in self.transmitters:
            transmitted_signals.append(transmitter.send(drop_duration))

        return transmitted_signals


from modem import Modem, Transmitter, Receiver
from channel import Channel
