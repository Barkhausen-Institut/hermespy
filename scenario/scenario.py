import numpy as np
from typing import List, Tuple

from parameters_parser.parameters_scenario import ParametersScenario
from parameters_parser.parameters_general import ParametersGeneral
from parameters_parser.parameters_channel import ParametersChannel
from simulator_core.random_streams import RandomStreams
import simulator_core.tools.constants as constants
from source.bits_source import BitsSource
from modem.modem import Modem

from channel.channel import Channel
from channel.multipath_fading_channel import MultipathFadingChannel
from channel.quadriga_channel import QuadrigaChannel
from channel.quadriga_interface import QuadrigaInterface
from channel.noise import Noise
from channel.rx_sampler import RxSampler


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

    def __init__(self, parameters: ParametersScenario = None, param_general: ParametersGeneral = None,
                 rnd: RandomStreams = None) -> None:
        self.sources: List[BitsSource] = []
        self.tx_modems: List[Modem] = []
        self.rx_modems: List[Modem] = []
        self.rx_samplers: List[RxSampler] = []
        self.channels: List[List[Channel]] = []
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
        """Intializes variables for each drop or creates new random numbers."""
        for source in self.sources:
            source.init_drop()

        for rx_channel in self.channels:
            for channel in rx_channel:
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
