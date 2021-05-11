import unittest
from unittest.mock import Mock, patch
import os

from scenario.scenario import Scenario
from parameters_parser.parameters_scenario import ParametersScenario
from parameters_parser.parameters_general import ParametersGeneral
from simulator_core.random_streams import RandomStreams
from source.bits_source import BitsSource
from channel.channel import Channel


class TestScenario(unittest.TestCase):

    def setUp(self) -> None:
        self.params_scenario = ParametersScenario()
        self.params_general = ParametersGeneral()

        params_path = os.path.join(
            os.getcwd(),
            'tests',
            'unit_tests',
            'scenario',
            'res')
        self.params_scenario.read_params(
            os.path.join(
                params_path,
                'settings_scenario.ini'))
        self.params_general.read_params(
            os.path.join(
                params_path,
                'settings_general.ini'))
        self.rnd = RandomStreams(42)

        self.scenario = Scenario()
        self.scenario.params = self.params_scenario
        self.scenario.param_general = self.params_general
        self.scenario.random = self.rnd

        # define tx and rx modems mocks here
        self.mock_tx_modem = Mock()
        self.mock_tx_modem.param.technology.sampling_rate = 1e6
        self.mock_tx_modem.param.carrier_frequency = 1e9
        self.mock_tx_modem.param.number_of_antennas = 1
        self.mock_tx_modems = [self.mock_tx_modem] * \
            self.params_scenario.number_of_tx_modems

        self.mock_rx_modem = Mock()
        self.mock_rx_modem.param.number_of_antennas = 1
        self.mock_rx_modem.waveform_generator.param.sampling_rate = 1e9
        self.mock_rx_modem.param.tx_modem = 0
        self.mock_rx_modems = [self.mock_rx_modem] * \
            self.params_scenario.number_of_rx_modems

    def test_proper_no_transmit_modems(self) -> None:
        sources, tx_modems = self.scenario._create_transmit_modems()

        self.assertEqual(
            self.params_scenario.number_of_tx_modems,
            len(tx_modems))
        self.assertEqual(
            self.params_scenario.number_of_tx_modems,
            len(sources))

    def test_proper_no_receiver_modem_generation(self) -> None:

        self.scenario.sources, self.scenario.tx_modems = self.scenario._create_transmit_modems()

        rx_samplers, rx_modems, noise = self.scenario._create_receiver_modems()

        self.assertEqual(
            len(rx_samplers),
            self.params_scenario.number_of_rx_modems)
        self.assertEqual(
            len(rx_modems),
            self.params_scenario.number_of_rx_modems)
        self.assertEqual(len(noise), self.params_scenario.number_of_rx_modems)

    def test_proper_no_channels_creation(self) -> None:
        # calculate expected values
        no_tx_modems_expected = self.params_scenario.number_of_tx_modems
        no_rx_modems_expected = self.params_scenario.number_of_rx_modems
        no_channels_expected = no_tx_modems_expected * no_rx_modems_expected

        self.scenario.rx_modems = self.mock_rx_modems  # type: ignore
        self.scenario.tx_modems = self.mock_tx_modems  # type: ignore

        channels = self.scenario._create_channels()
        channels_flattened = [item for sublist in channels for item in sublist]

        self.assertEqual(len(channels_flattened), no_channels_expected)

    def test_number_of_calls_init_drop(self) -> None:
        # test number of calls for sources
        with patch.object(BitsSource, 'init_drop', return_value=None) as mock_source_init_drop:
            self.scenario.sources, _ = self.scenario._create_transmit_modems()
            self.scenario.init_drop()

            self.assertEqual(
                mock_source_init_drop.call_count, len(
                    self.scenario.sources))

        # check number of calls for channel
        with patch.object(Channel, 'init_drop', return_value=None) as mock_channel_init_drop:
            self.scenario.rx_modems = self.mock_rx_modems  # type: ignore
            self.scenario.tx_modems = self.mock_tx_modems  # type: ignore
            self.scenario.channels = self.scenario._create_channels()

            channels_flattened = [
                item for sublist in self.scenario.channels for item in sublist]

            self.scenario.init_drop()
            self.assertEqual(
                mock_channel_init_drop.call_count,
                len(channels_flattened))

    def test_get_channel_instance(self) -> None:
        scenario = Scenario(
            self.params_scenario,
            self.params_general,
            self.rnd)

        channels = scenario._Scenario__get_channel_instance(  # type: ignore
            self.params_scenario.channel_model_params[0][0],
            self.mock_rx_modems[0],
            self.mock_tx_modems[0]
        )
        self.assertTrue(isinstance(channels, Channel))


if __name__ == '__main__':

    os.chdir("../../..")
    unittest.main()
