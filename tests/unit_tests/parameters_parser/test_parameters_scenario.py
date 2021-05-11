import unittest
from unittest.mock import patch, Mock
import os

from parameters_parser.parameters_scenario import ParametersScenario
from parameters_parser.parameters_tx_modem import ParametersTxModem
from parameters_parser.parameters_rx_modem import ParametersRxModem
from parameters_parser.parameters_channel import ParametersChannel


class TestParameterScenario(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 'res')
        self.params_scenario = ParametersScenario()
        self.params_scenario.read_params(
            os.path.join(
                self.parameters_dir,
                'settings_scenario.ini'))

        self.no_rx_modems = 2
        self.no_tx_modems = 3
        self.no_channels = self.no_rx_modems * self.no_tx_modems

    def test_read_parameters_correctly_tx_modem(self) -> None:
        self.assertEqual(
            self.params_scenario.number_of_tx_modems,
            self.no_tx_modems)
        self.assertEqual(
            len(self.params_scenario.tx_modem_params), self.no_tx_modems)
        self.assertTrue(
            isinstance(
                self.params_scenario.tx_modem_params[0],
                ParametersTxModem))

    def test_read_parameters_correctly_rx_modem(self) -> None:
        self.assertEqual(
            self.params_scenario.number_of_rx_modems,
            self.no_rx_modems)
        self.assertEqual(
            len(self.params_scenario.rx_modem_params), self.no_rx_modems)
        self.assertTrue(
            isinstance(
                self.params_scenario.rx_modem_params[0],
                ParametersRxModem))

    def test_read_parameters_correctly_channels(self) -> None:
        self.assertTrue(
            isinstance(
                self.params_scenario.channel_model_params[0][0],
                ParametersChannel))

    def test_check_correct_no_modems(self) -> None:
        self.params_scenario.number_of_tx_modems = 0
        self.assertRaises(
            ValueError,
            lambda: self.params_scenario._check_params(''))

        self.params_scenario.number_of_tx_modems = 1
        self.params_scenario.number_of_rx_modems = 0
        self.assertRaises(
            ValueError,
            lambda: self.params_scenario._check_params(''))

    @patch.object(ParametersChannel, 'check_params')
    @patch.object(ParametersRxModem, 'check_params')
    @patch.object(ParametersTxModem, 'check_params')
    def test_tx_modem_params_check(self, mock_tx_modem_check_params: Mock,
                                   mock_rx_modem_check_params: Mock,
                                   mock_params_channel_check_params: Mock
                                   ) -> None:
        self.params_scenario._check_params('')
        self.assertEqual(
            mock_tx_modem_check_params.call_count,
            self.no_tx_modems)
        self.assertEqual(
            mock_rx_modem_check_params.call_count,
            self.no_rx_modems)
        self.assertEqual(
            mock_params_channel_check_params.call_count,
            self.no_channels)
