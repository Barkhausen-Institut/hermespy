import unittest
from unittest.mock import Mock, patch

from parameters_parser.parameters_rx_modem import ParametersRxModem
from parameters_parser.parameters_modem import ParametersModem


class TestParametersRxModem(unittest.TestCase):

    def setUp(self) -> None:
        self.mock_tx_modem_params = [Mock()]
        self.mock_tx_modem_params[0].technology = 'LTE'
        self.mock_tx_modem_params[0].carrier_frequency = 1.6e9

        self.patcher_params_modem = patch.object(
            ParametersModem, 'check_params')
        self.mock_params_modem_check_params = self.patcher_params_modem.start()

        self.params_rx_modem = ParametersRxModem()
        self.tx_modem_idx = 0
        self.params_rx_modem.tx_modem = self.tx_modem_idx

    def test_if_parameters_modem_params_check_is_called(self) -> None:
        self.params_rx_modem.check_params(self.mock_tx_modem_params)  # type: ignore
        self.mock_params_modem_check_params.assert_called_once()

    def test_negative_tx_modem_idx(self) -> None:
        self.params_rx_modem.tx_modem = -1
        self.assertRaises(
            ValueError,
            lambda: self.params_rx_modem.check_params(
                self.mock_tx_modem_params))  # type: ignore

    def test_tx_modem_idx_larger_than_no_params(self) -> None:
        self.params_rx_modem.tx_modem = 1000
        self.assertRaises(
            ValueError,
            lambda: self.params_rx_modem.check_params(
                self.mock_tx_modem_params))  # type: ignore

    def test_member_assignment(self) -> None:
        self.params_rx_modem.check_params(self.mock_tx_modem_params)  # type: ignore
        self.assertEqual(
            self.params_rx_modem.technology,
            self.mock_tx_modem_params[self.tx_modem_idx - 1].technology
        )

        self.assertEqual(
            self.params_rx_modem.carrier_frequency,
            self.mock_tx_modem_params[self.tx_modem_idx - 1].carrier_frequency
        )
