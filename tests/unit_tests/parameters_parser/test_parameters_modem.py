import unittest
from unittest.mock import Mock
from typing import Any

import numpy as np

from parameters_parser.parameters_modem import ParametersModem
from parameters_parser.parameters_repetition_encoder import ParametersRepetitionEncoder


class TestParametersModem(unittest.TestCase):
    def setUp(self) -> None:
        self.stubbed_params_modem = StubParametersModem()
        self.stubbed_params_modem.position = [0, 0, 0]
        self.velocity = [1, 1, 1]
        self.stubbed_params_modem.velocity = self.velocity
        self.stubbed_params_modem.number_of_antennas = 2
        self.stubbed_params_modem.device_type = "UE"
        self.stubbed_params_modem._correlation = "LOW"

    def test_invalid_position(self) -> None:
        self.stubbed_params_modem.position = [0, 0]
        self.assertRaises(
            ValueError,
            lambda: self.stubbed_params_modem.check_params())

        self.stubbed_params_modem.position = [1, 'a', 4]  # type: ignore
        self.assertRaises(
            ValueError,
            lambda: self.stubbed_params_modem.check_params())

    def test_invalid_velocity(self) -> None:
        self.stubbed_params_modem.velocity = [0, 0]
        self.assertRaises(
            ValueError,
            lambda: self.stubbed_params_modem.check_params())

        self.stubbed_params_modem.velocity = ['3#', 3, 4]
        self.assertRaises(
            ValueError,
            lambda: self.stubbed_params_modem.check_params())

    def test_velocity_list_to_array_conversion(self) -> None:
        self.stubbed_params_modem.check_params()

        np.testing.assert_array_almost_equal(
            self.stubbed_params_modem.velocity, np.array(self.velocity)
        )

    def test_number_of_antennas_check(self) -> None:
        self.stubbed_params_modem.number_of_antennas = -1
        self.assertRaises(
            ValueError,
            lambda: self.stubbed_params_modem.check_params())

        self.stubbed_params_modem.number_of_antennas = 1.5  # type: ignore
        self.assertRaises(
            ValueError,
            lambda: self.stubbed_params_modem.check_params())

    def test_supportNoneEncoder(self) -> None:
        self.stubbed_params_modem._encoder_param_file = "NONE"
        self.stubbed_params_modem.check_params()

        self.assertTrue(isinstance(self.stubbed_params_modem.encoding_params[0], ParametersRepetitionEncoder))
        self.assertEqual(self.stubbed_params_modem.encoding_params[0].data_bits_k, 1)
        self.assertEqual(self.stubbed_params_modem.encoding_params[0].encoded_bits_n, 1)

    def test_invalid_antenna_spacing(self) -> None:
        self.stubbed_params_modem.antenna_spacing = -1
        self.assertRaises(
            ValueError,
            lambda: self.stubbed_params_modem.check_params()
        )

    def test_invalid_device_type(self) -> None:
        self.stubbed_params_modem.device_type = "invalid"
        self.assertRaises(
            ValueError,
            lambda: self.stubbed_params_modem.check_params()
        )

    def test_crc_bits_set_too_zero_if_no_encoder(self) -> None:
        self.stubbed_params_modem.encoding_type = ""
        self.stubbed_params_modem.encoding_params = Mock()
        self.stubbed_params_modem.check_params()
        self.assertEqual(self.stubbed_params_modem.crc_bits, 0)

    def test_crc_bits_set_too_zero_if_no_encoder_file(self) -> None:
        self.stubbed_params_modem._encoder_param_file = "NONE"
        self.stubbed_params_modem.encoding_params = Mock()
        self.stubbed_params_modem.check_params()
        self.assertEqual(self.stubbed_params_modem.crc_bits, 0)


class StubParametersModem(ParametersModem):
    def read_params(self, section: Any) -> None:
        super().read_params(section)

    def check_params(self) -> None:
        super().check_params()
