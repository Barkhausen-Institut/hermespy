import unittest
import os

from parameters_parser.parameters_psk_qam import ParametersPskQam


class TestParametersPskQam(unittest.TestCase):

    def setUp(self) -> None:
        self.params_psk_qam = ParametersPskQam()

        self.params_psk_qam.modulation_order = 2
        self.params_psk_qam.modulation_is_complex = True
        self.params_psk_qam.symbol_rate = 1
        self.params_psk_qam.oversampling_factor = 8
        self.params_psk_qam.filter_type = 'ROOT_RAISED_COSINE'
        self.params_psk_qam.roll_off_factor = 0.5
        self.params_psk_qam.filter_length_in_symbols = 10

        self.parameters_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 'res')

    def test_invalid_modulation_parameters(self) -> None:
        self.params_psk_qam.modulation_order = 200
        self.params_psk_qam.modulation_is_complex = True
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

        self.params_psk_qam.modulation_order = 64
        self.params_psk_qam.modulation_is_complex = False
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

        self.params_psk_qam.modulation_order = 2
        self.params_psk_qam.symbol_rate = 0
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

        self.params_psk_qam.symbol_rate = 1
        self.params_psk_qam.oversampling_factor = -8
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

        self.params_psk_qam.oversampling_factor = 8
        self.params_psk_qam.filter_type = 'INVALID_FILTER_TYPE'
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

        self.params_psk_qam.filter_type = 'ROOT_RAISED_COSINE'
        self.params_psk_qam.roll_off_factor = -1
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

        self.params_psk_qam.roll_off_factor = .5
        self.params_psk_qam.filter_length_in_symbols = 0
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

        self.params_psk_qam.filter_type = "FMCW"
        self.params_psk_qam._chirp_duration_str = "dummy"
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())
        self.params_psk_qam._chirp_duration_str = "-1/symbol_rate"
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

        self.params_psk_qam._chirp_duration_str = "10/symbol_rate"
        self.params_psk_qam._chirp_bandwidth_str = "-symbol_rate"
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

    def test_invalid_frame_parameters(self) -> None:

        self.params_psk_qam.number_preamble_symbols = -1
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

        self.params_psk_qam.number_preamble_symbols = 10
        self.params_psk_qam.number_data_symbols = 0
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

        self.params_psk_qam.number_data_symbols = 100
        self.params_psk_qam.number_postamble_symbols = -1
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

        self.params_psk_qam.number_postamble_symbols = 0
        self.params_psk_qam.guard_interval = -1
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

        self.params_psk_qam.guard_interval = 1e-6
        self.params_psk_qam._pilot_symbol_rate_str = "dummy"
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())
        self.params_psk_qam._pilot_symbol_rate_str = "-symbol_rate"
        self.assertRaises(
            ValueError,
            lambda: self.params_psk_qam._check_params())

        # now it should work
        self.params_psk_qam._pilot_symbol_rate_str = "10 * symbol_rate"
        self.params_psk_qam._check_params()

    def test_parameters_reading_with_raised_cosine(self) -> None:
        self.params = ParametersPskQam()
        self.params.read_params(
            os.path.join(
                self.parameters_dir,
                'settings_psk_qam_rc.ini'))

        self.assertEqual(self.params.modulation_order, 8)
        self.assertTrue(self.params.modulation_is_complex)
        self.assertEqual(self.params.symbol_rate, 125e3)
        self.assertEqual(self.params.filter_type, "ROOT_RAISED_COSINE")
        self.assertEqual(self.params.roll_off_factor, .5)
        self.assertEqual(self.params.oversampling_factor, 2)
        self.assertEqual(self.params.filter_length_in_symbols, 16)

        self.assertEqual(self.params.number_preamble_symbols, 2)
        self.assertEqual(self.params.number_data_symbols, 20)
        self.assertEqual(self.params.number_postamble_symbols, 2)

        self.assertEqual(self.params.pilot_symbol_rate, 125e3)
        self.assertEqual(self.params.guard_interval, 4e-6)
        self.assertEqual(self.params.bits_per_symbol, 3)
        self.assertEqual(self.params.bits_in_frame, 3 * 20)

    def test_parameters_reading_with_fmcw(self) -> None:
        self.params = ParametersPskQam()
        self.params.read_params(
            os.path.join(
                self.parameters_dir,
                'settings_psk_qam_fmcw.ini'))

        self.assertEqual(self.params.modulation_order, 16)
        self.assertFalse(self.params.modulation_is_complex)
        self.assertEqual(self.params.symbol_rate, 1e9)
        self.assertEqual(self.params.filter_type, "FMCW")
        self.assertEqual(self.params.chirp_bandwidth, 1e9)
        self.assertEqual(self.params.chirp_duration, 1e-8)
        self.assertEqual(self.params.oversampling_factor, 4)

        self.assertEqual(self.params.number_preamble_symbols, 5)
        self.assertEqual(self.params.number_data_symbols, 30)
        self.assertEqual(self.params.number_postamble_symbols, 0)

        self.assertEqual(self.params.pilot_symbol_rate, 1e-8)
        self.assertEqual(self.params.guard_interval, 0)


if __name__ == '__main__':
    unittest.main()
