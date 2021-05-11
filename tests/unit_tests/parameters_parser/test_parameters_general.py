import unittest
import os

import numpy as np

from parameters_parser.parameters_general import ParametersGeneral


class TestParametersGeneral(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), 'res')
        self.params = ParametersGeneral()
        self.params.read_params(
            os.path.join(
                self.parameters_dir,
                'settings_general.ini'))

    def test_parameter_read_correct_values(self) -> None:
        self.assertEqual(self.params.seed, 11)
        self.assertEqual(self.params.drop_length, 0.1)
        self.assertEqual(self.params.min_num_drops, 5)
        self.assertEqual(self.params.max_num_drops, 20)
        self.assertEqual(self.params.confidence_level, 0.5)
        self.assertEqual(self.params.confidence_margin, 0.1)
        self.assertEqual(self.params.confidence_metric, 'BER')
        self.assertEqual(self.params.snr_type, 'EB/N0(DB)')
        np.testing.assert_array_equal(self.params.snr_vector, np.arange(30))
        self.assertEqual(self.params.verbose, True)
        self.assertEqual(self.params.plot, True)
        self.assertEqual(self.params.calc_theory, True)
        self.assertEqual(self.params.calc_spectrum_tx, True)
        self.assertEqual(self.params.calc_stft_tx, True)
        self.assertEqual(self.params.spectrum_fft_size, 512)

    def test_parameter_check_for_drop_loop(self) -> None:
        self.params.min_num_drops = 0
        self.assertRaises(ValueError, lambda: self.params._check_params())

        self.params.min_num_drops = 1
        self.params.max_num_drops = 0
        self.assertRaises(ValueError, lambda: self.params._check_params())

        self.params.max_num_drops = 1
        self.params.min_num_drops = 2
        self.assertRaises(ValueError, lambda: self.params._check_params())

        self.params.max_num_drops = 3
        self.params.drop_length = -1
        self.assertRaises(ValueError, lambda: self.params._check_params())

        self.params.drop_length = 1
        self.params.confidence_metric = 'INVALID'
        self.assertRaises(ValueError, lambda: self.params._check_params())

        self.params.confidence_metric = 'BER'
        self.params.confidence_level = 0
        self.assertRaises(ValueError, lambda: self.params._check_params())

        self.params.confidence_margin = -1
        self.assertRaises(ValueError, lambda: self.params._check_params())

    def test_parameters_check_noise_loop(self) -> None:
        self.params.snr_type = 'INVALID'
        self.assertRaises(ValueError, lambda: self.params._check_params())

        self.params.snr_type = 'EB_N0(DB)'
        self.params.snr_vector = 'CANNOT_BE_EVALUATED'
        self.assertRaises(ValueError, lambda: self.params._check_params())

    def test_parameters_check_statistics(self) -> None:
        self.params.snr_vector = 'np.arange(30)'
        self.params.spectrum_fft_size = -1
        self.assertRaises(ValueError, lambda: self.params._check_params())
