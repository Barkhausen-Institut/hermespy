import unittest

from parameters_parser.parameters_chirp_fsk import ParametersChirpFsk


class TestParametersChirpFsk(unittest.TestCase):

    def setUp(self) -> None:
        self.params_chirp_fsk = ParametersChirpFsk()

        self.params_chirp_fsk.modulation_order = 2
        self.params_chirp_fsk.chirp_duration = 1
        self.params_chirp_fsk.chirp_bandwidth = 1
        self.params_chirp_fsk.freq_difference = 0.5
        self.params_chirp_fsk.oversampling_factor = 8

    def test_check_invalid_modulation_parameters(self) -> None:
        self.params_chirp_fsk.modulation_order = 0
        self.assertRaises(
            ValueError,
            lambda: self.params_chirp_fsk._check_params())

        self.params_chirp_fsk.modulation_order = 2
        self.params_chirp_fsk.chirp_duration = 0
        self.assertRaises(
            ValueError,
            lambda: self.params_chirp_fsk._check_params())

        self.params_chirp_fsk.chirp_duration = 1
        self.params_chirp_fsk.chirp_bandwidth = 0
        self.assertRaises(
            ValueError,
            lambda: self.params_chirp_fsk._check_params())

        self.params_chirp_fsk.chirp_bandwidth = 1
        self.params_chirp_fsk.freq_difference = -1
        self.assertRaises(
            ValueError,
            lambda: self.params_chirp_fsk._check_params())

        self.params_chirp_fsk.freq_difference = 2
        self.assertRaises(
            ValueError,
            lambda: self.params_chirp_fsk._check_params())

        self.params_chirp_fsk.freq_difference = 0.5
        self.params_chirp_fsk.oversampling_factor = -8
        self.assertRaises(
            ValueError,
            lambda: self.params_chirp_fsk._check_params())

    def test_check_invalid_frame_parameters(self) -> None:
        self.params_chirp_fsk.number_pilot_chirps = -1
        self.assertRaises(
            ValueError,
            lambda: self.params_chirp_fsk._check_params())

        self.params_chirp_fsk.number_pilot_chirps = 1
        self.params_chirp_fsk.number_data_chirps = -1
        self.assertRaises(
            ValueError,
            lambda: self.params_chirp_fsk._check_params())

        self.params_chirp_fsk.number_data_chirps = 1
        self.params_chirp_fsk.guard_interval = 0
        self.assertRaises(
            ValueError,
            lambda: self.params_chirp_fsk._check_params())
