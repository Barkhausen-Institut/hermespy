# -*- coding: utf-8 -*-

import unittest
import os
from math import ceil
from unittest.mock import Mock, patch, PropertyMock

import numpy as np

from hermespy.modem.modem import Symbols
from hermespy.modem.waveform_generator_chirp_fsk import ChirpFSKWaveform, ChirpFSKSynchronization,\
    ChirpFSKCorrelationSynchronization
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestChirpFSKWaveform(unittest.TestCase):
    """Test the chirp frequency shift keying waveform generation."""

    def setUp(self) -> None:

        self.generator = ChirpFSKWaveform.__new__(ChirpFSKWaveform)
        self.modem = Mock()
        self.modem.waveform_generator = self.generator

        self.parameters = {
            "modem": self.modem,
            "oversampling_factor": 4,
            "modulation_order": 32,
            "chirp_duration": 4e-6,
            "chirp_bandwidth": 200e6,
            "freq_difference": 5e6,
            "num_pilot_chirps": 2,
            "num_data_chirps": 20,
            "guard_interval": 4e-6,
        }

        self.modem.carrier_frequency = 1.0

        self.generator.__init__(**self.parameters)

        self.data_bits_per_symbol = 5
        self.data_bits_in_frame = 100
        self.data_bits = np.random.randint(0, 2, self.data_bits_in_frame)

        self.parent_dir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)), 'res')

    def test_init(self) -> None:
        """Test that the init routine properly assigns all parameters"""

        self.assertIs(self.modem, self.generator.modem,
                      "Modem init produced unexpected result")
        self.assertEqual(self.generator.oversampling_factor, self.parameters["oversampling_factor"],
                         "Oversampling factor init produced unexpected result")
        self.assertEqual(self.generator.modulation_order, self.parameters["modulation_order"],
                         "Modulation order init produced unexpected result")
        self.assertEqual(self.generator.chirp_duration, self.parameters["chirp_duration"],
                         "Chirp duration init produced unexpected result")
        self.assertEqual(self.generator.chirp_bandwidth, self.parameters["chirp_bandwidth"],
                         "Chirp bandwidth init produced unexpected result")
        self.assertEqual(self.generator.freq_difference, self.parameters["freq_difference"],
                         "Frequency difference init produced unexpected result")
        self.assertEqual(self.generator.num_pilot_chirps, self.parameters["num_pilot_chirps"],
                         "Number of pilot chirps init produced unexpected result")
        self.assertEqual(self.generator.num_data_chirps, self.parameters["num_data_chirps"],
                         "Number of data chirps init produced unexpected result")
        self.assertEqual(self.generator.guard_interval, self.parameters["guard_interval"],
                         "Guard interval init produced unexpected result")

    def test_frame_duration(self) -> None:
        """Test the valid calculation of frame durations."""

        frame_duration = self.parameters["chirp_duration"] * (self.parameters["num_pilot_chirps"] +
                                                              self.parameters["num_data_chirps"])
        expected_duration = self.parameters["guard_interval"] + frame_duration

        self.assertEqual(self.generator.frame_duration, expected_duration,
                         "Frame duration computation produced unexpected result")

    def test_chirp_duration(self) -> None:
        """Test the valid calculation of chirp durations."""

        expected_duration = self.parameters["chirp_duration"]
        self.assertEqual(expected_duration, self.generator.chirp_duration,
                         "Chirp duration get produced unexpected result")

    def test_chirp_duration_get_set(self) -> None:
        """Test that chirp duration getter returns set value."""

        chirp_duration = 20
        self.generator.chirp_duration = chirp_duration

        self.assertEqual(chirp_duration, self.generator.chirp_duration,
                         "Chirp duration set/get returned unexpected result")

    def test_chirp_duration_validation(self) -> None:
        """Test the validation of chirp duration parameters during set."""

        with self.assertRaises(ValueError):
            self.generator.chirp_duration = -1.0

        try:
            self.generator.chirp_duration = 0.0

        except ValueError:
            self.fail("Chirp duration set produced unexpected exception")

    def test_chirp_bandwidth_get_set(self) -> None:
        """Test that chirp bandwidth get returns set value."""

        bandwidth = 1.0
        self.generator.chirp_bandwidth = bandwidth

        self.assertEqual(bandwidth, self.generator.chirp_bandwidth,
                         "Chirp bandwidth set/get returned unexpected result")

    def test_chirp_bandwidth_validation(self) -> None:
        """Test the validation of chirp bandwidth parameters during set"""

        with self.assertRaises(ValueError):
            self.generator.chirp_bandwidth = -1.0

        with self.assertRaises(ValueError):
            self.generator.chirp_bandwidth = 0.0

    def test_freq_difference_get_set(self) -> None:
        """Test that frequency difference get returns set value."""

        freq_difference = 0.5
        self.generator.freq_difference = freq_difference

        self.assertEqual(freq_difference, self.generator.freq_difference,
                         "Frequency difference set/get returned unexpected result")

    def test_freq_difference_validation(self) -> None:
        """Test the validation of frequency difference during set"""

        with self.assertRaises(ValueError):
            self.generator.freq_difference = -1.0

        with self.assertRaises(ValueError):
            self.generator.freq_difference = 0.0

    def test_num_pilot_chirps_get_set(self) -> None:
        """Test that the number of pilot chirps get returns set value."""

        num_pilot_chirps = 2
        self.generator.num_pilot_chirps = num_pilot_chirps

        self.assertEqual(num_pilot_chirps, self.generator.num_pilot_chirps,
                         "Number of pilot chirps set/get returned unexpected result")

    def test_num_pilot_chirps_validation(self) -> None:
        """Test the validation of the number of pilot chirps during set"""

        with self.assertRaises(ValueError):
            self.generator.num_pilot_chirps = -1

    def test_num_data_chirps_get_set(self) -> None:
        """Test that the number of data chirps get returns set value."""

        num_data_chirps = 2
        self.generator.num_data_chirps = num_data_chirps

        self.assertEqual(num_data_chirps, self.generator.num_data_chirps,
                         "Number of data chirps set/get returned unexpected result")

    def test_num_data_chirps_validation(self) -> None:
        """Test the validation of the number of data chirps during set"""

        with self.assertRaises(ValueError):
            self.generator.num_data_chirps = -1

    def test_guard_interval_get_set(self) -> None:
        """Test that the guard interval get returns set value."""

        guard_interval = 2.0
        self.generator.guard_interval = guard_interval

        self.assertEqual(guard_interval, self.generator.guard_interval,
                         "Guard interval set/get returned unexpected result")

    def test_guard_interval_validation(self) -> None:
        """Test the validation of the guard interval during set"""

        with self.assertRaises(ValueError):
            self.generator.guard_interval = -1.0

        try:
            self.generator.guard_interval = 0.0

        except ValueError:
            self.fail("Guard interval set produced unexpected exception")

    def test_bits_per_symbol_calculation(self) -> None:
        """Test the calculation of bits per symbol."""

        expected_bits_per_symbol = int(np.log2(self.parameters["modulation_order"]))
        self.assertEqual(expected_bits_per_symbol, self.generator.bits_per_symbol,
                         "Bits per symbol calculation produced unexpected result")

    def test_bits_per_frame_calculation(self) -> None:
        """Test the calculation of number of bits contained within a single frame."""

        self.assertEqual(self.data_bits_in_frame, self.generator.bits_per_frame,
                         "Bits per frame calculation produced unexpected result")

    def test_samples_in_chirp_calculation(self) -> None:
        """Test the calculation for the number of samples within one chirp."""

        expected_samples_in_chirp = int(ceil(self.parameters["chirp_duration"] * self.generator.sampling_rate))
        self.assertEqual(expected_samples_in_chirp, self.generator.samples_in_chirp,
                         "Samples in chirp calculation produced unexpected result")

    def test_chirps_in_frame_calculation(self) -> None:
        """Test the calculation for the number of chirps per transmitted frame"""

        chirps_in_frame_expected = self.parameters["num_pilot_chirps"] + self.parameters["num_data_chirps"]

        self.assertEqual(chirps_in_frame_expected, self.generator.chirps_in_frame,
                         "Calculation of number of chirps in frame produced unexpected result")

    def test_prototype_chirps_for_modulation_symbols(self) -> None:
        """The generated prototypes for FSK signals must be proper."""

        cos_signal_expected = self.__read_saved_results_from_file('cos_signal.npy')
        sin_signal_expected = self.__read_saved_results_from_file('sin_signal.npy')

        prototypes, _ = self.generator._prototypes()
        expected_prototypes = .5 * cos_signal_expected + .5j * sin_signal_expected

        np.testing.assert_array_almost_equal(prototypes, expected_prototypes)

    def test_bit_energy_calculation(self) -> None:
        """Test the energy calculation for a single transmitted bit."""

        cos_signal_expected = self.__read_saved_results_from_file('cos_signal.npy')
        symbol_energy = sum(abs(cos_signal_expected[0, :]) ** 2)

        bit_energy_expected = symbol_energy / self.generator.bits_per_symbol
        bit_energy = self.generator.bit_energy
        self.assertAlmostEqual(bit_energy_expected, bit_energy)

    def test_symbol_energy_calculation(self) -> None:
        """Test the energy calculation for a single transmitted symbol."""

        cos_signal_expected = self.__read_saved_results_from_file('cos_signal.npy')
        symbol_energy_expected = sum(abs(cos_signal_expected[0, :]) ** 2)

        symbol_energy = self.generator.symbol_energy
        self.assertAlmostEqual(symbol_energy_expected, symbol_energy)

    def test_rx_signal_properly_demodulated(self) -> None:
        """Verify the proper demodulation of received signals."""

        rx_signal = self.__read_saved_results_from_file('rx_signal.npy')

        received_symbols = self.generator.demodulate(rx_signal)
        received_bits = self.generator.unmap(received_symbols)

        received_bits_expected = self.__read_saved_results_from_file('received_bits.npy').ravel()
        np.testing.assert_array_equal(received_bits[:len(received_bits_expected)], received_bits_expected)

    def test_proper_bit_energy_calculation(self) -> None:
        """Tests if theoretical bit energy is calculated correctly"""

        self.generator.guard_interval = 0.0
        self.generator.num_pilot_chirps = 0

        data_bits = np.random.randint(0, 2, self.generator.bits_per_frame)
        data_symbols = self.generator.map(data_bits)
        transmitted_signal = self.generator.modulate(data_symbols)

        symbol_energy = np.sum(abs(transmitted_signal.samples.flatten())**2) / (self.generator.num_pilot_chirps +
                                                                                self.generator.num_data_chirps)
        bit_energy = symbol_energy / self.generator.bits_per_symbol

        # compare the measured energy with the expected values
        self.assertAlmostEqual(bit_energy, self.generator.bit_energy,
                               msg="Unexpected bit energy transmitted")

    def test_proper_symbol_energy_calculation(self) -> None:
        """Tests if theoretical symbol energy is calculated correctly"""

        self.generator.guard_interval = 0.0
        self.generator.num_pilot_chirps = 0

        # define test parameters
        num_symbols = self.generator.chirps_in_frame

        data_bits = np.random.randint(0, 2, self.generator.bits_per_frame)
        data_symbols = self.generator.map(data_bits)
        transmitted_signal = self.generator.modulate(data_symbols)

        symbol_energy = np.sum(abs(transmitted_signal.samples.flatten())**2) / (num_symbols +
                                                                                self.generator.num_pilot_chirps)

        # compare the measured energy with the expected values
        self.assertAlmostEqual(symbol_energy, self.generator.symbol_energy,
                               msg="Unexpected symbol energy transmitted")

    def test_proper_power_calculation(self) -> None:
        """Tests if theoretical baseband_signal power is calculated correctly
        TODO: Check power calculation, since the delta is currently ~0.5, which seems kind of high
        """

        # define test parameters
        num_samples = self.generator.samples_in_frame

        data_bits = np.random.randint(0, 2, self.generator.bits_per_frame)
        data_symbols = self.generator.map(data_bits)
        transmitted_signal = self.generator.modulate(data_symbols)

        power = np.sum(abs(transmitted_signal.samples.flatten())**2) / num_samples
        self.assertAlmostEqual(power, self.generator.power, places=1,
                               msg="Unexpected baseband signal energy transmitted")

    def test_bandwidth(self) -> None:
        """Bandwidth property should return chirp bandwidth."""

        self.assertEqual(self.parameters['chirp_bandwidth'], self.generator.bandwidth)

    def __read_saved_results_from_file(self, file_name: str) -> dict[str, np.ndarray]:
        """Internal helper function for reading numpy arrays from save files.

        Args:
            file_name (str): The file location.

        Returns:
            np.ndarray: The contained numpy array.

        Raises:
            FileNotFoundError: If `file_name` does not exist within the parent directory.
        """

        if not os.path.exists(os.path.join(self.parent_dir, file_name)):
            raise FileNotFoundError(
                f"{file_name} must be in same folder as this file.")

        return np.load(os.path.join(self.parent_dir, file_name))
            
    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.modem.waveform_generator_chirp_fsk.ChirpFSKWaveform.modem', new_callable=PropertyMock) as blacklist:
        
            blacklist.return_value = {'modem'}
            test_yaml_roundtrip_serialization(self, self.generator, {'modem',})


class TestChirpFskSynchronization(unittest.TestCase):
    """Test chirp FSK synchronization base class."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        self.waveform_generator = ChirpFSKWaveform()

        self.synchronization = self.waveform_generator.synchronization

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as object attributes."""

        self.assertIs(self.waveform_generator, self.synchronization.waveform_generator)
        self.assertIsInstance(self.waveform_generator.synchronization, ChirpFSKSynchronization)


class TestChirpFskCorrelationSynchronization(unittest.TestCase):
    """Test correlation-based clock synchronization for the chirp FSK waveform."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        self.num_frames = 3
        self.max_offset = 200                   # Maximum synchronization offset in samples
        self.threshold = 0.94
        self.guard_ratio = 0.5

        self.modem = Mock()
        self.modem.carrier_frequency = 1e5
        self.waveform = ChirpFSKWaveform(modem=self.modem)
        self.waveform.num_pilot_chirps = 5
        self.waveform.num_data_chirps = 20
        self.waveform.oversampling_factor = 2

        self.synchronization = ChirpFSKCorrelationSynchronization(threshold=self.threshold,
                                                                  guard_ratio=self.guard_ratio,
                                                                  waveform_generator=self.waveform)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertEqual(self.threshold, self.synchronization.threshold)
        self.assertEqual(self.guard_ratio, self.synchronization.guard_ratio)

    def test_threshold_setget(self) -> None:
        """Threshold property getter should return setter argument."""

        threshold = 0.25
        self.synchronization.threshold = threshold

        self.assertEqual(threshold, self.synchronization.threshold)

    def test_threshold_validation(self) -> None:
        """Threshold property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.synchronization.threshold = -1.

        with self.assertRaises(ValueError):
            self.synchronization.threshold = 2.

        try:
            self.synchronization.threshold = 0.
            self.synchronization.threshold = 1.

        except ValueError:
            self.fail()
            
    def test_guard_ratio_setget(self) -> None:
        """Guard ratio property getter should return setter argument."""

        guard_ratio = 0.25
        self.synchronization.guard_ratio = guard_ratio

        self.assertEqual(guard_ratio, self.synchronization.guard_ratio)

    def test_guard_ratio_validation(self) -> None:
        """Guard ratio property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.synchronization.guard_ratio = -1.

        with self.assertRaises(ValueError):
            self.synchronization.guard_ratio = 2.

        try:
            self.synchronization.guard_ratio = 0.
            self.synchronization.guard_ratio = 1.

        except ValueError:
            self.fail()

    def test_synchronization(self) -> None:
        """Synchronization should properly partition signal samples into frame sections."""

        # Generate frame signal models
        num_samples = 2 * self.max_offset + self.num_frames * self.waveform.samples_in_frame
        samples = np.zeros((1, num_samples), dtype=complex)
        expected_frames = []
        pilot_indices = self.rng.integers(0, self.max_offset, self.num_frames) + np.arange(self.num_frames) * self.waveform.samples_in_frame
        
        for p in pilot_indices:

            data_symbols = Symbols(self.rng.integers(0, self.waveform.modulation_order,
                                                     self.waveform.symbols_per_frame))
            signal_samples = self.waveform.modulate(data_symbols).samples

            samples[:, p:p+self.waveform.samples_in_frame] += signal_samples
            expected_frames.append(samples[:, p:p+self.waveform.samples_in_frame])

        synchronized_frames = self.synchronization.synchronize(samples)

        if len(synchronized_frames) != len(expected_frames):
            self.fail()

    def test_synchronization_validation(self) -> None:
        """Synchronization should raise RuntimeError if no pilot signal is available."""

        self.waveform.num_pilot_chirps = 0
        with self.assertRaises(RuntimeError):
            _ = self.synchronization.synchronize(Mock())
