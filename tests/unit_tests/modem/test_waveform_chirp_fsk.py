# -*- coding: utf-8 -*-

import unittest
import os
from math import ceil
from unittest.mock import Mock
from typing_extensions import override

import numpy as np

from hermespy.modem.modem import Symbols
from hermespy.modem.waveform_chirp_fsk import ChirpFSKWaveform, ChirpFSKSynchronization, ChirpFSKCorrelationSynchronization
from unit_tests.core.test_factory import test_roundtrip_serialization  # type: ignore
from .test_waveform import TestCommunicationWaveform

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestChirpFSKWaveform(TestCommunicationWaveform):
    """Test the chirp frequency shift keying waveform generation"""

    waveform: ChirpFSKWaveform

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        self.bandwidth = 1e6
        self.modulation_order = 32
        self.chirp_duration = 256 / self.bandwidth
        self.freq_difference = 0.0
        self.num_pilot_chirps = 2
        self.num_data_chirps = 1000
        self.guard_interval = 4e-6

        self.waveform = ChirpFSKWaveform(
            chirp_duration=self.chirp_duration,
            freq_difference=self.freq_difference,
            modulation_order=self.modulation_order,
            num_pilot_chirps=self.num_pilot_chirps,
            num_data_chirps=self.num_data_chirps,
            guard_interval=self.guard_interval,
        )

        self.data_bits_per_symbol = 5
        self.data_bits_in_frame = int(np.log2(self.modulation_order)) * self.num_data_chirps
        self.data_bits = np.random.randint(0, 2, self.data_bits_in_frame)

        self.parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "res")

    def test_init(self) -> None:
        """Test that the init routine properly assigns all parameters"""

        self.assertEqual(self.waveform.modulation_order, self.modulation_order, "Modulation order init produced unexpected result")
        self.assertEqual(self.waveform.chirp_duration, self.chirp_duration, "Chirp duration init produced unexpected result")
        self.assertEqual(self.waveform.freq_difference, self.freq_difference, "Frequency difference init produced unexpected result")
        self.assertEqual(self.waveform.num_pilot_chirps, self.num_pilot_chirps, "Number of pilot chirps init produced unexpected result")
        self.assertEqual(self.waveform.num_data_chirps, self.num_data_chirps, "Number of data chirps init produced unexpected result")
        self.assertEqual(self.waveform.guard_interval, self.guard_interval, "Guard interval init produced unexpected result")

    def test_frame_duration(self) -> None:
        """Test the valid calculation of frame durations"""

        frame_duration = self.chirp_duration * (self.num_pilot_chirps + self.num_data_chirps)
        expected_duration = self.guard_interval + frame_duration

        self.assertEqual(self.waveform.frame_duration(self.bandwidth), expected_duration, "Frame duration computation produced unexpected result")

    def test_duration(self) -> None:
        """Test the valid calculation of chirp durations"""

        self.assertEqual(self.chirp_duration, self.waveform.chirp_duration, "Chirp duration get produced unexpected result")

    def test_chirp_duration_get_set(self) -> None:
        """Test that chirp duration getter returns set value"""

        expected_chirp_duration = 20
        self.waveform.chirp_duration = expected_chirp_duration
        self.assertEqual(expected_chirp_duration, self.waveform.chirp_duration, "Chirp duration set/get returned unexpected result")

    def test_chirp_duration_validation(self) -> None:
        """Test the validation of chirp duration parameters during set"""

        with self.assertRaises(ValueError):
            self.waveform.chirp_duration = -1.0

        try:
            self.waveform.chirp_duration = 0.0

        except ValueError:
            self.fail("Chirp duration set produced unexpected exception")

    def test_freq_difference_get_set(self) -> None:
        """Test that frequency difference get returns set value"""

        freq_difference = 0.5
        self.waveform.freq_difference = freq_difference

        self.assertEqual(freq_difference, self.waveform.freq_difference, "Frequency difference set/get returned unexpected result")

    def test_freq_difference_validation(self) -> None:
        """Test the validation of frequency difference during set"""

        with self.assertRaises(ValueError):
            self.waveform.freq_difference = -1.0

    def test_num_pilot_chirps_get_set(self) -> None:
        """Test that the number of pilot chirps get returns set value"""

        num_pilot_chirps = 2
        self.waveform.num_pilot_chirps = num_pilot_chirps

        self.assertEqual(num_pilot_chirps, self.waveform.num_pilot_chirps, "Number of pilot chirps set/get returned unexpected result")

    def test_num_pilot_chirps_validation(self) -> None:
        """Test the validation of the number of pilot chirps during set"""

        with self.assertRaises(ValueError):
            self.waveform.num_pilot_chirps = -1

    def test_num_data_chirps_get_set(self) -> None:
        """Test that the number of data chirps get returns set value"""

        num_data_chirps = 2
        self.waveform.num_data_chirps = num_data_chirps

        self.assertEqual(num_data_chirps, self.waveform.num_data_chirps, "Number of data chirps set/get returned unexpected result")

    def test_num_data_chirps_validation(self) -> None:
        """Test the validation of the number of data chirps during set"""

        with self.assertRaises(ValueError):
            self.waveform.num_data_chirps = -1

    def test_guard_interval_get_set(self) -> None:
        """Test that the guard interval get returns set value"""

        guard_interval = 2.0
        self.waveform.guard_interval = guard_interval

        self.assertEqual(guard_interval, self.waveform.guard_interval, "Guard interval set/get returned unexpected result")

    def test_guard_interval_validation(self) -> None:
        """Test the validation of the guard interval during set"""

        with self.assertRaises(ValueError):
            self.waveform.guard_interval = -1.0

        try:
            self.waveform.guard_interval = 0.0

        except ValueError:
            self.fail("Guard interval set produced unexpected exception")

    def test_bits_per_symbol_calculation(self) -> None:
        """Test the calculation of bits per symbol"""

        expected_bits_per_symbol = int(np.log2(self.modulation_order))
        self.assertEqual(expected_bits_per_symbol, self.waveform.bits_per_symbol, "Bits per symbol calculation produced unexpected result")

    def test_bits_per_frame_calculation(self) -> None:
        """Test the calculation of number of bits contained within a single frame"""

        self.assertEqual(self.data_bits_in_frame, self.waveform.bits_per_frame(self.waveform.num_data_symbols), "Bits per frame calculation produced unexpected result")

    def test_samples_in_chirp_calculation(self) -> None:
        """Test the calculation for the number of samples within one chirp"""

        expected_samples_in_chirp = int(ceil(self.chirp_duration * self.bandwidth * 2))
        self.assertEqual(expected_samples_in_chirp, self.waveform.samples_in_chirp(self.bandwidth, 2), "Samples in chirp calculation produced unexpected result")

    def test_chirps_in_frame_calculation(self) -> None:
        """Test the calculation for the number of chirps per transmitted frame"""

        chirps_in_frame_expected = self.num_data_chirps + self.num_pilot_chirps
        self.assertEqual(chirps_in_frame_expected, self.waveform.chirps_in_frame, "Calculation of number of chirps in frame produced unexpected result")

    def test_pick(self) -> None:
        """Test the picking of symbols"""

        symbols = Mock()
        picked_symbols = self.waveform.pick(symbols)
        self.assertIs(symbols, picked_symbols)

    @override
    def test_energy(self) -> None:
        self.waveform.guard_interval = 0.0
        self.waveform.num_pilot_chirps = 0
        return super().test_energy()

    @override
    def test_power(self) -> None:
        self.waveform.guard_interval = 0.0
        return super().test_power()

    def test_symbol_precoding_support(self) -> None:
        """Test if symbol precoding is supported"""

        self.assertFalse(self.waveform.symbol_precoding_support)

    def test_serialization(self) -> None:
        """Test waveform serialization"""

        test_roundtrip_serialization(self, self.waveform, {'modem'})


class TestChirpFskSynchronization(unittest.TestCase):
    """Test chirp FSK synchronization base class"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.waveform = ChirpFSKWaveform()

        self.synchronization = self.waveform.synchronization

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as object attributes"""

        self.assertIs(self.waveform, self.synchronization.waveform)
        self.assertIsInstance(self.waveform.synchronization, ChirpFSKSynchronization)

    def test_serialization(self) -> None:
        """Test synchronization serialization"""

        test_roundtrip_serialization(self, self.synchronization, {'waveform'})


class TestChirpFskCorrelationSynchronization(unittest.TestCase):
    """Test correlation-based clock synchronization for the chirp FSK waveform"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.num_frames = 3
        self.max_offset = 200  # Maximum synchronization offset in samples
        self.threshold = 0.94
        self.guard_ratio = 0.5
        self.bandwidth = 1e6
        self.oversampling_factor = 4
        self.chirp_duration = 100 / self.bandwidth

        self.waveform = ChirpFSKWaveform(self.chirp_duration)
        self.waveform.num_pilot_chirps = 5
        self.waveform.num_data_chirps = 20

        self.synchronization = ChirpFSKCorrelationSynchronization(threshold=self.threshold, guard_ratio=self.guard_ratio, waveform=self.waveform)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertEqual(self.threshold, self.synchronization.threshold)
        self.assertEqual(self.guard_ratio, self.synchronization.guard_ratio)

    def test_threshold_setget(self) -> None:
        """Threshold property getter should return setter argument"""

        threshold = 0.25
        self.synchronization.threshold = threshold

        self.assertEqual(threshold, self.synchronization.threshold)

    def test_threshold_validation(self) -> None:
        """Threshold property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.synchronization.threshold = -1.0

        with self.assertRaises(ValueError):
            self.synchronization.threshold = 2.0

        try:
            self.synchronization.threshold = 0.0
            self.synchronization.threshold = 1.0

        except ValueError:
            self.fail()

    def test_guard_ratio_setget(self) -> None:
        """Guard ratio property getter should return setter argument"""

        guard_ratio = 0.25
        self.synchronization.guard_ratio = guard_ratio

        self.assertEqual(guard_ratio, self.synchronization.guard_ratio)

    def test_guard_ratio_validation(self) -> None:
        """Guard ratio property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.synchronization.guard_ratio = -1.0

        with self.assertRaises(ValueError):
            self.synchronization.guard_ratio = 2.0

        try:
            self.synchronization.guard_ratio = 0.0
            self.synchronization.guard_ratio = 1.0

        except ValueError:
            self.fail()

    def test_synchronization(self) -> None:
        """Synchronization should properly partition signal samples into frame sections"""

        # Generate frame signal models
        num_samples = 2 * self.max_offset + self.num_frames * self.waveform.samples_per_frame(self.bandwidth, self.oversampling_factor)
        samples = np.zeros((1, num_samples), dtype=complex)
        expected_frames = []
        pilot_indices = self.rng.integers(0, self.max_offset, self.num_frames) + np.arange(self.num_frames) * self.waveform.samples_per_frame(self.bandwidth, self.oversampling_factor)

        for p in pilot_indices:
            data_symbols = Symbols(self.rng.integers(0, self.waveform.modulation_order, self.waveform.num_data_symbols))
            signal_samples = self.waveform.modulate(self.waveform.place(data_symbols), self.bandwidth, self.oversampling_factor)

            samples[:, p : p + self.waveform.samples_per_frame(self.bandwidth, self.oversampling_factor)] += signal_samples
            expected_frames.append(samples[:, p : p + self.waveform.samples_per_frame(self.bandwidth, self.oversampling_factor)])

        synchronized_frames = self.synchronization.synchronize(samples, self.bandwidth, self.oversampling_factor)
        self.assertSequenceEqual(list(pilot_indices), synchronized_frames[: self.num_frames])

    def test_synchronization_validation(self) -> None:
        """Synchronization should raise RuntimeError if no pilot signal is available"""

        self.waveform.num_pilot_chirps = 0
        mock_signal = np.zeros((1, 1000), dtype=complex)
        with self.assertRaises(RuntimeError):
            _ = self.synchronization.synchronize(mock_signal, self.bandwidth, self.oversampling_factor)

    def test_serialization(self) -> None:
        """Test synchronization serialization"""

        test_roundtrip_serialization(self, self.synchronization, {'waveform'})
