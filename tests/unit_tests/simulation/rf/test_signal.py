# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from unit_tests.core.test_factory import test_roundtrip_serialization
from unit_tests.utils import random_rf_signal, assert_signals_equal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRFSignal(TestCase):
    """Test the RF expansion to signal models."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)

        self.bandwidth = 1e6
        self.oversampling_factor = 4
        self.sampling_rate = self.bandwidth * self.oversampling_factor
        self.carrier_frequencies = np.array([1e6, 2e6, 3e6])
        self.noise_powers = np.array([1e-10, 1e-20, 1e-15])
        self.delay = 0.1

        self.signal = random_rf_signal(3, 100, self.bandwidth, self.oversampling_factor, self.rng)
        self.signal.carrier_frequencies = self.carrier_frequencies
        self.signal.noise_powers = self.noise_powers
        self.signal.delay = self.delay

    def test_copy(self) -> None:
        """Test copying an RF signal"""

        copied_signal = self.signal.copy()

        # Ensure the mutable content has actually been copied
        self.assertIsNot(self.signal, copied_signal)
        self.assertIsNot(self.signal.carrier_frequencies, copied_signal.carrier_frequencies)
        self.assertIsNot(self.signal.noise_powers, copied_signal.noise_powers)

        # Ensure the copied content's integrity
        assert_array_equal(copied_signal.carrier_frequencies, self.signal.carrier_frequencies)
        assert_array_equal(copied_signal.noise_powers, self.signal.noise_powers)
        self.assertEqual(copied_signal.delay, self.signal.delay)
        assert_signals_equal(self, self.signal, copied_signal)

    def test_slicing_single_stream_selection(self) -> None:
        """Test RF signal slicing in the first dimension (stream selection)"""

        sliced_signal = self.signal[[1], :]
        self.assertCountEqual(sliced_signal.shape, (1, self.signal.num_samples))
        self.assertCountEqual(sliced_signal.carrier_frequencies.shape, (1,))
        self.assertCountEqual(sliced_signal.noise_powers.shape, (1,))
        self.assertEqual(sliced_signal.carrier_frequencies[0], self.carrier_frequencies[1])
        self.assertEqual(sliced_signal.noise_powers[0], self.noise_powers[1])
        self.assertEqual(sliced_signal.delay, self.delay)

    def test_slicing_multi_stream_selection(self) -> None:
        """Test RF signal slicing in the first dimension (stream selection)"""

        sliced_signal = self.signal[[0, 2], :]
        self.assertCountEqual(sliced_signal.shape, (2, self.signal.num_samples))
        self.assertCountEqual(sliced_signal.carrier_frequencies.shape, (2,))
        self.assertCountEqual(sliced_signal.noise_powers.shape, (2,))
        self.assertEqual(sliced_signal.carrier_frequencies[0], self.carrier_frequencies[0])
        self.assertEqual(sliced_signal.carrier_frequencies[1], self.carrier_frequencies[2])
        self.assertEqual(sliced_signal.noise_powers[0], self.noise_powers[0])
        self.assertEqual(sliced_signal.noise_powers[1], self.noise_powers[2])
        self.assertEqual(sliced_signal.delay, self.delay)

    def test_scalar_slicing(self) -> None:
        """Test RF signal slicing in the first dimension (stream selection)"""

        sliced_signal = self.signal[1]
        self.assertCountEqual(sliced_signal.shape, (1, self.signal.num_samples))
        self.assertCountEqual(sliced_signal.carrier_frequencies.shape, (1,))
        self.assertCountEqual(sliced_signal.noise_powers.shape, (1,))
        self.assertEqual(sliced_signal.carrier_frequencies[0], self.carrier_frequencies[1])
        self.assertEqual(sliced_signal.noise_powers[0], self.noise_powers[1])
        self.assertEqual(sliced_signal.delay, self.delay)

    def test_serialization(self) -> None:
        """Test serialization of RF signals"""

        test_roundtrip_serialization(self, self.signal)
