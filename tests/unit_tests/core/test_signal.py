# -*- coding: utf-8 -*-
"""Test the HermesPy Signal Model."""

import unittest

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi

from hermespy.core.signal_model import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSignal(unittest.TestCase):
    """Test the signal model base class."""

    def setUp(self) -> None:

        self.random = default_rng(42)

        self.num_streams = 3
        self.num_samples = 100
        self.sampling_rate = 1e4
        self.carrier_frequency = 1e3
        self.delay = 0.

        self.samples = (self.random.random((self.num_streams, self.num_samples)) +
                        1j * self.random.random((self.num_streams, self.num_samples)))

        self.signal = Signal(samples=self.samples, sampling_rate=self.sampling_rate,
                             carrier_frequency=self.carrier_frequency, delay=self.delay)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as object attributes."""

        assert_array_equal(self.samples, self.signal.samples)
        self.assertEqual(self.sampling_rate, self.signal.sampling_rate)
        self.assertEqual(self.carrier_frequency, self.signal.carrier_frequency)
        self.assertEqual(self.delay, self.signal.delay)

    def test_empty(self) -> None:
        """Using the empty initializer should result in an empty signal model."""

        sampling_rate = 2
        num_streams = 5
        num_samples = 6
        empty_signal = Signal.empty(sampling_rate, num_streams=num_streams, num_samples=num_samples)

        self.assertEqual(sampling_rate, empty_signal.sampling_rate)
        self.assertEqual(num_samples, empty_signal.num_samples)
        self.assertEqual(num_streams, empty_signal.num_streams)

    def test_samples_setget(self) -> None:
        """Samples property getter should return setter argument."""

        samples = (self.random.random((self.num_streams + 1, self.num_samples + 1)) +
                   1j * self.random.random((self.num_streams + 1, self.num_samples + 1)))

        self.signal.samples = samples
        assert_array_equal(samples, self.signal.samples)

    def test_samples_validation(self) -> None:
        """Samples property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.signal.samples = self.random.random((1, 2, 3))

    def test_num_streams(self) -> None:
        """Number of streams property should return the correct number of streams."""

        self.assertEqual(self.num_streams, self.signal.num_streams)

    def test_num_samples(self) -> None:
        """Number of samples property should return the correct number of samples."""

        self.assertEqual(self.num_samples, self.signal.num_samples)

    def test_sampling_rate_setget(self) -> None:
        """Sampling rate property getter should return setter argument."""

        sampling_rate = 1.123e4
        self.signal.sampling_rate = sampling_rate

        self.assertEqual(sampling_rate, self.signal.sampling_rate)

    def test_sampling_rate_validation(self) -> None:
        """Sampling rate property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.signal.sampling_rate = -1.23

        with self.assertRaises(ValueError):
            self.signal.sampling_rate = 0.

    def test_carrier_frequency_setget(self) -> None:
        """Carrier frequency property getter should return setter argument."""

        carrier_frequency = 1.123
        self.signal.carrier_frequency = carrier_frequency

        self.assertEqual(carrier_frequency, self.signal.carrier_frequency)

    def test_carrier_frequency_validation(self) -> None:
        """Carrier frequency setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.signal.carrier_frequency = -1.0

        try:
            self.signal.carrier_frequency = 0.

        except ValueError:
            self.fail()

    def test_copy(self) -> None:
        """Copying a signal model should result in a completely independent instance."""

        samples = self.signal.samples.copy()
        signal_copy = self.signal.copy()
        signal_copy.samples += 1j

        assert_array_equal(samples, self.signal.samples)

    def test_resampling_power_up(self) -> None:
        """Resampling to a higher sampling rate should not affect the signal power."""

        # Create an oversampled sinusoid signal
        frequency = 0.1 * self.sampling_rate
        self.num_samples = 1000
        timestamps = np.arange(self.num_samples) / self.sampling_rate
        samples = np.outer(np.exp(2j * pi * np.array([0, .33, .66])), np.exp(2j * pi * timestamps * frequency))
        self.signal.samples = samples

        expected_sampling_rate = 2 * self.sampling_rate
        resampled_signal = self.signal.resample(expected_sampling_rate)

        expected_power = self.signal.power
        resampled_power = resampled_signal.power

        assert_array_almost_equal(expected_power, resampled_power, decimal=3)
        self.assertEqual(expected_sampling_rate, resampled_signal.sampling_rate)

    def test_resampling_power_down(self) -> None:
        """Resampling to a lower sampling rate should not affect the signal power."""

        # Create an oversampled sinusoid signal
        frequency = 0.1 * self.sampling_rate
        self.num_samples = 1000
        timestamps = np.arange(self.num_samples) / self.sampling_rate
        samples = np.outer(np.exp(2j * pi * np.array([0, .33, .66])), np.exp(2j * pi * timestamps * frequency))
        self.signal.samples = samples

        expected_sampling_rate = .5 * self.sampling_rate
        resampled_signal = self.signal.resample(expected_sampling_rate)

        expected_power = self.signal.power
        resampled_power = resampled_signal.power

        assert_array_almost_equal(expected_power, resampled_power, decimal=3)
        self.assertEqual(expected_sampling_rate, resampled_signal.sampling_rate)

    def test_resampling_circular(self) -> None:
        """Up- and subsequently down-sampling a signal model should result in the identical signal."""

        # Create an oversampled sinusoid signal
        frequency = 0.3 * self.sampling_rate
        timestamps = np.arange(self.num_samples) / self.sampling_rate
        samples = np.outer(np.exp(2j * pi * np.array([0, 0.33, 0.66])), np.exp(2j * pi * timestamps * frequency))
        self.signal.samples = samples

        # Up-sample and down-sample again
        up_signal = self.signal.resample(1.5 * self.sampling_rate)
        down_signal = up_signal.resample(self.sampling_rate)

        # Compare to the initial samples
        assert_array_almost_equal(samples, down_signal.samples, decimal=1)
        self.assertEqual(self.sampling_rate, down_signal.sampling_rate)

    def test_superimpose_power(self) -> None:
        """Superimposing two signal models should yield approximately the sum of both model's individual power."""

        expected_power = 4 * self.signal.power
        self.signal.superimpose(self.signal)

        assert_array_almost_equal(expected_power, self.signal.power)

    def test_timestamps(self) -> None:
        """Timestamps property should return the correct sampling times."""

        expected_timestamps = np.arange(self.num_samples) / self.sampling_rate
        assert_array_equal(expected_timestamps, self.signal.timestamps)

    def test_plot(self) -> None:
        """The plot routine should not raise any exceptions."""
        pass

    def test_append_samples(self) -> None:
        """Appending a signal model should yield the proper result."""

        samples = self.signal.samples.copy()
        append_samples = self.signal.samples + 1j
        append_signal = Signal(append_samples, self.signal.sampling_rate, self.signal.carrier_frequency)

        self.signal.append_samples(append_signal)

        assert_array_equal(np.append(samples, append_samples, axis=1), self.signal.samples)

    def test_append_samples_assert(self) -> None:
        """Appending to a signal model should raise a ValueError if the models don't match."""

        with self.assertRaises(ValueError):

            samples = self.signal.samples[0, :]
            append_signal = Signal(samples, self.signal.sampling_rate, self.signal.carrier_frequency)
            self.signal.append_samples(append_signal)

        with self.assertRaises(ValueError):

            samples = self.signal.samples
            append_signal = Signal(samples, self.signal.sampling_rate, 0.)
            self.signal.append_samples(append_signal)

    def test_append_streams(self) -> None:
        """Appending a signal model should yield the proper result."""

        samples = self.signal.samples.copy()
        append_samples = self.signal.samples + 1j
        append_signal = Signal(append_samples, self.signal.sampling_rate, self.signal.carrier_frequency)

        self.signal.append_streams(append_signal)

        assert_array_equal(np.append(samples, append_samples, axis=0), self.signal.samples)

    def test_append_stream_assert(self) -> None:
        """Appending to a signal model should raise a ValueError if the models don't match."""

        with self.assertRaises(ValueError):

            samples = self.signal.samples[:, 0]
            append_signal = Signal(samples, self.signal.sampling_rate, self.signal.carrier_frequency)
            self.signal.append_streams(append_signal)

        with self.assertRaises(ValueError):

            samples = self.signal.samples
            append_signal = Signal(samples, self.signal.sampling_rate, 0.)
            self.signal.append_streams(append_signal)
