# -*- coding: utf-8 -*-
"""Test the HermesPy Signal Model"""

from contextlib import nullcontext
from os import getenv
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch, PropertyMock

import numpy as np
from h5py import File
from os import path
from numpy.random import default_rng
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi
from scipy.fft import ifft, ifftshift

from hermespy.core.signal_model import Signal
from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSignal(TestCase):
    """Test the signal model base class"""

    def setUp(self) -> None:
        self.random = default_rng(42)

        self.num_streams = 3
        self.num_samples = 100
        self.sampling_rate = 1e4
        self.carrier_frequency = 1e3
        self.delay = 0.0

        self.samples = self.random.random((self.num_streams, self.num_samples)) + 1j * self.random.random((self.num_streams, self.num_samples))

        self.signal = Signal(samples=self.samples, sampling_rate=self.sampling_rate, carrier_frequency=self.carrier_frequency, delay=self.delay)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as object attributes"""

        assert_array_equal(self.samples, self.signal.samples)
        self.assertEqual(self.sampling_rate, self.signal.sampling_rate)
        self.assertEqual(self.carrier_frequency, self.signal.carrier_frequency)
        self.assertEqual(self.delay, self.signal.delay)

    def test_title(self) -> None:
        """Title property should return the correct string representation"""

        self.assertEqual("Signal Model", self.signal.title)

    def test_empty(self) -> None:
        """Using the empty initializer should result in an empty signal model"""

        sampling_rate = 2
        num_streams = 5
        num_samples = 6
        empty_signal = Signal.empty(sampling_rate, num_streams=num_streams, num_samples=num_samples)

        self.assertEqual(sampling_rate, empty_signal.sampling_rate)
        self.assertEqual(num_samples, empty_signal.num_samples)
        self.assertEqual(num_streams, empty_signal.num_streams)

    def test_samples_setget(self) -> None:
        """Samples property getter should return setter argument"""

        samples = self.random.random((self.num_streams + 1, self.num_samples + 1)) + 1j * self.random.random((self.num_streams + 1, self.num_samples + 1))

        self.signal.samples = samples
        assert_array_equal(samples, self.signal.samples)

    def test_samples_validation(self) -> None:
        """Samples property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.signal.samples = self.random.random((1, 2, 3))

    def test_num_streams(self) -> None:
        """Number of streams property should return the correct number of streams"""

        self.assertEqual(self.num_streams, self.signal.num_streams)

    def test_num_samples(self) -> None:
        """Number of samples property should return the correct number of samples"""

        self.assertEqual(self.num_samples, self.signal.num_samples)

    def test_sampling_rate_setget(self) -> None:
        """Sampling rate property getter should return setter argument"""

        sampling_rate = 1.123e4
        self.signal.sampling_rate = sampling_rate

        self.assertEqual(sampling_rate, self.signal.sampling_rate)

    def test_sampling_rate_validation(self) -> None:
        """Sampling rate property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.signal.sampling_rate = -1.23

        with self.assertRaises(ValueError):
            self.signal.sampling_rate = 0.0

    def test_carrier_frequency_setget(self) -> None:
        """Carrier frequency property getter should return setter argument"""

        carrier_frequency = 1.123
        self.signal.carrier_frequency = carrier_frequency

        self.assertEqual(carrier_frequency, self.signal.carrier_frequency)

    def test_carrier_frequency_validation(self) -> None:
        """Carrier frequency setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.signal.carrier_frequency = -1.0

        try:
            self.signal.carrier_frequency = 0.0

        except ValueError:
            self.fail()

    def test_noise_power_validation(self) -> None:
        """Noise power setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.signal.noise_power = -1.0

        try:
            self.signal.noise_power = 0.0

        except ValueError:
            self.fail()

    def test_noise_power_setget(self) -> None:
        """Noise power property getter should return setter argument"""

        noise_power = 1.123
        self.signal.noise_power = noise_power

        self.assertEqual(noise_power, self.signal.noise_power)

    def test_power(self) -> None:
        """Power property should return the correct power"""

        expected_power = np.mean(abs(self.samples) ** 2, axis=1)
        assert_array_almost_equal(expected_power, self.signal.power)

        self.signal.samples = np.empty((self.num_streams, 0))
        assert_array_equal(np.zeros(self.num_streams), self.signal.power)

    def test_copy(self) -> None:
        """Copying a signal model should result in a completely independent instance"""

        samples = self.signal.samples.copy()
        signal_copy = self.signal.copy()
        signal_copy.samples += 1j

        assert_array_equal(samples, self.signal.samples)

    def test_energy(self) -> None:
        """Energy property should return the correct energy"""

        expected_energy = np.sum(abs(self.samples) ** 2, axis=1)
        assert_array_almost_equal(expected_energy, self.signal.energy)

    def test_resample_validation(self) -> None:
        """Resampling should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = self.signal.resample(-1.0)

    def test_resample_empty(self) -> None:

        """Resampling an empty signal should just adapt the sampling rate"""
        empty_signal = Signal.empty(1, 1, 0)
        resampled_signal = empty_signal.resample(1.01)

        self.assertEqual(1.01, resampled_signal.sampling_rate)

    def test_resampling_power_up(self) -> None:
        """Resampling to a higher sampling rate should not affect the signal power"""

        # Create an oversampled sinusoid signal
        frequency = 0.1 * self.sampling_rate
        self.num_samples = 1000
        timestamps = np.arange(self.num_samples) / self.sampling_rate
        samples = np.outer(np.exp(2j * pi * np.array([0, 0.33, 0.66])), np.exp(2j * pi * timestamps * frequency))
        self.signal.samples = samples

        expected_sampling_rate = 2 * self.sampling_rate
        resampled_signal = self.signal.resample(expected_sampling_rate)

        expected_power = self.signal.power
        resampled_power = resampled_signal.power

        assert_array_almost_equal(expected_power, resampled_power, decimal=3)
        self.assertEqual(expected_sampling_rate, resampled_signal.sampling_rate)

    def test_resampling_power_down(self) -> None:
        """Resampling to a lower sampling rate should not affect the signal power"""

        # Create an oversampled sinusoid signal
        frequency = 0.1 * self.sampling_rate
        self.num_samples = 1000
        timestamps = np.arange(self.num_samples) / self.sampling_rate
        samples = np.outer(np.exp(2j * pi * np.array([0, 0.33, 0.66])), np.exp(2j * pi * timestamps * frequency))
        self.signal.samples = samples

        expected_sampling_rate = 0.5 * self.sampling_rate
        resampled_signal = self.signal.resample(expected_sampling_rate)

        expected_power = self.signal.power
        resampled_power = resampled_signal.power

        assert_array_almost_equal(expected_power, resampled_power, decimal=2)
        self.assertEqual(expected_sampling_rate, resampled_signal.sampling_rate)

    def test_resampling_circular_no_filter(self) -> None:
        """Up- and subsequently down-sampling a signal model should result in the identical signal"""

        initial_spectrum = np.zeros(self.num_samples, dtype=complex)
        initial_spectrum[int(0.25 * self.num_samples) : int(0.25 * self.num_samples) + 50] = np.exp(2j * np.pi * self.random.uniform(0, 2, 50))

        initial_samples = np.outer(np.exp(2j * np.pi * np.array([0.33, 0.66, 0.99])), ifft(ifftshift(initial_spectrum)))
        self.signal.samples = initial_samples

        # Up-sample and down-sample again
        up_signal = self.signal.resample(1.456 * self.sampling_rate, False)
        down_signal = up_signal.resample(self.sampling_rate, False)

        # Compare to the initial samples
        assert_array_almost_equal(initial_samples, down_signal.samples, decimal=2)
        self.assertEqual(self.sampling_rate, down_signal.sampling_rate)

    def test_resampling_circular_filter(self) -> None:
        """Up- and subsequently down-sampling a signal model should result in the identical signal"""

        # Create an oversampled sinusoid signal
        frequency = 0.3 * self.sampling_rate
        timestamps = np.arange(self.num_samples) / self.sampling_rate
        samples = np.outer(np.exp(2j * pi * np.array([0, 0.33, 0.66])), np.exp(2j * pi * timestamps * frequency))
        self.signal.samples = samples

        # Up-sample and down-sample again
        up_signal = self.signal.resample(1.5 * self.sampling_rate, aliasing_filter=True)
        down_signal = up_signal.resample(self.sampling_rate, aliasing_filter=True)

        # Compare to the initial samples
        assert_array_almost_equal(abs(samples[:, 10:]), abs(down_signal.samples[:, 10:]), decimal=1)
        self.assertEqual(self.sampling_rate, down_signal.sampling_rate)

    def test_superimpose_validation(self) -> None:
        """Superimposing should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.signal.superimpose(Signal.empty(self.sampling_rate, self.num_streams + 1, 0))

        with self.assertRaises(NotImplementedError):
            self.signal.superimpose(Signal.empty(self.sampling_rate, self.num_streams, 0, delay=1.0))

        with self.assertRaises(RuntimeError):
            self.signal.superimpose(Signal.empty(0.5 * self.sampling_rate, self.num_streams), resample=False)

    def test_superimpose_empty_stream_indices(self) -> None:
        """Given an empty list of stream indices, the superimpose method should do nothing"""

        copied_signal = self.signal.copy()
        copied_signal.samples = self.random.random((self.num_streams, self.num_samples)) + 1j * self.random.random((self.num_streams, self.num_samples))
        self.signal.superimpose(copied_signal, stream_indices=[])

        assert_array_equal(self.signal.samples, self.samples)

    def test_superimpose_no_overlap(self) -> None:
        """Superimposing two non-overlapping signal models should yield the original signal"""

        copied_signal = self.signal.copy()
        copied_signal.carrier_frequency = self.signal.carrier_frequency + 4 * self.signal.sampling_rate
        self.signal.superimpose(copied_signal)

        assert_array_equal(self.signal.samples, self.samples)

    def test_superimpose_power_full(self) -> None:
        """Superimposing two full overlapping signal models should yield approximately the sum of both model's individual power"""

        expected_power = 4 * self.signal.power
        self.signal.superimpose(self.signal)

        assert_array_almost_equal(expected_power, self.signal.power)

    def test_superimpose_power_partially(self) -> None:
        """Superimposing two partially overlapping signal models should yield approximately the sum of the overlapping power"""

        self.signal.samples = ifft(np.exp(2j * np.pi * self.random.uniform(0, 1, self.signal.samples.shape)))
        initial_power = self.signal.power

        added_signal = self.signal.copy()
        added_signal.carrier_frequency = 1e4

        expected_added_power = initial_power * (0.5 * (added_signal.sampling_rate + self.signal.sampling_rate) - abs(added_signal.carrier_frequency - self.signal.carrier_frequency)) / added_signal.sampling_rate
        self.signal.superimpose(added_signal)

        assert_array_almost_equal(expected_added_power, self.signal.power - initial_power, decimal=3)

        self.signal.samples = ifft(np.exp(2j * np.pi * self.random.uniform(0, 1, self.signal.samples.shape)))
        initial_power = self.signal.power

        added_signal = self.signal.copy()
        self.signal.carrier_frequency = 1e4

        expected_added_power = initial_power * (0.5 * (added_signal.sampling_rate + self.signal.sampling_rate) - abs(added_signal.carrier_frequency - self.signal.carrier_frequency)) / added_signal.sampling_rate
        self.signal.superimpose(added_signal)

        assert_array_almost_equal(expected_added_power, self.signal.power - initial_power, decimal=3)

    def test_superimpose_resample(self) -> None:
        """Superimposing two signal models with different sampling rates should yield the correct result"""

        added_signal = self.signal.copy()
        added_signal.sampling_rate = 0.5 * self.sampling_rate

        self.signal.superimpose(added_signal)
        self.assertEqual(2 * self.num_samples, self.signal.num_samples)

    def test_timestamps(self) -> None:
        """Timestamps property should return the correct sampling times"""

        expected_timestamps = np.arange(self.num_samples) / self.sampling_rate
        assert_array_equal(expected_timestamps, self.signal.timestamps)

    def test_frequencies(self) -> None:
        """Frequencies property should return the correct frequencies"""

        expected_frequencies = np.fft.fftfreq(self.num_samples, 1 / self.sampling_rate)
        assert_array_equal(expected_frequencies, self.signal.frequencies)

    def test_plot(self) -> None:
        """The plot routine should not raise any exceptions"""

        with patch("matplotlib.pyplot.figure") if getenv("HERMES_TEST_PLOT", "False").lower() == "true" else nullcontext():
            try:
                _ = self.signal.plot(space="time")
                _ = self.signal.plot(space="frequency", angle=True)
                _ = self.signal.plot(space="both")

                # Empty plotting
                _ = Signal.empty(1, 1, 0).plot(space="both")

            except Exception as e:
                self.fail(e)

        return

    def test_plot_eye(self) -> None:
        """Visualizing eye diagrams in time-dime domain should yield a plot"""

        with SimulationTestContext():
            try:
                _ = self.signal.eye(symbol_duration=10/self.signal.sampling_rate, domain="time")
                _ = self.signal.eye(symbol_duration=10/self.signal.sampling_rate, domain="complex")

            except Exception as e:
                self.fail(e)

        return

    def test_plot_eye_validation(self) -> None:
        """The eye plotting routine should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = self.signal.eye(symbol_duration=-1.0)

        with self.assertRaises(ValueError):
            _ = self.signal.eye(symbol_duration=1e-3, domain="blablabla")

        with self.assertRaises(ValueError):
            _ = self.signal.eye(symbol_duration=1e-3, linewidth=0.0)

        with self.assertRaises(ValueError):
            _ = self.signal.eye(symbol_duration=1e-3, symbol_cutoff=2.0)

    def test_append_samples(self) -> None:
        """Appending a signal model should yield the proper result"""

        samples = self.signal.samples.copy()
        append_samples = self.signal.samples + 1j
        append_signal = Signal(append_samples, self.signal.sampling_rate, self.signal.carrier_frequency)

        self.signal.append_samples(append_signal)
        assert_array_equal(np.append(samples, append_samples, axis=1), self.signal.samples)

        test_signal = Signal.empty(self.sampling_rate, 0, 0, carrier_frequency=self.carrier_frequency)
        test_signal.append_samples(self.signal)
        assert_array_equal(self.signal.samples, test_signal.samples)

    def test_append_samples_validation(self) -> None:
        """Appending to a signal model should raise a ValueError if the models don't match"""

        with self.assertRaises(ValueError):
            samples = self.signal.samples[0, :]
            append_signal = Signal(samples, self.signal.sampling_rate, self.signal.carrier_frequency)
            self.signal.append_samples(append_signal)

        with self.assertRaises(ValueError):
            samples = self.signal.samples
            append_signal = Signal(samples, self.signal.sampling_rate, 0.0)
            self.signal.append_samples(append_signal)

        with self.assertRaises(NotImplementedError):
            appended_signal = self.signal.copy()
            appended_signal.sampling_rate = 0.5 * self.sampling_rate
            self.signal.append_samples(appended_signal)

    def test_append_streams(self) -> None:
        """Appending a signal model should yield the proper result"""

        samples = self.signal.samples.copy()
        append_samples = self.signal.samples + 1j
        append_signal = Signal(append_samples, self.signal.sampling_rate, self.signal.carrier_frequency)

        self.signal.append_streams(append_signal)
        assert_array_equal(np.append(samples, append_samples, axis=0), self.signal.samples)

        test_signal = Signal.empty(self.sampling_rate, 0, 0, carrier_frequency=self.carrier_frequency)
        test_signal.append_streams(self.signal)
        assert_array_equal(self.signal.samples, test_signal.samples)

    def test_append_stream_validation(self) -> None:
        """Appending to a signal model should raise a ValueError if the models don't match"""

        with self.assertRaises(ValueError):
            samples = self.signal.samples[:, 0]
            append_signal = Signal(samples, self.signal.sampling_rate, self.signal.carrier_frequency)
            self.signal.append_streams(append_signal)

        with self.assertRaises(ValueError):
            samples = self.signal.samples
            append_signal = Signal(samples, self.signal.sampling_rate, 0.0)
            self.signal.append_streams(append_signal)

        with self.assertRaises(NotImplementedError):
            appended_signal = self.signal.copy()
            appended_signal.sampling_rate = 0.5 * self.sampling_rate
            self.signal.append_streams(appended_signal)

    def test_duration(self) -> None:
        """Duration property should return the correct duration"""

        self.assertEqual(self.num_samples / self.sampling_rate, self.signal.duration)

    def test_to_from_interleaved(self) -> None:
        """Interleaving and de-interleaving should yield the original signal"""

        interleaved_signal = self.signal.to_interleaved()
        deinterleaved_signal = Signal.from_interleaved(interleaved_signal, sampling_rate=self.sampling_rate, carrier_frequency=self.carrier_frequency)

        assert_array_almost_equal(np.angle(self.samples), np.angle(deinterleaved_signal.samples), decimal=3)
        self.assertEqual(self.signal.sampling_rate, deinterleaved_signal.sampling_rate)
        self.assertEqual(self.signal.carrier_frequency, deinterleaved_signal.carrier_frequency)

    def test_hdf_serialization(self) -> None:
        """Serialization to and from HDF5 should yield the correct object reconstruction"""

        signal: Signal = None

        with TemporaryDirectory() as tempdir:
            file_location = path.join(tempdir, "testfile.hdf5")

            with File(file_location, "a") as file:
                group = file.create_group("testgroup")
                self.signal.to_HDF(group)

            with File(file_location, "r") as file:
                group = file["testgroup"]
                signal = self.signal.from_HDF(group)

        self.assertEqual(self.signal.carrier_frequency, signal.carrier_frequency)
        self.assertEqual(self.signal.sampling_rate, signal.sampling_rate)
        self.assertEqual(self.signal.delay, signal.delay)
        self.assertEqual(self.signal.noise_power, signal.noise_power)
        assert_array_equal(self.signal.samples, signal.samples)
