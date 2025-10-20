# -*- coding: utf-8 -*-

from contextlib import nullcontext
from os import getenv
from pickle import dumps, loads
from unittest import TestCase
from unittest.mock import patch
from typing import List

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi
from scipy.fft import ifft, ifftshift

from hermespy.core.signal_model import Signal, SignalBlock, DenseSignal, SparseSignal
from unit_tests.utils import SimulationTestContext  # type: ignore[import-not-found]
from unit_tests.core.test_factory import test_roundtrip_serialization  # type: ignore[import-not-found]
from unit_tests.utils import assert_signals_equal  # type: ignore[import-not-found]

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Egor Achkasov"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSignalBlock(TestCase):
    """SignalBlock is a wrapper around numpy's ndarray.
    It must contain a 2D matrix of complex samples."""

    def setUp(self) -> None:
        self.num_streams = 3
        self.num_samples = 150
        self.shape = (self.num_streams, self.num_samples)

        self.rng = np.random.default_rng(42)
        self.samples = self.rng.random(self.shape) + 1.j * self.rng.random(self.shape)
        self.offset = 42

        self.block = SignalBlock(self.num_streams, self.num_samples, self.offset, buffer=self.samples)

    def test_init(self) -> None:
        """Test the constraints that SignalBlock puts on the provided samples"""

        # Validate the correctly-constructed block
        self.assertEqual(self.block.ndim, 2)
        self.assertEqual(self.block.shape, self.shape)
        self.assertTrue(np.iscomplexobj(self.block))

        assert_array_equal(self.block.view(np.ndarray), self.samples.view(np.ndarray))

    def test_validate_off(self) -> None:
        """off property must be non-negative."""

        # __new__ must set the value given
        self.assertEqual(self.block.offset, self.offset)

        # "Offset must be non-zero"
        with self.assertRaises(ValueError):
            self.block.offset = -1

        # Attemts to set an incorrect value should not change the previous value
        self.assertEqual(self.block.offset, self.offset)

    def test_validate_append_samples(self) -> None:
        """A ValueError should be raised on different num_streams"""

        with self.assertRaises(ValueError):
            samples_add = np.concatenate((self.samples, self.samples), axis=0)
            self.block.append_samples(samples_add)

    def test_pickle_unpickle(self) -> None:
        """Test pickling and unpickling of signal blocks"""

        # Pickle the block
        pickled_block = dumps(self.block)

        # Unpickle the block
        unpickled_block: SignalBlock = loads(pickled_block)

        # Ensure the data has not been corrupted
        self.assertEqual(unpickled_block.shape, self.block.shape)
        assert_array_equal(unpickled_block.view(np.ndarray), self.block.view(np.ndarray))
        self.assertEqual(unpickled_block.offset, self.block.offset)

    def test_serialization(self) -> None:
        """Test serialization and deserialization of signal blocks"""

        test_roundtrip_serialization(self, self.block)


class _TestSignal(TestCase):
    """Base class for TestCases of Singal implementations.
    Note that this is not a TestCase. Inherit your Test* class from this class."""

    NUM_STREAMS: int = 3
    NUM_SAMPLES: int = 1000
    SAMPLING_RATE: float = 1.234e6
    CARRIER_FREQUENCY: float = 4.56e9
    DELAY: float = 1000 / SAMPLING_RATE
    NOISE_POWER: float = 0.789e-3

    signal: Signal

    def test_num_samples(self) -> None:
        """Number of samples property should return the correct number of samples"""

        self.assertEqual(self.NUM_SAMPLES, self.signal.num_samples)

    def test_num_streams(self) -> None:
        """Number of streams property should return the correct number of streams"""

        self.assertEqual(self.NUM_STREAMS, self.signal.num_streams)

    def test_append_signal_samples(self) -> None:
        """Appending a signal model should yield the proper result"""

        # Create the samples to append by copying and shifting the original signal
        appended_signal = self.signal * (0.5 + 1j)
        new_signal = self.signal.append_samples(appended_signal)

        # Convert to ndarray and compare
        expected_samples = np.append(self.signal.view(np.ndarray), appended_signal.view(np.ndarray), axis=1)
        assert_array_equal(expected_samples, new_signal.view(np.ndarray))

    def test_append_ndarray_samples(self) -> None:
        """Appending a np.ndarray should yield the proper result"""

        # Create the samples to append by copying and shifting the original signal
        appended_signal = self.signal.view(np.ndarray) * (0.5 + 1j)
        new_signal = self.signal.append_samples(appended_signal)

        # Convert to ndarray and compare
        expected_samples = np.append(self.signal.view(np.ndarray), appended_signal.view(np.ndarray), axis=1)
        assert_array_equal(expected_samples, new_signal.view(np.ndarray))

    def test_append_samples_validation(self) -> None:
        """Appending to a signal model should raise a ValueError if the models don't match"""

        with self.assertRaises(ValueError):
            append_signal = self.signal[:self.NUM_STREAMS-1, :]
            self.signal.append_samples(append_signal)

    def test_append_signal_streams(self) -> None:
        """Appending streams to a signal model should yield the proper result"""

        # Create the samples to append by copying and shifting the original signal
        appended_signal = self.signal * (0.5 + 1j)
        new_signal = self.signal.append_streams(appended_signal)

        # Convert to ndarray and compare
        expected_samples = np.append(self.signal.view(np.ndarray), appended_signal.view(np.ndarray), axis=0)
        assert_array_equal(expected_samples, new_signal.view(np.ndarray))

    def test_append_ndarray_streams(self) -> None:
        """Appending a np.ndarray should yield the proper result"""

        # Create the samples to append by copying and shifting the original signal
        appended_signal = self.signal.view(np.ndarray) * (0.5 + 1j)
        new_signal = self.signal.append_streams(appended_signal)

        # Convert to ndarray and compare
        expected_samples = np.append(self.signal.view(np.ndarray), appended_signal.view(np.ndarray), axis=0)
        assert_array_equal(expected_samples, new_signal.view(np.ndarray))

    def test_append_streams_validation(self) -> None:
        """Appending to a signal model should raise a ValueError if the model dimensions don't match"""

        with self.assertRaises(ValueError):
            append_signal = np.ones((self.NUM_STREAMS-1, 2 * self.NUM_SAMPLES), dtype=complex)
            self.signal.append_streams(append_signal)

    def test_title(self) -> None:
        """Title property should return the correct string representation"""

        self.assertIsInstance(self.signal.title, str)

    def test_empty(self) -> None:
        """Using the empty initializer should result in an empty signal model"""

        empty_signal = Signal.Empty(self.SAMPLING_RATE, self.NUM_STREAMS, self.NUM_SAMPLES, carrier_frequency=self.CARRIER_FREQUENCY, delay=self.DELAY, noise_power=0.0)
        self.assertEqual(self.NUM_STREAMS, empty_signal.num_streams)

    def test_setgetitem(self) -> None:
        """Set- and getitem should return the correct slices of the original samples"""

        slice_full = slice(None, None)
        samples_dense = self.signal.to_dense().view(np.ndarray)

        keys = [
            # Whole selection
            # slice_full,  # [:]
            (slice_full, slice_full),  # [:, :]

            # Int indexing
            # [0],  # [0]
            # ([0], slice_full),  # [[0], :]
            #(slice_full, [10]),  # [:, [10]]
            #(slice_full, slice(None, self.NUM_SAMPLES // 2)),  # [:, :self.num_samples//2]

            # Steps
            #(slice(None, None, -1), slice_full),  # [::-1, :]
            (slice(None, None, 2), slice_full),  # [::2, :]
            #(slice_full, slice(None, None, -1)),  # [:, ::-1]
            (slice_full, slice(None, None, 2)),  # [:, ::2]

            # Boolean masks
            #np.array([*[[True]*self.NUM_SAMPLES]]*self.NUM_STREAMS, dtype=bool),
            #np.array([*[[False]*self.NUM_SAMPLES]]*self.NUM_STREAMS, dtype=bool),
        ]

        # getitem
        for key in keys:
            with self.subTest(msg="Getitem", key=key):
                sliced_signal = self.signal[key]
                assert_array_equal(samples_dense[key].flatten(), sliced_signal.view(np.ndarray).flatten())

        # __setitem__
        dummy_value = 13.37 + 73.31j
        dummy_samples_full = np.full((self.NUM_STREAMS, self.NUM_SAMPLES),
                                     dtype=np.complex128, fill_value=dummy_value)
        dummy_samples_diff = np.arange(0, self.NUM_STREAMS*self.NUM_SAMPLES).reshape(
            (self.NUM_STREAMS, self.NUM_SAMPLES)
        )
        for key in keys:
            with self.subTest(msg="Setitem", key=key):
                try:
                    # Try assigning a scalar
                    signal_new = self.signal.copy()
                    signal_new[key] = dummy_value
                    assert_array_equal(signal_new[key].view(np.ndarray), dummy_samples_full[key].view(np.ndarray))
                    # Try assigning a ndarray
                    signal_new = self.signal.copy()
                    signal_new[key] = dummy_samples_diff[key]
                    assert_array_equal(signal_new[key].view(np.ndarray).flatten(), dummy_samples_diff[key].view(np.ndarray).flatten())
                except NotImplementedError as e:
                    # Currently this happens only in the following situations:
                    # On non-positive steps
                    if ((isinstance(key, tuple)
                            and ((key[0].step is not None and key[0].step <= 0)
                                or (key[1].step is not None and key[1].step <= 0)))
                            or (isinstance(key, slice)
                                and key.step is not None and key.step <= 0)):
                        continue
                    # On any non-unit sample steps
                    if (isinstance(key, tuple) and isinstance(key[1], slice)
                            and key[1].step is not None and key[1].step != 1):
                        continue
                    # On a boolean mask
                    if isinstance(key, np.ndarray) and key.dtype == bool:
                        continue
                    raise AssertionError(f"Unexpected NotImplementedError:\n{e}")

        # Extra cases
        # samples_init = self.signal.view(np.ndarray)
        # # key, value, expected result
        # key_value_expectedRes = [
        #     # nothing should be done
        #     (slice(self.NUM_STREAMS+1, self.NUM_STREAMS+2),
        #      dummy_value,
        #      samples_init),
        #     # replace only one stream
        #     ((0, slice_full),
        #      np.arange(0, self.signal.num_samples),
        #      np.append(
        #          np.arange(0, self.signal.num_samples).reshape((1, self.signal.num_samples)),
        #          self.signal[1:, :].view(np.ndarray),
        #          0))
        # ]
        # for key, value, expected_result in key_value_expectedRes:
        #     signal_new = self.signal.copy()
        #     signal_new[key] = value
        #     assert_array_equal(signal_new.view(np.ndarray), expected_result[:, :])

    def test_setitem_validation(self) -> None:
        """__setitem__ should raise a ValueError or an IndexError on attempts to set incorrect data."""

        dummy_val = 13.37 + 73.31j
        valueError_keyvalues = [
            # shape mismatch
            ((slice(None, self.signal.num_streams+1), slice(None, None)),
             np.full((self.signal.num_streams+1, self.signal.num_samples), fill_value=dummy_val)),
        ]
        indexError_keyvalues = []

        for key, value in indexError_keyvalues:
            with self.assertRaises(IndexError):
                self.signal[key] = value
        for key, value in valueError_keyvalues:
            with self.assertRaises(ValueError):
                self.signal[key] = value

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

    def test_noise_power_setget(self) -> None:
        """Noise power property getter should return setter argument"""

        noise_power = 1.123
        self.signal.noise_power = noise_power
        self.assertEqual(noise_power, self.signal.noise_power)

    def test_noise_power_validation(self) -> None:
        """Noise power setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.signal.noise_power = -1.0

    def test_power_scaling(self) -> None:
        """Signal power should scale square with amplitude changes"""

        # Get the signal model's initial power
        initial_power = self.signal.power

        # Test the power scales squarely with amplitude scaling
        scale = 3.21
        scaled_signal = self.signal * scale
        assert_array_almost_equal(initial_power * scale**2, scaled_signal.power)

    def test_energy_scaling(self) -> None:
        """Energy should scale square with amplitude changes"""

        # Get the signal model's initial energy
        initial_energy = self.signal.energy

        # Test the energy scales squarely with amplitude scaling
        scale = 3.21
        scaled_signal = self.signal * scale
        assert_array_almost_equal(initial_energy * scale**2, scaled_signal.energy)

    def test_resample_validation(self) -> None:
        """Resampling should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = self.signal.resample(-1.0)

    def test_resample_empty(self) -> None:

        """Resampling an empty signal should just adapt the sampling rate"""
        empty_signal = self.signal[:, :0]
        resampled_empty_signal = empty_signal.resample(1.01)

        self.assertEqual(1.01, resampled_empty_signal.sampling_rate)

    def test_resampling_power_up(self) -> None:
        """Resampling to higher sampling rates should preserver the signal power"""

        expected_sampling_rate = 2 * self.SAMPLING_RATE
        resampled_signal = self.signal.resample(expected_sampling_rate)

        expected_power = self.signal.power
        resampled_power = resampled_signal.power

        self.assertEqual(expected_sampling_rate, resampled_signal.sampling_rate)
        assert_array_almost_equal(expected_power, resampled_power, decimal=1)

    def test_resampling_power_down(self) -> None:
        """Resampling to a lower sampling rate should not affect the signal power"""

        # Create an oversampled sinusoid signal
        frequency = 0.1 * self.SAMPLING_RATE
        timestamps = np.arange(self.NUM_SAMPLES) / self.SAMPLING_RATE
        samples = np.outer(np.exp(2j * pi * np.array([0, 0.33, 0.66])), np.exp(2j * pi * timestamps * frequency))
        self.signal[::] = samples

        expected_sampling_rate = 0.5 * self.SAMPLING_RATE
        resampled_signal = self.signal.resample(expected_sampling_rate)

        expected_power = self.signal.power
        resampled_power = resampled_signal.power

        self.assertEqual(expected_sampling_rate, resampled_signal.sampling_rate)
        assert_array_almost_equal(expected_power, resampled_power, decimal=1)

    def test_resampling_circular_no_filter(self) -> None:
        """Up- and subsequently down-sampling a signal model should result in the identical signal"""

        initial_spectrum = np.zeros(self.NUM_SAMPLES, dtype=complex)
        initial_spectrum[int(0.25 * self.NUM_SAMPLES) : int(0.25 * self.NUM_SAMPLES) + 50] = np.exp(2j * np.pi * self.random.uniform(0, 2, 50))

        initial_samples = np.outer(np.exp(2j * np.pi * np.array([0.33, 0.66, 0.99])), ifft(ifftshift(initial_spectrum)))
        self.signal[::] = initial_samples

        # Up-sample and down-sample again
        up_signal = self.signal.resample(1.456 * self.SAMPLING_RATE, False)
        down_signal = up_signal.resample(self.SAMPLING_RATE, False)

        # Compare to the initial samples
        assert_array_almost_equal(initial_samples, down_signal.view(np.ndarray), decimal=2)
        self.assertEqual(self.SAMPLING_RATE, down_signal.sampling_rate)

    def test_resampling_circular_filter(self) -> None:
        """Up- and subsequently down-sampling a signal model should result in the identical signal"""

        # Create an oversampled sinusoid signal
        frequency = 0.3 * self.SAMPLING_RATE
        timestamps = np.arange(self.NUM_SAMPLES) / self.SAMPLING_RATE
        samples = np.outer(np.exp(2j * pi * np.array([0, 0.33, 0.66])), np.exp(2j * pi * timestamps * frequency))
        self.signal[::] = samples

        # Up-sample and down-sample again
        up_signal = self.signal.resample(1.5 * self.SAMPLING_RATE, aliasing_filter=True)
        down_signal = up_signal.resample(self.SAMPLING_RATE, aliasing_filter=True)

        # Compare to the initial samples
        assert_array_almost_equal(abs(samples[:, 10:]), abs(down_signal[:, 10:].view(np.ndarray)), decimal=1)
        self.assertEqual(self.SAMPLING_RATE, down_signal.sampling_rate)

    def test_superimpose_resample(self) -> None:
        """Superimposing two signal models with different sampling rates should yield the correct result"""

        added_signal = self.signal.copy()
        added_signal.sampling_rate = 0.5 * self.SAMPLING_RATE

        superimposed_signal = self.signal.superimpose(added_signal)
        self.assertEqual(2 * self.NUM_SAMPLES, superimposed_signal.num_samples)

    def test_timestamps(self) -> None:
        """Timestamps property should return the correct sampling times"""

        expected_timestamps = np.arange(self.NUM_SAMPLES) / self.SAMPLING_RATE
        assert_array_equal(expected_timestamps, self.signal.timestamps)

    def test_frequencies(self) -> None:
        """Frequencies property should return the correct frequencies"""

        expected_frequencies = np.fft.fftfreq(self.NUM_SAMPLES, 1 / self.SAMPLING_RATE)
        assert_array_equal(expected_frequencies, self.signal.frequencies)

    def test_duration(self) -> None:
        """Duration property should return the correct duration"""

        self.assertEqual(self.NUM_SAMPLES / self.SAMPLING_RATE, self.signal.duration)

    def test_plot(self) -> None:
        """The plot routine should not raise any exceptions"""

        with patch("matplotlib.pyplot.figure") if getenv("HERMES_TEST_PLOT", "False").lower() == "true" else nullcontext():
            try:
                _ = self.signal.plot(space="time")
                _ = self.signal.plot(space="frequency", angle=True)
                _ = self.signal.plot(space="both")

                # Empty plotting
                _ = self.signal.Empty(1, 1, 0).plot(space="both")

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

    def test_superimpose_validation(self) -> None:
        """Superimposing should raise a ValueError on invalid arguments"""

        # On different sampling rates with resample=False
        with self.assertRaises(RuntimeError):
            self.signal.superimpose(self.signal.Empty(0.5 * self.SAMPLING_RATE, self.NUM_STREAMS, 1, delay=self.DELAY), resample=False)

    def test_superimpose_empty_stream_indices(self) -> None:
        """Given an empty list of stream indices, the superimpose method should do nothing"""

        copied_signal = self.signal.copy()
        superimposed_signal = self.signal.superimpose(copied_signal, stream_indices=[])

        assert_signals_equal(self, self.signal, superimposed_signal)

    def test_superimpose_empty_samples(self) -> None:
        """Superimposition to an empty signal should pad it with zeros"""

        signal_empty = self.signal.Empty(num_streams=self.NUM_STREAMS, num_samples=0, sampling_rate=self.SAMPLING_RATE, carrier_frequency=self.CARRIER_FREQUENCY, delay=self.DELAY, noise_power=self.NOISE_POWER)
        superimposed_signal = signal_empty.superimpose(self.signal)

        assert_signals_equal(self, self.signal, superimposed_signal)

    def test_superimpose_no_overlap(self) -> None:
        """Superimposing two non-overlapping signal models should yield the original signal"""

        copied_signal = self.signal.copy()
        copied_signal.carrier_frequency = self.signal.carrier_frequency + 4 * self.signal.sampling_rate
        superimposed_signal = self.signal.superimpose(copied_signal)

        assert_signals_equal(self, self.signal, superimposed_signal)

    def test_superimpose_power_full(self) -> None:
        """Superimposing two full overlapping signal models should yield approximately the sum of both model's individual power"""

        expected_power = 4 * self.signal.power
        superimposed_signal = self.signal.superimpose(self.signal)

        assert_array_almost_equal(expected_power, superimposed_signal.power)

    def test_superimpose_power_partially(self) -> None:
        """Superimposing two partially overlapping signal models should yield approximately the sum of the overlapping power"""

        # Update the signal to a random signal with flat spectrum
        for block in self.signal.blocks:
            self.signal[:, block.offset:block.end] = ifft(np.exp(2j * np.pi * self.random.uniform(0, 1, block.shape)))

        initial_power = self.signal.power

        added_signal = self.signal.copy()
        added_signal.carrier_frequency = self.CARRIER_FREQUENCY + .5 * self.SAMPLING_RATE

        overlap = (self.SAMPLING_RATE - abs(added_signal.carrier_frequency - self.signal.carrier_frequency) / 2) / self.SAMPLING_RATE
        expected_superimposed_power = (1 + overlap) * initial_power
        superimposed_signal = self.signal.superimpose(added_signal)

        assert_array_almost_equal(expected_superimposed_power, superimposed_signal.power, decimal=3)

    def test_superimpose_smaller_delay(self) -> None:
        """Superimposing two signals with different delays should yield the correct result"""

        sample_shift = self.NUM_SAMPLES // 2
        added_signal = self.signal.copy()
        added_signal.delay -= sample_shift / self.SAMPLING_RATE

        superimposed_signal = self.signal.superimpose(added_signal)

        # Ensure the number of samples has not changed, since the superimposed signal starts earlier
        self.assertEqual(self.signal.num_samples, superimposed_signal.num_samples)

        # The power should have increased
        self.assertGreater(superimposed_signal.power.sum(), added_signal.power.sum())

    def test_to_dense(self) -> None:
        """to_dense method should return a DenseSignal version of the signal."""

        assert_signals_equal(self, self.signal, self.signal.to_dense())

    def test_copy(self) -> None:
        """Test the copy method implementation of signal models"""

        copy = self.signal.copy()

        # Ensure it's actually a copy
        self.assertIsNot(copy, self.signal)

        assert_signals_equal(self, self.signal, copy)

    def test_serialization(self) -> None:
        """Test signal serialization"""

        test_roundtrip_serialization(self, self.signal)


class TestDenseSignal(_TestSignal, TestCase):
    """Test DenseSignal implementation of Signal."""

    signal: DenseSignal

    def setUp(self) -> None:
        self.random = default_rng(42)
        self.signal = DenseSignal.FromNDArray(
            self.random.standard_normal((self.NUM_STREAMS, self.NUM_SAMPLES)) + 1j* self.random.standard_normal((self.NUM_STREAMS, self.NUM_SAMPLES)),
            self.SAMPLING_RATE,
            self.CARRIER_FREQUENCY,
            self.NOISE_POWER,
            self.DELAY,
        )

    def test_title(self) -> None:
        self.assertEqual("Dense Signal Model", self.signal.title)

    def test_plusequal(self) -> None:
        """Test the sum update operation on dense signals"""

        signal = DenseSignal.Zeros(2, 10, self.SAMPLING_RATE, self.CARRIER_FREQUENCY, self.NOISE_POWER, self.DELAY)
        added_numbers = np.arange(10)

        signal[[0], :] += added_numbers

        assert_array_equal(signal.view(np.ndarray)[0], added_numbers)

    def test_to_from_interleaved(self) -> None:
        """Interleaving and de-interleaving should yield the original signal"""

        self.signal /= np.max(abs(self.signal))
        interleaved_signal = self.signal.to_interleaved(scale=True)
        deinterleaved_signal = self.signal.FromInterleaved(interleaved_signal, scale=True, sampling_rate=self.SAMPLING_RATE, carrier_frequency=self.CARRIER_FREQUENCY, noise_power=self.NOISE_POWER, delay=self.DELAY)

        assert_array_almost_equal(self.signal.view(np.ndarray), deinterleaved_signal.view(np.ndarray), decimal=3)

    def test_pickle_unpickle(self) -> None:
        """Test pickling and unpickling of dense signals"""

        # Pickle the dense signal
        pickled_block = dumps(self.signal)

        # Unpickle the dense signal
        unpickled_dense: DenseSignal = loads(pickled_block)

        # Ensure the data has not been corrupted
        assert_signals_equal(self, self.signal, unpickled_dense)


class TestSparseSignal(_TestSignal, TestCase):
    """Test SparseSignal implementation of Signal."""

    def setUp(self) -> None:
        self.random = default_rng(42)

        # Sample windows properties
        self.NUM_STREAMS = 3
        self.num_beg_zeros = 5  # number of starting zero columns in samples
        self.num_end_zeros = 5  # number of ending zero columns in samples
        self.num_mid_zeros = 5  # number of zero columns between each non-zero window
        self.num_windows = 5    # number of non-zero windows

        # number of columns in each non-zero window
        # (looks like [200, 300, 200, 300, 200])
        self.window_sizes = [[200, 300][i % 2] for i in range(self.num_windows)]

        # num_samples
        self.NUM_SAMPLES = self.num_beg_zeros + self.num_end_zeros
        self.num_samples_nonzero = np.sum(self.window_sizes)
        self.NUM_SAMPLES += self.num_samples_nonzero
        self.NUM_SAMPLES += self.num_mid_zeros * (self.num_windows - 1)

        # Generate samples in a sparse form
        self.samples_sparse: List[np.ndarray]
        self.samples_sparse = []
        for ws in self.window_sizes:
            re = self.random.random((self.NUM_STREAMS, ws))
            im = self.random.random((self.NUM_STREAMS, ws))
            self.samples_sparse.append(re + 1j+im)

        # Calculate offsets
        self.offsets = np.empty((self.num_windows,), dtype=int)
        self.offsets[0] = self.num_beg_zeros
        for i in range(1, self.num_windows):
            self.offsets[i] = self.offsets[i-1] + self.window_sizes[i-1] + self.num_mid_zeros

        # Convert these samples into a dense form
        self.samples_dense = np.zeros((self.NUM_STREAMS, self.NUM_SAMPLES), dtype=np.complex128)
        for i in range(self.num_windows):
            w_off = self.offsets[i]
            w_end = w_off + self.window_sizes[i]
            self.samples_dense[:, w_off:w_end] = self.samples_sparse[i]

        self.blocks = [SignalBlock(self.samples_sparse[i].shape[0], self.samples_sparse[i].shape[1], self.offsets[i], buffer=self.samples_sparse[i].tobytes())
                       for i in range(self.num_windows)]

        self.signal = SparseSignal(self.blocks, self.SAMPLING_RATE, self.CARRIER_FREQUENCY, self.NOISE_POWER, self.DELAY)

        # This implementation is expected to throw away trailing zeros
        self.NUM_SAMPLES -= self.num_end_zeros
        self.samples_dense = self.samples_dense[:, :self.NUM_SAMPLES]

    def test_title(self) -> None:
        self.assertEqual("Sparse Signal Model", self.signal.title)

    def _test_setgetitem_validation(self) -> None:
        """__setitem__ and getitem should raise IndexError on incorrect slicing."""

        keys_index_error = [
            # "Streams slice start must be lower then stop"
            (slice(0, 0), slice(None, None)),
            (slice(self.NUM_STREAMS, 0), slice(None, None)),
            slice(0, 0),
            slice(self.NUM_STREAMS, 0),
            # "Streams index is out of bounds"
            (self.NUM_STREAMS, slice(None, None)),
            (self.NUM_STREAMS + 1, slice(None, None)),
            self.NUM_STREAMS,
            self.NUM_STREAMS + 1,
            # "Samples index is out of bounds"
            (slice(None, None), self.NUM_SAMPLES),
            (slice(None, None), self.NUM_SAMPLES + 1),
        ]
        keys_type_error = [
            # "Expected to get streams index as an integer or a slice"
            ("str", slice(None, None)),
            (1.5, slice(None, None)),
            # "Samples key is of an unsupported type"
            (slice(None, None), "str"),
            (slice(None, None), 1.5),
            # "Unsupported key type"
            "str",
            1.5
        ]

        # __getitem__
        for key in keys_index_error:
            with self.subTest(msg="GetItem IndexError", key=key), self.assertRaises(IndexError):
                _ = self.signal[key]

        for key in keys_type_error:
            with self.subTest(msg="GetItem TypeError", key=key), self.assertRaises(TypeError):
                _ = self.signal[key]

        # __setitem__
        dummy_value = 13.37 + 73.31j
        for key in keys_index_error:
            with self.subTest(msg="SetItem IndexError", key=key), self.assertRaises(IndexError):
                self.signal[key] = dummy_value

        for key in keys_type_error:
            with self.subTest(msg="SetItem TypeError", key=key), self.assertRaises(TypeError):
                self.signal[key] = dummy_value


del _TestSignal
