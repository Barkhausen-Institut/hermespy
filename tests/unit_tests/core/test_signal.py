# -*- coding: utf-8 -*-
"""Test the HermesPy Signal Model"""

from contextlib import nullcontext
from os import getenv
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from typing import List
from abc import abstractmethod

import numpy as np
from h5py import File
from os import path
from numpy.random import default_rng
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi
from scipy.fft import ifft, ifftshift

from hermespy.core.signal_model import Signal, SignalBlock
from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
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

        self.block = SignalBlock(self.samples, self.offset)

    def test_validate_init(self) -> None:
        """Test the constraints that SignalBlock puts on the provided samples."""

        # Validate the correctly-constructed block
        self.assertEquals(self.block.ndim, 2)
        self.assertEquals(self.block.shape, self.shape)
        self.assertTrue(np.iscomplexobj(self.block))

        # "Samples must have ndim <= 2"
        with self.assertRaises(ValueError):
            SignalBlock(np.arange(0, 2*3*4).reshape((2, 3, 4)), 0)

    def test_validate_off(self) -> None:
        """off property must be non-negative."""

        # __new__ must set the value given
        self.assertEquals(self.block.offset, self.offset)

        # "Offset must be non-zero"
        with self.assertRaises(ValueError):
            self.block.offset = -1

        # Attemts to set an incorrect value should not change the previous value
        self.assertEquals(self.block.offset, self.offset)

    def test_validate_copy(self) -> None:
        """As it is inherited from numpy ndarray, copy must contain the \"order\" argument.
        The support for it is not implemented yet,
        so a NotImplementedError must be raised if order is not None."""

        with self.assertRaises(NotImplementedError):
            self.block.copy('K')

    def test_validate_append_samples(self) -> None:
        """A ValueError should be raised on different num_streams."""

        with self.assertRaises(ValueError):
            samples_add = np.concatenate((self.samples, self.samples), axis=0)
            self.block.append_samples(samples_add)


class TestSignal():
    """Base class for TestCases of Singal implementations.
    Note that this is not a TestCase. Inherit your Test* class from this class."""

    signal: Signal
    blocks: List[SignalBlock]

    def setUp(self) -> None:
        self.random = default_rng(42)

        # Signal properties
        self.sampling_rate = 1e4
        self.carrier_frequency = 1e3
        self.delay = 0.0
        self.noise_power = 0.0
        self.kwargs = {'sampling_rate': self.sampling_rate,
                       'carrier_frequency': self.carrier_frequency,
                       'delay': self.delay,
                       'noise_power': self.noise_power}

        # Sample windows properties
        self.num_streams = 3
        self.num_beg_zeros = 5  # number of starting zero columns in samples
        self.num_end_zeros = 5  # number of ending zero columns in samples
        self.num_mid_zeros = 5  # number of zero columns between each non-zero window
        self.num_windows = 5    # number of non-zero windows

        # number of columns in each non-zero window
        # (looks like [20, 30, 20, 30, 20])
        self.window_sizes = [[20, 30][i % 2] for i in range(self.num_windows)]

        # num_samples
        self.num_samples = self.num_beg_zeros + self.num_end_zeros
        self.num_samples_nonzero = np.sum(self.window_sizes)
        self.num_samples += self.num_samples_nonzero
        self.num_samples += self.num_mid_zeros * (self.num_windows - 1)

        # Generate samples in a sparse form
        self.samples_sparse: List[np.ndarray]
        self.samples_sparse = []
        for ws in self.window_sizes:
            re = self.random.random((self.num_streams, ws))
            im = self.random.random((self.num_streams, ws))
            self.samples_sparse.append(re + 1j+im)

        # Calculate offsets
        self.offsets = np.empty((self.num_windows,), dtype=int)
        self.offsets[0] = self.num_beg_zeros
        for i in range(1, self.num_windows):
            self.offsets[i] = self.offsets[i-1] + self.window_sizes[i-1] + self.num_mid_zeros

        # Convert these samples into a dense form
        self.samples_dense = np.zeros((self.num_streams, self.num_samples), dtype=np.complex128)
        for i in range(self.num_windows):
            w_off = self.offsets[i]
            w_end = w_off + self.window_sizes[i]
            self.samples_dense[:, w_off:w_end] = self.samples_sparse[i]

        # Your class instance should be created here.
        # Tests in this ABC require two more things: blocks and signal.
        # self.signal is an instance of your class.
        # self.blocks is expected self.signal._blocks.

        # Implement your setUp like this:
        # super().setUp()
        # self.blocks = [...]
        # self.signal = MySignal(...)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as object attributes"""

        # assert blocks
        self.assert_(len(self.signal) == len(self.blocks))
        for b_exp, b_act in zip(self.blocks, self.signal):
            assert_array_equal(b_exp, b_act)
            self.assertEqual(b_exp.offset, b_act.offset)

        # assert other properties
        self.assertEqual(self.num_samples, self.signal.num_samples)
        self.assertEqual(self.num_streams, self.signal.num_streams)
        self.assertEqual(self.sampling_rate, self.signal.sampling_rate)
        self.assertEqual(self.carrier_frequency, self.signal.carrier_frequency)
        self.assertEqual(self.delay, self.signal.delay)
        self.assertEqual(self.noise_power, self.signal.delay)

    @abstractmethod
    def test_title(self) -> None:
        """Title property should return the correct string representation"""
        raise NotImplementedError()

    def test_empty(self) -> None:
        """Using the empty initializer should result in an empty signal model"""

        kwargs = {'sampling_rate': self.signal.sampling_rate * 2 + 1.,
                  'carrier_frequency': self.carrier_frequency * 2 + 1.,
                  'delay': self.delay * 2 + 1.,
                  'noise_power': self.noise_power * 2 + 1.,
                  'num_streams': self.num_streams + 2,
                  'num_samples': self.num_samples + 2}
        empty_signal = self.signal.Empty(**kwargs)

        self.assertIsInstance(empty_signal, self.signal.__class__)

        self.assertEqual(kwargs['sampling_rate'], empty_signal.sampling_rate)
        self.assertEqual(kwargs['carrier_frequency'], empty_signal.carrier_frequency)
        self.assertEqual(kwargs['delay'], empty_signal.delay)
        self.assertEqual(kwargs['noise_power'], empty_signal.noise_power)
        self.assertEqual(kwargs['num_samples'], empty_signal.num_samples)
        self.assertEqual(kwargs['num_streams'], empty_signal.num_streams)

    def test_setgetitem(self) -> None:
        """__setitem__ and getitem should return the correct slices of the original samples."""

        slice_full = slice(None, None)

        keys = [
            # Whole selection
            slice_full,  # [:]
            (slice_full, slice_full),  # [:, :]
            # Int indexing
            0,  # [0]
            (0, slice_full),  # [0, :]
            (slice_full, 0),  # [:, 0]
            (0, 0),  # [0, 0]
            (self.num_streams // 2, self.num_samples // 2),  # [self.num_streams // 2, self.num_samples // 2]
            (slice_full, slice(None, self.num_samples // 2)),  # [:, :self.num_samples//2]
            # Steps
            (slice(None, None, -1), slice_full),  # [::-1, :]
            (slice(None, None, 2), slice_full),  # [::2, :]
            (slice_full, slice(None, None, -1)),  # [:, ::-1]
            (slice_full, slice(None, None, 2)),  # [:, ::2]
            # Windows
            (slice_full, slice(None, self.num_beg_zeros)),  # beginning zeros
            *[(slice_full, slice(self.offsets[i], self.offsets[i] + self.window_sizes[i]))
              for i in range(self.num_windows)],  # non-zero windows
            (slice_full, slice(-self.num_end_zeros, None)),  # ending zeros
            # Windows cuts
            (slice_full, (slice(self.offsets[0] + self.window_sizes[0] // 2,
                                self.offsets[-1] + self.window_sizes[-1]))),
            # Boolean masks
            np.array([*[[True]*self.num_samples]]*self.num_streams, dtype=bool),
            np.array([*[[False]*self.num_samples]]*self.num_streams, dtype=bool)
        ]

        # getitem
        for key in keys:
            assert_array_equal(self.samples_dense[key].flatten(), self.signal.getitem(key).flatten())

        # getitem with unflatten=False
        assert_array_equal(self.signal.getitem(0, False).shape,
                           (self.signal.num_samples,))
        assert_array_equal(self.signal.getitem((slice_full, 0), False).shape,
                           (self.signal.num_streams,))

        # __setitem__
        dummy_value = 13.37 + 73.31j
        dummy_samples_full = np.full((self.num_streams, self.num_samples),
                                     dtype=np.complex128, fill_value=dummy_value)
        dummy_samples_diff = np.arange(0, self.num_streams*self.num_samples).reshape(
            (self.num_streams, self.num_samples)
        )
        for key in keys:
            try:
                # Try assigning a scalar
                signal_new = self.signal.copy()
                signal_new[key] = dummy_value
                assert_array_equal(signal_new.getitem(key).flatten(), dummy_samples_full[key].flatten())
                # Try assigning a ndarray
                signal_new = self.signal.copy()
                signal_new[key] = dummy_samples_diff[key]
                assert_array_equal(signal_new.getitem(key).flatten(), dummy_samples_diff[key].flatten())
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
        samples_init = self.signal.getitem()
        # key, value, expected result
        key_value_expectedRes = [
            # nothing should be done
            (slice(self.num_streams+1, self.num_streams+2),
             dummy_value,
             samples_init),
            # replace only one stream
            ((0, slice_full),
             np.arange(0, self.signal.num_samples),
             np.append(
                 np.arange(0, self.signal.num_samples).reshape((1, self.signal.num_samples)),
                 self.signal.getitem(slice(1, None)),
                 0))
        ]
        for key, value, expected_result in key_value_expectedRes:
            signal_new = self.signal.copy()
            signal_new[key] = value
            assert_array_equal(signal_new.getitem(), expected_result[:, :])

    def test_getstreams(self) -> None:
        """getstreams should yield the same result a getitem, but casted to the Signal."""

        keys = [
            0, 1, -1,
            slice(None, None),
            slice(1, None),
            slice(1, -1),
            slice(None, None, 2),
            slice(None, None, -1)
        ]

        for key in keys:
            expected_samples = self.signal.getitem(key)
            actual = self.signal.getstreams(key)
            assert_array_equal(actual.getitem(), expected_samples)
            self.assertEqual(actual.__class__, self.signal.__class__)
            for prop_actual, prop_expected in zip(actual.kwargs, self.signal.kwargs):
                self.assertEqual(prop_actual[1], prop_expected[1])

    def test_setitem_validation(self) -> None:
        """__setitem__ should raise a ValueError or an IndexError on attempts to set incorrect data."""

        dummy_val = 13.37 + 73.31j
        valueError_keyvalues = [
            # shape mismatch
            ((slice(None, self.signal.num_streams+1), slice(None, None)),
             np.full((self.signal.num_streams+1, self.signal.num_samples), fill_value=dummy_val)),
        ]
        indexError_keyvalues = [
        ]

        for key, value in indexError_keyvalues:
            with self.assertRaises(IndexError):
                self.signal[key] = value
        for key, value in valueError_keyvalues:
            with self.assertRaises(ValueError):
                self.signal[key] = value

    def test_set_samples_validation(self) -> None:
        """set_samples method should completely replace the model's samples with the given samples and offsets.
        It can accept a sequence of SignalBlocks and must validate it properly."""

        # Try setting SignalBlocks with one of them containing a different number of streams
        blocks = []
        for s, o in zip(self.samples_sparse, self.offsets):
            blocks.append(SignalBlock(s.copy(), o))
        block = blocks[len(blocks) // 2]
        abnormal_block = np.arange(0, block.shape[1]).reshape((1, block.shape[1]))
        abnormal_block = np.append(block, abnormal_block, 0)
        abnormal_block = SignalBlock(abnormal_block, block.offset)
        blocks[len(blocks) // 2] = abnormal_block
        with self.assertRaises(ValueError):
            self.signal.set_samples(blocks)

        # Try setting SignalBlocks with incorrect offsets that cause overlap
        blocks = []
        for s, o in zip(self.samples_sparse, self.offsets):
            blocks.append(SignalBlock(s.copy(), o))
        idx = len(blocks) // 2
        blocks[idx].offset = blocks[idx - 1].end - 1
        with self.assertRaises(ValueError):
            self.signal.set_samples(blocks)

        # Try setting samples with incorrect offsets that cause overlap
        offsets = self.offsets.copy()
        idx = len(blocks) // 2
        offsets[idx] = offsets[idx - 1] + self.samples_sparse[idx - 1].shape[1] - 1
        with self.assertRaises(ValueError):
            self.signal.set_samples(self.samples_sparse, offsets)

        # Try setting samples and offsets with different array lengths
        offsets = np.append(self.offsets, [self.samples_sparse[-1].shape[1] + 1])
        with self.assertRaises(ValueError):
            self.signal.set_samples(self.samples_sparse, offsets)

        # Try setting samples with one of them having > 2 ndim
        samples = []
        for s in self.samples_sparse:
            samples.append(s.copy())
        idx = len(samples) // 2
        samples[idx] = np.tile(samples[idx], (1, 1, 1))
        with self.assertRaises(ValueError):
            self.signal.set_samples(samples, self.offsets)

        # Try setting a vector, intending for a single stream signal
        signal_new = self.signal.copy()
        signal_new.set_samples(self.samples_dense[0])
        self.assertEquals(signal_new.num_streams, 1)
        assert_array_equal(signal_new.shape, (1, self.samples_dense.shape[1]))

        # Try setting a sparse array of one stream signals as vectors
        signal_new = self.signal.copy()
        samples_new = [s[0] for s in self.samples_sparse]
        signal_new.set_samples(samples_new, self.offsets)
        self.assertEquals(signal_new.num_streams, 1)
        for s, o in zip(samples_new, self.offsets):
            assert_array_equal(signal_new.getitem((slice(None, None), slice(o, o+s.shape[1]))), s)

        # Try setting non-complex samples
        signal_new = self.signal.copy()
        samples_new = [s.astype(np.float64) for s in self.samples_sparse]
        signal_new.set_samples(samples_new, self.offsets)
        for s, o in zip(samples_new, self.offsets):
            assert_array_equal(signal_new.getitem((slice(None, None), slice(o, o+s.shape[1]))), s)
        for b in self.signal:
            self.assertTrue(np.iscomplexobj(b))

        # Try setting an empty array
        signal_new = self.signal.copy()
        signal_new.set_samples([])
        assert_array_equal(signal_new.shape, (0, 0))

        # Setting a list of sample blocks without offsets
        # should clamp the offsets to the ends of the windows
        signal_new = self.signal.copy()
        signal_new.set_samples(self.samples_sparse)
        assert_array_equal(signal_new.getitem(), np.concatenate(self.samples_sparse, axis=1))

        # Try setting samples as a 3D tensor
        signal_new = self.signal.copy()
        num_samples_min = np.min([s.shape[1] for s in self.samples_sparse])
        samples_new = [s[:, :num_samples_min] for s in self.samples_sparse]
        samples_new = np.asarray(samples_new)
        signal_new.set_samples(samples_new, self.offsets)
        for s, o in zip(samples_new, self.offsets):
            assert_array_equal(signal_new.getitem((slice(None, None), slice(o, o+s.shape[1]))), s)

        # Try setting a higher dim tensor
        with self.assertRaises(ValueError):
            self.signal.set_samples(np.tile(self.samples_dense, (2, 3, 1, 1)))

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

    def test_noise_power_setget(self) -> None:
        """Noise power property getter should return setter argument"""

        noise_power = 1.123
        self.signal.noise_power = noise_power
        self.assertEqual(noise_power, self.signal.noise_power)

    def test_noise_power_validation(self) -> None:
        """Noise power setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.signal.noise_power = -1.0

    def test_power(self) -> None:
        """Power property should return the correct power"""

        assert_array_almost_equal(np.mean(abs(self.samples_dense) ** 2, axis=1),
                                  self.signal.power)

        # Signal with zero samples must have zero power
        assert_array_equal(np.zeros(self.num_streams),
                           self.signal.Empty(self.sampling_rate, self.num_streams).power)

    def test_copy(self) -> None:
        """Copying a signal model should result in a completely independent instance"""

        signal_copy = self.signal.copy()
        self.assertIsInstance(signal_copy, self.signal.__class__)
        for b in signal_copy:
            self.assertIsInstance(b, SignalBlock)

        # Assert independence
        self.assertNotEqual(id(signal_copy), id(self.signal))
        for bc, ba in zip(signal_copy, self.signal):
            self.assertNotEqual(id(bc), id(ba))

        # Warning: copy-paste from test_init incoming
        # assert blocks
        self.assert_(len(signal_copy) == len(self.blocks))
        for b_exp, b_act in zip(self.blocks, signal_copy):
            assert_array_equal(b_exp, b_act)
            self.assertEqual(b_exp.offset, b_act.offset)
        # assert other properties
        self.assertEqual(self.num_samples, signal_copy.num_samples)
        self.assertEqual(self.num_streams, signal_copy.num_streams)
        self.assertEqual(self.sampling_rate, signal_copy.sampling_rate)
        self.assertEqual(self.carrier_frequency, signal_copy.carrier_frequency)
        self.assertEqual(self.delay, signal_copy.delay)
        self.assertEqual(self.noise_power, signal_copy.delay)

    def test_energy(self) -> None:
        """Energy property should return the correct energy"""

        expected_energy = [np.sum(abs(b) ** 2, axis=1) for b in self.signal]
        expected_energy = np.sum(expected_energy, 0)
        assert_array_almost_equal(expected_energy, self.signal.energy)

    def test_resample_validation(self) -> None:
        """Resampling should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = self.signal.resample(-1.0)

    def test_resample_empty(self) -> None:

        """Resampling an empty signal should just adapt the sampling rate"""
        empty_signal = self.signal.Empty(num_streams=1, num_samples=0, **self.kwargs)
        resampled_signal = empty_signal.resample(1.01)

        self.assertEqual(1.01, resampled_signal.sampling_rate)

    def test_resampling_power_up(self) -> None:
        """Resampling to a higher sampling rate should not affect the signal power"""

        # Create an oversampled sinusoid signal
        frequency = 0.1 * self.sampling_rate
        self.num_samples = 1000
        timestamps = np.arange(self.num_samples) / self.sampling_rate
        samples = np.outer(np.exp(2j * pi * np.array([0, 0.33, 0.66])), np.exp(2j * pi * timestamps * frequency))
        self.signal.set_samples(samples)

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
        self.signal.set_samples(samples)

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
        self.signal.set_samples(initial_samples)

        # Up-sample and down-sample again
        up_signal = self.signal.resample(1.456 * self.sampling_rate, False)
        down_signal = up_signal.resample(self.sampling_rate, False)

        # Compare to the initial samples
        assert_array_almost_equal(initial_samples, down_signal.getitem(), decimal=2)
        self.assertEqual(self.sampling_rate, down_signal.sampling_rate)

    def test_resampling_circular_filter(self) -> None:
        """Up- and subsequently down-sampling a signal model should result in the identical signal"""

        # Create an oversampled sinusoid signal
        frequency = 0.3 * self.sampling_rate
        timestamps = np.arange(self.num_samples) / self.sampling_rate
        samples = np.outer(np.exp(2j * pi * np.array([0, 0.33, 0.66])), np.exp(2j * pi * timestamps * frequency))
        self.signal.set_samples(samples)

        # Up-sample and down-sample again
        up_signal = self.signal.resample(1.5 * self.sampling_rate, aliasing_filter=True)
        down_signal = up_signal.resample(self.sampling_rate, aliasing_filter=True)

        # Compare to the initial samples
        assert_array_almost_equal(abs(samples[:, 10:]), abs(down_signal.getitem((slice(None, None), slice(10, None)))), decimal=1)
        self.assertEqual(self.sampling_rate, down_signal.sampling_rate)

    def test_superimpose_validation(self) -> None:
        """Superimposing should raise a ValueError on invalid arguments"""

        # on different num_streams
        with self.assertRaises(ValueError):
            self.signal.superimpose(self.signal.Empty(self.sampling_rate, self.num_streams + 1, 1))
        # on different delay
        with self.assertRaises(NotImplementedError):
            self.signal.superimpose(self.signal.Empty(self.sampling_rate, self.num_streams, 1, delay=1.0))
        # on different sampling_rate with resample=False
        with self.assertRaises(RuntimeError):
            self.signal.superimpose(self.signal.Empty(0.5 * self.sampling_rate, self.num_streams, 1), resample=False)

    def test_superimpose_empty_stream_indices(self) -> None:
        """Given an empty list of stream indices, the superimpose method should do nothing"""

        copied_signal = self.signal.copy()
        copied_signal.set_samples(self.random.random((self.num_streams, self.num_samples)) + 1j * self.random.random((self.num_streams, self.num_samples)))
        self.signal.superimpose(copied_signal, stream_indices=[])

        assert_array_equal(self.signal.getitem(), self.samples_dense)

    def test_superimpose_empty_samples(self) -> None:
        """Superimposition to an empty signal should pad it with zeros"""

        signal_empty = self.signal.Empty(num_streams=self.num_streams, num_samples=0, **self.signal.kwargs)
        signal_empty.superimpose(self.signal)

        assert_array_equal(signal_empty.getitem(), self.signal.getitem())

    def test_superimpose_no_overlap(self) -> None:
        """Superimposing two non-overlapping signal models should yield the original signal"""

        copied_signal = self.signal.copy()
        copied_signal.carrier_frequency = self.signal.carrier_frequency + 4 * self.signal.sampling_rate
        self.signal.superimpose(copied_signal)

        assert_array_equal(self.signal.getitem(), self.samples_dense)

    def test_superimpose_power_full(self) -> None:
        """Superimposing two full overlapping signal models should yield approximately the sum of both model's individual power"""

        expected_power = 4 * self.signal.power
        self.signal.superimpose(self.signal.copy())

        assert_array_almost_equal(expected_power, self.signal.power)

    def test_superimpose_power_partially(self) -> None:
        """Superimposing two partially overlapping signal models should yield approximately the sum of the overlapping power"""

        self.signal.set_samples(ifft(np.exp(2j * np.pi * self.random.uniform(0, 1, self.signal.shape))))
        initial_power = self.signal.power

        added_signal = self.signal.copy()
        added_signal.carrier_frequency = 1e4

        expected_added_power = initial_power * (0.5 * (added_signal.sampling_rate + self.signal.sampling_rate) - abs(added_signal.carrier_frequency - self.signal.carrier_frequency)) / added_signal.sampling_rate
        self.signal.superimpose(added_signal)

        assert_array_almost_equal(expected_added_power, self.signal.power - initial_power, decimal=3)

        self.signal.set_samples(ifft(np.exp(2j * np.pi * self.random.uniform(0, 1, self.signal.shape))))
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

    def test_append_samples(self) -> None:
        """Appending a signal model should yield the proper result"""

        # Init
        samples = self.signal.getitem().copy()
        append_samples = samples + 1j
        append_signal = self.signal.from_ndarray(append_samples)
        expected_samples = np.append(samples, append_samples, axis=1)

        # Try appending a signal
        signal_new = self.signal.copy()
        signal_new.append_samples(append_signal)
        assert_array_equal(expected_samples, signal_new.getitem())

        # Try appending a np.ndarray
        signal_new = self.signal.copy()
        signal_new.append_samples(append_samples)
        assert_array_equal(expected_samples, signal_new.getitem())

        # Try appending to an empty signal
        signal_new = self.signal.Empty(num_streams=self.signal.num_streams, **self.signal.kwargs)
        signal_new.append_samples(self.signal)
        assert_array_equal(self.signal.getitem(), signal_new.getitem())

    def test_append_samples_validation(self) -> None:
        """Appending to a signal model should raise a ValueError if the models don't match"""

        with self.assertRaises(ValueError):
            append_signal = self.signal.getstreams(0)
            self.signal.append_samples(append_signal)

        with self.assertRaises(ValueError):
            samples = self.signal.getitem()
            append_signal = self.signal.Create(samples, self.signal.sampling_rate, 0.0)
            self.signal.append_samples(append_signal)

    def test_append_streams(self) -> None:
        """Appending a signal model should yield the proper result"""

        # Init
        samples = self.signal.getitem().copy()
        append_samples = samples + 1j
        append_signal = self.signal.from_ndarray(append_samples)
        expected_samples = np.append(samples, append_samples, axis=0)

        # Try appending a signal
        signal_new = self.signal.copy()
        signal_new.append_streams(append_signal)
        assert_array_equal(expected_samples, signal_new.getitem())

        # Try appending a np.ndarray
        signal_new = self.signal.copy()
        signal_new.append_streams(append_samples)
        assert_array_equal(expected_samples, signal_new.getitem())

        # Try appending to an empty signal
        signal_new = self.signal.Empty(num_streams=0, **self.signal.kwargs)
        signal_new.append_streams(self.signal)
        assert_array_equal(self.signal.getitem(), signal_new.getitem())

    def test_append_stream_validation(self) -> None:
        """Appending to a signal model should raise a ValueError if the models don't match"""

        with self.assertRaises(ValueError):
            samples = self.signal.getitem((slice(None, None), 0))
            append_signal = Signal.Create(samples, self.signal.sampling_rate, self.signal.carrier_frequency)
            self.signal.append_streams(append_signal)

        with self.assertRaises(ValueError):
            samples = self.signal.getitem()
            append_signal = Signal.Create(samples, self.signal.sampling_rate, 0.0)
            self.signal.append_streams(append_signal)

    def test_duration(self) -> None:
        """Duration property should return the correct duration"""

        self.assertEqual(self.num_samples / self.sampling_rate, self.signal.duration)

    def test_to_from_interleaved(self) -> None:
        """Interleaving and de-interleaving should yield the original signal"""

        interleaved_signal = self.signal.to_interleaved(scale=True)
        deinterleaved_signal = self.signal.from_interleaved(interleaved_signal, **self.kwargs)

        assert_array_almost_equal(np.angle(self.samples_dense), np.angle(deinterleaved_signal.getitem()), decimal=3)
        self.assertEqual(self.signal.sampling_rate, deinterleaved_signal.sampling_rate)
        self.assertEqual(self.signal.carrier_frequency, deinterleaved_signal.carrier_frequency)
        self.assertEqual(self.signal.delay, deinterleaved_signal.delay)
        self.assertEqual(self.signal.noise_power, deinterleaved_signal.noise_power)

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
        for b_exp, b_act in zip(self.signal, signal):
            assert_array_equal(b_exp, b_act)
            self.assertEqual(b_exp.offset, b_act.offset)

    def test_to_dense(self) -> None:
        """to_dense method should return a DenseSignal version of the signal."""

        assert_array_equal(self.signal.getitem(), self.signal.to_dense().getitem())


class TestDenseSignal(TestSignal, TestCase):
    """Test DenseSignal implementation of Signal."""

    def setUp(self) -> None:
        super().setUp()
        self.blocks = [SignalBlock(self.samples_dense, 0)]
        self.signal = Signal.Create(self.samples_dense, **self.kwargs)

    def test_title(self) -> None:
        self.assertEqual("Dense Signal Model", self.signal.title)


class TestSparseSignal(TestSignal, TestCase):
    """Test SparseSignal implementation of Signal."""

    def setUp(self) -> None:
        super().setUp()
        self.blocks = [SignalBlock(self.samples_sparse[i], self.offsets[i])
                       for i in range(self.num_windows)]
        self.signal = Signal.Create(self.samples_sparse, **self.kwargs, offsets=self.offsets)

        # This implementation is expected to throw away trailing zeros
        self.num_samples -= self.num_end_zeros
        self.samples_dense = self.samples_dense[:, :self.num_samples]

    def test_title(self) -> None:
        self.assertEqual("Sparse Signal Model", self.signal.title)

    def test_setgetitem_validation(self) -> None:
        """__setitem__ and getitem should raise IndexError on incorrect slicing."""

        keys_index_error = [
            # "Streams slice start must be lower then stop"
            (slice(0, 0), slice(None, None)),
            (slice(self.num_streams, 0), slice(None, None)),
            slice(0, 0),
            slice(self.num_streams, 0),
            # "Streams index is out of bounds"
            (self.num_streams, slice(None, None)),
            (self.num_streams + 1, slice(None, None)),
            self.num_streams,
            self.num_streams + 1,
            # "Samples index is out of bounds"
            (slice(None, None), self.signal.num_samples),
            (slice(None, None), self.signal.num_samples + 1),
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

        # getitem
        for key in keys_index_error:
            with self.assertRaises(IndexError):
                self.signal.getitem(key)
        for key in keys_type_error:
            with self.assertRaises(TypeError):
                self.signal.getitem(key)

        # __setitem__
        dummy_value = 13.37 + 73.31j
        for key in keys_index_error:
            with self.assertRaises(IndexError):
                self.signal[key] = dummy_value
        for key in keys_type_error:
            with self.assertRaises(TypeError):
                self.signal[key] = dummy_value
