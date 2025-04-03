# -*- coding: utf-8 -*-

from copy import deepcopy
import unittest

import numpy as np

from hermespy.simulation.rf_chain.analog_digital_converter import AnalogDigitalConverter, Gain, AutomaticGainControl, QuantizerType, GainControlType
from hermespy.tools.math import rms_value
from hermespy.core.signal_model import Signal
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "André Noll-Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["André Noll-Barreto"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestQuantization(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.num_samples = 50

        self.num_quantization_bits = 8
        self.gain = Gain()
        self.quantizer_type = QuantizerType.MID_RISER

        self.quantizer = AnalogDigitalConverter(num_quantization_bits=self.num_quantization_bits, gain=self.gain, quantizer_type=self.quantizer_type)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as object attributes."""
        self.assertEqual(self.num_quantization_bits, self.quantizer.num_quantization_bits)
        self.assertEqual(self.gain, self.quantizer.gain)
        self.assertEqual(self.quantizer_type, self.quantizer.quantizer_type)

    def test_num_quantization_bits_setget(self) -> None:
        """Quantization bits property getter should return setter argument."""

        number_bits = 10
        self.quantizer.num_quantization_bits = number_bits
        self.assertEqual(number_bits, self.quantizer.num_quantization_bits)

        number_bits = 0
        self.quantizer.num_quantization_bits = number_bits
        self.assertEqual(None, self.quantizer.num_quantization_bits)

    def test_num_quantization_bits_validation(self) -> None:
        """Quantization bits property setter should raise ValueError on arguments smaller than zero or non-integer."""

        with self.assertRaises(ValueError):
            self.quantizer.num_quantization_bits = -1

        with self.assertRaises(ValueError):
            self.quantizer.num_quantization_bits = 2.5

    def test_num_quantization_levels_get(self) -> None:
        """Quantization bits property getter should return correct value"""

        number_bits = 8
        self.assertEqual(2**number_bits, self.quantizer.num_quantization_levels)

    def test_backoff_setget(self) -> None:
        """Backoff (linear) property getter should return setter argument."""

        self.gain = AutomaticGainControl()
        backoff = 3.1
        self.quantizer.gain.backoff = backoff
        self.assertEqual(backoff, self.quantizer.gain.backoff)

    def test_backoff_validation(self) -> None:
        """Backoff (linear) setter should raise ValueError on non-positive arguments."""

        self.quantizer.gain = AutomaticGainControl()

        with self.assertRaises(ValueError):
            self.quantizer.gain.backoff = -0.3

        with self.assertRaises(ValueError):
            self.quantizer.gain.backoff = 0

    def test_quantizer_type_setget(self) -> None:
        """Gain Control property getter should return setter argument."""

        quant_type = QuantizerType.MID_TREAD
        self.quantizer.quantizer_type = quant_type
        self.assertEqual(quant_type, self.quantizer.quantizer_type)

    def test_no_quantization(self) -> None:
        """Test correct quantizer output with infinite resolution"""

        self.quantizer.num_quantization_bits = 0

        input_signal = np.random.normal(size=self.num_samples) + 1j * np.random.normal(size=self.num_samples)

        input_signal = Signal.Create(sampling_rate=1.0, samples=input_signal)
        output_signal = self.quantizer.convert(input_signal)

        np.testing.assert_array_equal(input_signal.getitem(), output_signal.getitem())

    def test_quantization_no_gain_control(self):
        """Test correct quantizer output without gain control"""

        max_amplitude = 100
        self.quantizer.gain = Gain(1 / max_amplitude, True)

        # randomly choose quantization levels
        quantization_idx = self.rng.integers(self.quantizer.num_quantization_levels, size=self.num_samples)
        quantization_step = 2 * max_amplitude / self.quantizer.num_quantization_levels
        quantization_levels = -max_amplitude + quantization_step * (quantization_idx + 0.5)

        # add random noise within quantization interval
        random_noise = self.rng.uniform(-quantization_step / 2, quantization_step / 2, size=self.num_samples)
        input_signal = quantization_levels + random_noise

        # add saturated values
        max_quantization_level = max_amplitude - quantization_step / 2
        quantization_levels = np.append(quantization_levels, [-max_quantization_level, max_quantization_level])
        saturated_level = max_amplitude + 10.0
        input_signal = np.append(input_signal, [-saturated_level, saturated_level])

        input_signal = Signal.Create(samples=input_signal, sampling_rate=1.0)

        output_signal = self.quantizer.convert(input_signal)

        np.testing.assert_almost_equal(np.real(output_signal.getitem().flatten()), quantization_levels)

    def test_quantization_max_amplitude(self):
        """Test correct quantizer output with gain control to maximum amplitude"""

        self.quantizer.gain = AutomaticGainControl(GainControlType.MAX_AMPLITUDE, 1.0, True)

        max_amplitude = 123.7

        # randomly choose quantization levels
        quantization_idx = self.rng.integers(self.quantizer.num_quantization_levels, size=self.num_samples)

        quantization_step = 2 * max_amplitude / self.quantizer.num_quantization_levels
        quantization_levels = -max_amplitude + quantization_step * (quantization_idx + 0.5)

        # add random noise within quantization interval
        random_noise = self.rng.uniform(-quantization_step / 2, quantization_step / 2, size=self.num_samples)
        input_signal = quantization_levels + random_noise

        # add maximum amplitude value to input_vector
        input_signal = np.append(input_signal, [max_amplitude])
        quantization_levels = np.append(quantization_levels, [max_amplitude - quantization_step / 2])

        input_signal = Signal.Create(samples=input_signal, sampling_rate=1.0)

        output_signal = self.quantizer.convert(input_signal)

        np.testing.assert_almost_equal(np.real(output_signal.getitem().flatten()), quantization_levels)

    def test_quantization_rms(self):
        """Test correct quantizer output with gain control to rms amplitude"""

        rms_amplitude = 576577.79

        self.quantizer.gain = AutomaticGainControl(GainControlType.RMS_AMPLITUDE)

        # create signal with desired rms
        input_signal = self.rng.normal(size=self.num_samples)
        measured_rms = rms_value(input_signal)
        input_signal = input_signal / measured_rms * rms_amplitude
        test_signal = deepcopy(input_signal)

        # create non-adaptive quantizer with desired amplitude
        quantizer_no_gain_control = deepcopy(self.quantizer)
        quantizer_no_gain_control.gain = Gain(1 / rms_amplitude)

        # Output of both quantizers must be the same
        input_signal = Signal.Create(samples=input_signal, sampling_rate=1.0)

        output_signal_adaptive = self.quantizer.convert(input_signal)
        output_signal_non_adaptive = quantizer_no_gain_control.convert(input_signal)

        np.testing.assert_almost_equal(output_signal_adaptive.getitem(), output_signal_non_adaptive.getitem())

    def test_quantization_complex(self):
        """Test correct quantization of complex numbers"""
        max_amplitude = 100
        self.quantizer.gain = Gain(1 / max_amplitude, True)

        # randomly choose quantization levels
        quantization_idx = self.rng.integers(self.quantizer.num_quantization_levels, size=(2, self.num_samples))
        quantization_step = 2 * max_amplitude / self.quantizer.num_quantization_levels
        quantization_levels = -max_amplitude + quantization_step * (quantization_idx + 0.5)

        # add random noise within quantization interval
        random_noise = self.rng.uniform(-quantization_step / 2, quantization_step / 2, size=(2, self.num_samples))
        input_signal = quantization_levels + random_noise

        quantization_levels = quantization_levels[0, :] + 1j * quantization_levels[1, :]
        input_signal = input_signal[0, :] + 1j * input_signal[1, :]

        input_signal = Signal.Create(samples=input_signal, sampling_rate=1.0)

        output_signal = self.quantizer.convert(input_signal)

        np.testing.assert_almost_equal(output_signal.getitem().flatten(), quantization_levels)

    def test_quantization_mid_tread(self):
        """Test correct mid-tread quantizer output without gain control"""

        max_amplitude = 150.0
        self.quantizer.gain = Gain(1 / max_amplitude, True)
        self.quantizer.quantizer_type = QuantizerType.MID_TREAD

        # randomly choose quantization levels
        quantization_idx = self.rng.integers(self.quantizer.num_quantization_levels, size=self.num_samples)
        quantization_step = 2 * max_amplitude / self.quantizer.num_quantization_levels
        quantization_levels = -max_amplitude + quantization_step * quantization_idx

        # add random noise within quantization interval
        random_noise = self.rng.uniform(-quantization_step / 2, quantization_step / 2, size=self.num_samples)
        input_signal = quantization_levels + random_noise

        # add saturated values
        max_quantization_level = max_amplitude - quantization_step
        min_quantization_level = -max_amplitude
        quantization_levels = np.append(quantization_levels, [min_quantization_level, max_quantization_level])
        saturated_level_pos = max_quantization_level + 10.0
        saturated_level_neg = min_quantization_level - 10.0
        input_signal = np.append(input_signal, [saturated_level_neg, saturated_level_pos])

        input_signal = Signal.Create(samples=input_signal, sampling_rate=1.0)

        output_signal = self.quantizer.convert(input_signal)

        np.testing.assert_almost_equal(np.real(output_signal.getitem().flatten()), quantization_levels)

    def test_serialization(self) -> None:
        """Test serialization"""

        test_roundtrip_serialization(self, self.quantizer)
