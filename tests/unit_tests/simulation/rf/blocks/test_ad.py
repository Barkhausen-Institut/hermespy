# -*- coding: utf-8 -*-

from __future__ import annotations
from unittest import TestCase
from unittest.mock import Mock, patch
from copy import deepcopy

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.core import Signal
from hermespy.simulation.rf import RFSignal
from hermespy.simulation.rf.blocks.ad import ConverterBase, GainControlBase, Gain, AutomaticGainControl, GainControlType, QuantizerType, ADC, DAC
from hermespy.tools.math import rms_value
from unit_tests.core.test_factory import test_roundtrip_serialization
from unit_tests.utils import assert_signals_equal, random_rf_signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockGainControl(GainControlBase):
    """Mock gain control model for testing"""

    def estimate_gain(self, _: Signal) -> float:
        return 0.456


class TestConverterBase(TestCase):
    """Test base class of AD and DA converters"""

    converter: ConverterBase

    def test_num_quantization_bits_setget(self) -> None:
        """Quantization bits property getter should return setter argument."""

        number_bits = 10
        self.converter.num_quantization_bits = number_bits
        self.assertEqual(number_bits, self.converter.num_quantization_bits)

        number_bits = 0
        self.converter.num_quantization_bits = number_bits
        self.assertEqual(None, self.converter.num_quantization_bits)

    def test_num_quantization_bits_validation(self) -> None:
        """Quantization bits property setter should raise ValueError on arguments smaller than zero or non-integer."""

        with self.assertRaises(ValueError):
            self.converter.num_quantization_bits = -1

        with self.assertRaises(ValueError):
            self.converter.num_quantization_bits = 2.5  # type: ignore

    def test_num_quantization_levels_get(self) -> None:
        """Quantization levels property getter should return correct value"""

        number_bits = 8
        self.converter.num_quantization_bits = number_bits
        self.assertEqual(2**number_bits, self.converter.num_quantization_levels)

    def test_serialization(self) -> None:
        """Test serialization"""

        test_roundtrip_serialization(self, self.converter)


class TestGainControlBase(TestCase):
    """Test gain control base model"""

    def setUp(self) -> None:
        self.gain = MockGainControl()

    def test_rescale_quanization_setget(self) -> None:
        """Rescale quantization property getter should return setter argument"""

        expected_rescale = True
        self.gain.rescale_quantization = expected_rescale
        self.assertEqual(self.gain.rescale_quantization, expected_rescale)

    def test_adjust_signal(self) -> None:
        """Adjust signal should return correct signal"""

        gain = 0.456
        signal = RFSignal.FromNDArray(np.ones((2, 1)) / gain, 1.0)
        expected_signal = RFSignal.FromNDArray(np.ones((2, 1)), 1.0)

        adjusted_signal = self.gain.adjust_signal(signal, gain)
        assert_signals_equal(self, expected_signal, adjusted_signal)

    def test_scale_quantized_signal_disabled(self) -> None:
        """Scale quantized signal should return correct signal when the flag is disabled"""

        self.gain.rescale_quantization = False

        gain = 0.456
        signal = RFSignal.FromNDArray(np.ones((2, 1)), 1.0)
        expected_signal = RFSignal.FromNDArray(np.ones((2, 1)), 1.0)

        rescaled_signal = self.gain.scale_quantized_signal(signal, gain)
        assert_signals_equal(self, expected_signal, rescaled_signal)

    def test_scale_quantized_signal_enabled(self) -> None:
        """Scale quantized signal should return correct signal when the flag is enabled"""

        self.gain.rescale_quantization = True

        gain = 0.456
        signal = RFSignal.FromNDArray(np.ones((2, 1)), 1.0)
        expected_signal = RFSignal.FromNDArray(np.ones((2, 1)) / gain, 1.0)

        rescaled_signal = self.gain.scale_quantized_signal(signal, gain)
        assert_signals_equal(self, expected_signal, rescaled_signal)


class TestGain(TestCase):
    """Test gain model"""

    def setUp(self) -> None:
        self.gain = Gain()

    def test_gain_validation(self) -> None:
        """Gain property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.gain.gain = -1.0

        with self.assertRaises(ValueError):
            self.gain.gain = 0.0

    def test_gain_setget(self) -> None:
        """Gain property getter should return setter argument"""

        expected_gain = 0.5
        self.gain.gain = expected_gain

        self.assertEqual(self.gain.gain, expected_gain)

    def test_estimate_gain(self) -> None:
        """Gain estimation should return correct gain"""

        estimate = self.gain.estimate_gain(Mock())
        self.assertEqual(estimate, self.gain.gain)

    def test_serialization(self) -> None:
        """Test serialization"""

        test_roundtrip_serialization(self, self.gain)


class TestAutomaticGainControl(TestCase):
    """Test automatic gain control model"""

    def setUp(self) -> None:
        self.gain = AutomaticGainControl()

    def test_agc_type_setget(self) -> None:
        """AGC type property getter should return setter argument"""

        expected_type = GainControlType.RMS_AMPLITUDE
        self.gain.agc_type = expected_type

        self.assertEqual(self.gain.agc_type, expected_type)

    def test_backoff_validation(self) -> None:
        """Backoff property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.gain.backoff = -1.0

        with self.assertRaises(ValueError):
            self.gain.backoff = 0.0

    def test_backoff_setget(self) -> None:
        """Backoff property getter should return setter argument"""

        expected_backoff = 0.5
        self.gain.backoff = expected_backoff

        self.assertEqual(self.gain.backoff, expected_backoff)

    def test_estimate_gain_validation(self) -> None:
        """Gain estimation should raise RuntimeError on invalid internal states"""

        signal = Signal.Create(np.ones((2, 1)), 1.0, 0.0)
        self.gain.agc_type = Mock(spec=GainControlType)

        with self.assertRaises(RuntimeError):
            self.gain.estimate_gain(signal)

    def test_estimate_gain(self) -> None:
        """Gain estimation should return correct gain"""

        self.gain.agc_type = GainControlType.MAX_AMPLITUDE
        signal = Signal.Create(2 * np.ones((2, 1)), 1.0, 0.0)
        max_amplitude_gain_estimate = self.gain.estimate_gain(signal)

        self.gain.agc_type = GainControlType.RMS_AMPLITUDE
        signal = Signal.Create(2 * np.ones((2, 1)), 1.0, 0.0)
        rms_amplitude_gain_estimate = self.gain.estimate_gain(signal)

        self.assertEqual(0.5, max_amplitude_gain_estimate)
        self.assertEqual(0.5, rms_amplitude_gain_estimate)

    def test_serialization(self) -> None:
        """Test serialization"""

        test_roundtrip_serialization(self, self.gain)


class TestADC(TestConverterBase):
    """Test analog-digital converter model"""

    converter: ADC

    def setUp(self) -> None:
        self.num_samples = 61
        self.bandwidth = 20.0
        self.oversampling_factor = 4

        self.converter = ADC()
        self.realization = self.converter.realize(self.bandwidth, self.oversampling_factor, 0.0)

    def test_quantizer_type_setget(self) -> None:
        """Quantier type property getter should return setter argument"""

        quant_type = QuantizerType.MID_TREAD
        self.converter.quantizer_type = quant_type
        self.assertEqual(quant_type, self.converter.quantizer_type)

    def test_no_quantization(self) -> None:
        """Test correct quantizer output with infinite resolution"""

        self.converter.num_quantization_bits = 0
        input_rf_signal = random_rf_signal(1, self.num_samples, self.bandwidth, self.oversampling_factor)
        output_signal = self.converter.propagate(self.realization, input_rf_signal, False)
        assert_signals_equal(self, input_rf_signal, output_signal)

    def test_mid_riser_convert(self) -> None:
        """Convert method should return correctly quantized signal for mid_riser setting"""

        self.converter.num_quantization_bits = 2
        self.converter.quantizer_type = QuantizerType.MID_RISER

        signal = RFSignal.FromNDArray(np.array([[-0.75, -0.25, 0.25, 0.75]]) + 1j * np.array([[0.75, 0.25, -0.25, -0.75]]), self.bandwidth*self.oversampling_factor)
        expected_signal = RFSignal.FromNDArray(np.array([[-0.75, -0.25, 0.25, 0.75]]) + 1j * np.array([[0.75, 0.25, -0.25, -0.75]]), self.bandwidth*self.oversampling_factor)
        converted_signal = self.converter.propagate(self.realization, signal, False)

        assert_signals_equal(self, expected_signal, converted_signal)

    def test_mid_tread_convert(self) -> None:
        """Convert method should return correctly quantized signal for mid_tread setting"""

        self.converter.num_quantization_bits = 2
        self.converter.quantizer_type = QuantizerType.MID_TREAD

        signal = RFSignal.FromNDArray(np.array([[-1.2, 0.6]]) + 1j* np.array([[1.2, 0.6]]), self.bandwidth*self.oversampling_factor)
        expected_signal = RFSignal.FromNDArray(np.array([[-0.8, 0.4]]) + 1j* np.array([[0.4, 0.4]]), self.bandwidth*self.oversampling_factor)
        converted_signal = self.converter.propagate(self.realization, signal, False)

        assert_signals_equal(self, expected_signal, converted_signal)

    def test_quantization_no_gain_control(self) -> None:
        """Test correct quantizer output without gain control"""

        max_amplitude = 100
        self.converter.gain = Gain(1 / max_amplitude, True)
        self.converter.num_quantization_bits = 3

        # randomly choose quantization levels
        rng = np.random.default_rng(42)
        quantization_idx = rng.integers(self.converter.num_quantization_levels, size=self.num_samples)
        quantization_step = 2 * max_amplitude / self.converter.num_quantization_levels
        quantization_levels = -max_amplitude + quantization_step * (quantization_idx + 0.5)

        # add random noise within quantization interval
        random_noise = rng.uniform(-quantization_step / 2, quantization_step / 2, size=self.num_samples)
        input_signal = quantization_levels + random_noise

        # add saturated values
        max_quantization_level = max_amplitude - quantization_step / 2
        quantization_levels = np.append(quantization_levels, [-max_quantization_level, max_quantization_level])
        saturated_level = max_amplitude + 10.0
        input_signal = np.append(input_signal, [-saturated_level, saturated_level])

        input_signal = RFSignal.FromNDArray(input_signal.reshape((1, -1)), self.realization.sampling_rate)

        output_signal = self.converter.propagate(self.realization, input_signal, False)
        np.testing.assert_almost_equal(np.real(output_signal.view(np.ndarray).flatten()), quantization_levels)

    def test_quantization_max_amplitude(self) -> None:
        """Test correct quantizer output with gain control to maximum amplitude"""

        self.converter.gain = AutomaticGainControl(GainControlType.MAX_AMPLITUDE, 1.0, True)

        max_amplitude = 123.7

        # randomly choose quantization levels
        rng = np.random.default_rng(42)
        self.converter.num_quantization_bits = 4
        quantization_idx = rng.integers(self.converter.num_quantization_levels, size=self.num_samples)

        quantization_step = 2 * max_amplitude / self.converter.num_quantization_levels
        quantization_levels = -max_amplitude + quantization_step * (quantization_idx + 0.5)

        # add random noise within quantization interval
        random_noise = rng.uniform(-quantization_step / 2, quantization_step / 2, size=self.num_samples)
        input_signal = quantization_levels + random_noise

        # add maximum amplitude value to input_vector
        input_signal = np.append(input_signal, [max_amplitude])
        quantization_levels = np.append(quantization_levels, [max_amplitude - quantization_step / 2])

        input_signal = RFSignal.FromNDArray(input_signal.reshape((1, -1)), self.realization.sampling_rate)
        output_signal = self.converter.propagate(self.realization, input_signal, False)
        np.testing.assert_almost_equal(np.real(output_signal.view(np.ndarray)).flatten(), quantization_levels)

    def test_quantization_rms(self):
        """Test correct quantizer output with gain control to rms amplitude"""

        rms_amplitude = 576577.79

        self.converter.gain = AutomaticGainControl(GainControlType.RMS_AMPLITUDE)

        # create signal with desired rms
        rng = np.random.default_rng(42)
        input_signal = rng.normal(size=self.num_samples)
        measured_rms = rms_value(input_signal)
        input_signal = input_signal / measured_rms * rms_amplitude
        test_signal = deepcopy(input_signal)

        # create non-adaptive quantizer with desired amplitude
        quantizer_no_gain_control = deepcopy(self.converter)
        quantizer_no_gain_control.gain = Gain(1 / rms_amplitude)

        # Output of both quantizers must be the same
        input_signal = RFSignal.FromNDArray(input_signal.reshape((1, -1)), self.realization.sampling_rate)

        output_signal_adaptive = self.converter.propagate(self.realization, input_signal, False)
        output_signal_non_adaptive = quantizer_no_gain_control.propagate(self.realization, input_signal, False)
        assert_signals_equal(self, output_signal_adaptive, output_signal_non_adaptive)

    def test_plot_quantizers(self) -> None:
        """Plot quantizers method should return matplotlib figure"""

        axes = Mock()

        self.converter.plot_quantizer(np.ones((2, 2)), fig_axes=axes)
        axes.plot.assert_called_once()

        with patch("matplotlib.pyplot.figure") as figure_mock:
            self.converter.plot_quantizer(np.ones((2, 2)))
            figure_mock.assert_called_once()


class TestDAC(TestConverterBase):
    """Test digital-analog converter model"""

    def setUp(self) -> None:
        self.converter = DAC()


del TestConverterBase
