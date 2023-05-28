# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.core import Signal
from hermespy.simulation import Gain, AutomaticGainControl, GainControlType, QuantizerType, AnalogDigitalConverter
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestGain(TestCase):
    """Test gain model"""
    
    def setUp(self) -> None:
        
        self.gain = Gain()
        
    def test_gain_validation(self) -> None:
        """Gain property should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.gain.gain = -1.
            
        with self.assertRaises(ValueError):
            self.gain.gain = 0.
        
    def test_gain_setget(self) -> None:
        """Gain property getter should return setter argument"""
        
        expected_gain = 0.5
        self.gain.gain = expected_gain
        
        self.assertEqual(self.gain.gain, expected_gain)
        
    def test_multiply_signal(self) -> None:
        """Multiply method should multiply signal by gain"""
        
        expected_gain = 0.5
        self.gain.gain = expected_gain
        
        signal = Signal(np.ones((2, 1)), 1., 0.)
        self.gain.multiply_signal(signal)
        
        np.testing.assert_array_almost_equal(signal.samples, expected_gain * np.ones((2, 1)))
        
    def test_divide_signal(self) -> None:
        """Divide method should divide signal by gain"""
        
        expected_gain = 0.5
        self.gain.gain = expected_gain
        
        signal = Signal(np.ones((2, 1)), 1., 0.)
        self.gain.divide_signal(signal)
        
        np.testing.assert_array_almost_equal(signal.samples, (1. / expected_gain) * np.ones((2, 1)))
        
    def test_yaml_serialization(self) -> None:
        """Test YAML roundtrip serialization"""
        
        test_yaml_roundtrip_serialization(self, self.gain)

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
            self.gain.backoff = -1.
            
        with self.assertRaises(ValueError):
            self.gain.backoff = 0.
            
    def test_backoff_setget(self) -> None:
        """Backoff property getter should return setter argument"""
        
        expected_backoff = 0.5
        self.gain.backoff = expected_backoff
        
        self.assertEqual(self.gain.backoff, expected_backoff)
        
    def test_multiply_signal_validation(self) -> None:
        """Multiply method should raise RuntimeError on invalid internal states"""
        
        signal = Signal(np.ones((2, 1)), 1., 0.)
        self.gain.agc_type = Mock(spec=GainControlType)
        
        with self.assertRaises(RuntimeError):
            self.gain.multiply_signal(signal)
        
    def test_multiply_signal(self) -> None:
        """Multiply method should multiply signal by gain"""
        
        self.gain.agc_type = GainControlType.MAX_AMPLITUDE
        signal = Signal(2 * np.ones((2, 1)), 1., 0.)
        self.gain.multiply_signal(signal)
        
        assert_array_almost_equal(signal.samples, np.ones((2, 1), dtype=complex))
        
        self.gain.agc_type = GainControlType.RMS_AMPLITUDE
        signal = Signal(2 * np.ones((2, 1)), 1., 0.)
        self.gain.multiply_signal(signal)
        
        assert_array_almost_equal(signal.samples, np.ones((2, 1), dtype=complex))
        
    def test_yaml_serialization(self) -> None:
        """Test YAML roundtrip serialization"""
        
        test_yaml_roundtrip_serialization(self, self.gain)


class TestAnalogDigitalConverter(TestCase):
    """Test analog-digital converter model"""
    
    def setUp(self) -> None:
        
        self.adc = AnalogDigitalConverter()
        
    def test_num_quantization_bits_validation(self) -> None:
        """Number of quantization bits property should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.adc.num_quantization_bits = -1
            
    def test_num_quantization_bits_setget(self) -> None:
        """Number of quantization bits property getter should return setter argument"""
        
        expected_num_bits = 4
        self.adc.num_quantization_bits = expected_num_bits
        
        self.assertEqual(self.adc.num_quantization_bits, expected_num_bits)
        
        self.adc.num_quantization_bits = None
        self.assertIsNone(self.adc.num_quantization_bits)
        
    def test_num_quantization_levels(self) -> None:
        """Number of quantization levels property should return 2 ** num_quantization_bits"""
        
        expected_num_levels = 2 ** 4
        self.adc.num_quantization_bits = 4
        
        self.assertEqual(self.adc.num_quantization_levels, expected_num_levels)
        
        self.adc.num_quantization_bits = None
        self.assertEqual(np.inf, self.adc.num_quantization_levels)

    def test_quantizer_type_setget(self) -> None:
        """Quantizer type property getter should return setter argument"""
        
        expected_type = QuantizerType.MID_TREAD
        self.adc.quantizer_type = expected_type
        
        self.assertEqual(self.adc.quantizer_type, expected_type)
        
    def test_default_convert(self) -> None:
        """Convert method should return signal"""
        
        self.adc.num_quantization_bits = None
        expected_signal = Signal(np.ones((2, 2)), 1., 0.)
        
        converted_signal = self.adc.convert(expected_signal)
        assert_array_almost_equal(converted_signal.samples, expected_signal.samples)

    def test_mid_riser_convert(self) -> None:
        """Convert method should return correctly quantized signal for mid_riser setting"""
        
        self.adc.num_quantization_bits = 2
        self.adc.quantizer_type = QuantizerType.MID_RISER
        
        signal = Signal(np.ones((2, 2)), 1., 0.)
        expected_signal = Signal(.75 * np.ones((2, 2)) + .25j * np.ones((2, 2)), 1., 0.)
        converted_signal = self.adc.convert(signal)
        
        assert_array_almost_equal(converted_signal.samples, expected_signal.samples)

    def test_mid_tread_convert(self) -> None:
        """Convert method should return correctly quantized signal for mid_tread setting"""
        
        self.adc.num_quantization_bits = 2
        self.adc.quantizer_type = QuantizerType.MID_TREAD
        
        signal = Signal(np.ones((2, 2)), 1., 0.)
        expected_signal = Signal(.5 * np.ones((2, 2)), 1., 0.)
        converted_signal = self.adc.convert(signal)
        
        assert_array_almost_equal(converted_signal.samples, expected_signal.samples)
        
    def test_plot_quantizers(self) -> None:
        """Plot quantizers method should return matplotlib figure"""
        
        signal = Signal(np.ones((2, 2)), 1., 0.)
        axes = Mock()
        
        self.adc.plot_quantizer(signal.samples, fig_axes=axes)
        axes.plot.assert_called_once()
        
        with patch('matplotlib.pyplot.figure') as figure_mock:
            
            self.adc.plot_quantizer(signal.samples)
            figure_mock.assert_called_once()
