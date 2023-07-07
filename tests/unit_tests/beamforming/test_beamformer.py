# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.core import FloatingError, Signal, UniformArray, IdealAntenna
from hermespy.beamforming import BeamformerBase, FocusMode, ReceiveBeamformer, TransmitBeamformer

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestBeamformerBase(TestCase):
    """Test the base for all beamformers"""
    
    def setUp(self) -> None:
        
        self.operator = Mock()
        self.base = BeamformerBase(operator=self.operator)
        
    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""
        
        self.assertIs(self.operator, self.base.operator)
        
    def test_operator_setget(self) -> None:
        """Operator property getter should return setter argument"""
        
        operator = Mock()
        self.base.operator = operator
        
        self.assertIs(operator, self.base.operator)
        

class TransmitBeamformerMock(TransmitBeamformer):
    """Mock class to test transmitting beamformers"""
    
    @property
    def num_transmit_input_streams(self) -> int:
        return 2
    
    @property
    def num_transmit_output_streams(self) -> int:
        return 2
    
    @property
    def num_transmit_focus_angles(self) -> int:
        return 1
        
    def _encode(self, samples: np.ndarray, carrier_frequency: float, focus_angles: np.ndarray) -> np.ndarray:
        return samples
    

class TestTransmitBeamformer(TestCase):
    
    def setUp(self) -> None:
        
        self.operator = Mock()
        self.operator.device = Mock()
        
        self.beamformer = TransmitBeamformerMock(operator=self.operator)
        
    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""
        
        self.assertIs(self.operator, self.beamformer.operator)
        
    def test_encode_streams_validation(self) -> None:
        """Encode streams routine should raise exceptions on invalid arguments"""
        
        signal = Signal(np.zeros((3, 10), dtype=complex), 1.)
        with self.assertRaises(ValueError):
            self.beamformer.encode_streams(signal)
            
    def test_encode_streams(self) -> None:
        """Stream encoding should properly encode the argument signal"""    
        
        signal = Signal(np.ones((2, 10), dtype=complex), 1.)
        encoded_signal = self.beamformer.encode_streams(signal)
        
        assert_array_equal(signal.samples, encoded_signal.samples)
        
    def test_precoding_setget(self) -> None:
        """Precoding property getter should return setter argument"""
        
        precoding = Mock()
        self.beamformer.precoding = precoding
        
        self.assertIs(precoding, self.beamformer.precoding)
        self.assertIs(precoding.modem, self.beamformer.operator)
        
    def test_focus_point_validation(self) -> None:
        """Focus point property setter should raise ValueErrors on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.beamformer.transmit_focus = np.ones((1, 2, 3))
            
        with self.assertRaises(ValueError):
            self.beamformer.transmit_focus = np.ones((2, 2))
        
    def test_focus_points_setget(self) -> None:
        """Focus property getters should return focus points property setter arguments"""
        
        expected_mode = FocusMode.CARTESIAN
        expected_points = np.ones((self.beamformer.num_transmit_focus_angles, 2), dtype=float)
        
        self.beamformer.transmit_focus = (expected_points, expected_mode)
        points, mode = self.beamformer.transmit_focus
        
        assert_array_equal(expected_points, points)
        self.assertEqual(expected_mode, mode)
        
        self.beamformer.transmit_focus = expected_points
        
        assert_array_equal(expected_points, points)
        self.assertEqual(expected_mode, mode)

    def test_transmit_validation(self) -> None:
        """Transmit routine should raise exceptions on invalid configurations"""
        
        with self.assertRaises(RuntimeError):
            self.beamformer.transmit(Signal(np.zeros((1, 10), dtype=complex), 1.))
            
        self.operator.device = None
        
        with self.assertRaises(FloatingError):
            self.beamformer.transmit(Signal(np.zeros((2, 10), dtype=complex), 1.))
            
        self.beamformer.operator = None
        
        with self.assertRaises(FloatingError):
            self.beamformer.transmit(Signal(np.zeros((2, 10), dtype=complex), 1.))
            
    def test_transmit(self) -> None:
        """Transmit routine should correctly envoke the encode subroutine"""
        
        expected_signal = Signal(np.ones((2, 10), dtype=complex), 1.)
        focus = np.ones((2, self.beamformer.num_transmit_focus_angles), dtype=float)
        
        steered_signal = self.beamformer.transmit(expected_signal, focus)
        assert_array_equal(expected_signal.samples, steered_signal.samples)


class ReceiveBeamformerMock(ReceiveBeamformer):
    """Mock class to test receiving beamformers"""
    
    @property
    def num_receive_input_streams(self) -> int:
        return 2
    
    @property
    def num_receive_output_streams(self) -> int:
        return 2
    
    @property
    def num_receive_focus_angles(self) -> int:
        return 1
        
    def _decode(self, samples: np.ndarray, carrier_frequency: float, focus_angles: np.ndarray) -> np.ndarray:
        return np.repeat(samples[np.newaxis, ::], focus_angles.shape[0], axis=0)
    

class TestReceiveBeamformer(TestCase):
    
    def setUp(self) -> None:
        
        self.operator = Mock()
        self.operator.device = Mock()
        self.operator.device.carrier_frequency = 10e9
        self.operator.device.antennas = UniformArray(IdealAntenna, 1e-2, (4, 4))
        
        self.beamformer = ReceiveBeamformerMock(operator=self.operator)
        
    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""
        
        self.assertIs(self.operator, self.beamformer.operator)
        
    def test_decode_streams_validation(self) -> None:
        """Decode streams routine should raise exceptions on invalid arguments"""
        
        signal = Signal(np.zeros((3, 10), dtype=complex), 1.)
        with self.assertRaises(ValueError):
            self.beamformer.decode_streams(signal)
            
    def test_decode_streams(self) -> None:
        """Stream decoding should properly encode the argument signal"""    
        
        signal = Signal(np.ones((2, 10), dtype=complex), 1.)
        decoded_signal = self.beamformer.decode_streams(signal)
        
        assert_array_equal(signal.samples, decoded_signal.samples)
        
    def test_precoding_setget(self) -> None:
        """Precoding property getter should return setter argument"""
        
        precoding = Mock()
        self.beamformer.precoding = precoding
        
        self.assertIs(precoding, self.beamformer.precoding)
        self.assertIs(precoding.modem, self.beamformer.operator)
        
    def test_focus_point_validation(self) -> None:
        """Focus point property setter should raise ValueErrors on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.beamformer.receive_focus = np.ones((1, 2, 3))
            
        with self.assertRaises(ValueError):
            self.beamformer.receive_focus = np.ones((2, 2))
        
    def test_focus_points_setget(self) -> None:
        """Focus property getters should return focus points property setter arguments"""
        
        expected_mode = FocusMode.CARTESIAN
        expected_points = np.ones((self.beamformer.num_receive_focus_angles, 2), dtype=float)
        
        self.beamformer.receive_focus = (expected_points, expected_mode)
        points, mode = self.beamformer.receive_focus
        
        assert_array_equal(expected_points, points)
        self.assertEqual(expected_mode, mode)
        
        self.beamformer.receive_focus = expected_points
        
        assert_array_equal(expected_points, points)
        self.assertEqual(expected_mode, mode)
        
    def test_probe_focus_point_validation(self) -> None:
        """Focus point property setter should raise ValueErrors on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.beamformer.probe_focus_points = np.ones((2, 3))
            
        with self.assertRaises(ValueError):
            self.beamformer.probe_focus_points = np.ones((2, 3, 4, 1))
            
        with self.assertRaises(ValueError):
            self.beamformer.probe_focus_points = np.ones((2, 2))
            
    def test_probe_focus_setget(self) -> None:
        """Probe focus getter should return setter argument"""
        
        expected_points = np.array([[1, 2]], dtype=complex)
        self.beamformer.probe_focus_points = expected_points
        
        assert_array_equal(expected_points[np.newaxis, ::], self.beamformer.probe_focus_points)

    def test_receive_validation(self) -> None:
        """Receive routine should raise exceptions on invalid configurations"""
        
        with self.assertRaises(RuntimeError):
            self.beamformer.receive(Signal(np.zeros((1, 10), dtype=complex), 1.))
            
        self.operator.device = None
        
        with self.assertRaises(FloatingError):
            self.beamformer.receive(Signal(np.zeros((2, 10), dtype=complex), 1.))
            
        self.beamformer.operator = None
        
        with self.assertRaises(FloatingError):
            self.beamformer.receive(Signal(np.zeros((2, 10), dtype=complex), 1.))
            
    def test_receive(self) -> None:
        """Receive routine should correctly envoke the encode subroutine"""
        
        expected_signal = Signal(np.ones((2, 10), dtype=complex), 1.)
        focus = np.ones((2, self.beamformer.num_receive_focus_angles), dtype=float)
        
        steered_signal = self.beamformer.receive(expected_signal, focus)
        assert_array_equal(expected_signal.samples, steered_signal.samples)

    def test_probe_validation(self) -> None:
        """Probe routine should raise exceptions on invalid configurations"""
        
        with self.assertRaises(RuntimeError):
            self.beamformer.probe(Signal(np.zeros((1, 10), dtype=complex), 1.))
            
        self.operator.device = None
        
        with self.assertRaises(FloatingError):
            self.beamformer.probe(Signal(np.zeros((2, 10), dtype=complex), 1.))
            
        self.beamformer.operator = None
        
        with self.assertRaises(FloatingError):
            self.beamformer.probe(Signal(np.zeros((2, 10), dtype=complex), 1.))
            
    def test_probe(self) -> None:
        """Probe routine should correctly envoke the encode subroutine"""
        
        expected_samples = np.ones((2, 10), dtype=complex)
        expected_signal = Signal(expected_samples, 1.)
        focus = np.ones((1, 2, self.beamformer.num_receive_focus_angles), dtype=float)
        
        steered_signal = self.beamformer.probe(expected_signal, focus)
        assert_array_equal(expected_samples[np.newaxis, ::], steered_signal)

    def test_plot_receive_characteristics(self) -> None:
        """Plotting the receive beamforming characteristics should result in a proper figure generation"""
        
        with patch('matplotlib.pyplot.figure') as figure:
            
            _ = self.beamformer.plot_receive_pattern()
            figure.assert_called()
            
        with patch('matplotlib.pyplot.figure') as figure:
            
            _ = self.beamformer.PlotReceivePattern()
            figure.assert_called()
