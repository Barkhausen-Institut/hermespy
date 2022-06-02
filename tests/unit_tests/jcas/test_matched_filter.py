from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from scipy.constants import speed_of_light

from hermespy.core import Signal, ChannelStateInformation
from hermespy.modem import Symbols
from hermespy.jcas import MatchedFilterJcas

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "3.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMatchedFilterJoint(TestCase):
    """Matched filter joint testing."""
    
    def setUp(self) -> None:
        
        self.device = Mock()
        self.device.num_antennas = 1
        self.device.carrier_frequency = 0.
        self.waveform = Mock()
        self.waveform.frame_duration = 1e-5
        self.waveform.sampling_rate = speed_of_light
        self.waveform.bits_per_frame = 0
        self.waveform.symbols_per_frame = 0
        self.waveform.samples_in_frame = 5
        self.waveform.map.return_value = Symbols(np.empty(0, dtype=complex))
        self.waveform.unmap.return_value = np.empty(0, dtype=complex)
        self.waveform.modulate.return_value = Signal(np.ones((1, 5), dtype=complex), sampling_rate=self.waveform.sampling_rate)
        self.waveform.demodulate.return_value = Symbols(np.empty(0, dtype=complex)), ChannelStateInformation.Ideal(0, 1, 1), np.empty(0, dtype=float)
        self.waveform.synchronization.synchronize.return_value = [(np.ones((1, 5), dtype=complex), ChannelStateInformation.Ideal(5))]
        
        self.range_resolution = 10
        
        self.joint = MatchedFilterJcas(max_range=10)
        self.joint.device = self.device
        self.joint.waveform_generator = self.waveform
        
    def test_transmit_receive(self) -> None:
        
        num_delay_samples = 10
        signal, _, _ = self.joint.transmit()
        
        delay_offset = Signal(np.zeros((1, num_delay_samples), dtype=complex), signal.sampling_rate)
        delay_offset.append_samples(signal)
        
        self.joint._receiver.cache_reception(delay_offset)
        
        signal, _, _, cube = self.joint.receive()
        self.assertTrue(10, cube.data.argmax)

    def test_range_resolution_setget(self) -> None:
        """Maximum range property getter should return setter argument."""
        
        range_resolution = 5.
        self.joint.range_resolution = range_resolution
        
        self.assertEquals(range_resolution, self.joint.range_resolution)
        
    def test_range_resolution_validation(self) -> None:
        """Maximum range property setter should raise ValueError on non-positive arguments."""
        
        with self.assertRaises(ValueError):
            self.joint.range_resolution = -1.
            
        with self.assertRaises(ValueError):
            self.joint.range_resolution = 0.
            
    def test_range_resolution_setget(self) -> None:
        """Range resolution property getter should return setter argument."""
        
        range_resolution = 1e-3
        self.joint.range_resolution = range_resolution
        
        self.assertEqual(range_resolution, self.joint.range_resolution)
        
    def test_range_resolution_validation(self) -> None:
        """Range resolution property setter should raise ValueError on non-positive arguments."""
        
        with self.assertRaises(ValueError):
            self.joint.range_resolution = -1.
            
        with self.assertRaises(ValueError):
            self.joint.range_resolution = 0.
