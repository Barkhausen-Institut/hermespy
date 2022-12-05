# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import patch, PropertyMock

import numpy as np

from hermespy.core import Signal
from hermespy.modem import Symbols, WaveformGenerator
from hermespy.simulation import SimulatedDevice
from hermespy.jcas import MatchedFilterJcas
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockWaveformGenerator(WaveformGenerator):
    """Mock communication waveform for modem testing."""

    symbol_rate = 1e4

    @property
    def samples_in_frame(self) -> int:
        
        return self.oversampling_factor * self.symbols_per_frame
    
    @property
    def bits_per_frame(self) -> int:
        
        return self.symbols_per_frame * 1
    
    @property
    def symbols_per_frame(self) -> int:
        
        return 100
    
    @property
    def bit_energy(self) -> float:
    
        return 1.
    
    @property
    def symbol_energy(self) -> float:
        
        return 1.
    
    @property
    def power(self) -> float:
        
        return 1.

    @property
    def carrier_frequency(self) -> float:
        
        return 1.
    
    def map(self, data_bits: np.ndarray) -> Symbols:
        
        return Symbols(data_bits[np.newaxis, np.newaxis, :])
    
    def unmap(self, symbols: Symbols) -> np.ndarray:
        
        return symbols.raw.real.flatten()
    
    def modulate(self, data_symbols: Symbols) -> Signal:
        
        return Signal(data_symbols.raw.flatten().repeat(self.oversampling_factor), self.sampling_rate)

    def demodulate(self, signal: np.ndarray) -> Symbols:
        
        symbols = Symbols(signal[np.newaxis, np.newaxis, :self.oversampling_factor * self.symbols_per_frame:self.oversampling_factor])
        return symbols
    
    @property
    def bandwidth(self) -> float:
        
        return self.sampling_rate
    
    @property
    def sampling_rate(self) -> float:
        
        return self.symbol_rate * self.oversampling_factor


class TestMatchedFilterJoint(TestCase):
    """Matched filter joint testing."""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        self.carrier_frequency = 1e8

        self.joint = MatchedFilterJcas(max_range=10)
        self.waveform = MockWaveformGenerator()
        self.joint.waveform_generator = self.waveform
        
        self.device = SimulatedDevice(carrier_frequency=self.carrier_frequency)
        self.device._rng = self.rng
        self.device.transmitters.add(self.joint)
        self.device.receivers.add(self.joint)
        
    def test_transmit_receive(self) -> None:
        
        num_delay_samples = 10
        transmission = self.joint.transmit()
        
        delay_offset = Signal(np.zeros((1, num_delay_samples), dtype=complex),
                              transmission.signal.sampling_rate, carrier_frequency=self.carrier_frequency)
        delay_offset.append_samples(transmission.signal)
        
        self.joint.cache_reception(delay_offset)
        
        reception = self.joint.receive()
        self.assertTrue(10, reception.cube.data.argmax)
            
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
            
    def test_serialization(self) -> None:
        """Test YAML serialization"""

        with patch('hermespy.jcas.matched_filtering.MatchedFilterJcas.property_blacklist',
                   new_callable=PropertyMock) as blacklist, \
             patch('hermespy.jcas.matched_filtering.MatchedFilterJcas.waveform_generator',
                   new_callable=PropertyMock) as waveform_generator:
            
            blacklist.return_value = {'slot', 'waveform_generator'}
            waveform_generator.return_value = self.waveform
            
            test_yaml_roundtrip_serialization(self, self.joint)
