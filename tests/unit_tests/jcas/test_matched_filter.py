# -*- coding: utf-8 -*-

from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch, PropertyMock

import numpy as np
from h5py import File
from numpy.testing import assert_array_equal

from hermespy.core import Signal
from hermespy.modem import CommunicationReception, CommunicationTransmission, DuplexModem, Symbols, WaveformGenerator
from hermespy.radar import Radar, RadarCube, RadarReception
from hermespy.simulation import SimulatedDevice
from hermespy.jcas import JCASTransmission, JCASReception, MatchedFilterJcas
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockWaveformGenerator(WaveformGenerator):
    """Mock communication waveform for modem testing."""

    symbol_rate = 1e9

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


class TestJCASTransmission(TestCase):
    """Test JCAS transmission"""

    def setUp(self) -> None:

        self.signal = Signal(np.empty((1, 0), dtype=np.complex_), 1.)
        self.transmission = JCASTransmission(CommunicationTransmission(self.signal, []))

    def test_hdf_serialization(self) -> None:
        """Test proper serialization to HDF"""

        transmission: JCASTransmission

        with TemporaryDirectory() as tempdir:
            
            file_location = path.join(tempdir, 'testfile.hdf5')
            
            with File(file_location, 'a') as file:
                
                group = file.create_group('testgroup')
                self.transmission.to_HDF(group)
                
            with File(file_location, 'r') as file:
                
                group = file['testgroup']
                transmission = self.transmission.from_HDF(group)

        assert_array_equal(self.signal.samples, transmission.signal.samples)


class TestJCASReception(TestCase):
    """Test JCAS reception"""

    def setUp(self) -> None:

        self.signal = Signal(np.zeros((1, 10), dtype=np.complex_), 1.)
        self.communication_reception = CommunicationReception(self.signal)
        self.cube = RadarCube(np.zeros((1, 1, 10)))
        self.radar_reception = RadarReception(self.signal, self.cube)
        self.reception = JCASReception(self.communication_reception, self.radar_reception)

    def test_hdf_serialization(self) -> None:
        """Test proper serialization to HDF"""

        with TemporaryDirectory() as tempdir:
            
            file_location = path.join(tempdir, 'testfile.hdf5')
            
            with File(file_location, 'a') as file:
                
                group = file.create_group('testgroup')
                self.reception.to_HDF(group)
                
            with File(file_location, 'r') as file:
                
                group = file['testgroup']
                reception = JCASReception.from_HDF(group)

        assert_array_equal(self.signal.samples, reception.signal.samples)


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

    def test_receive_validation(self) -> None:
        """Receiving should raise a RuntimeError if there's no cached transmission"""

        with self.assertRaises(RuntimeError):
            self.joint.receive(Signal(np.zeros((1, 10)), 1.))
        
    def test_transmit_receive(self) -> None:
        
        num_delay_samples = 10
        transmission = self.joint.transmit()
        
        delay_offset = Signal(np.zeros((1, num_delay_samples), dtype=complex),
                              transmission.signal.sampling_rate, carrier_frequency=self.carrier_frequency)
        delay_offset.append_samples(transmission.signal)
        
        self.joint.cache_reception(delay_offset)
        
        reception = self.joint.receive()
        self.assertTrue(10, reception.cube.data.argmax)

        padded_reception = self.joint.receive(transmission.signal)
        self.assertTrue(10, padded_reception.cube.data.argmax)
            
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

    def test_sampling_rate_validation(self) -> None:
        """Sampling rate property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.joint.sampling_rate = 0.

    def test_sampling_rate_setget(self) -> None:
        """Sampling rate property getter should return setter argument"""

        self.joint.sampling_rate = 1.234e10
        self.assertEqual(1.234e10, self.joint.sampling_rate)

        self.joint.sampling_rate = None
        self.assertEqual(self.waveform.sampling_rate, self.joint.sampling_rate)

    def test_max_range_validation(self) -> None:
        """Max range property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.joint.max_range = 0.

    def test_device_setget(self) -> None:
        """Device property getter should return setter argument"""

        expected_device = SimulatedDevice()
        self.joint.device = expected_device

        self.assertIs(expected_device, self.joint.device)
        self.assertIs(expected_device, DuplexModem.device.fget(self.joint))
        self.assertIs(expected_device, Radar.device.fget(self.joint))
            
    def test_recall_transmission(self) -> None:
        """Test joint transmission recall from HDF"""
        
        transmission = self.joint.transmit()
        
        with TemporaryDirectory() as tempdir:
            
            file_location = path.join(tempdir, 'testfile.hdf5')

            with File(file_location, 'w') as file:
                group = file.create_group('testgroup')
                transmission.to_HDF(group)
            
            with File(file_location, 'r') as file:
                recalled_transmission = self.joint._recall_transmission(file['testgroup'])
                
        self.assertEqual(transmission.signal.num_samples, recalled_transmission.signal.num_samples)

    def test_recall_reception(self) -> None:
        """Test joint reception recall from HDF"""
        
        transmission = self.joint.transmit()
        reception = self.joint.receive(transmission.signal)
        
        with TemporaryDirectory() as tempdir:
            
            file_location = path.join(tempdir, 'testfile.hdf5')

            with File(file_location, 'w') as file:
                group = file.create_group('testgroup')
                reception.to_HDF(group)
            
            with File(file_location, 'r') as file:
                recalled_reception = self.joint._recall_reception(file['testgroup'])
                
        self.assertEqual(reception.signal.num_samples, recalled_reception.signal.num_samples)
        
    def test_serialization(self) -> None:
        """Test YAML serialization"""

        with patch('hermespy.jcas.matched_filtering.MatchedFilterJcas.property_blacklist',
                   new_callable=PropertyMock) as blacklist, \
             patch('hermespy.jcas.matched_filtering.MatchedFilterJcas.waveform_generator',
                   new_callable=PropertyMock) as waveform_generator:
            
            blacklist.return_value = {'slot', 'waveform_generator'}
            waveform_generator.return_value = self.waveform
            
            test_yaml_roundtrip_serialization(self, self.joint)
