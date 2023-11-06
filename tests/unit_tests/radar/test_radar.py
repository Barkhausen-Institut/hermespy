# -*- coding: utf-8 -*-

from os.path import join
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from h5py import File
from numpy.testing import assert_array_equal
from matplotlib.figure import Figure
from scipy.constants import speed_of_light

from hermespy.core import Signal, SNRType, IdealAntenna, UniformArray
from hermespy.radar import Radar, RadarCube, RadarWaveform, PointDetection, RadarReception, RadarPointCloud
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization


__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarWaveformMock(RadarWaveform):
    """Mock implementation of a a radar waveform"""

    def __init__(self) -> None:

        self.num_samples = 10
        self.rng = np.random.default_rng(42)

    def ping(self) -> Signal:

        return Signal(np.exp(2j * np.pi * self.rng.uniform(0, 1, size=(1, self.num_samples))), self.sampling_rate)
            
    def estimate(self, signal: Signal) -> np.ndarray:

        num_velocity_bins = len(self.relative_doppler_bins)
        num_range_bins = len(self.range_bins)

        velocity_range_estimate = np.zeros((num_velocity_bins, num_range_bins), dtype=float)
        velocity_range_estimate[int(.5 * num_velocity_bins), int(.5 * num_range_bins)] = 1.

        return velocity_range_estimate

    @property
    def sampling_rate(self) -> float:
        return 1.2345

    @property
    def range_bins(self) -> np.ndarray:
        return np.arange(10)
    
    @property
    def max_relative_doppler(self) -> float:
        return 1.
        
    @property
    def relative_doppler_resolution(self) -> float:
        return .5

    @property
    def relative_doppler_bins(self) -> np.ndarray:
        return np.arange(5)
    
    @property
    def energy(self) -> float:
        return 1.
    
    @property
    def power(self) -> float:
        return 1.

    @property
    def frame_duration(self) -> float:

        return 12.345


class TestRadarReception(TestCase):
    """Test the radar reception model"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)

        self.signal = Signal(self.rng.normal(size=(2, 1)), 1., 0.)
        self.cube = RadarCube(self.rng.normal(size=(5, 4, 4)), self.rng.normal(size=(5, 2)), self.rng.normal(size=4), self.rng.normal(size=4))
        self.cloud = RadarPointCloud(max_range=1.)

        self.reception = RadarReception(self.signal, self.cube, self.cloud)
        
    def test_hdf_serialization(self) -> None:
        """Test HDF roundtrip serialization"""

        with TemporaryDirectory() as tempdir:
            
            file_path = join(tempdir, 'test.hdf')

            file = File(file_path, 'w')
            group = file.create_group('g1')
            self.reception.to_HDF(group)
            file.close()
            
            file = File(file_path, 'r')
            recalled_reception = RadarReception.from_HDF(file['g1'])
            file.close()
            
        assert_array_equal(self.reception.signal.samples, recalled_reception.signal.samples)


class TestRadar(TestCase):
    """Test the radar operator"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        
        self.waveform = RadarWaveformMock()
        self.device = SimulatedDevice(carrier_frequency=1e8, antennas=UniformArray(IdealAntenna(), .5 * speed_of_light / 1e8, (2, 1, 1)))
        
        self.radar = Radar()
        self.radar.waveform = self.waveform
        self.radar.device = self.device
        self.radar.receive_beamformer = Mock()
        self.radar.receive_beamformer.num_receive_input_streams = 2
        self.radar.receive_beamformer.num_receive_output_streams = 1
        self.radar.receive_beamformer.probe.return_value = np.zeros((1, 1, 1), dtype=complex)
        self.radar.receive_beamformer.probe_focus_points = np.zeros((1, 1, 2), dtype=float)

    def test_transmit_beamfromer_setget(self) -> None:
        """Transmit beamformer property getter should return setter argument"""
        
        self.radar.transmit_beamformer = None
        self.assertEqual(None, self.radar.transmit_beamformer)
        
        beamformer = Mock()
        self.radar.transmit_beamformer = beamformer
        self.assertEqual(beamformer, self.radar.transmit_beamformer)
        
    def test_receive_beamfromer_setget(self) -> None:
        """Receive beamformer property getter should return setter argument"""
        
        self.radar.receive_beamformer = None
        self.assertEqual(None, self.radar.receive_beamformer)
        
        beamformer = Mock()
        self.radar.receive_beamformer = beamformer
        self.assertEqual(beamformer, self.radar.receive_beamformer)
        
    def test_sampling_rate(self) -> None:
        """Sampling rate property should return the waveform sampling rate"""
        
        self.assertEqual(self.waveform.sampling_rate, self.radar.sampling_rate)
        
    def test_default_frame_duration(self) -> None:
        """Frame duration property should return zero if no waveform is configured"""
        
        self.radar.waveform = None
        self.assertEqual(0., self.radar.frame_duration)
        
    def test_frame_duration(self) -> None:
        """Frame duration property should return the frame duration"""
        
        self.assertEqual(12.345, self.radar.frame_duration)
        
    def test_noise_power(self) -> None:
        """Noise power estimator should compute the correct powers"""
        
        self.assertEqual(1., self.radar.noise_power(1., SNRType.EN0))
        self.assertEqual(1., self.radar.noise_power(1., SNRType.PN0))
        
        with self.assertRaises(ValueError):
            _ = self.radar.noise_power(1., SNRType.EBN0)

        self.radar.waveform = None
        self.assertEqual(0., self.radar.noise_power(1., SNRType.PN0))
        
    def test_waveform_setget(self) -> None:
        """Waveform property getter should return setter argument"""
        
        self.radar.waveform = None
        self.assertEqual(None, self.radar.waveform)
        
        waveform = Mock()
        self.radar.waveform = waveform
        self.assertEqual(waveform, self.radar.waveform)
            
    def test_max_range(self) -> None:
        """Max range property getter should return the waveform's max range"""
        
        self.assertEqual(self.waveform.max_range, self.radar.max_range)
        
    def test_velocity_resolution_validation(self) -> None:
        """Velocity resolution property getter should raise errors on invalid internal states"""
 
        self.radar.carrier_frequency = 0.
        with self.assertRaises(RuntimeError):
            _ = self.radar.velocity_resolution
            
        self.radar.waveform = None
        with self.assertRaises(RuntimeError):
            _ = self.radar.velocity_resolution
            
    def test_velocity_resolution(self) -> None:
        """Velocity resolution property getter should return the correct value"""
        
        self.assertEqual(.5 * self.waveform.relative_doppler_resolution * speed_of_light / self.device.carrier_frequency, self.radar.velocity_resolution)
 
    def test_transmit_waveform_validation(self) -> None:
        """Transmitting should raise a RuntimeError if no waveform was configured"""
        
        self.radar.waveform = None
        
        with self.assertRaises(RuntimeError):
            _ = self.radar.transmit()
            
    def test_transmit_device_validation(self) -> None:
        """Transmitting should raise a RuntimeError if no device was configured"""
        
        self.radar.device = None
        
        with self.assertRaises(RuntimeError):
            _ = self.radar.transmit()

    def test_transmit_beamformer_input_stream_validation(self) -> None:
        """Transmitting should raise a RuntimeError if the configured beamformer is not supported"""

        beamformer = Mock()
        beamformer.num_transmit_input_streams = 2
        self.radar.transmit_beamformer = beamformer

        with self.assertRaises(RuntimeError):
            _ = self.radar.transmit()

    def test_transmit_beamformer_output_stream_validation(self) -> None:
        """Transmitting should raise a RuntimeError if the configured beamformer is not supported"""

        beamformer = Mock()
        beamformer.num_transmit_input_streams = 1
        beamformer.num_transmit_output_streams = 1
        self.radar.transmit_beamformer = beamformer

        with self.assertRaises(RuntimeError):
            _ = self.radar.transmit()

    def test_transmit_beamformer(self) -> None:
        """Transmitting should result in the beamforming routine being envoked if configured accordingly"""

        beamformer = Mock()
        beamformer.num_transmit_input_streams = 1
        beamformer.num_transmit_output_streams = 2
        
        ping = self.waveform.ping()
        ping.samples = np.repeat(ping.samples, 2, 0)
        beamformer.transmit.return_value = ping
        self.radar.transmit_beamformer = beamformer

        _ = self.radar.transmit()

        beamformer.transmit.assert_called()

    def test_transmit_no_beamformer(self) -> None:
        """Transmitting without a beamformer should infer the signal properly"""

        transmission = self.radar.transmit()
        self.assertEqual(self.device.antennas.num_antennas, transmission.signal.num_streams)

    def test_receive_waveform_validation(self) -> None:
        """Receiving should raise a RuntimeError if no waveform was configured"""
        
        self.radar.waveform = None
        
        with self.assertRaises(RuntimeError):
            _ = self.radar.receive(Mock())
            
    def test_receive_device_validation(self) -> None:
        """Receiving should raise a RuntimeError if no device was configured"""
        
        self.radar.device = None
        
        with self.assertRaises(RuntimeError):
            _ = self.radar.receive(Mock())

    def test_receive_no_beamformer_validation(self) -> None:
        """Receiving without a configured beamformer should raise a RuntimeError"""

        transmission = self.radar.transmit()
        self.radar.cache_reception(transmission.signal)
        self.radar.receive_beamformer = None

        with self.assertRaises(RuntimeError):
            _ = self.radar.receive()

    def test_receive_beamformer_output_streams_validation(self) -> None:
        """Receiving should raise a RuntimeError if the configured beamformer is not supported"""

        transmission = self.radar.transmit()
        self.radar.cache_reception(transmission.signal)

        beamformer = Mock()
        beamformer.num_receive_output_streams = 2
        self.radar.receive_beamformer = beamformer

        with self.assertRaises(RuntimeError):
            _ = self.radar.receive()

    def test_receive_beamformer_input_streams_validation(self) -> None:
        """Receiving should raise a RuntimeError if the configured beamformer is not supported"""

        transmission = self.radar.transmit()
        self.radar.cache_reception(transmission.signal)

        beamformer = Mock()
        beamformer.num_receive_output_streams = 1
        beamformer.num_receive_input_streams = 3
        self.radar.receive_beamformer = beamformer

        with self.assertRaises(RuntimeError):
            _ = self.radar.receive()

    def test_receive_no_beamformer(self) -> None:
        """Receiving without a beamformer should result in a valid radar cube"""

        self.device.antennas = UniformArray(IdealAntenna(), 1., (1,))
        self.radar.cache_reception(Signal(np.zeros((1, 5)), self.waveform.sampling_rate))
        self.radar.receive_beamformer = None

        reception = self.radar.receive()
        self.assertEqual(1, len(reception.cube.angle_bins))

    def test_receive_beamformer(self) -> None:
        """Receiving with a beamformer should result in a valid radar cube"""

        transmission = self.radar.transmit()
        self.radar.cache_reception(transmission.signal)

        reception = self.radar.receive()
        self.assertEqual(1, len(reception.cube.angle_bins))
        
    def test_recall_transmission(self) -> None:
        """Recalling a transmission should return the correct deserialization"""
        
        with patch('hermespy.radar.radar.RadarTransmission') as transmission_mock:
            
            recall_mock = Mock()
            transmission_mock.from_HDF.return_value = recall_mock
            
            group_mock = Mock()
            recall = self.radar.recall_transmission(group_mock)
            
            self.assertIs(recall, recall_mock)
            transmission_mock.from_HDF.assert_called_with(group_mock)
            
    def test_recall_reception(self) -> None:
        """Recalling a reception should return the correct deserialization"""
        
        with patch('hermespy.radar.radar.RadarReception') as reception_mock:
            
            recall_mock = Mock()
            reception_mock.from_HDF.return_value = recall_mock
            
            group_mock = Mock()
            recall = self.radar.recall_reception(group_mock)
            
            self.assertIs(recall, recall_mock)
            reception_mock.from_HDF.assert_called_with(group_mock)

    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.radar.Radar.property_blacklist', new_callable=PropertyMock) as blacklist, \
             patch('hermespy.radar.Radar.waveform', new_callable=PropertyMock) as waveform:
                 
            blacklist.return_value = {'slot', 'waveform', 'receive_beamformer', 'device'}
            waveform.return_value = self.waveform

            test_yaml_roundtrip_serialization(self, self.radar)
