# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from numpy.testing import assert_array_equal
from matplotlib.figure import Figure
from scipy.constants import speed_of_light

from hermespy.core import Signal, SNRType, IdealAntenna, UniformArray
from hermespy.radar import Radar, RadarCube, RadarWaveform, PointDetection
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization


__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestPointDetection(TestCase):
    """Test the base class for radar point detections"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        
        self.position = self.rng.normal(size=3)
        self.velocity = self.rng.normal(size=3)
        self.power = 1.2345
        
        self.point = PointDetection(self.position, self.velocity, self.power)
        
    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        assert_array_equal(self.position, self.point.position)
        assert_array_equal(self.velocity, self.point.velocity)
        self.assertEqual(self.power, self.point.power)


class TestRadarCube(TestCase):
    """Test the radar cube resulting from radar demodulations"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        
        self.angle_bins = self.rng.normal(size=(5, 2))
        self.velocity_bins = self.rng.normal(size=4)
        self.range_bins = self.rng.rayleigh(size=4)
        self.data = self.rng.rayleigh(size=(5, 4, 4))
        
        self.cube = RadarCube(self.data, self.angle_bins, self.velocity_bins, self.range_bins)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        assert_array_equal(self.angle_bins, self.cube.angle_bins)
        assert_array_equal(self.velocity_bins, self.cube.velocity_bins)
        assert_array_equal(self.range_bins, self.cube.range_bins)
        assert_array_equal(self.data, self.cube.data)

    def test_init_data_validation(self) -> None:
        """Initialization routine should raise ValueErrors on invalid argument combinations"""
        
        with self.assertRaises(ValueError):
            RadarCube(self.data, self.angle_bins[:1], self.velocity_bins, self.range_bins)
            
        with self.assertRaises(ValueError):
            RadarCube(self.data, self.angle_bins, self.velocity_bins[:1], self.range_bins)
            
        with self.assertRaises(ValueError):
            RadarCube(self.data, self.angle_bins, self.velocity_bins, self.range_bins[:1])
            
    def test_plot_range(self) -> None:
        """Plotting the range profile should result in a valid matplotlib figure"""
        
        figure = self.cube.plot_range()
        self.assertIsInstance(figure, Figure)
        
    def test_plot_range_velocity(self) -> None:
        """Plotting the range-velocity profile should result in a valid matplotlib figure"""
        
        figure = self.cube.plot_range_velocity()
        self.assertIsInstance(figure, Figure)


class RadarWaveformMock(RadarWaveform):
    """Mock implementation of a a radar waveform"""

    def __init__(self) -> None:

        self.num_samples = 10
        self.rng = np.random.default_rng(42)

    def ping(self) -> Signal:

        return Signal(np.exp(2j * np.pi * self.rng.uniform(0, 1, size=(1, self.num_samples))), self.sampling_rate)
            
    def estimate(self, signal: Signal) -> np.ndarray:

        num_velocity_bins = len(self.velocity_bins)
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
    def velocity_bins(self) -> np.ndarray:
        return np.arange(5)
    
    @property
    def energy(self) -> float:
        return 1.
    
    @property
    def power(self) -> float:
        return 1.


class TestRadar(TestCase):
    """Test the radar operator."""
    
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
        
    def test_frame_duration(self) -> None:
        """Frame duration property should return the frame duration"""
        
        self.assertEqual(1., self.radar.frame_duration)
        
    def test_noise_power(self) -> None:
        """Noise power estimator should compute the correct powers"""
        
        self.assertEqual(1., self.radar.noise_power(1., SNRType.EN0))
        self.assertEqual(1., self.radar.noise_power(1., SNRType.PN0))
        
        with self.assertRaises(ValueError):
            _ = self.radar.noise_power(1., SNRType.EBN0)

        self.radar.waveform = None
        self.assertEqual(0., self.radar.noise_power(1., SNRType.PN0))

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
            _ = self.radar.receive()
            
    def test_receive_device_validation(self) -> None:
        """Receiving should raise a RuntimeError if no device was configured"""
        
        self.radar.device = None
        
        with self.assertRaises(RuntimeError):
            _ = self.radar.receive()

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

    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.radar.Radar.property_blacklist', new_callable=PropertyMock) as blacklist, \
             patch('hermespy.radar.Radar.waveform', new_callable=PropertyMock) as waveform:
                 
            blacklist.return_value = {'slot', 'waveform', 'receive_beamformer'}
            waveform.return_value = self.waveform

            test_yaml_roundtrip_serialization(self, self.radar)
