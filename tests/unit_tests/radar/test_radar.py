# -*- coding: utf-8 -*-

from os.path import join
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from h5py import File
from numpy.testing import assert_array_equal
from scipy.constants import speed_of_light

from hermespy.beamforming import ConventionalBeamformer
from hermespy.core import Signal
from hermespy.radar import Radar, RadarCube, RadarWaveform, RadarReception, RadarPointCloud, RadarTransmission
from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarWaveformMock(RadarWaveform):
    """Mock implementation of a a radar waveform"""

    def __init__(self) -> None:
        self.num_samples = 10
        self.rng = np.random.default_rng(42)

    def ping(self) -> Signal:
        return Signal.Create(np.exp(2j * np.pi * self.rng.uniform(0, 1, size=(1, self.num_samples))), self.sampling_rate)

    def estimate(self, signal: Signal) -> np.ndarray:
        num_velocity_bins = len(self.relative_doppler_bins)
        num_range_bins = len(self.range_bins)

        velocity_range_estimate = np.zeros((num_velocity_bins, num_range_bins), dtype=float)
        velocity_range_estimate[int(0.5 * num_velocity_bins), int(0.5 * num_range_bins)] = 1.0

        return velocity_range_estimate

    @property
    def sampling_rate(self) -> float:
        return 1.2345

    @property
    def range_bins(self) -> np.ndarray:
        return np.arange(10)

    @property
    def max_relative_doppler(self) -> float:
        return 1.0

    @property
    def relative_doppler_resolution(self) -> float:
        return 0.5

    @property
    def relative_doppler_bins(self) -> np.ndarray:
        return np.arange(5)

    @property
    def energy(self) -> float:
        return 1.0

    @property
    def power(self) -> float:
        return 1.0

    @property
    def frame_duration(self) -> float:
        return 12.345


class TestRadarTransmission(TestCase):
    """Test the radar transmission model"""
    
    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.signal = Signal.Create(self.rng.normal(size=(2, 1)), 1.0, 0.0)
        self.transmission = RadarTransmission(self.signal)

    def test_serialization(self) -> None:
        """Test radar transmission serialization"""
        
        test_roundtrip_serialization(self, self.transmission)


class TestRadarReception(TestCase):
    """Test the radar reception model"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.signal = Signal.Create(self.rng.normal(size=(2, 1)), 1.0, 0.0)
        self.cube = RadarCube(self.rng.normal(size=(5, 4, 4)), self.rng.normal(size=(5, 2)), self.rng.normal(size=4), self.rng.normal(size=4))
        self.cloud = RadarPointCloud(max_range=1.0)

        self.reception = RadarReception(self.signal, self.cube, self.cloud)

    def test_serialization(self) -> None:
        """Test radar reception serialization"""

        test_roundtrip_serialization(self, self.reception)


class TestRadar(TestCase):
    """Test the radar operator"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.waveform = RadarWaveformMock()
        self.device = SimulatedDevice(carrier_frequency=1e8, antennas=SimulatedUniformArray(SimulatedIdealAntenna, 0.5 * speed_of_light / 1e8, (2, 1, 1)))
        self.device.transmit_coding[0] = ConventionalBeamformer()

        self.radar = Radar()
        self.radar.waveform = self.waveform
        self.radar.device = self.device
        self.radar.receive_beamformer = ConventionalBeamformer()

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
        self.assertEqual(0.0, self.radar.frame_duration)

    def test_frame_duration(self) -> None:
        """Frame duration property should return the frame duration"""

        self.assertEqual(12.345, self.radar.frame_duration)
        
    def test_default_power(self) -> None:
        """Power property should return zero if no waveform is configured"""

        self.radar.waveform = None
        self.assertEqual(0.0, self.radar.power)
        
    def test_power(self) -> None:
        """Power property should return the waveform power"""

        self.assertEqual(1.0, self.radar.power)

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

        with self.assertRaises(RuntimeError):
            _ = self.radar.velocity_resolution(0.0)

        self.radar.waveform = None
        with self.assertRaises(RuntimeError):
            _ = self.radar.velocity_resolution(1e5)

    def test_velocity_resolution(self) -> None:
        """Velocity resolution method should compute the correct value"""

        self.assertEqual(0.5 * self.waveform.relative_doppler_resolution * speed_of_light / self.device.carrier_frequency, self.radar.velocity_resolution(self.device.carrier_frequency))

    def test_transmit_waveform_validation(self) -> None:
        """Transmitting should raise a RuntimeError if no waveform was configured"""

        self.radar.waveform = None
        with self.assertRaises(RuntimeError):
            _ = self.radar.transmit(self.device.state())

    def test_transmit_beamformer_input_stream_validation(self) -> None:
        """Transmitting should raise a RuntimeError if the configured beamformer is not supported"""

        with patch("hermespy.simulation.simulated_device.SimulatedDevice.num_digital_transmit_ports", new_callable=PropertyMock) as num_digital_transmit_ports:
            num_digital_transmit_ports.return_value = 2
            with self.assertRaises(RuntimeError):
                _ = self.radar.transmit(self.device.state())

    def test_receive_waveform_validation(self) -> None:
        """Receiving should raise a RuntimeError if no waveform was configured"""

        self.radar.waveform = None

        with self.assertRaises(RuntimeError):
            _ = self.radar.receive(Signal.Create(np.zeros((self.device.num_receive_antennas, 5), dtype=complex), 1.0), self.device.state())

    def test_receive_no_beamformer_validation(self) -> None:
        """Receiving without a configured beamformer should raise a RuntimeError"""

        transmission = self.radar.transmit(self.device.state())
        received_signal = transmission.signal.from_ndarray(np.tile(transmission.signal.getitem(), (self.device.num_digital_receive_ports, 1)))
        self.radar.receive_beamformer = None

        with self.assertRaises(RuntimeError):
            _ = self.radar.receive(received_signal, self.device.state())

    def test_receive_beamformer_suport(self) -> None:
        """Receiving should raise a RuntimeError if the configured beamformer is not supported"""

        transmission = self.radar.transmit(self.device.state())
        received_signal = transmission.signal.from_ndarray(np.tile(transmission.signal.getitem(), (self.device.num_digital_receive_ports, 1)))

        beamformer = Mock()
        beamformer.num_receive_output_streams.return_value = -1
        self.radar.receive_beamformer = beamformer

        with self.assertRaises(RuntimeError):
            _ = self.radar.receive(received_signal, self.device.state())

    def test_receive_beamformer_output_streams_validation(self) -> None:
        """Receiving should raise a RuntimeError if the configured beamformer is not supported"""

        transmission = self.radar.transmit(self.device.state())
        received_signal = transmission.signal.from_ndarray(np.tile(transmission.signal.getitem(), (self.device.num_digital_receive_ports, 1)))

        beamformer = Mock()
        beamformer.num_receive_output_streams.return_value = 2
        self.radar.receive_beamformer = beamformer

        with self.assertRaises(RuntimeError):
            _ = self.radar.receive(received_signal, self.device.state())

    def test_receive_no_beamformer(self) -> None:
        """Receiving without a beamformer should result in a valid radar cube"""

        self.device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1.0, (1,))
        self.radar.receive_beamformer = None

        reception = self.radar.receive(Signal.Create(np.zeros((1, 5)), self.waveform.sampling_rate), self.device.state())
        self.assertEqual(1, len(reception.cube.angle_bins))

    def test_receive_beamformer(self) -> None:
        """Receiving with a beamformer should result in a valid radar cube"""

        state = self.device.state()
        transmission = self.radar.transmit(state)

        received_signal =  transmission.signal.from_ndarray(np.tile(transmission.signal.getitem(), (self.device.num_digital_receive_ports, 1)))
        reception = self.radar.receive(received_signal, state)

        self.assertEqual(1, len(reception.cube.angle_bins))

    def test_serialization(self) -> None:
        """Test radar serialization"""

        self.radar.waveform = None
        test_roundtrip_serialization(self, self.radar)
