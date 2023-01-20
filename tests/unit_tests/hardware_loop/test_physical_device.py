# -*- coding: utf-8 -*-
"""Test Physical Device functionalities."""

from os import path
from pickle import dump
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch
from tempfile import TemporaryDirectory

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal

from hermespy.core import Signal
from hermespy.hardware_loop.physical_device import PhysicalDevice, StaticOperator, SilentTransmitter, SignalTransmitter, PowerReceiver, SignalReceiver

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalDeviceMock(PhysicalDevice):
    """Mock for a physical device."""

    __sampling_rate: float

    def __init__(self,
                 sampling_rate: float) -> None:

        PhysicalDevice.__init__(self)
        self.__sampling_rate = sampling_rate

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

    @property
    def carrier_frequency(self) -> float:
        return 0.
    
    def configure(self) -> None:
        pass

    def trigger(self) -> None:
        pass
    
    def fetch(self) -> None:
        pass
    
    @property
    def max_sampling_rate(self) -> float:
        return self.sampling_rate
    
    
class TestStaticOperator(TestCase):
    """Test the static device operator base class"""
    
    def setUp(self) -> None:
        
        self.device = PhysicalDeviceMock(1.)
        
        self.num_samples = 100
        self.sampling_rate = self.device.sampling_rate
        self.operator = StaticOperator(self.num_samples, self.sampling_rate)
        self.device.transmitters.add(self.operator)
        
    def test_sampling_rate(self) -> None:
        """Sampling rate should be the configured sampling rate"""
        
        self.assertEqual(self.sampling_rate, self.operator.sampling_rate)
        
    def test_frame_duration(self) -> None:
        """Silent transmitter should report the correct frame duration"""
        
        expected_duration = self.num_samples / self.sampling_rate
        self.assertEqual(expected_duration, self.operator.frame_duration)


class TestSilentTransmitter(TestCase):
    """Test static silent signal transmission."""
    
    def setUp(self) -> None:
        
        self.device = PhysicalDeviceMock(1.)
        
        self.num_samples = 100
        self.sampling_rate = self.device.sampling_rate
        self.transmitter = SilentTransmitter(self.num_samples, self.sampling_rate)
        self.device.transmitters.add(self.transmitter)

    def test_transmit(self) -> None:
        """Silent transmission should generate a silent signal"""
        
        default_transmission = self.transmitter.transmit()
        custom_transmission = self.transmitter.transmit(10)
        
        self.assertEqual(self.num_samples, default_transmission.signal.num_samples)
        self.assertCountEqual([0] * 10, custom_transmission.signal.samples[0, :].tolist())


class TestSignalTransmitter(TestCase):
    """Test static arbitrary signal transmission."""
    
    def setUp(self) -> None:
        
        self.device = PhysicalDeviceMock(1.)
        
        self.num_samples = 100
        self.sampling_rate = self.device.sampling_rate
        self.transmitter = SignalTransmitter(self.num_samples, self.sampling_rate)
        self.device.transmitters.add(self.transmitter)
    
    def test_transmit(self) -> None:
        """Transmit routine should transmit the submitted signal samples"""
        
        signal = Signal(np.ones((1, 10)), sampling_rate=self.device.sampling_rate, carrier_frequency=self.device.carrier_frequency)
        transmission = self.transmitter.transmit(signal)
        
        assert_array_equal(signal.samples, transmission.signal.samples)
        

class TestPowerReceiver(TestCase):
    
    def setUp(self) -> None:
        
        self.rng = default_rng(42)
        self.device = PhysicalDeviceMock(1.)
        
        self.num_samples = 100
        self.receiver = PowerReceiver(self.num_samples)
        self.device.receivers.add(self.receiver)
        
    def test_init_assert(self) -> None:
        """Invalid initialization arguments should raise ValueErrors"""
        
        with self.assertRaises(ValueError):
            _ = PowerReceiver(0)
            
    def test_num_samples(self) -> None:
        """Number of samples property should report the correct count"""
        
        self.assertEqual(self.num_samples, self.receiver.num_samples)
        
    def test_sampling_rate(self) -> None:
        """Sampling rate should be identical to the device's sampling rate"""
        
        self.assertEqual(self.device.sampling_rate, self.receiver.sampling_rate)
        
    def test_energy(self) -> None:
        """Reported energy should be zero"""
        
        self.assertEqual(0, self.receiver.energy)
        
    def test_frame_duration(self) -> None:
        """Power receiver should report a zero frame duration"""
        
        self.assertEqual(100 * self.device.sampling_rate, self.receiver.frame_duration)

    def test_receive(self) -> None:
        """Reception should correctly estimate the received power"""
        
        power_signal = Signal(np.ones((1, 10)), sampling_rate=self.device.sampling_rate, carrier_frequency=self.device.carrier_frequency)
        self.device.receive(power_signal)
        
        power_estimate = self.receiver.receive().signal.power
        
        assert_array_equal(power_signal.power, power_estimate)


class TestSignalReceiver(TestCase):
    
    def setUp(self) -> None:
        
        self.device = PhysicalDeviceMock(1.)
        
        self.num_samples = 100
        self.sampling_rate = self.device.sampling_rate
        self.receiver = SignalReceiver(self.num_samples, self.sampling_rate)
        self.device.receivers.add(self.receiver)

    def test_energy(self) -> None:
        """Reported energy should be zero"""
        
        self.assertEqual(0, self.receiver.energy)
        
    def test_receive(self) -> None:
        """Receiver should receive a signal"""
        
        power_signal = Signal(np.ones((1, 10)), sampling_rate=self.device.sampling_rate, carrier_frequency=self.device.carrier_frequency)
        self.device.receive(power_signal)
        
        received_signal = self.receiver.receive().signal
        
        assert_array_equal(received_signal.samples, power_signal.samples)


class TestPhysicalDevice(TestCase):
    """Test the base class for all physical devices."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.sampling_rate = 1e6

        self.device = PhysicalDeviceMock(sampling_rate=self.sampling_rate)

    def test_calibration_delay_setget(self) -> None:
        """Calibration delay property getter should return setter argument"""
        
        delay = 1.23456
        self.device.calibration_delay = delay
        
        self.assertEqual(delay, self.device.calibration_delay)
        
    def test_adaptive_sampling_setget(self) -> None:
        """Adaptive sampling property getter should return setter argument"""
        
        self.device.adaptive_sampling = True
        self.assertTrue(self.device.adaptive_sampling)
        
    def test_lowpass_filter_setget(self) -> None:
        """Lowpass filter property getter should return setter argument"""
        
        self.device.lowpass_filter = True
        self.assertTrue(self.device.lowpass_filter)
        
    def test_lowpass_bandwidth_setget(self) -> None:
        """Lopwass bandwidth property getter should return setter argument"""
        
        bandwidth = 3.45567
        self.device.lowpass_bandwidth = bandwidth
        
        self.assertEqual(bandwidth, self.device.lowpass_bandwidth)
        
    def test_lowpass_bandwidth_validation(self) -> None:
        """Lowpass bandwidth property setter should raise ValueError on invalid arguments"""
        
        try:
            self.device.lowpass_bandwidth = 0.
            
        except ValueError:
            self.fail()
            
        with self.assertRaises(ValueError):
            self.device.lowpass_bandwidth = -1.
            
    def test_max_receive_delay_setget(self) -> None:
        """Max receive delay property getter should return setter argument"""
        
        max_receive_delay = 3.45567
        self.device.max_receive_delay = max_receive_delay
        
        self.assertEqual(max_receive_delay, self.device.max_receive_delay)
        
    def test_max_receive_delay_validation(self) -> None:
        """Max receive delay property setter should raise ValueError on invalid arguments"""
        
        try:
            self.device.max_receive_delay = 0.
            
        except ValueError:
            self.fail()
            
        with self.assertRaises(ValueError):
            self.device.max_receive_delay = -1.
            
    def test_velocity(self) -> None:
        """Accessing the velocity property should raise a NotImplementedError"""
        
        with self.assertRaises(NotImplementedError):
            _ = self.device.velocity

    @patch.object(PhysicalDeviceMock, '_download')
    def test_estimate_noise_power(self, patch_download) -> None:
        """Noise power estimation should return the correct power estimate."""

        num_samples = 10000
        expected_noise_power = .1
        samples = 2 ** -.5 * (self.rng.normal(size=num_samples, scale=expected_noise_power ** .5) + 1j *
                              self.rng.normal(size=num_samples, scale=expected_noise_power ** .5))
        signal = Signal(samples, sampling_rate=self.sampling_rate)
        patch_download.side_effect = lambda: signal

        noise_power = self.device.estimate_noise_power(num_samples)
        self.assertAlmostEqual(expected_noise_power, noise_power[0], places=2)

    @patch('hermespy.hardware_loop.physical_device.PhysicalDevice._upload')
    def test_transmit_no_adpative_sampling(self, _upload: MagicMock) -> None:
        """Test physical device extended transmit routine without adptive sampling"""
        
        transmitter = SignalTransmitter(10, self.device.sampling_rate)
        self.device.transmitters.add(transmitter)
        
        transmitted_signal = Signal(np.zeros((self.device.num_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)
        _ = transmitter.transmit(transmitted_signal)
        
        self.device.adaptive_sampling = False
        transmission = self.device.transmit()
        
        _upload.assert_called_once()
        assert_array_equal(transmitted_signal.samples, transmission.samples)
        
    @patch('hermespy.hardware_loop.physical_device.PhysicalDevice._upload')
    def test_transmit_adpative_sampling(self, _upload: MagicMock) -> None:
        """Test physical device extended transmit routine with adptive sampling"""
        
        transmitter_alpha = SignalTransmitter(10, self.device.sampling_rate)
        transmitter_beta = SignalTransmitter(10, self.device.sampling_rate)
        self.device.transmitters.add(transmitter_alpha)
        self.device.transmitters.add(transmitter_beta)

        _ = transmitter_alpha.transmit(Signal(np.zeros((self.device.num_antennas, 10), dtype=complex), self.device.sampling_rate, self.device.carrier_frequency))
        _ = transmitter_beta.transmit(Signal(np.zeros((self.device.num_antennas, 10), dtype=complex), self.device.sampling_rate, self.device.carrier_frequency))

        transmission = self.device.transmit()
        
        _upload.assert_called_once()
        assert_array_equal(np.zeros((self.device.num_antennas, 10), dtype=complex), transmission.samples)
        
    def test_transmit_validation(self) -> None:
        """Phyiscal device extended transmit routine should raise RuntimeErrors on invalid configurations"""
        
        self.device.adaptive_sampling = True
            
        transmitter_alpha = SignalTransmitter(10, self.device.sampling_rate)
        transmitter_beta = SignalTransmitter(10, self.device.sampling_rate)
        self.device.transmitters.add(transmitter_alpha)
        self.device.transmitters.add(transmitter_beta)

        _ = transmitter_alpha.transmit(Signal(np.zeros((self.device.num_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency))
        _ = transmitter_beta.transmit(Signal(np.zeros((self.device.num_antennas, 10)), 1 + self.device.sampling_rate, self.device.carrier_frequency))

        with self.assertRaises(RuntimeError):
            _ = self.device.transmit()

    @patch('hermespy.hardware_loop.physical_device.PhysicalDevice._download')
    def test_receive(self, _download: MagicMock) -> None:
        """Test physical device extended receive routine"""
        
        receiver = Mock()
        receiver.sampling_rate = self.device.sampling_rate
        
        _download.return_value = Signal(np.zeros((self.device.num_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)
        self.device.lowpass_filter = True
        self.device.receivers.add(receiver)
        
        _ = self.device.receive()
        receiver.cache_reception.assert_called_once()
        
        receiver.reset_mock()
        self.device.lowpass_bandwidth = 1.
        
        _ = self.device.receive()
        receiver.cache_reception.assert_called_once() 
        
    @patch('hermespy.hardware_loop.physical_device.PhysicalDevice._download')
    def test_receive_validation(self, _download: MagicMock) -> None:
        """Receive routine should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            
            _download.return_value = Signal(np.zeros((3, 10)), self.device.sampling_rate, self.device.carrier_frequency)
            _ = self.device.receive()
            
        with self.assertRaises(ValueError):
            
            _download.return_value = Signal(np.zeros((self.device.num_antennas, 10)), self.device.sampling_rate + 1, self.device.carrier_frequency)
            _ = self.device.receive()
            
    def test_download(self) -> None:
        """The download subroutine should raise a NotImplementedError"""
        
        with self.assertRaises(NotImplementedError):
            _ = self.device._download()
       
    @patch('hermespy.hardware_loop.physical_device.PhysicalDevice._download')
    @patch('hermespy.hardware_loop.physical_device.PhysicalDevice._upload')
    def test_calibrate(self, _upload: MagicMock, _download: MagicMock) -> None:
        """Test the physical device calibration routine"""

        for expected_delay_samples in [0, 10, 12]:
            
            expected_delay = expected_delay_samples / self.device.max_sampling_rate
        
            # Configure the download routine to mirror the uploaded samples back
            # Results in a zero second calibration time of flight delay
            def return_side_effect() -> Signal:
                
                # Prepend delay samples
                delayed_signal: Signal = _upload.call_args[0][0].copy()
                delayed_signal.samples = np.append(np.zeros((delayed_signal.num_streams, expected_delay_samples), dtype=complex), 
                                                   delayed_signal.samples)
                
                return delayed_signal
            
            _download.side_effect = return_side_effect
            
            with TemporaryDirectory() as tempdir:
                
                file = path.join(tempdir, 'testfile')
                _ = self.device.calibrate(12 / self.sampling_rate, file, 2)
            
                _download.assert_called()
                self.assertAlmostEqual(-expected_delay, self.device.calibration_delay)
                
                self.device.load_calibration(file)
                self.assertAlmostEqual(-expected_delay, self.device.calibration_delay)
        
    def test_calibrate_validation(self) -> None:
        """The physical device calibration routine should raise ValueErrors on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.device.calibrate(0., 'xxx')
            
        with self.assertRaises(ValueError):
            self.device.calibrate(10, 'xxx', wait=-1)
            
        with self.assertRaises(ValueError):
            self.device.calibrate(10, 'xxx', num_iterations=-1)
        
    def test_load_calibration(self) -> None:
        """Test the loading of a calibration file"""
        
        expected_delay = 1.23456
        
        with TemporaryDirectory() as tempdir:
            
            file = path.join(tempdir, 'testfile')
            
            with open(file, 'wb') as file_stream:
                dump(expected_delay, file_stream)
                
            self.device.load_calibration(file)
            
        self.assertEqual(expected_delay, self.device.calibration_delay)
