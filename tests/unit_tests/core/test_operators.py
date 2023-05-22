# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.core import Signal, StaticOperator, SilentTransmitter, SignalTransmitter, SignalReceiver
from hermespy.simulation import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestStaticOperator(TestCase):
    """Test the static device operator base class"""
    
    def setUp(self) -> None:
        
        self.device = SimulatedDevice(sampling_rate=1e3)
        
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
        
        self.device = SimulatedDevice(sampling_rate=1e3)
        
        self.num_samples = 100
        self.sampling_rate = self.device.sampling_rate
        self.transmitter = SilentTransmitter(self.num_samples, self.sampling_rate)
        self.device.transmitters.add(self.transmitter)

    def test_transmit(self) -> None:
        """Silent transmission should generate a silent signal"""
        
        default_transmission = self.transmitter.transmit()
        custom_transmission = self.transmitter.transmit(10 / self.device.sampling_rate)
        
        self.assertEqual(self.num_samples, default_transmission.signal.num_samples)
        self.assertCountEqual([0] * 10, custom_transmission.signal.samples[0, :].tolist())


class TestSignalTransmitter(TestCase):
    """Test static arbitrary signal transmission."""
    
    def setUp(self) -> None:
        
        self.device = SimulatedDevice(sampling_rate=1e3)
        
        self.num_samples = 100
        self.sampling_rate = self.device.sampling_rate
        self.signal = Signal(np.ones((1, 10)), sampling_rate=self.device.sampling_rate, carrier_frequency=self.device.carrier_frequency)
        self.transmitter = SignalTransmitter(self.signal)
        self.device.transmitters.add(self.transmitter)
    
    def test_transmit(self) -> None:
        """Transmit routine should transmit the submitted signal samples"""
        
        transmission = self.transmitter.transmit()
        
        assert_array_equal(self.signal.samples, transmission.signal.samples)


class TestSignalReceiver(TestCase):
    
    def setUp(self) -> None:
        
        self.device = SimulatedDevice(sampling_rate=1e3)
        
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
        self.device.process_input(power_signal)
        
        received_signal = self.receiver.receive().signal
        
        assert_array_equal(received_signal.samples, power_signal.samples)
