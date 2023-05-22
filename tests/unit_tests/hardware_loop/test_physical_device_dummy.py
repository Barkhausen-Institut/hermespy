# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal

from hermespy.core import Signal, SignalReceiver, SignalTransmitter
from hermespy.hardware_loop.physical_device_dummy import PhysicalDeviceDummy

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestPhysicalDeviceDummy(TestCase):
    """Test the physical device dummy"""
    
    def setUp(self) -> None:
        
        self.rng = default_rng(42)
        
        self.sampling_rate = 1.
        self.dummy = PhysicalDeviceDummy(sampling_rate=self.sampling_rate)
        
    def test_transmit_receive(self) -> None:
        """Test the proper transmit receive routine execution """
        
        expected_signal = Signal(self.rng.normal(size=(1, 100)), self.sampling_rate)        
        
        transmitter = SignalTransmitter(expected_signal)
        receiver = SignalReceiver(expected_signal.num_samples, self.sampling_rate)
        self.dummy.transmitters.add(transmitter)
        self.dummy.receivers.add(receiver)

        transmission = self.dummy.transmit()
        self.dummy.trigger()
        reception = self.dummy.receive()
        
        assert_array_almost_equal(expected_signal.samples, transmission.mixed_signal.samples)
        assert_array_almost_equal(expected_signal.samples, reception.operator_receptions[0].signal.samples)

    def test_receive_transmission_flag(self) -> None:
        """Device dummy should receive nothing if respective flag is enabled"""
        
        self.dummy.receive_transmission = False
        expected_signal = Signal(self.rng.normal(size=(1, 100)), self.sampling_rate)        
        
        transmitter = SignalTransmitter(expected_signal)
        receiver = SignalReceiver(expected_signal.num_samples, self.sampling_rate)
        self.dummy.transmitters.add(transmitter)
        self.dummy.receivers.add(receiver)

        _ = self.dummy.transmit()
        self.dummy.trigger()
        reception = self.dummy.receive()
        
        assert_array_almost_equal(np.zeros(expected_signal.samples.shape), reception.operator_receptions[0].signal.samples)

        direction_reception = self.dummy.trigger_direct(expected_signal)
        assert_array_almost_equal(np.zeros(expected_signal.samples.shape), direction_reception.samples)

    def test_trigger_direction(self) -> None:
        """Test trigger direct routine"""
        
        expected_signal = Signal(self.rng.normal(size=(1, 100)), self.sampling_rate)        
        direction_reception = self.dummy.trigger_direct(expected_signal)

        assert_array_almost_equal(expected_signal.samples, direction_reception.samples)
