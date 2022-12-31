# -*- coding: utf-8 -*-

from unittest import TestCase
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal

from hermespy.core import Signal
from hermespy.hardware_loop.physical_device import SignalTransmitter, SignalReceiver
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
        reception = self.dummy.receive()
        
        assert_array_almost_equal(expected_signal.samples, transmission.mixed_signal.samples)
        assert_array_almost_equal(expected_signal.samples, reception.operator_receptions[0].signal.samples)
