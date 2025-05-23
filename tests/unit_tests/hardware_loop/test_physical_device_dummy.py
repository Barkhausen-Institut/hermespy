# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import patch

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal

from hermespy.core import Signal, SignalReceiver, SignalTransmitter
from hermespy.hardware_loop.physical_device_dummy import PhysicalDeviceDummy, PhysicalScenarioDummy
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestPhysicalDeviceDummy(TestCase):
    """Test the physical device dummy"""

    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.sampling_rate = 1.0
        self.dummy = PhysicalDeviceDummy(sampling_rate=self.sampling_rate)

    def test_transmit_receive(self) -> None:
        """Test the proper transmit receive routine execution"""

        expected_signal = Signal.Create(self.rng.normal(size=(1, 100)), self.sampling_rate)

        transmitter = SignalTransmitter(expected_signal)
        receiver = SignalReceiver(expected_signal.num_samples, self.sampling_rate)
        self.dummy.transmitters.add(transmitter)
        self.dummy.receivers.add(receiver)

        transmission = self.dummy.transmit()
        self.dummy.trigger()
        reception = self.dummy.receive()

        assert_array_almost_equal(expected_signal.getitem(), transmission.mixed_signal.getitem())
        assert_array_almost_equal(expected_signal.getitem(), reception.operator_receptions[0].signal.getitem())

    def test_receive_transmission_flag(self) -> None:
        """Device dummy should receive nothing if respective flag is enabled"""

        self.dummy.receive_transmission = False
        expected_signal = Signal.Create(self.rng.normal(size=(1, 100)), self.sampling_rate)

        transmitter = SignalTransmitter(expected_signal)
        receiver = SignalReceiver(expected_signal.num_samples, self.sampling_rate)
        self.dummy.transmitters.add(transmitter)
        self.dummy.receivers.add(receiver)

        _ = self.dummy.transmit()
        self.dummy.trigger()
        reception = self.dummy.receive()

        assert_array_almost_equal(np.zeros(expected_signal.shape), reception.operator_receptions[0].signal.getitem())

        direct_reception = self.dummy.trigger_direct(expected_signal)
        assert_array_almost_equal(np.zeros(expected_signal.shape), direct_reception.getitem())

    def test_trigger_direction(self) -> None:
        """Test trigger direct routine"""

        expected_signal = Signal.Create(self.rng.normal(size=(1, 100)), self.sampling_rate)
        direction_reception = self.dummy.trigger_direct(expected_signal)

        assert_array_almost_equal(expected_signal.getitem(), direction_reception.getitem())

    def test_serialize(self) -> None:
        """Test physical device dummy serialization"""

        test_roundtrip_serialization(self, self.dummy)


class TestPhysicalScenarioDummy(TestCase):
    """Test Physical scenario dummy"""

    def setUp(self) -> None:
        self.scenario = PhysicalScenarioDummy()

    def test_new_device(self) -> None:
        """Test new device creation"""

        device = self.scenario.new_device()
        self.assertIn(device, self.scenario.devices)

    def test_add_device(self) -> None:
        """Test adding a device"""

        device = PhysicalDeviceDummy()
        self.scenario.add_device(device)
        self.assertIn(device, self.scenario.devices)

    def test_receive_devices(self) -> None:
        device_reception = self.scenario.receive_devices()
        self.assertSequenceEqual([], device_reception)

    def test_trigger(self) -> None:
        with patch("hermespy.simulation.SimulationScenario.drop") as drop_mock:
            self.scenario._trigger()
            drop_mock.assert_called()

    def test_serialize(self) -> None:
        """Test physical device dummy serialization"""
        
        test_roundtrip_serialization(self, self.scenario)
