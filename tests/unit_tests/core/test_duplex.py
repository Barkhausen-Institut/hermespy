# -*- coding: utf-8 -*-

from unittest import TestCase

from h5py import Group

from hermespy.core import ChannelStateInformation, DuplexOperator, Reception, Signal, Transmission
from hermespy.simulation import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DuplexOperatorMock(DuplexOperator[Transmission, Reception]):
    """Duplex operator mock implementation for testing purposes only"""

    __sampling_rate: float = 1e9
    __num_frame_samples = 10
    __frame_duration: float = 10e-9
    
    @property
    def power(self) -> float:
        return 1.0

    def _transmit(self, duration: float = 0) -> Transmission:
        samples = self._rng.normal(size=(self.device.antennas.num_transmit_antennas, self.__num_frame_samples))
        return Transmission(Signal.Create(samples, self.__sampling_rate, self.device.carrier_frequency))

    def _receive(self, signal: Signal, csi: ChannelStateInformation) -> Reception:
        return Reception(signal)

    def sampling_rate(self) -> float:
        return self.__sampling_rate

    def frame_duration(self) -> float:
        return self.__frame_duration

    def _recall_reception(self, group: Group) -> Reception:
        return Reception.from_HDF(group)

    def _recall_transmission(self, group: Group) -> Transmission:
        return Transmission.from_HDF(group)


class TestDuplexOperator(TestCase):
    def setUp(self) -> None:
        self.device = SimulatedDevice()
        self.operator = DuplexOperatorMock(device=self.device, seed=42)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class atttributes"""

        self.assertIs(self.device, self.operator.device)

    def test_device_setget(self) -> None:
        """Device property getter should return setter argument"""

        expected_device = SimulatedDevice()
        self.operator.device = expected_device

        # Verify that the duplex operator has been assigned the device
        self.assertIs(expected_device, self.operator.device)

        # Verify that the duplex operator is registered as both a transmitter and receiver
        self.assertIn(self.operator, expected_device.transmitters)
        self.assertIn(self.operator, expected_device.receivers)

        # Verify that the duplex operator has been dropped as an operator by the old device
        self.assertNotIn(self.operator, self.device.transmitters)
        self.assertNotIn(self.operator, self.device.receivers)

    def test_slot_set(self) -> None:
        """Setting the slot should register the operator as both a transmitter and receiver"""

        expected_device = SimulatedDevice()
        self.operator.slot = expected_device.transmitters

        # Verify that the duplex operator is registered as both a transmitter and receiver
        self.assertIn(self.operator, expected_device.transmitters)
        self.assertIn(self.operator, expected_device.receivers)

        # Verify that the duplex operator has been dropped as an operator by the old device
        self.assertNotIn(self.operator, self.device.transmitters)
        self.assertNotIn(self.operator, self.device.receivers)
