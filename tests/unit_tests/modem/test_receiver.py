# -*- coding: utf-8 -*-
"""Test HermesPy receiving modem class."""

import unittest
from unittest.mock import Mock

from modem import Receiver

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestReceiver(unittest.TestCase):

    def setUp(self) -> None:

        self.reference_transmitter = Mock()
        self.reference_transmitter.power = 0.6
        self.additional_transmitter = Mock()
        self.additional_transmitter.power = 0.5

        self.scenario = Mock()
        self.scenario.transmitters = [self.additional_transmitter, self.reference_transmitter]
        self.scenario.num_receivers = 1
        self.scenario.num_transmitters = len(self.scenario.transmitters)

        self.receiver = Receiver()
        self.receiver.scenario = self.scenario
        self.receiver.reference_transmitter = self.reference_transmitter
        self.scenario.receivers = [self.receiver]

    def test_init(self) -> None:
        """Init parameters should be properly stored."""

        noise = Mock()
        self.receiver = Receiver(noise=noise)

        self.assertIs(noise, self.receiver.noise)

    def test_index(self) -> None:
        """Index property should return the index of the receive modem within the scenario's modem list."""

        self.scenario.receivers = [self.receiver]
        self.assertEqual(0, self.receiver.index)

    def test_noise_setget(self) -> None:
        """Noise property getter should return setter parameter registered to the receiver."""

        noise = Mock()
        self.receiver.noise = noise

        self.assertIs(noise, self.receiver.noise)
        self.assertIs(noise.receiver, self.receiver)

    def test_reference_transmitter_setget(self) -> None:
        """Reference transmitter property getter should return setter argument."""

        self.assertIs(self.reference_transmitter, self.receiver.reference_transmitter)

    def test_reference_transmitter_get_auto(self) -> None:
        """Reference transmitter property should return transmitter diagonal to the receiver if not specified."""

        self.receiver.reference_transmitter = None
        self.assertIs(self.additional_transmitter, self.receiver.reference_transmitter)

    def test_reference_transmitter_setget_none(self) -> None:
        """Reference transmitter property setter should allow None argument."""

        self.receiver.reference_transmitter = None
        self.assertEqual(self.additional_transmitter, self.receiver.reference_transmitter)

    def test_reference_transmitter_validation(self) -> None:
        """Reference transmitter property setter should raise RuntimeError if floating, ValueError if the modem is
        not registered with the scenario."""

        with self.assertRaises(ValueError):
            self.receiver.reference_transmitter = Mock()

        floating_receiver = Receiver()
        floating_receiver.transmitters = []
        with self.assertRaises(RuntimeError):
            floating_receiver.reference_transmitter = Mock()

    def test_received_power(self) -> None:
        """Received power property should return the reference transmitter's power,
        alternatively 1.0 if no reference has been configured."""

        self.assertEqual(self.reference_transmitter.power, self.receiver.received_power)

        self.receiver.reference_transmitter = None
        self.assertEqual(self.additional_transmitter.power, self.receiver.received_power)

        self.scenario.transmitters = []
        self.scenario.num_transmitters = 0
        self.assertEqual(1.0, self.receiver.received_power)
