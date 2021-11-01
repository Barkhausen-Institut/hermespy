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

        self.scenario = Mock()
        self.scenario.transmitters = []

        self.receiver = Receiver()
        self.receiver.scenario = self.scenario

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

        transmitter = Mock()
        self.scenario.transmitters = [transmitter]
        self.receiver.reference_transmitter = transmitter

        self.assertIs(transmitter, self.receiver.reference_transmitter)

    def test_reference_transmitter_setget_none(self) -> None:
        """Reference transmitter property setter should allow None argument."""

        self.receiver.reference_transmitter = None
        self.assertEqual(None, self.receiver.reference_transmitter)

    def test_reference_transmitter_validation(self) -> None:
        """Reference transmitter property setter should raise RuntimeError if floating, ArgumentError if the modem is
        not registered with the scenario."""

        with self.assertRaises(ValueError):
            self.receiver.reference_transmitter = Mock()

        floating_receiver = Receiver()
        floating_receiver.transmitters = []
        with self.assertRaises(RuntimeError):
            floating_receiver.reference_transmitter = Mock()
