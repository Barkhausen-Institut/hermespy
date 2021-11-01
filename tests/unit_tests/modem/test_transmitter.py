# -*- coding: utf-8 -*-
"""Test HermesPy transmit modem class."""

import unittest
import numpy.random as rnd
from unittest.mock import Mock

from modem import Transmitter

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestTransmitter(unittest.TestCase):
    """Test the transmit modem implementation."""

    def setUp(self) -> None:

        self.power = 0.9
        self.scenario = Mock()
        self.scenario.random_generator = rnd.default_rng(0)
        self.waveform_generator = Mock()
        self.waveform_generator.bits_per_frame = 101

        self.transmitter = Transmitter(power=self.power)

        self.transmitter.scenario = self.scenario
        self.transmitter.waveform_generator = self.waveform_generator

    def test_init(self) -> None:
        """Object initialization arguments should be properly stored."""

        self.assertEqual(self.power, self.transmitter.power)

    def test_generate_data_bits_length(self) -> None:
        """The number of data bits generated must be the number of required bits for one frame."""

        self.assertEqual(self.transmitter.num_data_bits_per_frame, len(self.transmitter.generate_data_bits()))

    def test_power_setget(self) -> None:
        """Transmit power property getter should return setter argument."""

        power = 0.5
        self.transmitter.power = power

        self.assertEqual(power, self.transmitter.power)

    def test_power_validation(self) -> None:
        """Transmit power property setter should raise ValueError on negative arguments."""

        with self.assertRaises(ValueError):
            self.transmitter.power = -0.2
