# -*- coding: utf-8 -*-
"""HermesPy testing for modem base class."""

import unittest
from unittest.mock import Mock

import numpy as np
from numpy import random as rnd
from numpy.testing import assert_array_equal, assert_almost_equal
from scipy.constants import speed_of_light

from hermespy.modem.modem import Modem

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestModem(unittest.TestCase):
    """Modem Base Class Test Case"""

    def setUp(self) -> None:

        self.random_generator = rnd.default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.random_generator

        self.device = Mock()
        self.device.position = np.zeros(3)
        self.device.orientation = np.zeros(3)
        self.device.carrier_frequency = 1e9
        self.device.num_antennas = 3

        self.encoding = Mock()
        self.precoding = Mock()
        self.waveform = Mock()

        self.modem = Modem(encoding=self.encoding, precoding=self.precoding, waveform=self.waveform)
        self.modem.device = self.device
        self.modem.random_mother = self.random_node

    def test_initialization(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertIs(self.encoding, self.modem.encoder_manager)
        self.assertIs(self.precoding, self.modem.precoding)
        self.assertIs(self.waveform, self.modem.waveform_generator)

    def test_num_streams(self) -> None:
        """Number of streams property should return proper number of streams."""

        self.device.num_antennas = 3
        self.assertEqual(3, self.modem.num_streams)

        self.device.num_antennas = 2
        self.assertEqual(2, self.modem.num_streams)

    def test_bits_source_setget(self) -> None:
        """Bits source property getter should return setter argument."""

        bits_source = Mock()
        self.modem.bits_source = bits_source

        self.assertIs(bits_source, self.modem.bits_source)

    def test_encoder_manager_setget(self) -> None:
        """Encoder manager property getter should return setter argument."""

        encoder_manager = Mock()
        self.modem.encoder_manager = encoder_manager

        self.assertIs(encoder_manager, self.modem.encoder_manager)
        self.assertIs(encoder_manager.modem, self.modem)

    def test_waveform_generator_setget(self) -> None:
        """Waveform generator property getter should return setter argument."""

        waveform_generator = Mock()
        self.modem.waveform_generator = waveform_generator

        self.assertIs(waveform_generator, self.modem.waveform_generator)
        self.assertIs(waveform_generator.modem, self.modem)
        
    def test_precoding_setget(self) -> None:
        """Precoding configuration property getter should return setter argument."""

        precoding = Mock()
        self.modem.precoding = precoding

        self.assertIs(precoding, self.modem.precoding)
        self.assertIs(precoding.modem, self.modem)

    def test_transmit(self) -> None:
        """Test modem data transmission."""
        pass

    def test_receive(self) -> None:
        """Test modem data reception."""