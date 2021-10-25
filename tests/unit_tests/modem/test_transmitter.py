# -*- coding: utf-8 -*-
"""Test HermesPy transmit modem class."""

import unittest
import numpy as np
import numpy.random as rnd
from typing import List
from unittest.mock import Mock
from itertools import product
from numpy.testing import assert_array_equal

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

        self.scenario = Mock()
        self.scenario.random_generator = rnd.default_rng(0)
        self.waveform_generator = Mock()
        self.waveform_generator.bits_per_frame = 101

        self.transmitter = Transmitter()

        self.transmitter.scenario = self.scenario
        self.transmitter.waveform_generator = self.waveform_generator

    def test_generate_data_bits_length(self) -> None:
        """The number of data bits generated must be the number of required bits for one frame."""

        self.assertEqual(self.transmitter.num_data_bits_per_frame, len(self.transmitter.generate_data_bits()))