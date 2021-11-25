# -*- coding: utf-8 -*-
"""Test Precoding configuration."""

import unittest
from unittest.mock import Mock
from fractions import Fraction

import numpy as np

from hermespy.precoding import SymbolPrecoding

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMMSEqualizer(unittest.TestCase):

    def setUp(self) -> None:

        # Random generator
        self.generator = np.random.default_rng(42)

        # Mock modem
        self.modem = Mock()

        # Precoder to be tested
        self.precoding = SymbolPrecoding(modem=self.modem)

    def test_init(self) -> None:
        """Object init arguments should be properly stored as class attributes."""

        self.assertIs(self.modem, self.precoding.modem)

    def test_rate(self) -> None:
        """Rate should be the multiplication of all precoder-rates."""

        precoder_alpha = Mock()
        precoder_beta = Mock()
        precoder_alpha.rate = Fraction(1, 2)
        precoder_beta.rate = Fraction(1, 6)

        self.precoding[0] = precoder_alpha
        self.precoding[1] = precoder_beta

        expected_rate = precoder_alpha.rate * precoder_beta.rate
        self.assertEqual(expected_rate, self.precoding.rate)

    def test_decode_validation(self) -> None:
        """Decoding should result in a RuntimeError, if multiple streams result."""

        precoder = Mock()
        precoder.decode = lambda symbols, streams, noise: (symbols.repeat(2, axis=0),
                                                           streams.repeat(2, axis=0),
                                                           noise.repeat(2, axis=0))

        self.precoding[0] = precoder
        symbols = self.generator.random((1, 10))
        channel = self.generator.random((1, 10, 1))
        noise = self.generator.random((1, 10))

        with self.assertRaises(RuntimeError):
            _ = self.precoding.decode(symbols, channel, noise)
