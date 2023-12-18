# -*- coding: utf-8 -*-
"""Test Precoding configuration."""

import unittest
from unittest.mock import Mock, patch, PropertyMock
from fractions import Fraction

import numpy as np

from hermespy.modem import StatedSymbols, SymbolPrecoding, SymbolPrecoder, DFT
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSymbolPrecoding(unittest.TestCase):
    def setUp(self) -> None:
        # Random rng
        self.generator = np.random.default_rng(42)

        # Mock modem
        self.modem = Mock()

        # Precoder to be tested
        self.precoding = SymbolPrecoding(modem=self.modem)

    def test_init(self) -> None:
        """Object init arguments should be properly stored as class attributes."""

        self.assertIs(self.modem, self.precoding.modem)

    def test_encode(self) -> None:
        """Encoding should be delegated to the registeded precoders"""

        encoder = Mock(spec=SymbolPrecoder)
        encoder.rate = Fraction(1, 1)
        self.precoding[0] = encoder
        symbols = Mock(spec=StatedSymbols)
        symbols.copy.return_value = symbols
        symbols.num_blocks = 1

        self.precoding.encode(symbols)

        encoder.encode.assert_called_once_with(symbols)

    def test_decode(self) -> None:
        """Decoding should be delegated to the registeded precoders"""

        decoder = Mock(spec=SymbolPrecoder)
        self.precoding[0] = decoder
        symbols = Mock()

        self.precoding.decode(symbols)

        decoder.decode.assert_called_once_with(symbols.copy())

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

    def test_num_encoded_blocks(self) -> None:
        """Number of encoded blocks should be the multiplication of all precoder-rates."""

        precoder_alpha = Mock()
        precoder_alpha.rate = Fraction(1, 2)
        self.precoding[0] = precoder_alpha

        self.assertEqual(10, self.precoding.num_encoded_blocks(5))

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        self.precoding[0] = DFT()

        with patch("hermespy.modem.precoding.SymbolPrecoding.modem", new_callable=PropertyMock) as modem:
            modem.return_value = self.modem
            test_yaml_roundtrip_serialization(self, self.precoding)
