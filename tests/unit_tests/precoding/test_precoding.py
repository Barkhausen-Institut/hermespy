# -*- coding: utf-8 -*-
"""Test Precoding configuration."""

import unittest
from unittest.mock import Mock, patch, PropertyMock
from fractions import Fraction

import numpy as np

from hermespy.precoding import SymbolPrecoding
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
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
        
    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.precoding.SymbolPrecoding.modem', new_callable=PropertyMock) as modem:
        
            modem.return_value = self.modem
            test_yaml_roundtrip_serialization(self, self.precoding)
