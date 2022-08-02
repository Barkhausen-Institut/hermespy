# -*- coding: utf-8 -*-
"""Test Cyclic Redundancy Check bit encoding."""

import unittest

import numpy as np

from hermespy.fec import CyclicRedundancyCheck

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestCrcEncoder(unittest.TestCase):
    """Test Redundancy Check Bit Encoding"""

    def setUp(self) -> None:

        self.generator = np.random.default_rng(42)
        self.bit_block_size = 10
        self.check_block_size = 3

        self.encoder = CyclicRedundancyCheck(bit_block_size=self.bit_block_size,
                                             check_block_size=self.check_block_size)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as class attributes."""

        self.assertEqual(self.bit_block_size, self.encoder.bit_block_size)
        self.assertEqual(self.check_block_size, self.encoder.check_block_size)

    def test_encoding(self) -> None:
        pass

    def test_decoding(self) -> None:
        pass

    def test_bit_block_size_setget(self) -> None:
        """Bit block size property getter should return setter argument."""

        block_size = 50
        self.encoder.bit_block_size = block_size

        self.assertEqual(block_size, self.encoder.bit_block_size)

    def test_bit_block_size_validation(self) -> None:
        """Bit block size property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.encoder.bit_block_size = -1

        with self.assertRaises(ValueError):
            self.encoder.bit_block_size = 0

    def test_check_block_size_setget(self) -> None:
        """Check block size property getter should return setter argument."""

        block_size = 50
        self.encoder.check_block_size = block_size

        self.assertEqual(block_size, self.encoder.check_block_size)

    def test_check_block_size_validation(self) -> None:
        """Check block size property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.encoder.check_block_size = -1

        try:
            self.encoder.check_block_size = 0

        except ValueError:
            self.fail()
