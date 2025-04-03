# -*- coding: utf-8 -*-
"""Test Cyclic Redundancy Check bit encoding."""

import unittest

import numpy as np

from hermespy.fec import CyclicRedundancyCheck
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestCrcEncoder(unittest.TestCase):
    """Test Redundancy Check Bit Encoding"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.bit_block_size = 10
        self.check_block_size = 3

        self.encoder = CyclicRedundancyCheck(bit_block_size=self.bit_block_size, check_block_size=self.check_block_size)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as class attributes."""

        self.assertEqual(self.bit_block_size, self.encoder.bit_block_size)
        self.assertEqual(self.check_block_size, self.encoder.check_block_size)

    def test_encode(self) -> None:
        """Cyclic redundancy check should properly pad the input block with bits"""

        data = self.rng.integers(0, 2, self.bit_block_size)
        code = self.encoder.encode(data)

        self.assertEqual(self.bit_block_size + self.check_block_size, len(code))

    def test_decoding(self) -> None:
        """Cyclic redundancy check should properly remove checksum"""

        code = self.rng.integers(0, 2, self.encoder.code_block_size)
        data = self.encoder.decode(code)

        self.assertEqual(self.encoder.bit_block_size, len(data))

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

    def test_serialization(self) -> None:
        """Test CRC serialization"""

        test_roundtrip_serialization(self, self.encoder)
