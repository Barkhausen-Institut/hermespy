# -*- coding: utf-8 -*-
"""Interleaver encoder testing."""

import unittest
import numpy as np

from hermespy.fec import BlockInterleaver
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization


__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestBlockInterleaver(unittest.TestCase):
    def setUp(self) -> None:
        self.block_size = 12
        self.interleave_blocks = 2
        self.interleaver = BlockInterleaver(self.block_size, self.interleave_blocks)

    def test_init(self) -> None:
        """Test that the init properly stores all parameters."""

        self.assertEqual(self.block_size, self.interleaver.block_size, "Block size init failed")
        self.assertEqual(self.interleave_blocks, self.interleaver.interleave_blocks, "Number of interleaved blocks init failed")

    def test_init_validation(self) -> None:
        """The interleaver init must raise a `ValueError` if blocks can't be sectioned properly"""

        with self.assertRaises(ValueError):
            _ = BlockInterleaver(1, 2)

    def test_bit_block_size(self) -> None:
        """Bit block size must be equal to the configured block size."""

        self.assertEqual(self.block_size, self.interleaver.bit_block_size)

    def test_code_block_size(self) -> None:
        """Code block size must be equal to the configured block size."""

        self.assertEqual(self.block_size, self.interleaver.code_block_size)

    def test_block_size(self) -> None:
        """Block size getter must return setter value."""

        self.interleaver.block_size = 1
        self.assertEqual(1, self.interleaver.block_size)

    def test_block_size_validation(self) -> None:
        """Block size setter must throw ValueException on values smaller than one."""

        with self.assertRaises(ValueError):
            self.interleaver.block_size = 0

        with self.assertRaises(ValueError):
            self.interleaver.block_size = -1

    def test_interleave_blocks(self) -> None:
        """Interleave blocks getter must return setter value."""

        self.interleaver.interleave_blocks = 1
        self.assertEqual(1, self.interleaver.interleave_blocks)

    def test_interleave_blocks_validation(self) -> None:
        """Interleave blocks setter must throw ValueException on values smaller than one."""

        with self.assertRaises(ValueError):
            self.interleaver.interleave_blocks = 0

        with self.assertRaises(ValueError):
            self.interleaver.interleave_blocks = -1

    def test_rate(self) -> None:
        """Rate must always return 1.0"""

        self.assertEqual(1.0, self.interleaver.rate, "Reported code rate is not 1.0")

    def test_interleaving(self) -> None:
        """Interleaving must produce the expected results."""

        bits = np.arange(self.block_size)
        expected_code = np.array([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])

        np.testing.assert_array_equal(expected_code, self.interleaver.encode(bits))

    def test_deinterleaving(self) -> None:
        """De-Interleaving must produce the expected results."""

        code = np.array([0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11])
        expected_bits = np.arange(self.block_size)

        np.testing.assert_array_equal(expected_bits, self.interleaver.decode(code))

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.interleaver)
