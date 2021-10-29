# -*- coding: utf-8 -*-
"""Repetition encoder testing."""

import unittest
from copy import deepcopy
import numpy as np
from numpy.testing import assert_array_equal

from coding.repetition_encoder import RepetitionEncoder

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRepetitionEncoder(unittest.TestCase):
    """Test the `Repetition` encoding instance."""

    def setUp(self) -> None:

        # Default parameters
        self.block_size = 8
        self.repetitions = 3

        self.encoder = RepetitionEncoder(self.block_size,
                                         self.repetitions)

    def test_init(self) -> None:
        """Test valid parameter initialization during object construction."""

        self.assertEqual(self.block_size, self.encoder.bit_block_size,
                         "Unexpected bit block size initialized")
        self.assertEqual(self.repetitions, self.encoder.repetitions,
                         "Unexpected repetition count initialized")

    def test_encoding_repetitions(self) -> None:
        """Test the encoding behaviour of a single data block being repeated multiple times."""

        data = np.arange(self.encoder.bit_block_size)
        expected_code = np.repeat(data, self.repetitions)

        code = self.encoder.encode(deepcopy(data))
        assert_array_equal(expected_code, code)

    def test_encoding_default(self) -> None:
        """Test the encoding behaviour of a single data block without repetitions."""

        self.repetitions = 1
        self.encoder.repetitions = self.repetitions

        data = np.arange(self.encoder.bit_block_size)
        code = self.encoder.encode(deepcopy(data))

        assert_array_equal(data, code)

    def test_encode_block_length(self) -> None:
        """Length of the code block after encoding must match the code block size property."""

        code = self.encoder.encode(np.random.randint(0, 2, self.encoder.bit_block_size))
        self.assertEqual(len(code), self.encoder.code_block_size)

    def test_decoding_repetitions(self) -> None:
        """Test the decoding behaviour of a single code block
           consisting of data being repeated multiple times."""

        expected_data = np.array([1, 0, 1, 0, 1, 0, 0, 1])

        code = np.repeat(expected_data, self.repetitions)
        code[0::self.repetitions] = np.array([0, 1, 0, 1, 0, 1, 1, 0])

        data = self.encoder.decode(code)
        assert_array_equal(expected_data, data)

    def test_decoding_default(self) -> None:
        """Test the decoding behaviour of a single code block without repetitions."""

        self.encoder.repetitions = 1
        code = np.array([1, 0, 1, 0, 1, 0, 0, 1])
        expected_data = code.copy()

        data = self.encoder.decode(code)
        assert_array_equal(expected_data, data)

    def test_decode_block_length(self) -> None:
        """Length of the data block after decoding must match the bit block size property."""

        data = self.encoder.decode(np.random.randint(0, 2, self.encoder.code_block_size))
        self.assertEqual(len(data), self.encoder.bit_block_size)

    def test_bit_block_size_setget(self) -> None:
        """Test that the bit block size getter returns the setter value."""

        bit_block_size = 99
        self.encoder.bit_block_size = bit_block_size

        self.assertEqual(bit_block_size, self.encoder.bit_block_size)

    def test_bit_block_size_validation(self) -> None:
        """Bit block size setter must raise exception on invalid arguments."""

        with self.assertRaises(ValueError):
            self.encoder.bit_block_size = 0

        with self.assertRaises(ValueError):
            self.encoder.bit_block_size = -1

    def test_code_block_size_calculation(self) -> None:
        """Code block size must be the number of repetitions times the bit block size."""

        bit_block_size = 10
        repetitions = 5
        expected_code_block_size = bit_block_size * repetitions

        self.encoder.bit_block_size = bit_block_size
        self.encoder.repetitions = repetitions

        self.assertEqual(expected_code_block_size, self.encoder.code_block_size)

    def test_repetitions_setget(self) -> None:
        """Repetitions property getter must return setter value."""

        repetitions = 5
        self.encoder.repetitions = repetitions

        self.assertEqual(repetitions, self.encoder.repetitions)

    def test_repetitions_validation(self) -> None:
        """Repetitions property setter must raise exception on invalid arguments"""

        with self.assertRaises(ValueError):
            self.encoder.repetitions = 0

        with self.assertRaises(ValueError):
            self.encoder.repetitions = -1

    def test_to_yaml(self) -> None:
        """Test YAML serialization dump validity."""
        pass

    def test_from_yaml(self) -> None:
        """Test YAML serialization recall validity."""
        pass
