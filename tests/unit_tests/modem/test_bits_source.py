# -*- coding: utf-8 -*-
"""Test source of bit streams to be transmitted."""

from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from hermespy.modem.bits_source import RandomBitsSource, StreamBitsSource

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRandomBitsSource(TestCase):
    def setUp(self) -> None:
        self.random_generator = np.random.default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.random_generator

        self.source = RandomBitsSource()
        self.source.random_mother = self.random_node

    def test_get_bits(self) -> None:
        """
        Test if BitsSource.get_bits method generates the right number of blocks and bits (all 0's and 1's) in a list of
        numpy.nd_array.
        Also test if BitsSource.get_number_of_bits returns the right number of bits and blocks
        """

        number_of_frames = 20
        for frame_idx in range(number_of_frames):
            number_of_bits = 123 * number_of_frames
            bits = self.source.generate_bits(number_of_bits)

            # Assert that the requested number of bits is returned
            self.assertEqual(bits.ndim, 1)
            self.assertEqual(bits.shape[0], number_of_bits)

            # Assert that all bits are actually either zeros or ones
            self.assertEqual(True, np.any((bits == 1) | (bits == 0)))


class TestStreamBitsSource(TestCase):
    """Test bits source that reads bits from a file stream"""

    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        self.file_path = path.join(self.temp_dir.name, "file")
        self.text = b"Hello World"

        with open(self.file_path, "wb") as file_stream:
            file_stream.write(self.text)

        self.source = StreamBitsSource(self.file_path)

    def tearDown(self) -> None:
        del self.source
        self.temp_dir.cleanup()
        
    def test_get_bits_validation(self) -> None:
        """Get bits routine should raise RuntimeError if the requested number of bits is not a multiple of 8"""

        with self.assertRaises(RuntimeError):
            self.source.generate_bits(1)

    def test_get_bits_validation(self) -> None:
        """Get bits routine should raise RuntimeError if the requested number of bits is not a multiple of 8"""

        with self.assertRaises(RuntimeError):
            self.source.generate_bits(1)

    def test_get_bits(self) -> None:
        """Test fetching the next block of bits from the file stream"""

        bits = self.source.generate_bits(len(self.text) * 8)
        text = np.packbits(bits).tobytes()

        self.assertEqual(self.text, text)
