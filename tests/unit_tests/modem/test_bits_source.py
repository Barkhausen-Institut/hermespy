"""Test source of bit streams to be transmitted."""

from hermespy.modem.bits_source import RandomBitsSource

from unittest.mock import Mock
import unittest
import numpy as np

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRandomBitsSource(unittest.TestCase):

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