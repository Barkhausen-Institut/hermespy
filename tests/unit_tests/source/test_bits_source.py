# -*- coding: utf-8 -*-
"""Test source of bit streams to be transmitted."""

from hermespy.source import BitsSource

from unittest.mock import Mock
import unittest
import numpy as np

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronauer@barkhauseninstitut.org"
__status__ = "Prototype"


class TestBitsSource(unittest.TestCase):

    def setUp(self) -> None:

        self.transmitter = Mock()
        self.transmitter.random_generator = np.random.default_rng(0)
        self.random_generator = np.random.default_rng(0)

        self.source = BitsSource(transmitter=self.transmitter,
                                 random_generator=self.random_generator)

    def test_get_bits(self) -> None:
        """
        Test if BitsSource.get_bits method generates the right number of blocks and bits (all 0's and 1's) in a list of
        numpy.nd_array.
        Also test if BitsSource.get_number_of_bits returns the right number of bits and blocks
        """

        number_of_frames = 20
        for frame_idx in range(number_of_frames):

            number_of_bits = 123 * number_of_frames
            bits = self.source.get_bits(number_of_bits)

            # Assert that the requested number of bits is returned
            self.assertEqual(bits.ndim, 1)
            self.assertEqual(bits.shape[0], number_of_bits)

            # Assert that all bits are actually either zeros or ones
            self.assertEqual(True, np.any((bits == 1) | (bits == 0)))

    def test_random_generator_setget(self) -> None:
        """Random rng property getter should return setter argument."""

        generator = np.random.default_rng()
        self.source.random_generator = generator

        self.assertIs(generator, self.source.random_generator)

    def test_random_generator_set_seed(self) -> None:
        """Random rng property should allow for array-like seeds."""

        self.source.random_generator = "123asdbasbd"

    def test_random_generator_get_default(self) -> None:
        """Random rng property getter should return transmitter rng if not specified."""

        self.transmitter.random_generator = Mock()
        self.source.random_generator = None

        self.assertIs(self.transmitter.random_generator, self.source.random_generator)

    def test_transmitter_setget(self) -> None:
        """Scenario property getter should return setter argument."""

        transmitter = Mock()
        self.source = BitsSource()
        self.source.transmitter = transmitter

        self.assertIs(transmitter, self.source.transmitter)

    def test_transmitter_get_validation(self) -> None:
        """Scenario property getter should raise RuntimeError if transmitter is not set."""

        self.source = BitsSource()
        with self.assertRaises(RuntimeError):
            _ = self.source.transmitter

    def test_transmitter_set_validation(self) -> None:
        """Scenario property setter should raise RuntimeError if transmitter is already set."""

        with self.assertRaises(RuntimeError):
            self.source.transmitter = Mock()


if __name__ == '__main__':
    unittest.main()
