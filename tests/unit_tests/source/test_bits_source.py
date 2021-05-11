from source.bits_source import BitsSource

import unittest
import numpy as np

from copy import deepcopy


class TestBitsSource(unittest.TestCase):

    def setUp(self) -> None:
        self.rnd = np.random.RandomState()
        self.source = BitsSource(self.rnd)

    def test_get_bits(self) -> None:
        """
        Test if BitsSource.get_bits method generates the right number of blocks and bits (all 0's and 1's) in a list of
        numpy.nd_array.
        Also test if BitsSource.get_number_of_bits returns the right number of bits and blocks
        """
        total_number_of_bits = 0
        total_number_of_blocks = 0
        bits_in_drop = list()

        number_of_frames = self.rnd.randint(2, 20)

        self.source.init_drop()

        for idx in range(number_of_frames):
            number_of_blocks = self.rnd.randint(1, 10)
            block_size = self.rnd.randint(1, 5000)
            bits = self.source.get_bits(block_size, number_of_blocks)
            bits_in_drop.extend(bits)

            self.assertIsInstance(bits, list)

            self.assertEqual(len(bits), number_of_blocks)
            for block in bits:
                self.assertIsInstance(block, np.ndarray)

                self.assertEqual(block.ndim, 1)
                self.assertEqual(block.size, block_size)

            total_number_of_bits += block_size * number_of_blocks
            total_number_of_blocks += number_of_blocks

        self.assertIsInstance(self.source.bits_in_drop, list)
        for block in self.source.bits_in_drop:
            self.assertIsInstance(block, np.ndarray)

        self.assertEqual(len(self.source.bits_in_drop), total_number_of_blocks)

        self.assertEqual(self.source.bits_in_drop, bits_in_drop)

        for block in self.source.bits_in_drop:
            self.assertTrue(np.all(np.logical_or(block == 0, block == 1)))

        # test method get_number_of_bits
        data_size = self.source.get_number_of_generated_bits()
        self.assertEqual(data_size['number_of_blocks'], total_number_of_blocks)
        self.assertEqual(data_size['number_of_bits'], total_number_of_bits)

    def test_get_number_of_errors(self) -> None:
        """
        Test if BitsSource.get_number_of_errors method calculates the right number of errors
        """

        block_error_probability = self.rnd.rand()

        self.source.init_drop()

        number_of_frames = self.rnd.randint(2, 20)
        bits_in_drop = list()

        for idx in range(number_of_frames):
            number_of_blocks = self.rnd.randint(1, 10)
            block_size = self.rnd.randint(1, 5000)
            bits = self.source.get_bits(block_size, number_of_blocks)
            frame = np.ndarray((number_of_blocks, block_size))

            for block_iter, block in enumerate(bits):
                frame[block_iter, :] = deepcopy(block)
            bits_in_drop.append(frame)

        number_of_errors = self.source.get_number_of_errors(bits_in_drop)

        self.assertEqual(number_of_errors.number_of_bit_errors, 0)
        self.assertEqual(number_of_errors.number_of_block_errors, 0)

        number_of_block_errors = 0
        number_of_bit_errors = 0

        for frame_idx, frame in enumerate(bits_in_drop):
            for block in frame:
                if self.rnd.rand() < block_error_probability:
                    number_of_block_errors += 1
                    # all blocks in the frame have the same size
                    errors_in_block = self.rnd.randint(1, block.size)
                    number_of_bit_errors += errors_in_block
                    error_idx = self.rnd.choice(
                        block.size, errors_in_block, replace=False)
                    block[error_idx] = np.logical_not(block[error_idx])

        number_of_errors = self.source.get_number_of_errors(bits_in_drop)
        self.assertEqual(
            number_of_errors.number_of_block_errors,
            number_of_block_errors)
        self.assertEqual(
            number_of_errors.number_of_bit_errors,
            number_of_bit_errors)


if __name__ == '__main__':
    unittest.main()
