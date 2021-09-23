import unittest

import numpy as np

from modem.coding.interleaver import BlockInterleaver
from parameters_parser.parameters_block_interleaver import ParametersBlockInterleaver
from .utils import assert_frame_equality


class TestBlockInterleaver(unittest.TestCase):
    def setUp(self) -> None:
        self.M = 4
        self.N = 3
        self.bits_in_frame = 30
        self.params = ParametersBlockInterleaver(self.M, self.N)
        self.interleaver = BlockInterleaver(self.params, self.bits_in_frame)

    def test_no_code_blocks(self) -> None:
        self.assertTrue(self.interleaver.code_blocks, 2)

    def test_source_bits(self) -> None:
        self.assertTrue(self.interleaver.source_bits, 24)

    def test_interleaving(self) -> None:
        data_bits = [np.arange(24)]
        expected_encoded_bits = [
            np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]),
            np.array([12, 15, 18, 21, 13, 16, 19, 22, 14, 17, 20, 23]),
            np.zeros(6)
        ]
        assert_frame_equality(expected_encoded_bits, self.interleaver.encode(data_bits))

    def test_deinterleaving(self) -> None:
        interleaved_bits = [
            np.array([0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11]),
            np.array([12, 15, 18, 21, 13, 16, 19, 22, 14, 17, 20, 23]),
            np.zeros(6)
        ]
        expected_data_bits = [np.arange(12), np.arange(12, 24)]
        assert_frame_equality(expected_data_bits, self.interleaver.decode(interleaved_bits))


