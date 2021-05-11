import unittest
from unittest.mock import Mock
from typing import List

import numpy as np

from modem.coding.encoder import Encoder


class StubEncoder(Encoder):
    def encode(self, bits: List[np.array]) -> List[np.array]:
        return bits

    def decode(self, bits: List[np.array]) -> List[np.array]:
        return bits


class TestEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.params_encoder = Mock()
        self.params_encoder.encoded_bits_n = 3
        self.params_encoder.data_bits_k = 2
        self.bits_in_frame = 100

        self.encoder = StubEncoder(self.params_encoder, self.bits_in_frame)

    def test_no_code_blocks_calculation(self) -> None:
        no_code_blocks = np.floor(
            self.bits_in_frame / self.params_encoder.encoded_bits_n
        )

        self.assertEqual(no_code_blocks, self.encoder.code_blocks)

    def test_no_bits_for_source_calculation(self) -> None:
        no_bits = (np.floor(
            self.bits_in_frame / self.params_encoder.encoded_bits_n
        ) * self.params_encoder.data_bits_k)

        self.assertEqual(
            no_bits,
            self.encoder.source_bits
        )
