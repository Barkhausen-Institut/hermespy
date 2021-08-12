import unittest
from typing import List

import numpy as np

from parameters_parser.parameters_encoder import ParametersEncoder
from modem.coding.crc_encoder import CrcEncoder
from .utils import assert_frame_equality


class TestCrcEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.global_seed = 42
        self.rng = np.random.RandomState(self.global_seed)

        self.params = ParametersEncoder()
        self.params.data_bits_k = 3
        self.crc_bits = 2
        self.params.encoded_bits_n = self.params.data_bits_k + self.crc_bits
        self.bits_in_frame = 11

        self.encoder = CrcEncoder(self.params, self.bits_in_frame, self.rng)

    def test_encoding(self) -> None:
        rng = np.random.RandomState(self.global_seed)
        data_bits = [np.arange(self.encoder.source_bits)]
        encoded_bits = self.encoder.encode(data_bits)
        expected_encoded_bits = [
            np.concatenate(
                (
                    np.arange(i*self.params.data_bits_k, (i+1)*self.params.data_bits_k),
                    rng.randint(2, size=self.crc_bits)
                )

            ) for i in range(self.encoder.code_blocks)
        ]
        expected_encoded_bits.append(np.array(0))


        assert_frame_equality(expected_encoded_bits, encoded_bits)

    def test_decoding(self) -> None:
        encoded_bits: List[np.array] = [
            np.concatenate
            (
                (np.ones(self.params.data_bits_k), np.zeros(self.crc_bits))
            ),
            np.concatenate
            (
                (np.ones(self.params.data_bits_k)*2, np.zeros(self.crc_bits)),
            ),    
            np.array(-1)
        ]
        expected_decoded_bits: List[np.array] = [
            np.ones(self.params.data_bits_k), 
            np.ones(self.params.data_bits_k) * 2,
        ]
        decoded_bits = self.encoder.decode(encoded_bits)

        assert_frame_equality(decoded_bits, expected_decoded_bits)