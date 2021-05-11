import unittest
from typing import List
from copy import deepcopy

import numpy as np

from modem.coding.repetition_encoder import RepetitionEncoder
from parameters_parser.parameters_repetition_encoder import ParametersRepetitionEncoder


class TestRepetitionEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.params_encoder = ParametersRepetitionEncoder()
        self.params_encoder.encoded_bits_n = 1
        self.params_encoder.data_bits_k = 1
        self.bits_in_frame = 10

        self._rEncoder = RepetitionEncoder(
            self.params_encoder, self.bits_in_frame)

    def test_encoding_one_block_nk1(self) -> None:
        data_bits = [np.arange(self.bits_in_frame)]
        encoded_bits = self._rEncoder.encode(deepcopy(data_bits))

        _assert_frame_equality(data_bits, encoded_bits)

    def test_encoding_multiple_blocks_nk1(self) -> None:
        data_bits = [
            np.arange(
                self.bits_in_frame /
                2),
            np.arange(
                self.bits_in_frame /
                2)]
        encoded_bits = self._rEncoder.encode(deepcopy(data_bits))

        _assert_frame_equality(data_bits, encoded_bits)

    def test_encoding_one_block_n3k1(self) -> None:
        self.params_encoder.encoded_bits_n = 3
        self.params_encoder.data_bits_k = 1
        encoder = RepetitionEncoder(self.params_encoder, self.bits_in_frame)

        data_bits = [np.arange(encoder.source_bits)]
        encoded_bits = encoder.encode(deepcopy(data_bits))
        expected_encoded_bits = [
            np.concatenate(
                (
                    np.repeat(
                        data_bits[0],
                        self.params_encoder.encoded_bits_n),
                    np.array([0])
                )
            )
        ]

        _assert_frame_equality(encoded_bits, expected_encoded_bits)

    def test_encoding_two_blocks_n3k1(self) -> None:
        self.params_encoder.encoded_bits_n = 3
        self.params_encoder.data_bits_k = 1
        encoder = RepetitionEncoder(self.params_encoder, self.bits_in_frame)

        first_block = np.arange(int(encoder.source_bits / 2))
        second_block = np.arange(int(encoder.source_bits / 2))
        data_bits = [
            deepcopy(first_block),
            deepcopy(second_block)
        ]

        encoded_bits = encoder.encode(data_bits)
        expected_encoded_bits = [
            np.concatenate(
                (
                    np.repeat(first_block, self.params_encoder.encoded_bits_n),
                    np.zeros(2)
                )
            ),
            np.concatenate(
                (
                    np.repeat(
                        second_block,
                        self.params_encoder.encoded_bits_n),
                    np.zeros(2)
                )
            )
        ]
        _assert_frame_equality(encoded_bits, expected_encoded_bits)

    def test_encoding_one_block_n3k2(self) -> None:
        self.params_encoder.encoded_bits_n = 3
        self.params_encoder.data_bits_k = 2
        encoder = RepetitionEncoder(self.params_encoder, self.bits_in_frame)

        data_bits = [np.arange(encoder.source_bits)]
        encoded_bits = encoder.encode(deepcopy(data_bits))
        expected_encoded_bits = [
            np.concatenate(
                (
                    np.repeat(
                        data_bits[0],
                        self.params_encoder.encoded_bits_n),
                    np.zeros(1)
                )
            )
        ]

        _assert_frame_equality(encoded_bits, expected_encoded_bits)

    def test_data_bits_set_to_1(self) -> None:
        self.params_encoder.data_bits_k = 2
        encoder = RepetitionEncoder(self.params_encoder, self.bits_in_frame)

        self.assertEqual(encoder.data_bits_k, 1)

    def test_decoding_two_blocks_n3k2(self) -> None:
        self.params_encoder.encoded_bits_n = 3
        self.params_encoder.data_bits_k = 2
        encoder = RepetitionEncoder(self.params_encoder, self.bits_in_frame)

        data_bits = [np.array([1]), np.array([0])]
        encoded_bits = [np.array([1, 1, 0.9, 0.4, 0.5]), 
                        np.array([-0.3, -0.7, -1, -0.9, -0.4])]

        decoded_bits = encoder.decode(encoded_bits)
        _assert_frame_equality(data_bits, decoded_bits)

    def test_decoding_one_block_n2k1(self) -> None:
        self.params_encoder.encoded_bits_n = 2
        self.params_encoder.data_bits_k = 1
        encoder = RepetitionEncoder(
            self.params_encoder, self.bits_in_frame
        )
        data_bits_soft = [np.array([-1, -0.2, 1, 1, 1])]
        data_bits_hard = [np.array([0, 0, 1, 1, 1])]
        encoded_bits = [np.repeat(data_bits_soft, 2)]

        decoded_bits = encoder.decode(encoded_bits)
        _assert_frame_equality(data_bits_hard, decoded_bits)


def _assert_frame_equality(
        data_bits: List[np.array], encoded_bits: List[np.array]) -> None:
    for data_block, encoded_block in zip(data_bits, encoded_bits):
        np.testing.assert_array_equal(data_block, encoded_block)
