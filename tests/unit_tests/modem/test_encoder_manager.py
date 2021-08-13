import unittest
from typing import List
import numpy as np

from parameters_parser.parameters_encoder import ParametersEncoder
from modem.coding.encoder import Encoder
from modem.coding.encoder_manager import EncoderManager


class StubEncoder(Encoder):

    def __init__(self, params: ParametersEncoder, bits_in_frame: int) -> None:
        super().__init__(params, bits_in_frame)

    def encode(self, data_bits: List[np.array]) -> List[np.array]:
        return data_bits

    def decode(self, encoded_bits: List[np.array]) -> List[np.array]:
        return encoded_bits


class TestEncoderManager(unittest.TestCase):
    def test_correct_order_of_encoders(self) -> None:
        params_encoder1 = ParametersEncoder()
        params_encoder1.data_bits_k = 3
        params_encoder1.encoded_bits_n = 5

        params_encoder2 = ParametersEncoder()
        params_encoder2.data_bits_k = 5
        params_encoder2.encoded_bits_n = 7

        encoder1 = StubEncoder(params_encoder1, 100)
        encoder2 = StubEncoder(params_encoder2, 100)

        encoder_manager = EncoderManager()
        self.assertTrue(encoder_manager.add_encoder(encoder1))
        self.assertTrue(encoder_manager.add_encoder(encoder2))
        self.assertEqual(len(encoder_manager.encoders), 2)


        