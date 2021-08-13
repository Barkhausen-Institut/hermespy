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
    def setUp(self) -> None:
        self.params_encoder1 = ParametersEncoder()
        self.params_encoder1.data_bits_k = 3
        self.params_encoder1.encoded_bits_n = 5

        self.params_encoder2 = ParametersEncoder()
        self.params_encoder2.data_bits_k = 6
        self.params_encoder2.encoded_bits_n = 7

        self.encoder1 = StubEncoder(self.params_encoder1, 100)
        self.encoder2 = StubEncoder(self.params_encoder2, 100)

    def test_correct_order_of_encoders(self) -> None:
        encoder_manager = EncoderManager()
        self.assertTrue(encoder_manager.add_encoder(self.encoder1))
        self.assertTrue(encoder_manager.add_encoder(self.encoder2))
        self.assertEqual(len(encoder_manager.encoders), 2)

    def test_ordering_of_encoders(self) -> None:
        encoder_manager = EncoderManager()
        self.assertTrue(encoder_manager.add_encoder(self.encoder2))
        self.assertTrue(encoder_manager.add_encoder(self.encoder1))
        self.assertEqual(len(encoder_manager.encoders), 2)
