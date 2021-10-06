import unittest
from unittest.mock import Mock, call
from typing import List
import numpy as np

from parameters_parser.parameters_encoder import ParametersEncoder
from modem.coding.encoder import Encoder
from modem.coding.encoder_manager import EncoderManager


class StubEncoder(Encoder):

    def __init__(self, params: ParametersEncoder,
                       bits_in_frame: int,
                       rng: np.random.RandomState) -> None:
        super().__init__(params, bits_in_frame, rng)

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

        self.bits_in_frame = 100

        self.encoder1 = StubEncoder(self.params_encoder1, self.bits_in_frame, None)
        self.encoder2 = StubEncoder(self.params_encoder2, self.bits_in_frame, None)

        self.encoder_manager = EncoderManager()

    def test_correct_order_of_encoders(self) -> None:
        self.encoder_manager.add_encoder(self.encoder1)
        self.encoder_manager.add_encoder(self.encoder2)
        self.assertEqual(id(self.encoder_manager.encoders[0]), id(self.encoder1))
        self.assertEqual(id(self.encoder_manager.encoders[1]), id(self.encoder2))

    def test_ordering_of_encoders(self) -> None:
        self.encoder_manager.add_encoder(self.encoder2)
        self.encoder_manager.add_encoder(self.encoder1)
        self.assertEqual(id(self.encoder_manager.encoders[0]), id(self.encoder1))
        self.assertEqual(id(self.encoder_manager.encoders[1]), id(self.encoder2))

    def test_encoding_functions_called(self) -> None:
        mock_parent = Mock()
        mock_parent.mock_encoder1 = Mock()
        mock_parent.mock_encoder1.data_bits_k = 2
        mock_parent.mock_encoder2 = Mock()
        mock_parent.mock_encoder2.data_bits_k = 3
        mock_parent.mock_encoder3 = Mock()
        mock_parent.mock_encoder3.data_bits_k = 5

        self.encoder_manager.add_encoder(mock_parent.mock_encoder1)
        self.encoder_manager.add_encoder(mock_parent.mock_encoder2)
        self.encoder_manager.add_encoder(mock_parent.mock_encoder3)

        _ = self.encoder_manager.encode([np.array([0])])
        mock_parent.assert_has_calls([
            call.mock_encoder1.encode([np.array([0])]),
            call.mock_encoder2.encode(mock_parent.mock_encoder1.encode(np.array([0]))),
            call.mock_encoder3.encode(
                mock_parent.mock_encoder2.encode(
                    mock_parent.mock_encoder1.encode(np.array([0]))
                )
            )])

    def test_decoding_functions_called(self) -> None:
        mock_parent = Mock()
        mock_parent.mock_encoder1 = Mock()
        mock_parent.mock_encoder1.data_bits_k = 2
        mock_parent.mock_encoder2 = Mock()
        mock_parent.mock_encoder2.data_bits_k = 3
        mock_parent.mock_encoder3 = Mock()
        mock_parent.mock_encoder3.data_bits_k = 5

        self.encoder_manager.add_encoder(mock_parent.mock_encoder1)
        self.encoder_manager.add_encoder(mock_parent.mock_encoder2)
        self.encoder_manager.add_encoder(mock_parent.mock_encoder3)

        _ = self.encoder_manager.decode([np.array([0])])

        mock_parent.assert_has_calls([
            call.mock_encoder3.decode([np.array([0])]),
            call.mock_encoder2.decode(
                mock_parent.mock_encoder3.decode(np.array([0]))
            ),
            call.mock_encoder1.decode(
                mock_parent.mock_encoder2.decode(
                    mock_parent.mock_encoder3.decode(np.array([0]))
                )
            )]
        )

    def test_code_rate_calculation(self) -> None:
        self.encoder_manager.add_encoder(self.encoder1)
        self.encoder_manager.add_encoder(self.encoder2)
        expected_code_rate = (
            self.params_encoder1.data_bits_k / self.params_encoder1.encoded_bits_n
            * self.params_encoder2.data_bits_k / self.params_encoder2.encoded_bits_n
        )
        self.assertAlmostEqual(expected_code_rate, self.encoder_manager.code_rate)

    def test_source_bits_calculation(self) -> None:
        self.encoder_manager.add_encoder(self.encoder2)
        self.encoder_manager.add_encoder(self.encoder1)
        self.assertEqual(
            self.encoder_manager.source_bits,
            self.encoder1.source_bits
        )