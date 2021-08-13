from parameters_parser.parameters_ldpc_encoder import ParametersLdpcEncoder
from modem.coding.encoder_factory import EncoderFactory
import unittest
from unittest.mock import (Mock, patch)

from parameters_parser.parameters_repetition_encoder import ParametersRepetitionEncoder

from modem.coding.repetition_encoder import RepetitionEncoder
from modem.coding.ldpc_encoder import LdpcEncoder


class TestEncoderFactory(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = EncoderFactory()
        self.mock_params_ldpc_encoder = Mock()

    def test_repetition_encoder_return(self) -> None:
        encoder = self.factory.get_encoder(
            ParametersRepetitionEncoder(), "Repetition", 10)
        self.assertTrue(isinstance(encoder, RepetitionEncoder))

    def test_ldpc_encoder_return(self) -> None:
        with patch.object(LdpcEncoder, "__init__", lambda x, y, z: None):
            encoder = self.factory.get_encoder(
                ParametersLdpcEncoder(), "ldpc", 10)
            self.assertTrue(isinstance(encoder, LdpcEncoder))

    def test_default_encoder_is_repetition(self) -> None:
        encoder = self.factory.get_encoder(
            ParametersRepetitionEncoder(), "bla", 10
        )
        self.assertTrue(isinstance(encoder, RepetitionEncoder))
        self.assertTrue(encoder.data_bits_k, 1)
        self.assertTrue(encoder.encoded_bits_n, 1)