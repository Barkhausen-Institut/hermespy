import unittest
from unittest.mock import (Mock, patch)

from parameters_parser.parameters_block_interleaver import ParametersBlockInterleaver
from parameters_parser.parameters_repetition_encoder import ParametersRepetitionEncoder
from parameters_parser.parameters_ldpc_encoder import ParametersLdpcEncoder
from parameters_parser.parameters_scrambler import ParametersScrambler
from modem.coding.encoder_factory import EncoderFactory

from modem.coding.repetition_encoder import RepetitionEncoder
from modem.coding.ldpc_encoder import LdpcEncoder
from modem.coding.interleaver import BlockInterleaver
from modem.coding.scrambler import Scrambler3GPP, Scrambler80211a


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

    def test_block_interleaver_return(self) -> None:
        encoder = self.factory.get_encoder(
            ParametersBlockInterleaver(4, 3), "block_interleaver", 30
        )
        self.assertTrue(isinstance(encoder, BlockInterleaver))

    def test_scrambler_80211a_return(self) -> None:
        encoder = self.factory.get_encoder(
            ParametersScrambler(), Scrambler80211a.factory_tag, 30
        )
        self.assertTrue(isinstance(encoder, Scrambler80211a))

    def test_scrambler_3GPP_return(self) -> None:
        encoder = self.factory.get_encoder(
            ParametersScrambler(), Scrambler3GPP.factory_tag, 30
        )
        self.assertTrue(isinstance(encoder, Scrambler3GPP))
