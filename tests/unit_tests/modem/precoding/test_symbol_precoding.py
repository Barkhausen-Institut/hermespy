# -*- coding: utf-8 -*-

import unittest
from fractions import Fraction

import numpy as np

from hermespy.modem import StatedSymbols
from hermespy.modem.precoding.symbol_precoding import TransmitSymbolCoding, ReceiveSymbolCoding, TransmitSymbolEncoder, ReceiveSymbolDecoder
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockTransmitSymbolEncoder(TransmitSymbolEncoder):
    """Transmit symbol encoder for unit testing purposes."""

    def encode_symbols(self, symbols: StatedSymbols, num_output_streams: int) -> StatedSymbols:
        tiled_symbols = np.tile(symbols.raw[[0], ::], (num_output_streams, 1, 2))
        tiled_states = np.tile(symbols.states[:, [0], ::], (num_output_streams, 1, 1, 2))
        return StatedSymbols(tiled_symbols, tiled_states)

    @property
    def num_transmit_input_symbols(self) -> int:
        return 1

    @property
    def num_transmit_output_symbols(self) -> int:
        return 2

    def num_transmit_input_streams(self, num_output_streams: int) -> int:
        return 1


class MockReceiveSymbolDecoder(ReceiveSymbolDecoder):
    """Receive symbol decoder for unit testing purposes."""

    def decode_symbols(self, symbols: StatedSymbols, num_output_streams: int) -> StatedSymbols:
        # Select the first stream and every other symbol block
        return StatedSymbols(symbols.raw[::2, :, ::2], symbols.states[::2, :, :, ::2])

    @property
    def num_receive_input_symbols(self) -> int:
        return 2

    @property
    def num_receive_output_symbols(self) -> int:
        return 1

    def num_receive_output_streams(self, num_input_streams: int) -> int:
        return num_input_streams // 2


class TestReceiveSymbolCoding(unittest.TestCase):
    """Test the receive symbol precoding configuration."""

    coding: ReceiveSymbolCoding

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.coding = ReceiveSymbolCoding()

    def test_decode_symbols(self) -> None:
        """Test proper symbol decoding"""

        self.coding[0] = MockReceiveSymbolDecoder()
        self.coding[1] = MockReceiveSymbolDecoder()

        symbols = StatedSymbols(self.rng.random((4, 2, 4)), self.rng.random((4, 1, 2, 4)))
        decoded_symbols = self.coding.decode_symbols(symbols)

        self.assertEqual(1, decoded_symbols.num_streams)
        self.assertEqual(1, decoded_symbols.num_transmit_streams)
        self.assertEqual(2, decoded_symbols.num_blocks)
        self.assertEqual(1, decoded_symbols.num_symbols)

    def test_decode_rate(self) -> None:
        """Decode rate should be the multiplication of all decoders' rates"""

        self.coding[0] = MockReceiveSymbolDecoder()
        self.coding[1] = MockReceiveSymbolDecoder()

        self.assertEqual(Fraction(1, 4), self.coding.decode_rate)

    def test_yaml_serialization(self) -> None:
        """Test serialization from and to YAML"""
        
        test_roundtrip_serialization(self, self.coding)


class TestTransmitSymbolCoding(unittest.TestCase):
    """Test the transmit symbol precoding configuration."""

    coding: TransmitSymbolCoding

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.selected_transmit_ports = [0, 1, 2, 3]
        self.coding = TransmitSymbolCoding()

    def test_encode_symbols(self) -> None:
        """Test proper symbol encoding"""

        self.coding[0] = MockTransmitSymbolEncoder()
        self.coding[1] = MockTransmitSymbolEncoder()

        symbols = StatedSymbols(self.rng.random((1, 2, 1)), self.rng.random((1, 1, 2, 1)))
        encoded_symbols = self.coding.encode_symbols(symbols, 4)

        self.assertEqual(4, encoded_symbols.num_streams)
        self.assertEqual(1, encoded_symbols.num_transmit_streams)
        self.assertEqual(2, encoded_symbols.num_blocks)
        self.assertEqual(4, encoded_symbols.num_symbols)

    def test_encode_rate(self) -> None:
        """Encode rate should be the multiplication of all encoders' rates"""

        self.coding[0] = MockTransmitSymbolEncoder()
        self.coding[1] = MockTransmitSymbolEncoder()

        self.assertEqual(Fraction(1, 4), self.coding.encode_rate)

    def test_yaml_serialization(self) -> None:
        """Test serialization from and to YAML"""
        
        test_roundtrip_serialization(self, self.coding)
