# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from fractions import Fraction

from hermespy.core import (
    ReceiveDecoder,
    ReceivePrecoding,
    Serializable,
    TransmitEncoder,
    TransmitPrecoding,
)
from ..symbols import StatedSymbols


__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TransmitSymbolEncoder(TransmitEncoder["TransmitSymbolCoding"]):

    def __init__(self) -> None:
        # Initialize base class, required for static type checking
        TransmitEncoder.__init__(self)

    @abstractmethod
    def encode_symbols(self, symbols: StatedSymbols, num_output_streams: int) -> StatedSymbols:
        """Encode a data stream before transmission.

        This operation may modify the number of streams as well as the number of data symbols per stream.

        Args:

            symbols (StatedSymbols):
                Symbols to be encoded.

            num_output_streams (int):
                Number of required output streams after encoding.

        Returns: Encoded symbols.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def num_transmit_input_symbols(self) -> int:
        """Required number of input streams during encoding."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def num_transmit_output_symbols(self) -> int:
        """Resulting number of output streams after encoding."""
        ...  # pragma: no cover

    @property
    def encode_rate(self) -> Fraction:
        """Rate between input symbol slots and output symbol slots."""
        return Fraction(self.num_transmit_input_symbols, self.num_transmit_output_symbols)


class ReceiveSymbolDecoder(ReceiveDecoder["ReceiveSymbolCoding"]):

    def __init__(self) -> None:
        # Initialize base class, required for static type checking
        ReceiveDecoder.__init__(self)

    @abstractmethod
    def decode_symbols(self, symbols: StatedSymbols, num_output_streams: int) -> StatedSymbols:
        """Decode a data stream before reception.

        This operation may modify the number of streams as well as the number of data symbols per stream.

        Args:

            symbols (StatedSymbols):
                Symbols to be decoded.

            num_output_streams (int):
                Number of required output streams after decoding.

        Returns: Decoded symbols.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def num_receive_input_symbols(self) -> int:
        """Required number of input streams during decoding."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def num_receive_output_symbols(self) -> int:
        """Resulting number of output streams after decoding."""
        ...  # pragma: no cover

    @property
    def decode_rate(self) -> Fraction:
        """Rate between input symbol slots and output symbol slots."""
        return Fraction(self.num_receive_output_symbols, self.num_receive_input_symbols)


class TransmitSymbolCoding(TransmitPrecoding[TransmitSymbolEncoder], Serializable):
    """Channel precoding configuration for wireless transmission of modulated data symbols.

    Transmit symbol precoding occurs as an intermediate step between bit-mapping and base-band symbol modulation.
    """

    def encode_symbols(self, symbols: StatedSymbols, num_output_streams: int) -> StatedSymbols:
        """Encode a data stream before transmission.

        This operation may modify the number of streams as well as the number of data symbols per stream.

        Args:

            symbols (StatedSymbols): Symbols to be encoded.
            num_output_streams (int): Number of required output streams after encoding.

        Returns: Encoded symbols.
        """

        num_encoder_io_streams = self._collect_encoder_num_io_streams(num_output_streams)

        # Iteratively apply each encoding step
        encoded_symbols = symbols.copy()
        for precoder, num_output_streams in zip(self.__iter__(), num_encoder_io_streams[1:]):
            encoded_symbols = precoder.encode_symbols(encoded_symbols, num_output_streams)

        return encoded_symbols

    @property
    def encode_rate(self) -> Fraction:
        """Rate between input symbol slots and output symbol slots."""

        r = Fraction(1, 1)
        for symbol_precoder in self:
            r *= symbol_precoder.encode_rate

        return r

    def num_encoded_blocks(self, num_input_blocks: int) -> int:
        """Number of resulting symbol blocks after encoding.

        Args:

            num_input_blocks (int):
                Number of blocks before encoding.

        Returns: Number of blocks after encoding.
        """

        num_blocks = Fraction(num_input_blocks, 1)

        for precoder in self:
            num_blocks /= precoder.encode_rate

        return int(num_blocks)


class ReceiveSymbolCoding(ReceivePrecoding[ReceiveSymbolDecoder], Serializable):
    """Channel precoding configuration for wireless reception of modulated data symbols.

    Receive symbol precoding occurs as an intermediate step between base-band demodulation and bit unmapping.
    """

    def decode_symbols(self, symbols: StatedSymbols) -> StatedSymbols:
        """Decode a data stream before reception.

        This operation may modify the number of streams as well as the number of data symbols per stream.

        Args:

            symbols (StatedSymbols):
                Symbols to be decoded.

        Returns: Decoded symbols.
        """

        # Collect number of output streams for each decoder
        num_decoder_io_streams = self._collect_decoder_num_io_streams(symbols.num_streams)

        decoded_symbols = symbols.copy()
        for decoder, num_output_streams in zip(self, num_decoder_io_streams[1:]):
            decoded_symbols = decoder.decode_symbols(decoded_symbols, num_output_streams)

        return decoded_symbols

    @property
    def decode_rate(self) -> Fraction:
        """Rate between input symbol slots and output symbol slots."""

        r = Fraction(1, 1)
        for symbol_precoder in self:
            r *= symbol_precoder.decode_rate

        return r

    def num_decoded_blocks(self, num_input_blocks: int) -> int:
        """Number of resulting symbol blocks after decoding.

        Args:

            num_input_blocks (int):
                Number of blocks before decoding.

        Returns: Number of blocks after decoding.
        """

        num_blocks = Fraction(num_input_blocks, 1)

        for precoder in reversed(self):
            num_blocks /= precoder.decode_rate

        return int(num_blocks)
