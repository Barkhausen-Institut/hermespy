# -*- coding: utf-8 -*-
"""
================
Symbol Precoding
================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from hermespy.core.factory import Serializable
from .precoding import Precoder, Precoding

if TYPE_CHECKING:

    from hermespy.modem import StatedSymbols
    from hermespy.modem.modem import BaseModem

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SymbolPrecoder(Precoder, ABC):
    """Abstract base class for signal processing algorithms operating on complex data symbols streams.

    A symbol precoder represents one coding step of a full symbol precoding configuration.
    It features the `encoding` and `decoding` routines, meant to encode and decode multidimensional symbol streams
    during transmission and reception, respectively.
    """

    @abstractmethod
    def encode(self, symbols: StatedSymbols) -> StatedSymbols:
        """Encode a data stream before transmission.

        This operation may modify the number of streams as well as the number of data symbols per stream.

        Args:

            symbols (StatedSymbols):
                Symbols to be encoded.

        Returns: Encoded symbols.

        Raises:

            NotImplementedError: If an encoding operation is not supported.
        """
        ...  # pragma no cover

    @abstractmethod
    def decode(self, symbols: StatedSymbols) -> StatedSymbols:
        """Decode a data stream before reception.

        This operation may modify the number of streams as well as the number of data symbols per stream.

        Args:

            symbols (Symbols):
                Symbols to be decoded.

        Returns: Decoded symbols.

        Raises:

            NotImplementedError: If an encoding operation is not supported.
        """
        ...  # pragma no cover


class SymbolPrecoding(Precoding[SymbolPrecoder], Serializable):
    """Channel SymbolPrecoding configuration for wireless transmission of modulated data symbols.

    Symbol precoding may occur as an intermediate step between bit-mapping and base-band symbol modulations.
    In order to account for the possibility of multiple antenna data-streams,
    waveform generators may access the `SymbolPrecoding` configuration to encode one-dimensional symbol
    streams into multi-dimensional symbol streams during transmission and subsequently decode during reception.
    """

    yaml_tag = "SymbolCoding"
    """YAML serialization tag."""

    def __init__(self, modem: BaseModem = None) -> None:
        """Symbol Precoding object initialization.

        Args:

            modem (Modem, Optional):
                The modem this `SymbolPrecoding` configuration is attached to.
        """

        Precoding.__init__(self, modem=modem)

    def encode(self, symbols: StatedSymbols) -> StatedSymbols:
        """Encode a data stream before transmission.

        This operation may modify the number of streams as well as the number of data symbols per stream.

        Args:

            symbols (Symbols): Symbols to be encoded.

        Returns: Encoded symbols.

        Raises:

            NotImplementedError: If an encoding operation is not supported.
        """

        # Iteratively apply each encoding step
        encoded_symbols = symbols.copy()
        for precoder in self:
            encoded_symbols = precoder.encode(encoded_symbols)

        return encoded_symbols

    def decode(self, symbols: StatedSymbols) -> StatedSymbols:
        """Decode a data stream before reception.

        This operation may modify the number of streams as well as the number of data symbols per stream.

        Args:

            symbols (Symbols):
                Symbols to be decoded.

        Returns: Decoded symbols.

        Raises:

            NotImplementedError: If an encoding operation is not supported.
        """

        decoded_symbols = symbols.copy()
        for precoder in reversed(self):
            decoded_symbols = precoder.decode(decoded_symbols)

        return decoded_symbols
