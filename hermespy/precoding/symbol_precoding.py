# -*- coding: utf-8 -*-
"""
================
Symbol Precoding
================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import  TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from hermespy.core.factory import Serializable
from .precoding import Precoder, Precoding

if TYPE_CHECKING:
    from hermespy.modem import Modem, StatedSymbols

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
            
            symbols (Symbols):
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


class SymbolPrecoding(Serializable, Precoding[SymbolPrecoder]):
    """Channel SymbolPrecoding configuration for wireless transmission of modulated data symbols.

    Symbol precoding may occur as an intermediate step between bit-mapping and base-band symbol modulations.
    In order to account for the possibility of multiple antenna data-streams,
    waveform generators may access the `SymbolPrecoding` configuration to encode one-dimensional symbol
    streams into multi-dimensional symbol streams during transmission and subsequently decode during reception.
    """

    yaml_tag = u'Symbol-Precoding'
    """YAML serialization tag."""

    debug: bool

    def __init__(self,
                 modem: Modem = None) -> None:
        """Symbol Precoding object initialization.

        Args:
        
            modem (Modem, Optional):
                The modem this `SymbolPrecoding` configuration is attached to.
        """

        self.debug = False
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

        if input_stream.shape[0] != channel_states.num_receive_streams:
            raise ValueError("Input streams and channel states must have identical number of streams")

        # if input_stream.shape[1] != channel_states.num_symbols:
        #     raise ValueError("Input streams and channel states must have identical number of symbols")

        # If only a nuclear noise variance is provided, expand it to an array
        if isinstance(stream_noises, float) or isinstance(stream_noises, int):
            stream_noises = np.array([[stream_noises]], dtype=float).repeat(input_stream.shape[0], axis=0)\
                .repeat(input_stream.shape[1], axis=1)

        symbols_iteration = input_stream.copy()
        channel_state_iteration = channel_states
        noises_iteration = stream_noises.copy()

        if self.debug:
            fig, ax = plt.subplots(3, 1+len(self.__symbol_precoders), squeeze=False)

            ax[0, 0].set_title("Input")
            ax[0, 0].set_ylabel("Signal")
            ax[0, 0].plot(abs(symbols_iteration.flatten()))
            ax[1, 0].set_ylabel("CSI")
            ax[1, 0].plot(abs(channel_state_iteration.state.sum(axis=1).sum(axis=2).flatten()))
            ax[2, 0].set_ylabel("Noise")
            ax[2, 0].plot(abs(noises_iteration.flatten()))
            i = 0

        # Recursion through all precoders, each one may update the stream as well as the responses
        for precoder in reversed(self.__symbol_precoders):
            symbols_iteration, channel_state_iteration, noises_iteration = precoder.decode(symbols_iteration,
                                                                                           channel_state_iteration,
                                                                                           noises_iteration)

            if self.debug:
                i += 1
                ax[0, i].set_title(precoder.__class__.__name__)
                ax[0, i].plot(abs(symbols_iteration.flatten()))
                ax[1, i].plot(abs(channel_state_iteration.state.sum(axis=1).sum(axis=2).flatten()))
                ax[2, i].plot(abs(noises_iteration.flatten()))

        if self.debug:
            plt.show()

        return symbols_iteration
