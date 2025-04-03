# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np

from hermespy.core import (
    Serializable,
    SerializableEnum,
    SerializationProcess,
    DeserializationProcess,
)
from ..symbols import StatedSymbols
from .symbol_precoding import TransmitSymbolEncoder, ReceiveSymbolDecoder

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DFTNorm(SerializableEnum):
    """Discrete Fourier Transform precoding normalization options."""

    BACKWARD = "backward"
    ORTHO = "ortho"
    FORWARD = "forward"


class DFT(TransmitSymbolEncoder, ReceiveSymbolDecoder, Serializable):
    """A precoder applying the Discrete Fourier Transform to each data stream."""

    __DEFAULT_FFT_NORM: DFTNorm = DFTNorm.ORTHO
    __fft_norm: DFTNorm

    def __init__(self, fft_norm: DFTNorm = __DEFAULT_FFT_NORM) -> None:
        """

        Args:
            fft_norm:
                The norm applied to the discrete fourier transform.
                Defaults to DFTNorm.ORTHO.
                See also numpy.fft.fft for details
        """

        # Initialize base classes
        TransmitSymbolEncoder.__init__(self)
        ReceiveSymbolDecoder.__init__(self)
        Serializable.__init__(self)

        # Initialize attributes
        self.__fft_norm = fft_norm

    def encode_symbols(self, symbols: StatedSymbols, num_output_streams: int) -> StatedSymbols:
        encoded_symbols = symbols.copy()
        encoded_symbols.raw = np.fft.fft(symbols.raw, axis=1, norm=self.__fft_norm.value)

        return encoded_symbols

    def decode_symbols(self, symbols: StatedSymbols, num_output_streams: int) -> StatedSymbols:
        decoded_symbols = symbols.copy()
        decoded_symbols.raw = np.fft.ifft(symbols.raw, axis=1, norm=self.__fft_norm.value)

        return decoded_symbols

    def num_transmit_input_streams(self, num_output_streams: int) -> int:
        return num_output_streams

    def num_receive_output_streams(self, num_input_streams: int) -> int:
        return num_input_streams

    @property
    def num_transmit_input_symbols(self) -> int:
        return 1

    @property
    def num_transmit_output_symbols(self) -> int:
        return 1

    @property
    def num_receive_input_symbols(self) -> int:
        return 1

    @property
    def num_receive_output_symbols(self) -> int:
        return 1

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_string(self.__fft_norm.value, "fft_norm")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> DFT:
        return cls(DFTNorm(process.deserialize_string("fft_norm", cls.__DEFAULT_FFT_NORM.value)))
