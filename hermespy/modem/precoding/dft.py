# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Literal

import numpy as np

from hermespy.core import Serializable
from ..symbols import StatedSymbols
from .symbol_precoding import SymbolPrecoder

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DFT(SymbolPrecoder, Serializable):
    """A precoder applying the Discrete Fourier Transform to each data stream."""

    yaml_tag = "DFT"
    __fft_norm: Literal["backward", "ortho", "forward"]

    def __init__(self, fft_norm: Literal["backward", "ortho", "forward"] = "ortho") -> None:
        """

        Args:
            fft_norm (str, optional):
                The norm applied to the discrete fourier transform.
                See also numpy.fft.fft for details
        """

        # Initialize base class
        SymbolPrecoder.__init__(self)

        # Initialize attributes
        self.__fft_norm = fft_norm

    def encode(self, symbols: StatedSymbols) -> StatedSymbols:
        encoded_symbols = symbols.copy()
        encoded_symbols.raw = np.fft.fft(symbols.raw, axis=1, norm=self.__fft_norm)

        return encoded_symbols

    def decode(self, symbols: StatedSymbols) -> StatedSymbols:
        decoded_symbols = symbols.copy()
        decoded_symbols.raw = np.fft.ifft(symbols.raw, axis=1, norm=self.__fft_norm)

        return decoded_symbols

    @property
    def num_input_streams(self) -> int:
        # DFT precoding does not alter the number of symbol streams
        return self.precoding.required_outputs(self)

    @property
    def num_output_streams(self) -> int:
        # DFT precoding does not alter the number of symbol streams
        return self.precoding.required_outputs(self)
