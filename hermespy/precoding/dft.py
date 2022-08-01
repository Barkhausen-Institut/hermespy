# -*- coding: utf-8 -*-
"""
====================================
Discrete Fourier Transform Precoding
====================================
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from hermespy.core.factory import Serializable
from . import SymbolPrecoder

if TYPE_CHECKING:
    from hermespy.modem import StatedSymbols

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DFT(SymbolPrecoder, Serializable):
    """A precoder applying the Discrete Fourier Transform to each data stream.
    """

    yaml_tag = u'DFT'
    __fft_norm: str

    def __init__(self,
                 fft_norm: str = None) -> None:
        """Object initialization.

        Args:
            fft_norm (str, optional):
                The norm applied to the discrete fourier transform.
                See also numpy.fft.fft for details
        """

        self.__fft_norm = 'ortho'

        if fft_norm is not None:
            self.__fft_norm = fft_norm

        SymbolPrecoder.__init__(self)

    def encode(self, symbols: StatedSymbols) -> StatedSymbols:

        encoded_symbols = symbols.copy()
        encoded_symbols.raw = np.fft.fft(symbols.raw, axis=2, norm=self.__fft_norm)

        return encoded_symbols

    def decode(self, symbols: StatedSymbols) -> StatedSymbols:

        decoded_symbols = symbols.copy()
        decoded_symbols.raw = np.fft.ifft(symbols.raw, axis=2, norm=self.__fft_norm)

        return decoded_symbols

    @property
    def num_input_streams(self) -> int:

        # DFT precoding does not alter the number of symbol streams
        return self.precoding.required_outputs(self)

    @property
    def num_output_streams(self) -> int:

        # DFT precoding does not alter the number of symbol streams
        return self.precoding.required_outputs(self)
