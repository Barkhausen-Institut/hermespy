# -*- coding: utf-8 -*-
"""
=======================
Maximum Ratio Combining
=======================
"""

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING

import numpy as np

from hermespy.core import Serializable
from .symbol_precoding import SymbolPrecoder

if TYPE_CHECKING:
    from hermespy.modem import StatedSymbols

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MaximumRatioCombining(SymbolPrecoder, Serializable):
    """Maximum ratio combining symbol decoding step"""

    yaml_tag: str = "MRC"

    def encode(self, _: StatedSymbols) -> StatedSymbols:
        raise NotImplementedError("Maximum ratio combining only supports decoding operations")

    def decode(self, symbols: StatedSymbols) -> StatedSymbols:

        # Decode data using MRC receive diversity with N_rx received antennas.
        #
        # Received signal with equal noise power is assumed, the decoded signal has same noise
        # level as input. It is assumed that all data have equal noise levels.

        resulting_symbols = np.sum(symbols.raw * symbols.states.conj(), axis=0, keepdims=True)
        # resulting_noises = np.sum(stream_noises * (np.abs(stream_responses) ** 2), axis=0, keepdims=True)
        resulting_states = np.sum(np.abs(symbols.states) ** 2, axis=0)

        return StatedSymbols(resulting_symbols, resulting_states)

    @property
    def num_input_streams(self) -> int:
        return -1

    @property
    def num_output_streams(self) -> int:
        return 1
