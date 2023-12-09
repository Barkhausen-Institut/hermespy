# -*- coding: utf-8 -*-
"""
=======================
Single Carrier Encoding
=======================
"""

from __future__ import annotations

import numpy as np
from numpy import argmax

from hermespy.core import Serializable
from ..symbols import StatedSymbols
from .symbol_precoding import SymbolPrecoder

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SingleCarrier(SymbolPrecoder, Serializable):
    """Single Carrier data symbol precoding step.

    Takes a on-dimensional input stream and distributes the symbols to multiple output streams.
    """

    yaml_tag = "SingleCarrier"

    def __init__(self) -> None:
        """Single Carrier object initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, symbols: StatedSymbols) -> StatedSymbols:
        if symbols.num_streams != 1:
            raise RuntimeError(
                "Single-Carrier spatial multiplexing only supports one-dimensional input streams during encoding"
            )

        repeated_symbols = symbols.copy()
        repeated_symbols.raw = np.repeat(repeated_symbols.raw, self.num_output_streams, axis=0)
        repeated_symbols.states = np.repeat(
            repeated_symbols.states, self.num_output_streams, axis=0
        )

        return repeated_symbols

    def decode(self, symbols: StatedSymbols) -> StatedSymbols:
        # Decode data using SC receive diversity with N_rx received antennas.
        #
        # Received signal with equal noise power is assumed, the decoded signal has same noise
        # level as input. It is assumed that all data points equal noise levels.

        # Essentially, over all symbol streams for each symbol the one with the strongest response will be selected
        symbols = symbols.copy()
        squeezed_channel_state = symbols.states.sum(axis=1, keepdims=False)

        # Select proper antenna for each symbol timestamp
        antenna_selection = argmax(abs(squeezed_channel_state), axis=0)

        new_symbols = np.take_along_axis(symbols.raw, antenna_selection[np.newaxis, ::], axis=0)
        # stream_noises = np.take_along_axis(symbols, antenna_selection.T, axis=0)

        channel_state_selection = antenna_selection[np.newaxis, np.newaxis, ::].repeat(2, axis=1)
        new_states = np.take_along_axis(symbols.states, channel_state_selection, axis=0)

        symbols.raw = new_symbols
        symbols.states = new_states
        return symbols

    @property
    def num_input_streams(self) -> int:
        return 1

    @property
    def num_output_streams(self) -> int:
        return self.required_num_output_streams
