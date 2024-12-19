# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from hermespy.core import Serializable
from ..symbols import StatedSymbols
from .symbol_precoding import ReceiveSymbolDecoder

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SingleCarrier(ReceiveSymbolDecoder, Serializable):
    """Single Carrier data symbol precoding step.

    Takes a on-dimensional input stream and distributes the symbols to multiple output streams.
    """

    yaml_tag = "SingleCarrier"

    def decode_symbols(self, symbols: StatedSymbols, num_output_streams: int) -> StatedSymbols:
        # Decode data using SC receive diversity with N_rx received antennas.
        #
        # Received signal with equal noise power is assumed, the decoded signal has same noise
        # level as input. It is assumed that all data points equal noise levels.

        # Essentially, over all symbol streams for each symbol the one with the strongest response will be selected
        symbols = symbols.copy()
        dense_states = symbols.dense_states()
        squeezed_channel_state = dense_states.sum(axis=1, keepdims=False)

        # Select proper antenna for each symbol timestamp
        antenna_selection = np.argmax(np.abs(squeezed_channel_state), axis=0)

        new_symbols = np.take_along_axis(symbols.raw, antenna_selection[np.newaxis, ::], axis=0)
        # stream_noises = np.take_along_axis(symbols, antenna_selection.T, axis=0)

        channel_state_selection = antenna_selection[np.newaxis, np.newaxis, ::].repeat(2, axis=1)
        new_states = np.take_along_axis(dense_states, channel_state_selection, axis=0)

        symbols.raw = new_symbols
        symbols.states = new_states
        return symbols

    def num_receive_output_streams(self, num_input_streams: int) -> int:
        return 1

    @property
    def num_receive_input_symbols(self) -> int:
        return 1

    @property
    def num_receive_output_symbols(self) -> int:
        return 1
