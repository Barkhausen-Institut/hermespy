# -*- coding: utf-8 -*-
"""Single carrier encoding step of communication data symbols."""

from __future__ import annotations
from typing import Tuple
from numpy import real, imag, argmax
import numpy as np

from .symbol_precoder import SymbolPrecoder
from hermespy.channel import ChannelStateInformation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SingleCarrier(SymbolPrecoder):
    """Single Carrier data symbol precoding step.

    Takes a on-dimensional input stream and distributes the symbols to multiple output streams.
    """

    yaml_tag: str = u'SC'

    def __init__(self) -> None:
        """Single Carrier object initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, symbol_stream: np.ndarray) -> np.ndarray:

        if symbol_stream.shape[0] != 1:
            raise RuntimeError("Single-Carrier spatial multiplexing only supports "
                               "one-dimensional input streams during encoding")

        return np.repeat(symbol_stream, self.num_output_streams, axis=0)

    def decode(self,
               symbol_stream: np.ndarray,
               channel_state: ChannelStateInformation,
               stream_noises: np.ndarray) -> Tuple[np.ndarray, ChannelStateInformation, np.ndarray]:

        # Decode data using SC receive diversity with N_rx received antennas.
        #
        # Received signal with equal noise power is assumed, the decoded signal has same noise
        # level as input. It is assumed that all data have equal noise levels.

        # TODO: Check this approach with André
        # Essentially, over all symbol streams for each symbol the one with the strongest response will be selected
        squeezed_channel_state = channel_state.state.sum(axis=1, keepdims=False)

        # Select proper antenna for each symbol timestamp
        antenna_selection = argmax(abs(squeezed_channel_state), axis=0)

        symbol_stream = np.take_along_axis(symbol_stream, antenna_selection.T, axis=0)
        stream_noises = np.take_along_axis(stream_noises, antenna_selection.T, axis=0)

        channel_state_selection = antenna_selection.T[:, np.newaxis, :, np.newaxis]\
            .repeat(2, axis=1).repeat(channel_state.state.shape[3], axis=3)
        channel_state.state = np.take_along_axis(channel_state.state, channel_state_selection, axis=0)

        return symbol_stream, channel_state, stream_noises

    @property
    def num_input_streams(self) -> int:
        return 1

    @property
    def num_output_streams(self) -> int:
        return self.required_num_output_streams
