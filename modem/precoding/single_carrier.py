# -*- coding: utf-8 -*-
"""Single carrier encoding step of communication data symbols."""

from __future__ import annotations
from typing import Tuple
from numpy import real, imag, argmax
import numpy as np

from .spatial_multiplexing import SpatialMultiplexing

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SingleCarrier(SpatialMultiplexing):
    """Single Carrier data symbol precoding step.

    Takes a on-dimensional input stream and distributes the symbols to multiple output streams.
    """

    yaml_tag: str = u'SC'

    def __init__(self) -> None:
        """Single Carrier object initialization."""

        SpatialMultiplexing.__init__(self)

    def decode(self,
               symbol_stream: np.ndarray,
               stream_responses: np.ndarray,
               stream_noises: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Decode data using SC receive diversity with N_rx received antennas.
        #
        # Received signal with equal noise power is assumed, the decoded signal has same noise
        # level as input. It is assumed that all data have equal noise levels.

        # TODO: Check this approach with André
        # Esentially, over all symbol streams for each symbol the one with the strongest response will be selected
        channel_estimation = real(stream_responses) ** 2 + imag(stream_responses) ** 2

        # Select proper antenna for each symbol timestamp
        antenna_selection = argmax(channel_estimation, axis=0)
        indices = np.arange(symbol_stream.shape[1])

        output_stream: np.ndarray = symbol_stream[antenna_selection, indices]
        symbol_responses = stream_responses[antenna_selection, indices]
        symbol_noises = stream_noises[antenna_selection, indices]

        return output_stream, symbol_responses, symbol_noises
