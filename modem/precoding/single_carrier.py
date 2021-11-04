# -*- coding: utf-8 -*-
"""Single carrier encoding step of communication data symbols."""

from __future__ import annotations
from numpy import real, imag, sum, argmax, take_along_axis
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

    def decode(self, symbol_stream: np.ndarray) -> np.ndarray:

        # Decode data using SC receive diversity with N_rx received antennas.
        #
        # Received signal with equal noise power is assumed, the decoded signal has same noise
        # level as input. It is assumed that all data have equal noise levels.

        channel_estimation = self.precoding.modem.reference_channel.estimate()
        noise_var = 1.0

        # TODO: Check this approach with André
        # channel_estimation = np.squeeze(channel_estimation, axis=1)
        channel_estimation = sum(real(channel_estimation) ** 2 + imag(channel_estimation) ** 2, axis=-1)

        # Select proper antenna for each symbol timestamp
        antenna_selection = argmax(channel_estimation, axis=-1).T
        # antenna_index = np.argmax(np.abs(channel_estimation) ** 2 / noise_var, axis=0)

        # Debug, for now, simply take all the symbols from the first stream
        # TODO: Re-implement proper symbol decoding
        output_stream: np.ndarray = take_along_axis(symbol_stream, antenna_selection, 0)

        # Flatten the output stream to a single dimension
        output_stream = np.reshape(output_stream, (1, -1), 'F')

        # output_stream = np.take_along_axis(symbol_stream, antenna_index[np.newaxis, :], axis=0)
        # channel_estimation = np.take_along_axis(channel_estimation, antenna_index[np.newaxis, :], axis=0)
        # noise_var = np.take_along_axis(noise_var, antenna_index[np.newaxis, :], axis=0)

        return output_stream
