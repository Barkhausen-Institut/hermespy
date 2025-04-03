# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

import numpy as np

from hermespy.core import Serializable, SerializationProcess, DeserializationProcess
from ..symbols import StatedSymbols
from .symbol_precoding import ReceiveSymbolDecoder

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MaximumRatioCombining(ReceiveSymbolDecoder, Serializable):
    """Maximum ratio combining symbol decoding step.

    Refer to :footcite:t:`1954:kahn` for further information.
    """

    def decode_symbols(
        self, symbols: StatedSymbols, num_output_streams: int
    ) -> StatedSymbols:  # Decode data using MRC receive diversity with N_rx received antennas.
        #
        # Received signal with equal noise power is assumed, the decoded signal has same noise
        # level as input. It is assumed that all data have equal noise levels.

        if symbols.num_transmit_streams != 1:
            raise RuntimeError("Maximum ratio combining only supports a  single transmit stream")

        dense_states = symbols.dense_states()
        simo_states = dense_states.reshape(
            (symbols.num_streams, symbols.num_symbols * symbols.num_blocks)
        )
        symbols_raw = symbols.raw.reshape(
            (symbols.num_streams, symbols.num_symbols * symbols.num_blocks)
        )

        symbol_estimates = np.sum(simo_states.conj() * symbols_raw, axis=0, keepdims=True) / np.sum(
            np.abs(simo_states) ** 2, axis=0, keepdims=True
        )
        state_estimates = np.sum(np.abs(dense_states) ** 2, axis=0)
        # resulting_noises = np.sum(stream_noises * (np.abs(stream_responses) ** 2), axis=0, keepdims=True)

        symbol_estimates = symbol_estimates.reshape((1, symbols.num_blocks, symbols.num_symbols))
        state_estimates = state_estimates.reshape((1, 1, symbols.num_blocks, symbols.num_symbols))
        return StatedSymbols(symbol_estimates, state_estimates)

    def num_receive_output_streams(self, num_input_streams: int) -> int:
        return 1

    @property
    def num_receive_input_symbols(self) -> int:
        return 1

    @property
    def num_receive_output_symbols(self) -> int:
        return 1

    @override
    def serialize(self, process: SerializationProcess) -> None:
        pass

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> MaximumRatioCombining:
        return cls()
