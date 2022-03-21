# -*- coding: utf-8 -*-
"""
=======================
Maximum Ratio Combining
=======================
"""

from __future__ import annotations
from typing import Type, Tuple

import numpy as np
from ruamel.yaml import SafeConstructor, SafeRepresenter, ScalarNode

from . import SymbolPrecoder
from hermespy.core.factory import Serializable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MaximumRatioCombining(SymbolPrecoder, Serializable):
    """Maximum ratio combining symbol decoding step"""

    yaml_tag: str = u'MRC'

    def encode(self, symbol_stream: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Maximum ratio combining only supports decoding operations")

    def decode(self,
               symbol_stream: np.ndarray,
               stream_responses: np.ndarray,
               stream_noises: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Decode data using MRC receive diversity with N_rx received antennas.
        #
        # Received signal with equal noise power is assumed, the decoded signal has same noise
        # level as input. It is assumed that all data have equal noise levels.

        resulting_symbols = np.sum(symbol_stream * stream_responses.conj(), axis=0, keepdims=True)
        resulting_noises = np.sum(stream_noises * (np.abs(stream_responses) ** 2), axis=0, keepdims=True)
        resulting_responses = np.sum(np.abs(stream_responses) ** 2, axis=0)

        return resulting_symbols, resulting_responses, resulting_noises

    @property
    def num_input_streams(self) -> int:
        return -1

    @property
    def num_output_streams(self) -> int:
        return 1
