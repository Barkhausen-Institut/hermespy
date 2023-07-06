# -*- coding: utf-8 -*-
"""
====================
Spatial Multiplexing
====================
"""

from __future__ import annotations
from fractions import Fraction

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


class SpatialMultiplexing(SymbolPrecoder, Serializable):
    """Spatial Multiplexing data symbol precoding step.

    Takes a on-dimensional input stream and distributes the symbols to multiple output streams.
    """

    yaml_tag: str = "SM"

    def __init__(self) -> None:
        """Spatial Multiplexing object initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, symbols: StatedSymbols) -> StatedSymbols:
        return symbols

    def decode(self, symbols: StatedSymbols) -> StatedSymbols:
        # Collect data symbols from the stream
        return symbols

    @property
    def num_input_streams(self) -> int:
        return self.required_num_output_streams

    @property
    def num_output_streams(self) -> int:
        # Always outputs the required number of streams
        return self.required_num_output_streams

    @property
    def rate(self) -> Fraction:
        # Spatial multiplexing distributes the incoming stream symbols
        # equally over the outgoing streams.
        return Fraction(1, 1)
