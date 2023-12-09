# -*- coding: utf-8 -*-

from .symbol_precoding import SymbolPrecoder, SymbolPrecoding
from .single_carrier import SingleCarrier
from .spatial_multiplexing import SpatialMultiplexing
from .dft import DFT
from .space_time_block_coding import Alamouti, Ganesan
from .ratio_combining import MaximumRatioCombining

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = [
    "SymbolPrecoder",
    "SymbolPrecoding",
    "SingleCarrier",
    "SpatialMultiplexing",
    "DFT",
    "Alamouti",
    "Ganesan",
    "MaximumRatioCombining",
]
