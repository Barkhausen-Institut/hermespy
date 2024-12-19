# -*- coding: utf-8 -*-

from .symbol_precoding import (
    TransmitSymbolCoding,
    ReceiveSymbolCoding,
    TransmitSymbolEncoder,
    ReceiveSymbolDecoder,
)
from .single_carrier import SingleCarrier
from .dft import DFT
from .space_time_block_coding import Alamouti, Ganesan
from .ratio_combining import MaximumRatioCombining

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = [
    "TransmitSymbolCoding",
    "ReceiveSymbolCoding",
    "TransmitSymbolEncoder",
    "ReceiveSymbolDecoder",
    "SingleCarrier",
    "DFT",
    "Alamouti",
    "Ganesan",
    "MaximumRatioCombining",
]
