# -*- coding: utf-8 -*-
"""HermesPy data drop."""

from __future__ import annotations
from typing import List
import numpy as np

from .executable import Executable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Drop:
    """Data generated within a single execution drop."""

    __transmitted_bits: List[np.ndarray]
    __transmitted_signals: List[np.ndarray]
    __received_signals: List[np.ndarray]
    __received_bits: List[np.ndarray]

    def __init__(self,
                 transmitted_bits: List[np.ndarray],
                 transmitted_signals: List[np.ndarray],
                 received_signals: List[np.ndarray],
                 received_bits: List[np.ndarray]) -> None:
        """Object initialization.

        Args:
            transmitted_bits (List[np.ndarray]): Bits fed into the transmitting modems.
            transmitted_signals (List[np.ndarray]): Modulated signals emitted by transmitting modems.
            received_signals (List[np.ndarray]): Modulated signals impinging onto receiving modems.
            received_bits (List[np.ndarray]): Bits output by receiving modems.
        """

        self.__transmitted_bits = transmitted_bits
        self.__transmitted_signals = transmitted_signals
        self.__received_signals = received_signals
        self.__received_bits = received_bits

    @property
    def transmitted_bits(self) -> List[np.ndarray]:
        """Access transmitted bits."""

        return self.__transmitted_bits

    @property
    def transmitted_signals(self) -> List[np.ndarray]:
        """Access transmitted signals."""

        return self.__transmitted_signals

    @property
    def received_signals(self) -> List[np.ndarray]:
        """Access received signals."""

        return self.__received_bits

    @property
    def received_bits(self) -> List[np.ndarray]:
        """Access received bits."""

        return self.__received_bits
