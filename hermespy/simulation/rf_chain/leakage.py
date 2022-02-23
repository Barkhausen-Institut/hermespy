# -*- coding: utf-8 -*-
"""
=================================
Transmit-Receive Leakage Modeling
=================================
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from ...core.factory import Serializable
from ...core.signal_model import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.6"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Leakage(Serializable, ABC):
    """Leakage model between transmit and receive hardware chains."""

    @abstractmethod
    def transmit(self,
                 transmission: Signal) -> Signal:
        """Model a leaking transmission.

        Args:

            transmission (Signal):
                Signal transmitted over the leaking hardware.

        Returns:

            Signal:
                Signal model after transmission.
        """

        # Default leakage modeling is just a stub
        return transmission

    @abstractmethod
    def receive(self,
                reception: Signal) -> Signal:
        """Model a leaking reception.

        Args:

            reception (Signal):
                Signal received over the leaking hardware.

        Returns:

            Signal:
                Signal model after reception.
        """

        # Default leakage modeling is just a stub
        return reception


class LinearLeakage(Leakage):
    """Linear leakage model between transmit and receive hardware chains."""

    __transmit_cache: Signal


    def __init__(self) -> None: