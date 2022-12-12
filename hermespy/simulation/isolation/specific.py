# -*- coding: utf-8 -*-
"""
==================
Specific Isolation
==================
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np

from hermespy.core import Serializable, Signal
from .isolation import Isolation

if TYPE_CHECKING:
    from ..simulated_device import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SpecificIsolation(Serializable, Isolation):
    """Specific leakage between RF chains."""

    yaml_tag = "Specific"

    __isolation: Optional[np.ndarray]

    def __init__(self, device: Optional[SimulatedDevice] = None, isolation: Optional[np.ndarray] = None) -> None:

        Isolation.__init__(self, device=device)
        self.isolation = isolation

    @property
    def isolation(self) -> np.ndarray:

        return self.__isolation

    @isolation.setter
    def isolation(self, value: Optional[np.ndarray]) -> None:

        if value is None:

            self.__isolation = None
            return

        if value.ndim != 2:
            raise ValueError("Isolation specification must be a two dimensional array")

        self.__isolation = value

    def _leak(self, signal: Signal) -> Signal:

        if self.__isolation is None:
            raise RuntimeError("Error trying to model specific isolaion leakage with undefined isolations")

        if self.__isolation.shape[0] != self.device.antennas.num_receive_antennas:
            raise RuntimeError("Number of receiving antennas in isolation specifications ({self.__isolation.shape[0]}) " "don't match the antenna array ({self.device.antennas.num_receive_antennas})")

        if self.__isolation.shape[1] != self.device.antennas.num_transmit_antennas:
            raise RuntimeError("Number of receiving antennas in isolation specifications ({self.__isolation.shape[0]}) " "don't match the antenna array ({self.device.antennas.num_receive_antennas})")

        leaked_samples = self.isolation @ signal.samples
        return Signal(leaked_samples, signal.sampling_rate, signal.carrier_frequency)
