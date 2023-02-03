# -*- coding: utf-8 -*-
"""
==================
Specific Isolation
==================
"""

from __future__ import annotations
from typing import Optional, Union, TYPE_CHECKING

import numpy as np

from hermespy.core import dimension, Serializable, Signal
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

    __leakage_factors: Optional[np.ndarray]

    def __init__(self,
                 isolation: Union[None, np.ndarray, float, int] = None,
                 device: Optional[SimulatedDevice] = None) -> None:

        # Initialize base class
        Isolation.__init__(self, device=device)

        # Initialize class attributes
        self.__leakage_factors = None
        self.isolation = isolation

    @dimension
    def isolation(self) -> np.ndarray:
        """Linear power isolation between transmit and receive chains.

        Returns: Numpy matrix (two-dimensional array).
        """

        return self.__isolation

    @isolation.setter(title='Isolation')
    def isolation(self, value: Union[None, np.ndarray, float, int]) -> None:

        if value is None:

            self.__isolation = None
            return
        
        if isinstance(value, (float, int)):
            
            if self.device is not None and self.device.num_antennas != 1:
                raise ValueError("Scalar isolation definition is only allowed for devices with a single antenna")

            value = np.array([[value]], dtype=float)

        if value.ndim != 2:
            raise ValueError("Isolation specification must be a two dimensional array")

        self.__isolation = value

        # The leaking power is the square root of the inverse isolation
        self.__leakage_factors = np.power(value, -.5)

    def _leak(self, signal: Signal) -> Signal:

        if self.__leakage_factors is None:
            raise RuntimeError("Error trying to model specific isolaion leakage with undefined isolations")

        if self.__leakage_factors.shape[0] != self.device.antennas.num_receive_antennas:
            raise RuntimeError("Number of receiving antennas in isolation specifications ({self.__leakage_factors.shape[0]}) " "don't match the antenna array ({self.device.antennas.num_receive_antennas})")

        if self.__leakage_factors.shape[1] != self.device.antennas.num_transmit_antennas:
            raise RuntimeError("Number of receiving antennas in isolation specifications ({self.__leakage_factors.shape[0]}) " "don't match the antenna array ({self.device.antennas.num_receive_antennas})")

        leaked_samples = self.__leakage_factors @ signal.samples
        return Signal(leaked_samples, signal.sampling_rate, signal.carrier_frequency)
