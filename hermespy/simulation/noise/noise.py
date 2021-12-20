# -*- coding: utf-8 -*-
"""
==============
Noise Modeling
==============
"""

from __future__ import annotations
from abc import abstractmethod

import numpy as np
from numpy.random import default_rng

from hermespy.signal import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Noise(object):
    """Noise modeling base class."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def add(self, signal: Signal) -> None:
        """Add noise to a signal model.

        Args:

            signal (Signal):
                The signal to which the noise should be added.
        """
        ...


class AWGN(Noise):
    """Additive White Gaussian Noise."""

    __slots__ = ['__power']

    __power: float       # Variance of the added noise

    def __init__(self,
                 power: float = 0.) -> None:
        """
        Args:

            power (float, optional):
                Power of the added noise.
        """
        
        self.power = power
        
    @property
    def power(self) -> float:
        """Power of the added noise.

        Note that for white Gaussian noise the power is equivalent to the
        variance of the added random variable.

        Returns:
            power (float): Power of the added noise.

        Raises:
            ValueError: If the `power` is smaller than zero.
        """

        return self.__power

    @power.setter
    def power(self, value: float) -> None:
        """Set power of the added noise."""

        if value < 0.:
            raise ValueError("Additive white Gaussian noise power must be greater or equal to zero")

        self.__power = value
