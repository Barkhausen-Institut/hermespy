# -*- coding: utf-8 -*-
"""
==============
Noise Modeling
==============
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Generic, Optional, TypeVar

import numpy as np

from hermespy.core import RandomNode, Serializable, Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class NoiseRealization(object):
    """Realization of a noise model"""

    __power: float

    def __init__(self, power: float) -> None:
        """
        Args:

            power (float): Power of the noise realization.
        """

        self.__power = power

    @property
    def power(self) -> float:
        """Power of the noise realization.

        Returns: Power in Watt.
        """

        return self.__power
    
    @abstractmethod
    def add_to(self, signal: Signal) -> Signal:
        """
        Args:
            signal (Signal):
                The signal to which the noise should be added.

            realization (NoiseRealizationType):
                Realization of the noise model to be added to `signal`.

            power (float, optional)
                Power of the added noise.
        """
        ...  # pragma no cover


NoiseRealizationType = TypeVar('NoiseRealizationType', bound=NoiseRealization)
"""Type of noise realization"""


class Noise(RandomNode, Generic[NoiseRealizationType]):
    """Noise modeling base class."""

    __power: float  # Power of the added noise

    def __init__(self, power: float = 0.0, seed: Optional[int] = None) -> None:
        """
        Args:

            power (float, optional):
                Power of the added noise.
        """

        self.power = power
        RandomNode.__init__(self, seed=seed)

    @abstractmethod
    def realize(self,
                signal: Signal,
                power: Optional[float] = None) -> NoiseRealizationType:
        """Realize the noise model.

        Args:

            signal (Signal):
                Signal model for which to realize noise.

            power (float, optional):
                Power of the added noise.
                If not specified, the class :meth:`.power` configuration will be applied.

        Returns: Noise model realization.
        """
        ...  # pragma no cover

    def add(self,
            signal: Signal,
            realization: NoiseRealizationType) -> Signal:
        """Add noise to a signal model.

        Args:

            signal (Signal):
                The signal to which the noise should be added.

            realization (NoiseRealizationType):
                Realization of the noise model to be added to `signal`.

        Returns: Signal model with added noise.
        """

        return realization.add_to(signal, self.power)

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

        if value < 0.0:
            raise ValueError("Additive white Gaussian noise power must be greater or equal to zero")

        self.__power = value


class AWGNRealization(NoiseRealization):
    """Realization of additive white Gaussian noise"""

    __samples: np.ndarray

    def __init__(self, samples: np.ndarray, power: float) -> None:
        """
        Args:

            samples (np.ndarray): Samples of the additive noise.
            power (float): Power of the additive noise.
        """

        self.__samples = samples
        NoiseRealization.__init__(self, power)

    def add_to(self, signal: Signal) -> Signal:

        noisy_signal = signal.copy()
        noisy_signal.samples += self.__samples


class AWGN(Serializable, Noise[AWGNRealization]):
    """Additive White Gaussian Noise."""

    yaml_tag = "AWGN"
    property_blacklist = {"random_mother"}

    def __init__(self, power: float = 0.0, seed: Optional[int] = None) -> None:
        """
        Args:

            power (float, optional):
                Power of the added noise.
        """

        Noise.__init__(self, power=power, seed=seed)

    def realize(self,
                signal: Signal,
                power: Optional[float] = None) -> AWGNRealization:

        power = self.power if power is None else power
        noise_samples = (self._rng.normal(0, power**0.5, signal.samples.shape) + 1j * self._rng.normal(0, power**0.5, signal.samples.shape)) / 2**0.5

        return AWGNRealization(noise_samples, power)
