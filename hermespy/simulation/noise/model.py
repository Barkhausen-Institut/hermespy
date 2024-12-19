# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from typing import Generic, TypeVar

from hermespy.core import RandomNode, RandomRealization, Serializable, Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class NoiseRealization(RandomRealization):
    """Realization of a noise model"""

    __power: float

    def __init__(self, noise: NoiseModel, power: float) -> None:
        """
        Args:

            noise (Noise): Noise model to be realized.
            power (float): Power indicator of the noise model.
        """

        # Validate attributes
        if power < 0:
            raise ValueError("Noise power of a noise realization must be non-negative.")

        # Initialize base class
        RandomRealization.__init__(self, noise)

        # Initialize attributes
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

        Returns: Signal model with added noise.
        """
        ...  # pragma: no cover


NRT = TypeVar("NRT", bound=NoiseRealization)
"""Type of noise realization"""


class NoiseModel(RandomNode, Generic[NRT]):
    """Noise modeling base class."""

    def __init__(self, seed: int | None = None) -> None:
        """
        Args:

            seed (int, optional):
                Random seed for initializating the pseud-random number generator.
        """

        RandomNode.__init__(self, seed=seed)

    @abstractmethod
    def realize(self, power: float) -> NRT:
        """Realize the noise model.

        Args:

            power (float, optional):
                Power of the added noise in Watt.

        Returns: Noise model realization.
        """
        ...  # pragma: no cover

    def add_noise(self, signal: Signal, power: float) -> Signal:
        """Add noise to a signal model.

        Args:

            signal (Signal):
                The signal to which the noise should be added.

            power (float):
                Power of the added noise in Watt.

        Returns: Signal model with added noise.
        """

        realization = self.realize(power)
        return realization.add_to(signal)


class AWGNRealization(NoiseRealization):
    """Realization of additive white Gaussian noise"""

    def add_to(self, signal: Signal) -> Signal:
        # Create random number generator
        rng = self.generator()

        noise_samples = (0.5 * self.power) ** 0.5 * (
            rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
        )

        noisy_signal = signal.copy()
        for block in noisy_signal:
            block += noise_samples
        noisy_signal.noise_power = self.power

        return noisy_signal


class AWGN(Serializable, NoiseModel[AWGNRealization]):
    """Additive White Gaussian Noise."""

    yaml_tag = "AWGN"
    property_blacklist = {"random_mother"}

    def __init__(self, seed: int | None = None) -> None:
        """
        Args:

            seed (int, optional):
                Random seed for initializating the pseud-random number generator.
        """

        NoiseModel.__init__(self, seed=seed)

    def realize(self, power: float) -> AWGNRealization:
        return AWGNRealization(self, power)
