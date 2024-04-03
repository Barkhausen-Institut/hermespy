# -*- coding: utf-8 -*-


from __future__ import annotations
from abc import abstractmethod
from typing import Generic, TypeVar

from hermespy.core import RandomNode, RandomRealization, Serializable, Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class NoiseRealization(RandomRealization):
    """Realization of a noise model"""

    __power: float

    def __init__(self, noise: Noise, power: float) -> None:
        """
        Args:

            noise (Noise): Noise model to be realized.
            power (float): Power indicator of the noise model.
        """

        self.__power = power
        RandomRealization.__init__(self, noise)

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

        Returns: Signal model with added noise.
        """
        ...  # pragma: no cover


NoiseRealizationType = TypeVar("NoiseRealizationType", bound=NoiseRealization)
"""Type of noise realization"""


class Noise(RandomNode, Generic[NoiseRealizationType]):
    """Noise modeling base class."""

    __power: float  # Power of the added noise

    def __init__(self, power: float = 0.0, seed: int | None = None) -> None:
        """
        Args:

            power (float, optional):
                Power of the added noise.

            seed (int, optional):
                Random seed for initializating the pseud-random number generator.
        """

        self.power = power
        RandomNode.__init__(self, seed=seed)

    @abstractmethod
    def realize(self, power: float | None = None) -> NoiseRealizationType:
        """Realize the noise model.

        Args:

            power (float, optional):
                Power of the added noise.
                If not specified, the class :meth:`Noise.power` configuration will be applied.

        Returns: Noise model realization.
        """
        ...  # pragma: no cover

    def add(self, signal: Signal, realization: NoiseRealizationType | None = None) -> Signal:
        """Add noise to a signal model.

        Args:

            signal (Signal):
                The signal to which the noise should be added.

            realization (NoiseRealizationType):
                Realization of the noise model to be added to `signal`.

        Returns: Signal model with added noise.
        """

        realization = self.realize() if realization is None else realization
        return realization.add_to(signal)

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

    def add_to(self, signal: Signal) -> Signal:
        # Create random number generator
        rng = self.generator()

        noise_samples = (
            rng.normal(0, self.power**0.5, signal.samples.shape)
            + 1j * rng.normal(0, self.power**0.5, signal.samples.shape)
        ) / 2**0.5

        noisy_signal = signal.copy()
        noisy_signal.samples += noise_samples
        noisy_signal.noise_power = self.power

        return noisy_signal


class AWGN(Serializable, Noise[AWGNRealization]):
    """Additive White Gaussian Noise."""

    yaml_tag = "AWGN"
    property_blacklist = {"random_mother"}

    def __init__(self, power: float = 0.0, seed: int | None = None) -> None:
        """
        Args:

            power (float, optional):
                Power of the added noise.
        """

        Noise.__init__(self, power=power, seed=seed)

    def realize(self, power: float | None = None) -> AWGNRealization:
        power = self.power if power is None else power
        return AWGNRealization(self, power)
