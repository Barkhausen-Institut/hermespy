# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from sys import maxsize
from typing import Generic, Type, TypeVar
from typing_extensions import override

from hermespy.core import (
    RandomNode,
    RandomRealization,
    Serializable,
    SerializationProcess,
    DeserializationProcess,
)
from hermespy.core.signal_model import ST

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class NoiseRealization(RandomRealization):
    """Realization of a noise model"""

    __power: float

    def __init__(self, power: float, seed: int) -> None:
        """
        Args:

            power: Power indicator of the noise model.
            seed: Seed for the random number generator.
        """

        # Validate attributes
        if power < 0.0:
            raise ValueError("Noise power of a noise realization must be non-negative.")

        # Initialize base class
        RandomRealization.__init__(self, seed)

        # Initialize attributes
        self.__power = power

    @property
    def power(self) -> float:
        """Power of the noise realization.

        Returns: Power in Watt.
        """

        return self.__power

    @abstractmethod
    def add_to(self, signal: ST) -> ST:
        """
        Args:
            signal: The signal to which the noise should be added.

        Returns:
            Signal model with added noise.
        """
        ...  # pragma: no cover

    @override
    def serialize(self, process: SerializationProcess) -> None:
        RandomRealization.serialize(self, process)
        process.serialize_floating(self.power, "power")

    @classmethod
    @override
    def _DeserializeParameters(cls, process: DeserializationProcess) -> dict[str, object]:
        params = RandomRealization._DeserializeParameters(process)
        params["power"] = process.deserialize_floating("power")
        return params

    @classmethod
    @override
    def Deserialize(
        cls: Type[NoiseRealization], process: DeserializationProcess
    ) -> NoiseRealization:
        params = cls._DeserializeParameters(process)
        return cls(**params)  # type: ignore[arg-type]


NRT = TypeVar("NRT", bound=NoiseRealization)
"""Type of noise realization"""


class NoiseModel(Serializable, RandomNode, Generic[NRT]):
    """Noise modeling base class."""

    def __init__(self, seed: int | None = None) -> None:
        """
        Args:

            seed:
                Random seed for initializating the pseud-random number generator.
        """

        RandomNode.__init__(self, seed=seed)

    @abstractmethod
    def realize(self, power: float) -> NRT:
        """Realize the noise model.

        Args:

            power:
                Power of the added noise in Watt.

        Returns: Noise model realization.
        """
        ...  # pragma: no cover

    def add_noise(self, signal: ST, power: float) -> ST:
        """Add noise to a signal model.

        Args:

            signal:
                The signal to which the noise should be added.

            power:
                Power of the added noise in Watt.

        Returns: Signal model with added noise.
        """

        realization = self.realize(power)
        return realization.add_to(signal)


class AWGNRealization(NoiseRealization):
    """Realization of additive white Gaussian noise"""

    def add_to(self, signal: ST) -> ST:
        # Abort if the power is zero
        if self.power == 0.0:
            return signal

        # Create random number generator
        rng = self.generator()

        noise_samples = (0.5 * self.power) ** 0.5 * (
            rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
        )

        noisy_signal = signal.copy()
        for block in noisy_signal.blocks:
            block += noise_samples  # type: ignore
        noisy_signal.noise_power = self.power

        return noisy_signal


class AWGN(NoiseModel[AWGNRealization]):
    """Additive White Gaussian Noise."""

    def __init__(self, seed: int | None = None) -> None:
        """
        Args:

            seed:
                Random seed for initializating the pseud-random number generator.
        """

        NoiseModel.__init__(self, seed)

    @override
    def realize(self, power: float) -> AWGNRealization:
        return AWGNRealization(power, int(self._rng.integers(0, maxsize)))

    @override
    def serialize(self, process: SerializationProcess) -> None:
        return

    @override
    @classmethod
    def Deserialize(cls: Type[AWGN], process: DeserializationProcess) -> AWGN:
        return cls()
