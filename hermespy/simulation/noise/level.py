# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod

from hermespy.core import Device, ScalarDimension, Transmitter, Receiver, Serializable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class NoiseLevel(ScalarDimension, Serializable):
    """Base class for all noise level configuration classes."""

    @property
    @abstractmethod
    def level(self) -> float:
        """Scalar level of the represented noise.

        Raises:

            ValueError: If the noise level is negative.
        """
        ...  # pragma: no cover

    @level.setter
    @abstractmethod
    def level(self, value: float) -> None: ...  # pragma: no cover

    @abstractmethod
    def get_power(self) -> float:
        """Power of the noise level.

        Returns: Power in Watt.
        """
        ...  # pragma: no cover

    def __lshift__(self, level: float) -> None:
        """Shorthand for updating the represented noise level

        Args:

            level (float): New noise level.

        Raises:

            ValueError: If the noise level is negative.
        """

        self.level = level


class N0(NoiseLevel):
    """Fixed noise power configuration."""

    yaml_tag = "N0"
    __power: float

    def __init__(self, power: float) -> None:
        """
        Args:
            power (float): Noise power in Watt.
        """

        self.power = power

    @property
    def level(self) -> float:
        """Power of the noise in Watt.

        Raises:

            ValueError: If the noise power is negative.
        """

        return self.power

    @level.setter
    def level(self, value: float) -> None:
        self.power = value

    @property
    def power(self) -> float:
        """Power of the noise level in Watt.

        Raises:

            ValueError: If the noise power is negative.
        """

        return self.__power

    @power.setter
    def power(self, value: float) -> None:
        if value < 0:
            raise ValueError("Noise power must be non-negative.")

        self.__power = value

    def __lshift__(self, power: float) -> None:
        """Shorthand for updating the represented noise power.

        Args:

            power (float): New noise power.
        """

        self.power = power

    def get_power(self) -> float:
        """Power of the noise level.

        Returns: Power in Watt.
        """

        return self.power

    @property
    def title(self) -> str:
        return "Noise Power"


class SNR(NoiseLevel):
    """Signal-to-noise ratio configuration."""

    yaml_tag = "SNR"
    __snr: float
    __reference: Device | Transmitter | Receiver

    def __init__(self, snr: float, reference: Device | Transmitter | Receiver) -> None:
        """
        Args:
            snr (float): Signal-to-noise ratio.
            reference (Device |Transmitter | Receiver): Reference of the noise level.
        """

        # Initialize base class
        NoiseLevel.__init__(self)

        # Initialize class attributes
        self.snr = snr
        self.reference = reference

    @property
    def level(self) -> float:
        """Linear signal power to noise power ratio.

        Raises:

            ValueError: If the ratio negative.
        """
        return self.snr

    @level.setter
    def level(self, value: float) -> None:
        self.snr = value

    def get_power(self) -> float:
        return self.reference.power / self.snr

    @property
    def title(self) -> str:
        return "SNR"

    @property
    def snr(self) -> float:
        """Linear signal power to noise power ratio.

        Raises:

            ValueError: If the ratio negative.
        """
        return self.__snr

    @snr.setter
    def snr(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Linear SNR must be greater than zero.")

        self.__snr = value

    @property
    def reference(self) -> Device | Transmitter | Receiver:
        """Reference to which the noise level is scaled.

        Can be either a device, transmitter or receiver.
        If assigned to a device,
        """
        return self.__reference

    @reference.setter
    def reference(self, value: Device | Transmitter | Receiver) -> None:
        self.__reference = value
