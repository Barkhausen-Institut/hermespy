# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from typing import Type
from typing_extensions import override

from hermespy.core import (
    Device,
    ScalarDimension,
    Transmitter,
    Receiver,
    Serializable,
    SerializationProcess,
    DeserializationProcess,
)
from hermespy.channel import Channel, ChannelSample

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
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

            level: New noise level.

        Raises:

            ValueError: If the noise level is negative.
        """

        self.level = level


class N0(NoiseLevel):
    """Fixed noise power configuration."""

    __power: float

    def __init__(self, power: float) -> None:
        """
        Args:
            power: Noise power in Watt.
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

            power: New noise power.
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

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.power, "power")

    @override
    @classmethod
    def Deserialize(cls: Type[N0], process: DeserializationProcess) -> N0:
        return cls(process.deserialize_floating("power"))


class SNR(NoiseLevel):
    """Signal-to-noise ratio configuration."""

    __snr: float
    __reference: Device | Transmitter | Receiver
    __expected_channel_scale: float

    def __init__(
        self, snr: float, reference: Device | Transmitter | Receiver, channel: Channel | None = None
    ) -> None:
        """
        Args:
            snr:
                Expected signal-to-noise ratio.

            reference:
                Reference of the noise level, i.e. with which power or energy was the signal generated.

            channel:
                Channel instance over which the signal was propagated.
                For channel models that consider propagation losses the noise power is scaled accordingly.
        """

        # Initialize base class
        NoiseLevel.__init__(self)

        # Initialize class attributes
        self.snr = snr
        self.reference = reference
        self.__expected_channel_scale = 1.0

        if channel is not None:
            channel.add_sample_hook(self.__update_expected_channel_scale)

    def __update_expected_channel_scale(self, sample: ChannelSample) -> None:
        """Update the expected channel scale.

        Args:
            sample: Channel sample.
        """

        self.__expected_channel_scale = sample.expected_energy_scale

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
        return self.reference.power / self.snr * self.__expected_channel_scale

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

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.snr, "snr")
        process.serialize_object(self.reference, "reference")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> SNR:
        return cls(
            process.deserialize_floating("snr"), process.deserialize_object("reference", Device)
        )
