# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Type
from typing_extensions import override

import numpy as np

from .device import Transmission, Transmitter, Receiver, Reception
from .factory import Serializable, SerializationProcess, DeserializationProcess
from .signal_model import Signal
from .state import ReceiveState, TransmitState

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class StaticOperator(object):
    """Base class for static device operators"""

    __num_samples: int  # Number of samples per transmission
    __sampling_rate: float  # Sampling rate of transmission

    def __init__(self, num_samples: int, sampling_rate: float) -> None:
        """
        Args:

            num_samples (int):
                Number of samples per transmission.

            sampling_rate (float):
                Sampling rate of transmission.
        """

        self.__num_samples = num_samples
        self.sampling_rate = sampling_rate

    @property
    def num_samples(self) -> int:
        """Number of samples per transmission.

        Returns: Number of samples.
        """

        return self.__num_samples

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"Sampling rate must be positive (not {value})")

        self.__sampling_rate = value

    @property
    def frame_duration(self) -> float:
        return self.__num_samples / self.sampling_rate

    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.__num_samples, "num_samples")
        process.serialize_floating(self.__sampling_rate, "sampling_rate")


class SilentTransmitter(StaticOperator, Transmitter[Transmission], Serializable):
    """Silent transmitter mock."""

    serialized_attributes = {"num_samples", "sampling_rate"}

    def __init__(self, num_samples: int, sampling_rate: float, *args, **kwargs) -> None:
        """
        Args:

            num_samples (int):
                Number of samples per transmission.

            sampling_rate (float):
                Sampling rate of transmission.
        """

        # Init base classes
        StaticOperator.__init__(self, num_samples, sampling_rate)
        Transmitter.__init__(self, *args, **kwargs)

    @property
    def power(self) -> float:
        return 0.0

    def _transmit(self, device: TransmitState, duration: float) -> Transmission:
        # Compute the number of samples to be transmitted
        num_samples = self.num_samples if duration <= 0.0 else int(duration * self.sampling_rate)

        silence = Signal.Create(
            np.zeros((device.num_digital_transmit_ports, num_samples), dtype=complex),
            sampling_rate=self.sampling_rate,
            carrier_frequency=device.carrier_frequency,
        )

        return Transmission(silence)

    @classmethod
    def Deserialize(
        cls: Type[SilentTransmitter], process: DeserializationProcess
    ) -> SilentTransmitter:
        return SilentTransmitter(
            process.deserialize_integer("num_samples"),
            process.deserialize_floating("sampling_rate"),
        )


class SignalTransmitter(StaticOperator, Transmitter[Transmission], Serializable):
    """Custom signal transmitter."""

    __signal: Signal

    def __init__(self, signal: Signal, *args, **kwargs) -> None:
        """
        Args:

            signal (Signal):
                Signal to be transmittered by the static operator for each transmission.
        """

        # Init base classes
        StaticOperator.__init__(self, signal.num_samples, signal.sampling_rate)
        Transmitter.__init__(self, *args, **kwargs)

        # Init class attributes
        self.__signal = signal

    @property
    def power(self) -> float:
        return np.mean(self.signal.power)

    @property
    def signal(self) -> Signal:
        """Signal to be transmitted by the static operator for each transmission."""

        return self.__signal

    @signal.setter
    def signal(self, value: Signal) -> None:
        self.__signal = value

    def _transmit(self, device: TransmitState, duration) -> Transmission:
        transmitted_signal = self.__signal.copy()

        # Update the transmitted signal's carrier frequency if it is specified as base-band
        if transmitted_signal.carrier_frequency == 0.0:
            transmitted_signal.carrier_frequency = device.carrier_frequency

        transmission = Transmission(transmitted_signal)
        return transmission

    @override
    def _recall_transmission(self, group):
        return super()._recall_transmission(group)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.__signal, "signal")

    @override
    @classmethod
    def Deserialize(
        cls: Type[SignalTransmitter], process: DeserializationProcess
    ) -> SignalTransmitter:
        return SignalTransmitter(process.deserialize_object("signal", Signal))


class SignalReceiver(StaticOperator, Receiver[Reception], Serializable):
    """Custom signal receiver."""

    __expected_power: float

    def __init__(
        self, num_samples: int, sampling_rate: float, expected_power: float = 0.0, *args, **kwargs
    ) -> None:
        # Initialize base classes
        StaticOperator.__init__(self, num_samples, sampling_rate)
        Receiver.__init__(self, *args, **kwargs)

        # Initialize class attributes
        if expected_power < 0.0:
            raise ValueError(f"Expected power must be non-negative (not {expected_power})")
        self.__expected_power = expected_power

    @property
    def energy(self) -> float:
        return self.__expected_power * self.num_samples

    @property
    def power(self) -> float:
        return self.__expected_power

    def _receive(self, signal: Signal, device: ReceiveState) -> Reception:
        received_signal = signal.resample(self.sampling_rate)
        return Reception(received_signal)

    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.num_samples, "num_samples")
        process.serialize_floating(self.sampling_rate, "sampling_rate")
        process.serialize_floating(self.__expected_power, "expected_power")

    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> SignalReceiver:
        return cls(
            process.deserialize_integer("num_samples"),
            process.deserialize_floating("sampling_rate"),
            process.deserialize_floating("expected_power"),
        )
