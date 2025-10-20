# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Type, Sequence
from typing_extensions import override

import numpy as np

from .device import Transmission, Transmitter, Receiver, Reception
from .factory import Serializable, SerializationProcess, DeserializationProcess
from .signal_model import Signal
from .state import ReceiveState, TransmitState

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class StaticOperator(object):
    """Base class for static device operators"""

    __num_samples: int  # Number of samples per transmission

    def __init__(self, num_samples: int) -> None:
        """
        Args:

            num_samples (int):
                Number of samples per transmission.
        """

        self.__num_samples = num_samples

    @property
    def num_samples(self) -> int:
        """Number of samples per transmission.

        Returns: Number of samples.
        """

        return self.__num_samples

    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.__num_samples, "num_samples")


class SilentTransmitter(StaticOperator, Transmitter[Transmission], Serializable):
    """Silent transmitter mock."""

    def __init__(
        self, num_samples: int, selected_transmit_ports: Sequence[int] | None = None
    ) -> None:
        """
        Args:

            num_samples: Number of samples per transmission.
            selected_transamit_ports: Digital transmit prots this dsp algorithm operates on.

        """

        # Init base classes
        StaticOperator.__init__(self, num_samples)
        Transmitter.__init__(self, selected_transmit_ports=selected_transmit_ports)

    @property
    def power(self) -> float:
        return 0.0

    @override
    def _transmit(self, state: TransmitState, duration: float) -> Transmission:
        # Compute the number of samples to be transmitted
        num_samples = self.num_samples if duration <= 0.0 else int(duration * state.sampling_rate)

        silence = Signal.Create(
            np.zeros((state.num_transmit_dsp_ports, num_samples), dtype=complex),
            sampling_rate=state.sampling_rate,
            carrier_frequency=state.carrier_frequency,
        )

        return Transmission(silence)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.num_samples, "num_samples")
        if self.selected_transmit_ports is not None:
            process.serialize_array(
                np.asarray(self.selected_transmit_ports, np.int64), "selected_transmit_ports"
            )

    @classmethod
    @override
    def Deserialize(
        cls: Type[SilentTransmitter], process: DeserializationProcess
    ) -> SilentTransmitter:
        transmit_ports = process.deserialize_array("selected_transmit_ports", np.int64, None)
        return SilentTransmitter(
            process.deserialize_integer("num_samples"),
            transmit_ports.flatten().tolist() if transmit_ports is not None else None,
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
        StaticOperator.__init__(self, signal.num_samples)
        Transmitter.__init__(self, *args, **kwargs)

        # Init class attributes
        self.__signal = signal

    @property
    def power(self) -> float:
        return float(np.mean(self.signal.power))

    @property
    def signal(self) -> Signal:
        """Signal to be transmitted by the static operator for each transmission."""

        return self.__signal

    @signal.setter
    def signal(self, value: Signal) -> None:
        self.__signal = value

    def frame_duration(self, bandwidth: float) -> float:
        return self.num_samples / bandwidth

    @override
    def _transmit(self, state: TransmitState, duration: float) -> Transmission:
        transmitted_signal = self.__signal.copy()

        # Update the transmitted signal's sampling rate
        transmitted_signal.sampling_rate = state.sampling_rate

        transmission = Transmission(transmitted_signal)
        return transmission

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
        self,
        num_samples: int,
        selected_receive_ports: Sequence[int] | None = None,
        expected_power: float = 0.0,
    ) -> None:
        # Initialize base classes
        StaticOperator.__init__(self, num_samples)
        Receiver.__init__(self, selected_receive_ports=selected_receive_ports)

        # Initialize class attributes
        if expected_power < 0.0:
            raise ValueError(f"Expected power must be non-negative (not {expected_power})")
        self.__expected_power = expected_power

    @property
    def energy(self) -> float:
        return self.__expected_power * self.num_samples

    def frame_duration(self, bandwidth: float) -> float:
        return self.num_samples / bandwidth

    @property
    @override
    def power(self) -> float:
        return self.__expected_power

    @override
    def samples_per_frame(self, bandwidth: float, oversampling_factor: int) -> int:
        return self.num_samples

    @override
    def _receive(self, signal: Signal, state: ReceiveState) -> Reception:
        return Reception(signal)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.num_samples, "num_samples")
        if self.selected_receive_ports is not None:
            process.serialize_array(
                np.asarray(self.selected_receive_ports, np.float64), "selected_receive_ports"
            )
        process.serialize_floating(self.__expected_power, "expected_power")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> SignalReceiver:
        selected_receive_ports = process.deserialize_array("selected_receive_ports", np.int64, None)
        return cls(
            process.deserialize_integer("num_samples"),
            (
                selected_receive_ports.flatten().tolist()
                if selected_receive_ports is not None
                else None
            ),
            process.deserialize_floating("expected_power"),
        )
