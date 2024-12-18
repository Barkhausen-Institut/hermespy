# -*- coding: utf-8 -*-
"""
===============
Extra Operators
===============

This module contains convenience operators not part of the standard library.
"""

from __future__ import annotations

import numpy as np
from h5py import Group

from .device import Transmission, Transmitter, Receiver, Reception
from .factory import Serializable
from .signal_model import Signal
from .state import ReceiveState, TransmitState

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
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


class SilentTransmitter(StaticOperator, Transmitter[Transmission], Serializable):
    """Silent transmitter mock."""

    yaml_tag = "SilentTransmitter"
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

    def _recall_transmission(self, group: Group) -> Transmission:
        return Transmission.from_HDF(group)


class SignalTransmitter(StaticOperator, Transmitter[Transmission], Serializable):
    """Custom signal transmitter."""

    yaml_tag = "SignalTransmitter"

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

    def _recall_transmission(self, group: Group) -> Transmission:
        return Transmission.from_HDF(group)


class SignalReceiver(StaticOperator, Receiver[Reception], Serializable):
    """Custom signal receiver."""

    yaml_tag = "SignalReceiver"
    serialized_attributes = {"num_samples", "sampling_rate"}

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

    def _recall_reception(self, group: Group) -> Reception:
        return Reception.from_HDF(group)
