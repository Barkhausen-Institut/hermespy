# -*- coding: utf-8 -*-
"""
=================
Duplex Operation
=================
"""

from __future__ import annotations
from abc import abstractmethod
from h5py import Group
from typing import Generic

from .channel_state_information import ChannelStateInformation
from .device import Device, OperatorSlot, ReceptionType, Receiver, SNRType, TransmissionType, Transmitter
from .signal_model import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DuplexOperator(Transmitter[TransmissionType], Receiver[ReceptionType], Generic[TransmissionType, ReceptionType]):
    """Operator binding to both transmit and receive slots of any device."""

    __device: Device | None

    def __init__(self, device: Device | None = None, reference: Device | None = None, seed: int | None = None):
        """
        Args:

            device (Device, optional):
                Device the duplex operator operates.
        """

        Transmitter.__init__(self, seed=seed)
        Receiver.__init__(self, seed=seed, reference=reference)

        self.__device = None
        self.device = device

    @property
    def device(self) -> Device | None:
        """Device this operator is operating.

        Returns:
            Handle to the operated device.
            `None` if the operator is currently operating no device and considered floating.
        """

        return self.__device

    @device.setter
    def device(self, value: Device | None) -> None:
        """Set the device this operator is operating."""

        if self.__device is not None:
            self.__device.transmitters.remove(self)
            self.__device.receivers.remove(self)

        self.__device = value

        if value is not None:
            value.transmitters.add(self)
            value.receivers.add(self)

    @Transmitter.slot.setter
    def slot(self, value: OperatorSlot[Transmitter]) -> None:
        if value is not None and self.device is not value.device:
            self.device = value.device

        Transmitter.slot.fset(self, value)

    @property
    def csi(self) -> ChannelStateInformation | None:
        return Receiver.csi.fget(self)  # type: ignore

    @abstractmethod
    def _transmit(self, duration: float = 0.0) -> TransmissionType:
        ...  # pragma: no cover

    @abstractmethod
    def _receive(self, signal: Signal, csi: ChannelStateInformation) -> ReceptionType:
        ...  # pragma: no cover

    @property
    @abstractmethod
    def sampling_rate(self) -> float:
        ...  # pragma: no cover

    @property
    @abstractmethod
    def frame_duration(self) -> float:
        ...  # pragma: no cover

    @abstractmethod
    def _noise_power(self, strength: float, snr_type: SNRType) -> float:
        ...  # pragma: no cover

    @abstractmethod
    def _recall_transmission(self, group: Group) -> TransmissionType:
        ...  # pragma: no cover

    @abstractmethod
    def _recall_reception(self, group: Group) -> ReceptionType:
        ...  # pragma: no cover
