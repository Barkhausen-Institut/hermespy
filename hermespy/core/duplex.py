# -*- coding: utf-8 -*-
"""
=================
Duplex Operation
=================
"""

from __future__ import annotations
from abc import abstractmethod
from h5py import Group
from typing import Generic, Sequence

from .device import Device, OperatorSlot, ReceptionType, Receiver, TransmissionType, Transmitter
from .signal_model import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DuplexOperator(
    Generic[TransmissionType, ReceptionType], Transmitter[TransmissionType], Receiver[ReceptionType]
):
    """Operator binding to both transmit and receive slots of any device."""

    __device: Device | None

    def __init__(
        self,
        device: Device | None = None,
        reference: Device | None = None,
        selected_transmit_ports: Sequence[int] | None = None,
        selected_receive_ports: Sequence[int] | None = None,
        seed: int | None = None,
    ):
        """
        Args:

            device (Device, optional):
                Device the duplex operator operates.
        """

        self.__device = None

        Transmitter.__init__(self, seed, selected_transmit_ports)
        Receiver.__init__(self, seed, reference, selected_receive_ports)

        self.device = device

    @property
    def device(self) -> Device | None:
        """Device this object is assigned to.

        :obj:`None` if this object is currently considered floating / unassigned.
        """

        return self.__device

    @device.setter
    def device(self, value: Device | None) -> None:
        # Abort if the device is already assigned
        if self.__device is value:
            return

        if value is not self.__device and self.__device is not None:
            self.__device.transmitters.remove(self)
            self.__device.receivers.remove(self)

        self.__device = value

        if value is not None:
            value.transmitters.add(self)
            value.receivers.add(self)

        return

    @Transmitter.slot.setter
    def slot(self, value: OperatorSlot[Transmitter]) -> None:
        if value is not None and self.device is not value.device:
            self.device = value.device

        Transmitter.slot.fset(self, value)

    @abstractmethod
    def _transmit(self, duration: float = 0.0) -> TransmissionType: ...  # pragma: no cover

    @abstractmethod
    def _receive(self, signal: Signal) -> ReceptionType: ...  # pragma: no cover

    @property
    @abstractmethod
    def sampling_rate(self) -> float: ...  # pragma: no cover

    @property
    @abstractmethod
    def frame_duration(self) -> float: ...  # pragma: no cover

    @property
    @abstractmethod
    def power(self) -> float: ...  # pragma: no cover

    @abstractmethod
    def _recall_transmission(self, group: Group) -> TransmissionType: ...  # pragma: no cover

    @abstractmethod
    def _recall_reception(self, group: Group) -> ReceptionType: ...  # pragma: no cover
