# -*- coding: utf-8 -*-
"""
====
Drop
====
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Generic, Type, TypeVar

from h5py import Group

from .device import Device, DeviceReception, DeviceTransmission, DRT, DTT
from .factory import HDFSerializable
from .signal_model import Signal
from .monte_carlo import Artifact

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Drop(Generic[DTT, DRT], HDFSerializable):
    """Drop containing the information transmitted and received by all devices
    within a scenario."""

    __timestamp: float  # Time at which the drop was generated
    __device_transmissions: Sequence[DTT]  # Transmitted device information
    __device_receptions: Sequence[DRT]  # Received device information

    def __init__(
        self,
        timestamp: float,
        device_transmissions: Sequence[DTT],
        device_receptions: Sequence[DRT],
    ) -> None:
        """
        Args:

            timestamp (float):
                Time at which the drop was generated.

            device_transmissions (Sequence[DTT]):
                Transmitted device information.

            device_receptions (Sequence[DRT]):
                Received device information.
        """

        self.__timestamp = timestamp
        self.__device_transmissions = device_transmissions
        self.__device_receptions = device_receptions

    @property
    def timestamp(self) -> float:
        """Time at which the drop was generated."""

        return self.__timestamp

    @property
    def device_transmissions(self) -> Sequence[DTT]:
        """Transmitted device information within this drop."""

        return self.__device_transmissions

    @property
    def device_receptions(self) -> Sequence[DRT]:
        """Received device information within this drop."""

        return self.__device_receptions

    @property
    def num_device_transmissions(self) -> int:
        """Number of device transmissions within this drop."""

        return len(self.__device_transmissions)

    @property
    def num_device_receptions(self) -> int:
        """Number of device receptions within this drop."""

        return len(self.__device_receptions)

    @property
    def operator_inputs(self) -> Sequence[Sequence[Signal]]:
        """Signals feeding into device's operators during reception.

        Returns: Operator inputs.
        """

        return [reception.operator_inputs for reception in self.device_receptions]

    @classmethod
    def from_HDF(cls: Type[Drop], group: Group, devices: Sequence[Device] | None = None) -> Drop:
        # Recall attributes
        timestamp = group.attrs.get("timestamp", 0.0)
        num_transmissions: int = group.attrs.get("num_transmissions", 0)
        num_receptions: int = group.attrs.get("num_receptions", 0)

        if devices is None:
            transmissions = [
                DeviceTransmission.from_HDF(group[f"transmission_{t:02d}"])
                for t in range(num_transmissions)
            ]
            receptions = [
                DeviceReception.from_HDF(group[f"reception_{r:02d}"]) for r in range(num_receptions)
            ]
        else:
            transmissions = [
                device.recall_transmission(group[f"transmission_{t:02d}"])
                for t, device in zip(range(num_transmissions), devices)
            ]
            receptions = [
                device.recall_reception(group[f"reception_{r:02d}"])
                for r, device in zip(range(num_receptions), devices)
            ]

        drop = cls(
            timestamp=timestamp, device_transmissions=transmissions, device_receptions=receptions
        )
        return drop

    def to_HDF(self, group: Group) -> None:
        # Serialize groups
        for t, transmission in enumerate(self.device_transmissions):
            transmission.to_HDF(group.create_group(f"transmission_{t:02d}"))

        for r, reception in enumerate(self.device_receptions):
            reception.to_HDF(group.create_group(f"reception_{r:02d}"))

        # Serialize attributes
        group.attrs["timestamp"] = self.timestamp
        group.attrs["num_transmissions"] = self.num_device_transmissions
        group.attrs["num_receptions"] = self.num_device_receptions


DropType = TypeVar("DropType", bound=Drop)
"""Type of a drop."""


class EvaluatedDrop(Drop):
    """Drop containing the information transmitted and received by all devices
    within a scenario as well as their evaluations."""

    # Evaluation artifacts generated for this drop.
    __artifacts: Sequence[Artifact]

    def __init__(
        self,
        timestamp: float,
        device_transmissions: Sequence[DeviceTransmission],
        device_receptions: Sequence[DeviceReception],
        artifacts: Sequence[Artifact],
    ) -> None:
        """
        Args:

            timestamp (float):
                Time at which the drop was generated.

            device_transmissions (Sequence[DeviceTransmission]):
                Transmitted device information.

            device_receptions (Sequence[DeviceReception]):
                Received device information.

            artifacts (Sequence[Artifact]):
                Evaluation artifacts generated for this scenario drop.
        """

        Drop.__init__(self, timestamp, device_transmissions, device_receptions)
        self.__artifacts = artifacts

    @property
    def num_artifacts(self) -> int:
        """Number of evaluation artifacts.

        Returns: The number of artifacts.
        """

        return len(self.__artifacts)

    @property
    def artifacts(self) -> Sequence[Artifact]:
        """Evaluation artifacts generated from the drop's data.

        Returns: List of artifacts.
        """

        return self.__artifacts
