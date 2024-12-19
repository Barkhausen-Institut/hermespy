# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Sequence, Type

from h5py import Group

from hermespy.channel import Channel, ChannelRealization
from hermespy.core import Device, Drop
from .simulated_device import SimulatedDeviceReception, SimulatedDeviceTransmission

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulatedDrop(Drop[SimulatedDeviceTransmission, SimulatedDeviceReception]):
    """Drop containing all information generated during a simulated wireless scenario transmission,
    channel propagation and reception."""

    __channel_realizations: Sequence[ChannelRealization]

    def __init__(
        self,
        timestamp: float,
        device_transmissions: Sequence[SimulatedDeviceTransmission],
        channel_realizations: Sequence[ChannelRealization],
        device_receptions: Sequence[SimulatedDeviceReception],
    ) -> None:
        """
        Args:

            timestamp (float):
                Time at which the drop was generated.

            device_transmissions (Sequence[DeviceTransmission]):
                Transmitted device information.

            channel_realizations (Sequence[ChannelRealization]):
                Matrix of channel realizations linking the simulated devices.

            device_receptions (Sequence[ProcessedSimulatedDeviceReception]):
                Received device information.
        """

        # Initialize attributes
        self.__channel_realizations = channel_realizations

        # Initialize base class
        Drop.__init__(self, timestamp, device_transmissions, device_receptions)

    @property
    def channel_realizations(self) -> Sequence[ChannelRealization]:
        """Squence of channel realizations linking the simulated devices."""

        return self.__channel_realizations

    def to_HDF(self, group: Group) -> None:
        # Serialize attributes
        group.attrs["timestamp"] = self.timestamp
        group.attrs["num_transmissions"] = self.num_device_transmissions
        group.attrs["num_receptions"] = self.num_device_receptions
        group.attrs["num_devices"] = self.num_device_transmissions

        # Serialize groups
        for t, transmission in enumerate(self.device_transmissions):
            transmission.to_HDF(self._create_group(group, f"transmission_{t:02d}"))

        for r, reception in enumerate(self.device_receptions):
            reception.to_HDF(self._create_group(group, f"reception_{r:02d}"))

        for cr, channel_realization in enumerate(self.channel_realizations):
            realization_group = self._create_group(group, f"channel_realization_{cr:02d}")
            channel_realization.to_HDF(realization_group)

    @classmethod
    def from_HDF(
        cls: Type[SimulatedDrop],
        group: Group,
        devices: Sequence[Device] | None = None,
        channels: Sequence[Channel] | None = None,
    ) -> SimulatedDrop:
        """Recall a simulated drop from a HDF5 group.

        Args:

            group (Group): The HDF5 group containing the serialized drop.
            devices (Sequence[Device], optional): The devices participating in the scenario.
            channels (Sequence[Channel], optional): The channels used in the scenario.
        """

        # Recall attributes
        timestamp = group.attrs.get("timestamp", 0.0)
        num_transmissions = group.attrs.get("num_transmissions", 0)
        num_receptions = group.attrs.get("num_receptions", 0)
        num_devices = group.attrs.get("num_devices", 1)
        _devices = [None] * num_devices if devices is None else devices

        # Recall groups
        transmissions = [
            SimulatedDeviceTransmission.from_HDF(
                group[f"transmission_{t:02d}"], None if d is None else list(d.transmitters)
            )
            for t, d in zip(range(num_transmissions), _devices)
        ]
        receptions = [
            SimulatedDeviceReception.from_HDF(
                group[f"reception_{r:02d}"], None if d is None else list(d.receivers)
            )
            for r, d in zip(range(num_receptions), _devices)
        ]

        channel_realizations: List[ChannelRealization] = []
        for c, channel in enumerate(channels):
            realization = channel.recall_realization(group[f"channel_realization_{c:02d}"])
            channel_realizations.append(realization)

        return SimulatedDrop(timestamp, transmissions, channel_realizations, receptions)
