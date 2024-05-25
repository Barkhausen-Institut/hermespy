# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import List, Sequence, Type, TYPE_CHECKING

from h5py import Group

from hermespy.channel import Channel, ChannelRealization
from hermespy.core import Drop
from .simulated_device import SimulatedDeviceReception, SimulatedDeviceTransmission

if TYPE_CHECKING:
    from hermespy.simulation import SimulationScenario  # pragma: no cover

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulatedDrop(Drop):
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

        self.__channel_realizations = channel_realizations
        Drop.__init__(self, timestamp, device_transmissions, device_receptions)

    @property
    def channel_realizations(self) -> Sequence[ChannelRealization]:
        """Squence of channel realizations linking the simulated devices."""

        return self.__channel_realizations

    def to_HDF(self, group: Group) -> None:
        num_devices = self.num_device_transmissions

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

        i = 0
        for d_out in range(num_devices):
            for d_in in range(d_out + 1):
                realization_group = self._create_group(group, f"channel_realization_{i:02d}")
                self.channel_realizations[i].to_HDF(realization_group)
                i += 1

    @classmethod
    def from_HDF(
        cls: Type[SimulatedDrop], group: Group, scenario: SimulationScenario | None = None
    ) -> SimulatedDrop:
        # Require a scenario to be specified
        # Maybe there is a workaround possible since this is validates object-oriented principles
        if scenario is None:
            raise ValueError("Simulation drops must be deserialized with a scenario instance")

        # Recall attributes
        timestamp = group.attrs.get("timestamp", 0.0)
        num_transmissions = group.attrs.get("num_transmissions", 0)
        num_receptions = group.attrs.get("num_receptions", 0)
        num_devices = group.attrs.get("num_devices", 1)

        # Assert that the scenario parameters match the serialization
        if scenario.num_devices != num_devices:
            raise ValueError(
                f"Number of scenario devices does not match the serialization ({scenario.num_devices} != {num_devices})"
            )

        # Recall groups
        transmissions = [
            SimulatedDeviceTransmission.from_HDF(group[f"transmission_{t:02d}"])
            for t in range(num_transmissions)
        ]
        receptions = [
            SimulatedDeviceReception.from_HDF(group[f"reception_{r:02d}"])
            for r in range(num_receptions)
        ]

        channel_realizations: List[ChannelRealization] = []
        i = 0
        for device_beta_idx in range(num_devices):
            for device_alpha_idx in range(device_beta_idx + 1):

                # Recall the channel realization
                channel: Channel = scenario.channels[device_beta_idx, device_alpha_idx]
                realization = channel.recall_realization(group[f"channel_realization_{i:02d}"])
                channel_realizations.append(realization)
                i += 1

        return SimulatedDrop(timestamp, transmissions, channel_realizations, receptions)
