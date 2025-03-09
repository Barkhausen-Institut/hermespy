# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Sequence, Type

from hermespy.channel import ChannelRealization
from hermespy.core import DeserializationProcess, Drop, SerializationProcess
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

    def serialize(self, process: SerializationProcess) -> None:
        Drop.serialize(self, process)
        process.serialize_object_sequence(self.__channel_realizations, "channel_realizations")

    @classmethod
    def Deserialize(cls: Type[SimulatedDrop], process: DeserializationProcess) -> SimulatedDrop:
        drop = Drop.Deserialize(process)
        channel_realizations = process.deserialize_object_sequence(
            "channel_realiations", ChannelRealization
        )
        return SimulatedDrop(
            drop.timestamp, drop.device_transmissions, channel_realizations, drop.device_receptions
        )
