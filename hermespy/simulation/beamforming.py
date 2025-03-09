# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Type
from typing_extensions import override

import numpy as np

from hermespy.beamforming import BeamFocus
from hermespy.core import DeserializationProcess, Direction, SerializationProcess, State
from .simulated_device import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DeviceFocus(BeamFocus):
    """Focus point targeting a device."""

    __focused_device: SimulatedDevice  # Device focused by the beamformer

    def __init__(self, focused_device: SimulatedDevice) -> None:
        """
        Args:

            focused_device (SimulatedDevice): Device being focused.
        """

        # Initialize base class
        BeamFocus.__init__(self)

        # Initialize class members
        self.__focused_device = focused_device

    def copy(self) -> DeviceFocus:
        return DeviceFocus(self.focused_device)

    @property
    def focused_device(self) -> SimulatedDevice:
        """Device being focused by the beamformer."""

        return self.__focused_device

    @focused_device.setter
    def focused_device(self, value: SimulatedDevice) -> None:
        self.__focused_device = value

    def spherical_angles(self, device: State) -> np.ndarray:
        transmitter_position = device.position
        receiver_position = self.focused_device.trajectory.sample(device.timestamp).pose.translation

        direction = Direction.From_Cartesian(receiver_position - transmitter_position, True)
        return direction.to_spherical()

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.focused_device, "focus")

    @classmethod
    @override
    def Deserialize(cls: Type[DeviceFocus], process: DeserializationProcess) -> DeviceFocus:
        focused_device = process.deserialize_object("focus", SimulatedDevice)
        return DeviceFocus(focused_device)
