# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from hermespy.beamforming import BeamFocus
from hermespy.core import Direction
from .simulated_device import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DeviceFocus(BeamFocus):
    """Focus point targeting a device."""

    yaml_tag = "DeviceFocus"
    __transmitting_device: SimulatedDevice  # Device focused by the beamformer
    __receiving_device: SimulatedDevice  # Device being focused
    __timestamp: float  # Time in the devices' trajectory the focus is applied

    def __init__(
        self,
        transmitting_device: SimulatedDevice,
        receiving_device: SimulatedDevice,
        timestamp: float = 0.0,
    ) -> None:
        """
        Args:

            transmitting_device (Device): Device focusing the `receiving_device`.
            receiving_device (Device): Device being focused.
            timestamp (float): At which time in the devices' trajectory the focus is applied.
        """

        # Initialize base class
        BeamFocus.__init__(self)

        # Initialize class members
        self.__transmitting_device = transmitting_device
        self.__receiving_device = receiving_device
        self.__timestamp = timestamp

    def copy(self) -> DeviceFocus:
        return DeviceFocus(self.__transmitting_device, self.__receiving_device, self.__timestamp)

    @property
    def transmitting_device(self) -> SimulatedDevice:
        """Device focusing the receiving device."""

        return self.__transmitting_device

    @transmitting_device.setter
    def transmitting_device(self, device: SimulatedDevice) -> None:
        self.__transmitting_device = device

    @property
    def receiving_device(self) -> SimulatedDevice:
        """Device focused by the beamformer."""

        return self.__receiving_device

    @receiving_device.setter
    def receiving_device(self, device: SimulatedDevice) -> None:
        self.__receiving_device = device

    @property
    def timestamp(self) -> float:
        """Time in the devices' trajectory the focus is applied."""

        return self.__timestamp

    @timestamp.setter
    def timestamp(self, timestamp: float) -> None:
        self.__timestamp = timestamp

    @property
    def spherical_angles(self) -> np.ndarray:
        transmitter_position, receiver_position = (
            d.trajectory.sample(self.timestamp).pose.translation
            for d in (self.transmitting_device, self.receiving_device)
        )

        direction = Direction.From_Cartesian(receiver_position - transmitter_position, True)
        return direction.to_spherical()
