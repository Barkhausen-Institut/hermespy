# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np

from hermespy.beamforming import BeamFocus
from hermespy.core import Direction, State
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
