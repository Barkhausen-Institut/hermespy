# -*- coding: utf-8 -*-
"""
=================
Wireless Scenario
=================
"""

from __future__ import annotations
import numpy as np
import numpy.random as rnd
from typing import Generic, List, Type, Optional
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node
from collections.abc import Iterable

from .device import DeviceType, Transmitter, Receiver, Operator
from .random_node import RandomNode


__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "3.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Scenario(RandomNode, Generic[DeviceType]):
    """A simulation scenario.

    Scenarios consist of several devices transmitting and receiving electromagnetic signals.
    Each device can be operated by multiple operators simultaneously.
    """

    __slots__ = ['__devices']

    __devices: List[DeviceType]         # Registered devices within this scenario.
    __drop_duration: float              # Drop duration in seconds.

    def __init__(self,
                 seed: Optional[int] = None) -> None:
        """
        Args:

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.
        """

        RandomNode.__init__(self, seed=seed)
        self.__devices = []
        self.drop_duration = 0.

    def add_device(self, device: DeviceType) -> None:
        """Add new device to scenario.

        Args:

            device (DeviceType):
                New device to be added to the scenario.

        Raises `ValueError` if the device already exists.
        """

        if self.device_registered(device):
            raise ValueError("Error trying to add an already registered device to a scenario")

        # Add device to internal device list
        self.__devices.append(device)
        
        # Register scenario at the device
        device.random_mother = self
        device.scenario = self

    def device_registered(self, device: DeviceType) -> bool:
        """Check if an device is registered in this scenario.

        Args:
            device (DeviceType): The device to be checked.

        Returns:
            bool: The device's registration status.
        """

        return device in self.__devices

    @property
    def devices(self) -> List[DeviceType]:
        """Devices registered in this scenario.

        Returns:
            List[DeviceType]: List of devices.
        """

        return self.__devices.copy()

    @property
    def num_devices(self) -> int:
        """Number of devices in this scenario.

        Returns:
            int: Number of devices
        """

        return len(self.__devices)

    @property
    def transmitters(self) -> List[Transmitter]:
        """All transmitting operators within this scenario.

        Returns:
            List[Transmitter]: List of all transmitting operators.
        """

        transmitters: List[Transmitter] = []

        for device in self.__devices:
            transmitters.extend(device.transmitters)

        return transmitters
    
    @property
    def receivers(self) -> List[Receiver]:
        """All receiving operators within this scenario.

        Returns:
            List[Receiver]: List of all transmitting operators.
        """

        receivers: List[Receiver] = []

        for device in self.__devices:
            receivers.extend(device.receivers)

        return receivers

    @property
    def num_receivers(self) -> int:
        """Number of receiving operators within this scenario.

        Returns:
            int: The number of receivers.
        """

        num = 0
        for device in self.__devices:
            num += device.receivers.num_operators

        return num

    @property
    def num_transmitters(self) -> int:
        """Number of transmitting operators within this scenario.

        Returns:
            int: The number of transmitters.
        """

        num = 0
        for device in self.__devices:
            num += device.transmitters.num_operators

        return num

    @property
    def operators(self) -> List[Operator]:
        """All operators within this scenario.
        
        Returns:
            List[Operator]: List of all operators.
        """

        operators: List[Receiver] = []

        for device in self.__devices:

            operators.extend(device.receivers)
            operators.extend(device.transmitters)

        return operators

    @property
    def num_operators(self) -> int:
        """Number of operators within this scenario.

        Returns:
            int: The number of operators.
        """

        num = 0
        for device in self.__devices:
            num += device.transmitters.num_operators + device.receivers.num_operators

        return num

    @property
    def drop_duration(self) -> float:
        """The scenario's default drop duration in seconds.

        If the drop duration is set to zero, the property will return the maximum frame duration
        over all registered transmitting modems as drop duration!

        Returns:
            float: The default drop duration in seconds.

        Raises:
            ValueError: For durations smaller than zero.
        """

        # Return the largest frame length as default drop duration
        if self.__drop_duration == 0.0:

            duration = 0.

            for device in self.__devices:
                duration = max(duration, device.max_frame_duration)

            return duration

        else:
            return self.__drop_duration

    @drop_duration.setter
    def drop_duration(self, value: float) -> None:
        """Set the scenario's default drop duration."""

        if value < 0.0:
            raise ValueError("Drop duration must be greater or equal to zero")

        self.__drop_duration = value
