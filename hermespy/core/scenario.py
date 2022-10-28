# -*- coding: utf-8 -*-
"""
=================
Wireless Scenario
=================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from enum import IntEnum
from os import path, remove
from typing import Generic, List, Optional, TypeVar

from h5py import File

from .device import DeviceType, Reception, Transmission, Transmitter, Receiver, Operator
from .drop import Drop
from .factory import Serializable
from .random_node import RandomNode
from .signal_model import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ScenarioMode(IntEnum):
    """Current scenario mode."""
    
    DEFAULT = 0
    """Default scenario state.
    
    For configuration and generating drops.
    """
    
    RECORD = 1
    """Recording scenario state.
    
    For recording datasets.
    """
    
    REPLAY = 2
    """Replay scenario state.
    
    For replaying already recorded datasets.
    """


class Scenario(ABC, RandomNode, Generic[DeviceType], Serializable):
    """A simulation scenario.

    Scenarios consist of several devices transmitting and receiving electromagnetic signals.
    Each device can be operated by multiple operators simultaneously.
    """

    __mode: ScenarioMode              # Current scenario operating mode
    __devices: List[DeviceType]       # Registered devices within this scenario.
    __drop_duration: float            # Drop duration in seconds.
    __file: Optional[File]            # HDF5 file handle
    __file_location: Optional[str]    # HDF5 file location
    __drop_counter: int               # Internal drop counter

    def __init__(self,
                 seed: Optional[int] = None) -> None:
        """
        Args:

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.
        """

        RandomNode.__init__(self, seed=seed)
        self.__mode = ScenarioMode.DEFAULT
        self.__devices = []
        self.drop_duration = 0.
        self.__file = None
        self.__file_location = None
        self.__drop_counter = 0
        
    def __del__(self) -> None:
        
        # Stop recording / playing if not in default mode
        if self.mode != ScenarioMode.DEFAULT:
            self.stop()
        
    @property
    def mode(self) -> ScenarioMode:
        """Current operating mode of the scenario.
        
        Returns: Operating mode flag.
        """
        
        return self.__mode

    def add_device(self, device: DeviceType) -> None:
        """Add a new device to the scenario.

        Args:

            device (DeviceType):
                New device to be added to the scenario.

        Raises:
        
            ValueError: If the device already exists.
            RuntimeError: If the scenario is not in default mode.
        """

        if self.device_registered(device):
            raise ValueError("Error trying to add an already registered device to a scenario")

        if self.mode != ScenarioMode.DEFAULT:
            raise RuntimeError("Modifying a scenario is only allowed in default mode")

        # Add device to internal device list
        self.__devices.append(device)
        
        # Register scenario at the device
        device.random_mother = self
        device.scenario = self
        
    def new_device(self, *args, **kwargs) -> DeviceType:
        """Add a new device to the scenario.
        
        Convenience function pointing to :meth:`hermespy.core.scenario.Scenario.new_device`.

        Returns: Handle to the created device.
        
        Raises:
        
            ValueError: If the device already exists.
            RuntimeError: If the scenario is not in default mode.
            RuntimeError: If the scenario does not allow for the creation or addition of new devices.
        """
        
        raise RuntimeError("Error trying to create a new device within a scenario not supporting the operation")

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
            RuntimeError: If the scenario is not in default mode.
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

        if value < 0.0:
            raise ValueError("Drop duration must be greater or equal to zero")
        
        if self.mode != ScenarioMode.DEFAULT:
            raise RuntimeError("Modifying scenario parameters is only allowed in default mode")

        self.__drop_duration = value

    def record(self,
               file: str,
               override: bool = False) -> None:
        """Start recording drop information generated from this scenario.
        
        After the scenario starts recording, changing the device and operator configuration
        is not permitted.
        
        Args:
        
            file (str):
                The system path where to store the generated recording data.

            override (bool, optional):
                Override the file if a recording already exists.
                Disabled by default.
                
        Raises:
        
            RuntimeError: If the scenario is not in default mode.
        """
        
        if self.mode != ScenarioMode.DEFAULT:
            raise RuntimeError("Initialize a recording is only possible in default mode. Please stop before starting a new recording.")
        
        if override and path.exists(file):
            remove(file)

        # Compute drop duration
        drop_duration = self.drop_duration
        
        # Initialize dataset
        self.__file = File(file, 'w-')
        self.__file_location = file
        self.__drop_counter = 0
        
        # Switch mode
        self.__mode = ScenarioMode.RECORD

        # ToDo: Write scenario state to the set
        # factory = Factory()
        # self.__file.attrs['serialization'] = factory.to_str(self)
        self.__file.attrs['drop_duration'] = drop_duration
        self.__file.attrs['num_devices'] = self.num_devices
        
    def replay(self, path: Optional[str] = None) -> None:
        
        if self.mode != ScenarioMode.DEFAULT:
            raise RuntimeError("Initializing a replay is only possible in default mode. Please stop before starting a new replay.")
        
        # Initialize dataset
        self.__file = File(path, 'r')
        self.__file_location = path
        self.__drop_counter = 0
        
        # Switch mode
        self.__mode = ScenarioMode.REPLAY
        
    def stop(self) -> None:
        """Stop a running recording / playback session."""
            
        # In default mode, nothing needs to be done
        if self.mode == ScenarioMode.DEFAULT:
            return
        
        # Save the overall number of recorder drops if in record mode
        if self.mode == ScenarioMode.RECORD:
            
            self.__file.attrs['num_drops'] = self.__drop_counter
        
        # Close HDF5 file handle properly
        self.__file.close()
        self.__file = None
        self.__drop_counter = 0
        
        # Reset the mode
        self.__mode = ScenarioMode.DEFAULT
        
    def transmit_operators(self) -> List[List[Transmission]]:
        """Generate information transmitted by all registered device operators.
        
        Returns:
            The generated information sorted into devices and their respective operators.
        """

        transmissions = [[o.transmit() for o in d.transmitters] for d in self.devices]
        return transmissions
            
    def transmit_devices(self) -> List[Signal]:
        """Generated information transmitted by all registered devices.

        Returns:
            The generated information.
        """
        
        transmissions = [device.transmit(False) for device in self.devices]
        return transmissions
    
    def receive_devices(self, receptions: List[Signal]) -> List[Signal]:
        """Receive over all devices.

        Args:

            receptions (List[Signal]):
                The signal models to be received be each device.

        Returns:
            The signal models after device processing.
        """
        
        if len(receptions) != len(self.__devices):
            raise ValueError(f"Number of receptions must be equal to number of devices ({len(receptions)} != {len(self.__devices)})")
        
        device_receptions = [d.receive(r) for d, r in zip(self.devices, receptions)]
        return device_receptions

    def receive_operators(self) -> List[List[Reception]]:
        """Generate information received by all registered device operators.
        
        Returns:
            The generated information sorted into devices and their respective operators.
        """
        
        receptions = [[o.receive() for o in d.receivers] for d in self.devices]
        return receptions
    
    @abstractmethod
    def _drop(self) -> Drop:
        """Generate a single scenario drop.

        Wrapped by the scenario base class :meth:`.drop` method.

        Returns:
            The drop object containing all information.
        """
        ...  # pragma no cover
        
    def drop(self) -> Drop:
        """Generate a single data drop from all scenario devices.
        
        Return: The generated drop information.
        """
        
        if self.mode == ScenarioMode.REPLAY:
            
            # Recall the drop from the savefile
            drop = Drop.from_HDF(self.__file[f'drop_{self.__drop_counter:02d}'])
            self.__drop_counter = (self.__drop_counter + 1) % self.__file.attrs['num_drops']
            
            # Replay device operator transmissions
            for device, device_transmission in zip(self.devices, drop.device_transmissions):
                
                for transmitter, transmission in zip(device.transmitters, device_transmission.operator_transmissions):
                    device.transmitters.add_transmission(transmitter, transmission)
            
            # Replay device operator receptions
            for device, device_reception in zip(self.devices, drop.device_receptions):
                
                for receiver, reception in zip(device.receivers, device_reception.operator_receptions):
                    receiver.cache_reception(reception.signal, device_reception.csi)
            
        else:
            
            # Generate a new drop
            drop = self._drop()
            
            # Serialize the drop to HDF if in record mode
            if self.mode == ScenarioMode.RECORD:
                
                drop.to_HDF(self.__file.create_group(f'drop_{self.__drop_counter:02d}'))
                self.__drop_counter += 1
            
        return drop

ScenarioType = TypeVar('ScenarioType', bound=Scenario)
"""Type of scenario."""