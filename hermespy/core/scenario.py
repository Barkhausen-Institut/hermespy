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
from typing import Generic, Iterable, List, Optional, Tuple, TypeVar, Union

from h5py import File

from .channel_state_information import ChannelStateInformation
from .device import DeviceInput, DeviceOutput, DeviceReception, DeviceType, ProcessedDeviceInput, Reception, Transmission, Transmitter, Receiver, Operator
from .drop import Drop
from .random_node import RandomNode
from .signal_model import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
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


class Scenario(ABC, RandomNode, Generic[DeviceType]):
    """A simulation scenario.

    Scenarios consist of several devices transmitting and receiving electromagnetic signals.
    Each device can be operated by multiple operators simultaneously.
    """

    __mode: ScenarioMode  # Current scenario operating mode
    # Registered devices within this scenario.
    __devices: List[DeviceType]
    __drop_duration: float  # Drop duration in seconds.
    __file: Optional[File]  # HDF5 file handle
    __file_location: Optional[str]  # HDF5 file location
    __drop_counter: int  # Internal drop counter

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Args:

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.
        """

        RandomNode.__init__(self, seed=seed)
        self.__mode = ScenarioMode.DEFAULT
        self.__devices = []
        self.drop_duration = 0.0
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
    
    def device_index(self, device: DeviceType) -> int:
        """Index of device
        
        Args:
        
            device (DeviceType): Device for which to lookup the index.
            
        Returns: The device index.
        
        Raises:

            ValueError: If `device` is not registered in this scenario.
        """
        
        if not self.device_registered(device):
            raise ValueError("Device not registered")
        
        return self.devices.index(device)

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

            duration = 0.0

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

    def record(self, file: str, override: bool = False) -> None:
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
        self.__file = File(file, "w-")
        self.__file_location = file
        self.__drop_counter = 0

        # Switch mode
        self.__mode = ScenarioMode.RECORD

        # ToDo: Write scenario state to the set
        # factory = Factory()
        # self.__file.attrs['serialization'] = factory.to_str(self)
        self.__file.attrs["drop_duration"] = drop_duration
        self.__file.attrs["num_devices"] = self.num_devices

    def replay(self, path: Optional[str] = None) -> None:

        if self.mode != ScenarioMode.DEFAULT:
            raise RuntimeError("Initializing a replay is only possible in default mode. Please stop before starting a new replay.")

        # Initialize dataset
        self.__file = File(path, "r")
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

            self.__file.attrs["num_drops"] = self.__drop_counter

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

    def generate_outputs(self,
                         transmissions: Optional[List[List[Transmission]]] = None) -> List[DeviceOutput]:
        """Generate signals emitted by devices.
        
        Args:

            transmissions ([List[List[Transmission]], optional):
                Transmissions by operators.
                If none were provided, cached operator transmissions are assumed.

        Returns: List of device outputs.
        """

        transmissions = [None] * self.num_devices if transmissions is None else transmissions

        if len(transmissions) != self.num_devices:
            raise ValueError(f"Number of device transmissions ({len(transmissions)}) does not match number of registered devices ({self.num_devices}")

        outputs = [d.generate_output(t) for d, t in zip(self.devices, transmissions)]
        return outputs

    def transmit_devices(self) -> List[DeviceOutput]:
        """Generated information transmitted by all registered devices.

        Returns: List of generated information transmitted by each device.
        """

        transmissions = [device.transmit(False) for device in self.devices]
        return transmissions

    def process_inputs(self,
                       impinging_signals: List[Union[DeviceInput, Signal, Iterable[Signal]]],
                       cache: bool = True) -> List[ProcessedDeviceInput]:
        """Process input signals impinging onto the scenario's devices.
        
        Args:

            impinging_signals (List[Union[DeviceInput, Signal, Iterable[Signal]]]):
                List of signals impinging onto the devices.

            cache (bool, optional):
                Cache the operator inputs at the registered receive operators for further processing.
                Enabled by default.

        Returns: List of the processed device input information.

        Raises:

            ValueError: If the number of `impinging_signals` does not match the number of registered devices.
        """

        if len(impinging_signals) != self.num_devices:
            raise ValueError(f"Number of impinging signals ({len(impinging_signals)}) does not match the number if registered devices ({self.num_devices}) within this scenario")

        # Call the process input method for each device
        processed_inputs = [d.process_input(i, cache) for d, i in zip(self.devices, impinging_signals)]

        return processed_inputs

    def receive_operators(self,
                          operator_inputs: Optional[List[List[Union[Signal, Tuple[Signal, ChannelStateInformation]]]]] = None,
                          cache: bool = True) -> List[Reception]:
        """Receive over the registered operators.
        
        Args:

            operator_inputs (List[List[Union[Signal, Tuple[Signal, ChannelStateInformation]]]], optional):
                Signal models fed to the receive operators of each device.
                If not provided, the operatores are expected to have inputs cached

            cache (bool, optional):
                Cache the generated received information at the device's receive operators.
                Enabled by default.

        Returns: List of information generated by receiving over the device's operators.

        Raises:
        
            ValueError: If the number of operator inputs does not match the number of receive devices.
            RuntimeError: If no operator inputs were specified and an operator has no cached inputs.
        """

        operator_inputs = [None for _ in range(self.num_devices)] if operator_inputs is None else operator_inputs

        if len(operator_inputs) != self.num_devices:
            raise ValueError(f"Number of operator inputs ({len(operator_inputs)}) does not match the number of registered scenario devices ({self.num_devices})")

        # Generate receptions
        receptions = [d.receive_operators(i, cache) for d, i in zip(self.devices, operator_inputs)]
        return receptions

    def receive_devices(self,
                        impinging_signals: List[Union[DeviceInput, Signal, Iterable[Signal]]],
                        cache: bool = True) -> List[DeviceReception]:
        """Receive over all scenario devices.

        Internally calls :meth:`.process_inputs` and :meth:`.receive_devices`.
        
        Args:

            impinging_signals (List[Union[DeviceInput, Signal, Iterable[Signal]]]):
                List of signals impinging onto the devices.

            cache (bool, optional):
                Cache the operator inputs at the registered receive operators for further processing.
                Enabled by default.

        Returns: List of the processed device input information.

        Raises:

            ValueError: If the number of `impinging_signals` does not match the number of registered devices.
        """

        # Generate inputs
        device_inputs = [d.process_input(i, cache) for d, i in zip(self.devices, impinging_signals)]

        # Generate operator receptions
        receptions = self.receive_operators(device_inputs)

        # Generate device receptions
        return [DeviceReception.From_ProcessedDeviceInput(i, r) for i, r in zip(device_inputs, receptions)]
    
    @property
    def num_drops(self) -> Optional[int]:
        """Number of drops within the scenario.
        
        If the scenario is in replay mode, this property represents the 
        recorded number of drops
        
        If the scenario is in record mode, this property represnts the
        current number of recorded drops.
        
        Returns: Number of drops. `None` if not applicable.
        """
        
        if self.mode == ScenarioMode.DEFAULT:
            return None
        
        if self.mode == ScenarioMode.RECORD:
            return self.__drop_counter
        
        if self.mode == ScenarioMode.REPLAY:
            return self.__file.attrs['num_drops']

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
            drop = Drop.from_HDF(self.__file[f"drop_{self.__drop_counter:02d}"])
            self.__drop_counter = (self.__drop_counter + 1) % self.__file.attrs["num_drops"]

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

                drop.to_HDF(self.__file.create_group(f"drop_{self.__drop_counter:02d}"))
                self.__drop_counter += 1

        return drop


ScenarioType = TypeVar("ScenarioType", bound=Scenario)
"""Type of scenario."""
