# -*- coding: utf-8 -*-
"""
=================
Wireless Scenario
=================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import IntEnum
from itertools import chain
from os import path, remove
from typing import Generic, overload, Type, TypeVar, Union

from h5py import File, Group

from .device import (
    Device,
    DeviceInput,
    DeviceOutput,
    DeviceReception,
    DeviceTransmission,
    DeviceType,
    ProcessedDeviceInput,
    Reception,
    Transmission,
    Transmitter,
    Receiver,
    Operator,
)
from .drop import Drop, DropType
from .factory import Factory
from .random_node import RandomNode
from .signal_model import Signal
from .transformation import TransformableBase

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
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


class Scenario(ABC, RandomNode, TransformableBase, Generic[DeviceType, DropType]):
    """A wireless scenario.

    Scenarios consist of several devices transmitting and receiving electromagnetic signals.
    Each device can be operated by multiple operators simultaneously.
    """

    yaml_tag = "Scenario"
    serialized_attributes = {"devices"}

    @classmethod
    def _arg_signature(cls: Type[Scenario]) -> set[str]:
        return {"seed", "devices"}

    __mode: ScenarioMode  # Current scenario operating mode
    __devices: list[DeviceType]  # Registered devices within this scenario.
    __drop_duration: float  # Drop duration in seconds.
    __file: File | None  # HDF5 file handle
    __drop_counter: int  # Internal drop counter
    __campaign: str  # Measurement campaign name
    __num_replayed_drops: int  # Number of replayed drops

    def __init__(
        self, seed: int | None = None, devices: Sequence[DeviceType] | None = None
    ) -> None:
        """
        Args:

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.

            devices (Sequence[Device], optional):
                Devices to be added to the scenario during initialization.

        """

        # Initialize base classes
        RandomNode.__init__(self, seed=seed)
        TransformableBase.__init__(self)

        # Initialize attributes
        self.__mode = ScenarioMode.DEFAULT
        self.__devices = list()
        self.drop_duration = 0.0
        self.__file = None
        self.__drop_counter = 0
        self.__campaign = "default"
        self.__num_replayed_drops = 0

        # Add devices if specified
        if devices is not None:
            for device in devices:
                self.add_device(device)

    @property
    def mode(self) -> ScenarioMode:
        """Current operating mode of the scenario.

        Returns: Operating mode flag.
        """

        return self.__mode

    def add_device(self, device: DeviceType) -> None:
        """Add a new device to the scenario.

        Args:

            device (Device):
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

        # Assign the scenario as the device's coordinate system base
        device.set_base(self)

    def new_device(self, *args, **kwargs) -> DeviceType:
        """Add a new device to the scenario.

        Convenience function pointing to :meth:`hermespy.core.scenario.Scenario.new_device`.

        Returns: Handle to the created device.

        Raises:

            ValueError: If the device already exists.
            RuntimeError: If the scenario is not in default mode.
            RuntimeError: If the scenario does not allow for the creation or addition of new devices.
        """

        raise RuntimeError(
            "Error trying to create a new device within a scenario not supporting the operation"
        )

    def device_registered(self, device: DeviceType) -> bool:
        """Check if an device is registered in this scenario.

        Args:
            device (Device): The device to be checked.

        Returns:
            bool: The device's registration status.
        """

        return device in self.__devices

    def device_index(self, device: DeviceType) -> int:
        """Index of device

        Args:

            device (Device): Device for which to lookup the index.

        Returns: The device index.

        Raises:

            ValueError: If `device` is not registered in this scenario.
        """

        if not self.device_registered(device):
            raise ValueError("Device not registered")

        return self.devices.index(device)

    @property
    def devices(self) -> list[DeviceType]:
        """Devices registered in this scenario.

        Returns: list of devices.
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
    def transmitters(self) -> list[Transmitter]:
        """All transmitting operators within this scenario.

        Returns:
            list[Transmitter]: list of all transmitting operators.
        """

        transmitters: list[Transmitter] = []

        for device in self.__devices:
            transmitters.extend(device.transmitters)

        return transmitters

    @property
    def receivers(self) -> list[Receiver]:
        """All receiving operators within this scenario.

        Returns:
            list[Receiver]: list of all transmitting operators.
        """

        receivers: list[Receiver] = []

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
    def operators(self) -> set[Operator]:
        """All operators within this scenario.

        Returns: A set containing all unique operators within this scenario
        """

        operators: set[Operator] = set()

        # Iterate over all devices and collect operators
        for device in self.devices:
            for operator in chain(device.transmitters, device.receivers):
                operators.add(operator)  # type: ignore

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

    @property
    def campaign(self) -> str:
        """Measurement campaign identifier.

        If not specified, the scenario will assume the campaign name to be `default`.

        Returns:
            Name of the current measurement campaign.

        Raises:

            ValueError: If in replay mode and the requested campaign name is not available.
        """

        return self.__campaign

    @campaign.setter
    def campaign(self, value: str) -> None:
        # Do nothing if value matches the current campaign
        if value == self.__campaign:
            return

        # If in replay mode, make sure the campaign exists
        if self.mode == ScenarioMode.REPLAY:
            if not self.__campaign_exists(value, self.__file):
                raise ValueError(
                    f"The requested measurement campaign '{value}' does not exists within the currently replayed savefile"
                )

            self.__drop_counter = 0

        elif self.mode == ScenarioMode.RECORD:
            # Create the campaign if it doesn't exists
            if not self.__campaign_exists(value, self.__file):
                self.__file.create_group("/campaigns/" + value)

            self.__drop_counter = self.__file["/campaigns/" + value].len

        # Update the campaign identifier
        self.__campaign = value

    def _state_to_HDF(self, factory: Factory, group: Group) -> None:
        """Serialize the scenario's state to an HDF5 group.

        Args:

            factory (Factory):
                Reference to the serialization factory.

            group (Group):
                Reference to an empty HDF5 group.
        """

        # Serialize required attributes
        group.attrs["num_devices"] = self.num_devices
        group.attrs["num_operators"] = self.num_operators

        # Serialize device states
        for d, device in enumerate(self.devices):
            group.attrs[f"device_{d:02d}"] = factory.to_str(device)

            # Serialize operator states
            for o, operator in enumerate(self.operators):
                group.attrs[f"operator_{o:02d}"] = factory.to_str(operator)

        # Serialize full state
        group.attrs["state"] = factory.to_str(
            {"devices": self.devices, "operators": self.operators}
        )

    @classmethod
    def _state_from_HDF(cls: Type[Scenario], factory: Factory, group: Group) -> Scenario:
        # Initialize class
        scenario = cls()

        # Recall serialization
        state: dict = factory.from_str(group.attrs["state"])  # type: ignore

        # Add devices to the scenario
        device: DeviceType
        for device in state["devices"]:
            scenario.add_device(device)

        # Return initialize scenario
        return scenario

    def record(
        self,
        file: str,
        overwrite: bool = False,
        campaign: str = "default",
        state: Scenario | None = None,
        serialize_state: bool = True,
    ) -> None:
        """Start recording drop information generated from this scenario.

        After the scenario starts recording, changing the device and operator configuration
        is not permitted.

        Args:

            file (str):
                The system path where to store the generated recording data.

            overwrite (bool, optional):
                Overwrite the file if a recording already exists.
                Disabled by default.

            campaign (str, optional):
                Name of the measurement campaign.

            state (scenario, optional):
                Scenario to be used for state serialization.
                By default, this scenario is assumed.

            serialize_state (bool, optional):
                Serialize the scenario state to the recording.
                Enabled by default.

        Raises:

            RuntimeError: If the scenario is not in default mode.
        """

        if self.mode != ScenarioMode.DEFAULT:
            raise RuntimeError(
                "Initialize a recording is only possible in default mode. Please stop before starting a new recording."
            )

        # Check wether the specified file already exists within the filesystem
        file_exists = path.exists(file)

        # Remove the existing file if the overwrite flag is enabled
        if overwrite and file_exists:
            remove(file)
            file_exists = False

        # Compute drop duration
        drop_duration = self.drop_duration

        # Initialize dataset
        file_mode = "w-" if overwrite else "a"
        self.__file = File(file, file_mode)
        self.__drop_counter = 0
        self.__campaign = campaign

        # Switch mode
        self.__mode = ScenarioMode.RECORD

        # Write required attributes
        self.__file.attrs["drop_duration"] = drop_duration

        # Write required groups
        if "/campaigns" not in self.__file:
            self.__file.create_group("/campaigns")

        # Write scenario state to the dataset for easy recollection
        # Future feature: Write a locking mechanism during recording
        if serialize_state:
            if "/state" not in self.__file:
                self.__file.create_group("state")

            factory = Factory()

            if state is None:
                self._state_to_HDF(factory, self.__file["/state"])

            else:
                state._state_to_HDF(factory, self.__file["/state"])

        # Write meta-information
        self.__file.attrs["hermes_version"] = __version__
        self.__file.attrs["hermes_status"] = __status__

        # Update the campaign, will create the respective group if it doesn't exist yet
        self.campaign = campaign

    def __campaign_exists(self, campaign: str, file: File) -> bool:
        """Check whether a campaign identifier exists within the current dataset.

        Args:

            campaign (str):
                The campaign identifier string.

            file (File):
                The HDF5 file to check for campaign existence.

        Returns: Boolean indicator.

        Raises:

            RuntimeError: If the scenario is currently in default mode and `file` was not specified.
        """

        return "/campaigns/" + campaign in file

    def replay(self, file: str | File | None = None, campaign: str = "default") -> None:
        """Replay the scenario from and HDF5 savefile.

        Args:

            file (Union[None, str, File], optional):
                File from which the scenario should be replayed.
                May be a file system location or an HDF5 `File` handle.

            campaign (str, optional):
                Identifier of the campaign to replay.
                If not specified, the assumed campaign name is `default`.

        Raises:

            RuntimeError: If `file` is not specified and can't be inferred from previous record executions.
            ValueError: If `campaign` is specified and is not contained within the savefile.
        """

        if file is None:
            if self.__file is None:
                raise ValueError(
                    "A file location must be specified or the scenario most be in record or replay mode"
                )

            file = self.__file.filename

        # If only a file system location was specified, open the file
        _file = File(file, "r") if isinstance(file, str) else file

        # Check if the campaign is available (if a campaign was specified)
        if not self.__campaign_exists(campaign, _file):
            filename = _file.filename
            _file.close()
            raise ValueError(
                f"The requested measurement campaign '{campaign}' does not exists within the savefile '{filename}'"
            )

        # Stop any action and close file handles if required
        self.stop()

        # Initialize dataset
        self.__file = _file
        self.__drop_counter = 0
        self.__campaign = campaign
        self.__num_replayed_drops = len(_file["/campaigns/" + campaign])

        # Switch mode
        self.__mode = ScenarioMode.REPLAY

    @classmethod
    def Replay(cls: Type[Scenario], file: Union[str, File], campaign: str = "default") -> Scenario:
        """Replay a scenario from an HDF5 save file.

        Args:

            file (str):
                File system location of the HDF5 save file.


            campaign (str, optional):
                Identifier of the campaign to replay.
                If not specified, the assumed campaign name is `default`.
        """

        # Load the dataset
        if isinstance(file, str):
            file = File(file, "r")

        # Recall the class state from the respective HDF5 group
        factory = Factory()
        scenario = cls._state_from_HDF(factory, file["state"])

        # Enable the replay mode
        scenario.replay(file, campaign)

        # Return the scenario (initialized and in replay mode)
        return scenario

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

    def transmit_operators(self) -> Sequence[Sequence[Transmission]]:
        """Generate information transmitted by all registered device operators.

        Returns:
            The generated information sorted into devices and their respective operators.
        """

        transmissions = [[o.transmit() for o in d.transmitters] for d in self.devices]
        return transmissions

    def generate_outputs(
        self, transmissions: list[list[Transmission]] | None = None
    ) -> Sequence[DeviceOutput]:
        """Generate signals emitted by devices.

        Args:

            transmissions (list[list[Transmission]], optional):
                Transmissions by operators.
                If none were provided, cached operator transmissions are assumed.

        Returns: list of device outputs.
        """

        # Assume cached operator transmissions if none were provided
        _transmissions: list[None] | list[list[Transmission]] = (
            [None] * self.num_devices if not transmissions else transmissions
        )

        if len(_transmissions) != self.num_devices:
            raise ValueError(
                f"Number of device transmissions ({len(_transmissions)}) does not match number of registered devices ({self.num_devices}"
            )

        outputs = [d.generate_output(t) for d, t in zip(self.devices, _transmissions)]
        return outputs

    def transmit_devices(self) -> Sequence[DeviceTransmission]:
        """Generated information transmitted by all registered devices.

        Returns: list of generated information transmitted by each device.
        """

        transmissions = [device.transmit() for device in self.devices]
        return transmissions

    @overload
    def process_inputs(
        self, impinging_signals: Sequence[DeviceInput], cache: bool = True
    ) -> Sequence[ProcessedDeviceInput]: ...  # pragma: no cover

    @overload
    def process_inputs(
        self, impinging_signals: Sequence[Signal], cache: bool = True
    ) -> Sequence[ProcessedDeviceInput]: ...  # pragma: no cover

    @overload
    def process_inputs(
        self, impinging_signals: Sequence[Sequence[Signal]], cache: bool = True
    ) -> Sequence[ProcessedDeviceInput]: ...  # pragma: no cover

    def process_inputs(
        self,
        impinging_signals: Sequence[DeviceInput] | Sequence[Signal] | Sequence[Sequence[Signal]],
        cache: bool = True,
    ) -> Sequence[ProcessedDeviceInput]:
        """Process input signals impinging onto the scenario's devices.

        Args:

            impinging_signals (Sequence[DeviceInput | Signal | Sequence[Signal]]):
                list of signals impinging onto the devices.

            cache (bool, optional):
                Cache the operator inputs at the registered receive operators for further processing.
                Enabled by default.

        Returns: list of the processed device input information.

        Raises:

            ValueError: If the number of `impinging_signals` does not match the number of registered devices.
        """

        if len(impinging_signals) != self.num_devices:
            raise ValueError(
                f"Number of impinging signals ({len(impinging_signals)}) does not match the number if registered devices ({self.num_devices}) within this scenario"
            )

        # Call the process input method for each device
        processed_inputs = [d.process_input(i, cache) for d, i in zip(self.devices, impinging_signals)]  # type: ignore

        return processed_inputs

    @overload
    def receive_operators(
        self, operator_inputs: Sequence[ProcessedDeviceInput], cache: bool = True
    ) -> Sequence[Sequence[Reception]]: ...  # pragma: no cover

    @overload
    def receive_operators(
        self, operator_inputs: Sequence[Sequence[Signal]], cache: bool = True
    ) -> Sequence[Sequence[Reception]]: ...  # pragma: no cover

    @overload
    def receive_operators(self) -> Sequence[Sequence[Reception]]: ...  # pragma: no cover

    def receive_operators(
        self,
        operator_inputs: Sequence[ProcessedDeviceInput] | Sequence[Sequence[Signal]] | None = None,
        cache: bool = True,
    ) -> Sequence[Sequence[Reception]]:
        """Receive over the registered operators.

        Args:

            operator_inputs (Sequence[Sequence[Signal]] | ProcessedDeviceInput, optional):
                Signal models fed to the receive operators of each device.
                If not provided, the operatores are expected to have inputs cached

            cache (bool, optional):
                Cache the generated received information at the device's receive operators.
                Enabled by default.

        Returns: list of information generated by receiving over the device's operators.

        Raises:

            ValueError: If the number of operator inputs does not match the number of receive devices.
            RuntimeError: If no operator inputs were specified and an operator has no cached inputs.
        """

        _operator_inputs = (
            [None for _ in range(self.num_devices)] if operator_inputs is None else operator_inputs
        )

        if len(_operator_inputs) != self.num_devices:
            raise ValueError(
                f"Number of operator inputs ({len(_operator_inputs)}) does not match the number of registered scenario devices ({self.num_devices})"
            )

        # Generate receptions
        receptions = [d.receive_operators(i, cache) for d, i in zip(self.devices, _operator_inputs)]  # type: ignore
        return receptions

    @overload
    def receive_devices(
        self, impinging_signals: Sequence[DeviceInput], cache: bool = True
    ) -> Sequence[DeviceReception]: ...  # pragma: no cover

    @overload
    def receive_devices(
        self, impinging_signals: Sequence[Signal], cache: bool = True
    ) -> Sequence[DeviceReception]: ...  # pragma: no cover

    @overload
    def receive_devices(
        self, impinging_signals: Sequence[Sequence[Signal]], cache: bool = True
    ) -> Sequence[DeviceReception]: ...  # pragma: no cover

    def receive_devices(
        self,
        impinging_signals: Sequence[DeviceInput] | Sequence[Signal] | Sequence[Sequence[Signal]],
        cache: bool = True,
    ) -> Sequence[DeviceReception]:
        """Receive over all scenario devices.

        Internally calls :meth:`Scenario.process_inputs` and :meth:`Scenario.receive_operators`.

        Args:

            impinging_signals (list[Union[DeviceInput, Signal, Iterable[Signal]]]):
                list of signals impinging onto the devices.

            cache (bool, optional):
                Cache the operator inputs at the registered receive operators for further processing.
                Enabled by default.

        Returns: list of the processed device input information.

        Raises:

            ValueError: If the number of `impinging_signals` does not match the number of registered devices.
        """

        # Generate inputs
        processed_inputs = self.process_inputs(impinging_signals, cache)

        # Generate operator receptions
        operator_receptions = self.receive_operators([i.operator_inputs for i in processed_inputs])

        # Generate device receptions
        device_receptions = [
            DeviceReception.From_ProcessedDeviceInput(i, r)
            for i, r in zip(processed_inputs, operator_receptions)
        ]
        return device_receptions

    @property
    def num_drops(self) -> int | None:
        """Number of drops within the scenario.

        If the scenario is in replay mode, this property represents the
        recorded number of drops

        If the scenario is in record mode, this property represnts the
        current number of recorded drops.

        Returns: Number of drops. `None` if not applicable.
        """

        if self.mode == ScenarioMode.RECORD:
            return self.__drop_counter

        if self.mode == ScenarioMode.REPLAY:
            return self.__num_replayed_drops

        return None

    @abstractmethod
    def _drop(self) -> DropType:
        """Generate a single scenario drop.

        Wrapped by the scenario base class :meth:`.drop` method.

        Returns:
            The drop object containing all information.
        """
        ...  # pragma no cover

    @abstractmethod
    def _recall_drop(self, group: Group) -> DropType:
        """Recall a recorded drop from a HDF5 group.

        Args:

            group (Group):
                HDF5 group containing the drop information.

        Returns: The recalled drop.
        """
        ...  # pragma no cover

    def drop(self) -> DropType:
        """Generate a single data drop from all scenario devices.

        Return: The generated drop information.
        """

        if self.mode == ScenarioMode.REPLAY:
            # Recall the drop from the savefile
            for _ in range(self.__num_replayed_drops):
                drop_path = f"/campaigns/{self.__campaign}/drop_{self.__drop_counter:02d}"
                self.__drop_counter = (self.__drop_counter + 1) % self.__num_replayed_drops

                if drop_path in self.__file:
                    drop = self._recall_drop(self.__file[drop_path])
                    break

            # Replay device operator transmissions
            for device, device_transmission in zip(self.devices, drop.device_transmissions):
                device.cache_transmission(device_transmission)

            # Replay device operator receptions
            _ = self.receive_operators(drop.operator_inputs, cache=True)

        else:
            # Generate a new drop
            drop = self._drop()

            # Serialize the drop to HDF if in record mode
            if self.mode == ScenarioMode.RECORD:
                drop.to_HDF(
                    self.__file.create_group(
                        f"campaigns/{self.__campaign}/drop_{self.__drop_counter:02d}"
                    )
                )
                self.__drop_counter += 1

        return drop


ScenarioType = TypeVar("ScenarioType", bound="Scenario")
"""Type of scenario."""


class ReplayScenario(Scenario[Device, Drop]):
    """Scenario which is unable to generate drops."""

    def _drop(self) -> Drop:
        raise RuntimeError("Replay scenario may not generate data drops.")

    def _recall_drop(self, group: Group) -> Drop:
        return Drop.from_HDF(group)
