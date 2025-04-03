# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import IntEnum
from itertools import chain
from os import path, remove
from typing import Generic, Type, TypeVar
from typing_extensions import override

from .device import (
    Device,
    DeviceInput,
    DeviceOutput,
    DeviceReception,
    DeviceState,
    DeviceTransmission,
    DeviceType,
    DST,
    ProcessedDeviceInput,
    Reception,
    Transmission,
    Transmitter,
    Receiver,
    Operator,
)
from .drop import Drop, DropType
from .factory import (
    DeserializationProcess,
    Factory,
    Serializable,
    SerializationBackend,
    SerializationProcess,
)
from .random_node import RandomNode
from .signal_model import Signal
from .transformation import TransformableBase

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
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


class Scenario(
    ABC, Serializable, RandomNode, TransformableBase, Generic[DeviceType, DST, DropType]
):
    """A wireless scenario.

    Scenarios consist of several devices transmitting and receiving electromagnetic signals.
    Each device can be operated by multiple operators simultaneously.
    """

    __mode: ScenarioMode  # Current scenario operating mode
    __devices: list[DeviceType]  # Registered devices within this scenario.
    __drop_duration: float  # Drop duration in seconds.
    __serialization_process: SerializationProcess | None  # Available during recording
    __deserialization_process: DeserializationProcess | None  # Available during replay
    __drop_counter: int  # Internal drop counter
    __campaign: str  # Measurement campaign name
    __replay_file: str | None  # File path for replay

    def __init__(
        self, seed: int | None = None, devices: Sequence[DeviceType] | None = None
    ) -> None:
        """
        Args:

            seed:
                Random seed used to initialize the pseudo-random number generator.

            devices:
                Devices to be added to the scenario during initialization.

        """

        # Initialize base classes
        RandomNode.__init__(self, seed=seed)
        TransformableBase.__init__(self)

        # Initialize attributes
        self.__mode = ScenarioMode.DEFAULT
        self.__devices = list()
        self.drop_duration = 0.0
        self.__serialization_process = None
        self.__drop_counter = 0
        self.__campaign = "default"
        self.__replay_file = None
        self.__serialization_process = None
        self.__deserialization_process = None

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

            device:
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

        device = self._device_type()(*args, **kwargs)
        self.add_device(device)
        return device

    def device_registered(self, device: DeviceType) -> bool:
        """Check if an device is registered in this scenario.

        Args:
            device: The device to be checked.

        Returns:
            bool: The device's registration status.
        """

        return device in self.__devices

    def device_index(self, device: DeviceType) -> int:
        """Index of device

        Args:

            device: Device for which to lookup the index.

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
        # For serialization compatibility, the campaign name 'state' is reserved
        if value == "state":
            raise ValueError("The campaign name 'state' is reserved for serialization purposes")

        # Do nothing if value matches the current campaign
        if value == self.__campaign:
            return

        # Setting the campaign will always reset the internal drop counter
        self.__drop_counter = 0

        # When in replay mode, changing the campaign changes the number of replayed drops
        if self.mode == ScenarioMode.REPLAY:
            raise NotImplementedError(
                "Changing the campaign name during replay is currently not supported"
            )

        # Update the campaign identifier
        self.__campaign = value

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object_sequence(list(self.devices), "devices")
        process.serialize_floating(self.drop_duration, "drop_duration")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> Scenario:
        # Deserialize class instance
        instance = cls()

        # Configure drop duration parameter
        instance.drop_duration = process.deserialize_floating("drop_duration", 0.0)

        # Add devices and (implicitly) their respective operators
        for device in process.deserialize_object_sequence("devices", cls._device_type()):
            instance.add_device(device)

        return instance

    def record(
        self,
        file: str,
        campaign: str | None = None,
        overwrite: bool = False,
        state: Scenario | None = None,
        serialize_state: bool = True,
        backend: SerializationBackend = SerializationBackend.HDF,
    ) -> None:
        """Start recording drop information generated from this scenario.

        After the scenario starts recording, changing the device and operator configuration
        is not permitted.

        Args:

            file:
                The system path where to store the generated recording data.

            campaign:
                Name of the measurement campaign.

            overwrite:
                Overwrite the file if a recording already exists.
                Disabled by default.

            state:
                Scenario to be used for state serialization.
                By default, this scenario is assumed.

            serialize_state:
                Serialize the scenario state to the recording.
                Enabled by default.

            backend:
                Serialization backend to be used for serialization.
                Default is HDF5.

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

        # Start the serialization process
        factory = Factory()
        self.__serialization_process = factory.serialize(file, campaign, backend)

        # Serialize the scenario's state, i.e. the device and operator configurations
        if serialize_state:
            self.__serialization_process.serialize_object(
                self if state is None else state, "state", True
            )

        # Switch mode flag and reset the drop counter
        self.__mode = ScenarioMode.RECORD
        self.__drop_counter = 0

        # Create the campaign if it doesn't exists
        self.campaign = campaign

    def replay(
        self,
        file: str | None = None,
        campaign: str | None = None,
        backend: SerializationBackend = SerializationBackend.HDF,
    ) -> int:
        """Replay the scenario from and HDF5 savefile.

        Args:

            file:
                File system location from which the scenario should be replayed.
                Must be specified if the scenario is not in replay mode.

            campaign:
                Identifier of the campaign to replay.

            backend:
                Serialization backend to be used for deserialization.
                Default is HDF5.

        Raises:

            RuntimeError: If `file` is not specified and scenario is not in replay mode.
            ValueError: If `campaign` is specified and is not contained within the savefile.

        Returns: Number of recorded drops to be replayed.
        """

        _file: str
        if file is None:
            if self.__replay_file is None:
                raise ValueError("Initial replay requires a valid file path argument")
            else:
                _file = self.__replay_file
        else:
            _file = file

        # Stop any action and close file handles if required
        self.stop()

        # Start a new deserialization process
        self.__deserialization_process = Factory().deserialize(_file, campaign, backend)
        self.__replay_file = _file
        self.__drop_counter = 0
        self.__campaign = campaign

        # Switch mode flag
        self.__mode = ScenarioMode.REPLAY

        # Return the number of recorded drops
        return self.__deserialization_process.sequence_length("drops")

    @classmethod
    def Replay(
        cls: Type[Scenario],
        file: str,
        campaign: str | None = None,
        backend: SerializationBackend = SerializationBackend.HDF,
    ) -> tuple[Scenario, int]:
        """Replay a scenario from an HDF5 save file.

        .. note::
           This function is currently highly experimental and may be subject to change.
           It does not fully deserialize all configurations in the case of simulation scenarios.

        Args:

            file:
                File system location of the HDF5 save file.

            campaign:
                Identifier of the campaign to replay.

            backend:
                Serialization backend to be used for deserialization.
                Default is HDF5.

        Returns: Tuple containing the replayed scenario and the number of recorded drops.
        """

        # Deserialize the respective scenario state
        deserialization_process = Factory().deserialize(file, campaign, backend)
        scenario: Scenario = deserialization_process.deserialize_object("state", cls)
        deserialization_process.finalize()

        # Enable the replay mode
        num_drops = scenario.replay(file, campaign, backend)

        # Return the scenario (initialized and in replay mode)
        return scenario, num_drops

    def stop(self) -> None:
        """Stop a running recording / playback session."""

        # Finalize the serialization process if in record mode
        if self.__serialization_process is not None:
            self.__serialization_process.finalize()
            self.__serialization_process = None
            self.__drop_counter = 0

        # Finalize the deserialization process if in replay mode
        if self.__deserialization_process is not None:
            self.__deserialization_process.finalize()
            self.__deserialization_process = None
            self.__drop_counter = 0

        # Reset the mode
        self.__mode = ScenarioMode.DEFAULT

    def transmit_operators(
        self, states: Sequence[DST] = None, notify: bool = True
    ) -> Sequence[Sequence[Transmission]]:
        """Generate information transmitted by all registered device operators.

        Args:

            states:
                States of the transmitting devices.
                If not specified, the current device states will be queried by calling :meth:`Device.state<hermespy.core.device.Device.state>`.

            notify:
                Notify the DSP layer's callbacks about the transmission results.
                Enabled by default.

        Returns:
            The generated information sorted into devices and their respective operators.
        """

        _states = [d.state() for d in self.devices] if states is None else states
        transmissions = [
            [o.transmit(s, 0.0, notify) for o in d.transmitters]
            for d, s in zip(self.devices, _states)
        ]
        return transmissions

    def generate_outputs(
        self, transmissions: list[list[Transmission]], states: Sequence[DST | None] | None = None
    ) -> Sequence[DeviceOutput]:
        """Generate signals emitted by devices.

        Args:

            transmissions:
                Results of all transmitting DSP algorithms.

            states:
                States of the transmitting devices.
                If not specified, the current device states will be queried by calling :meth:`Device.state<hermespy.core.device.Device.state>`.

        Returns: List of device outputs.
        """

        _states = [None] * self.num_devices if states is None else states

        if len(transmissions) != self.num_devices:
            raise ValueError(
                f"Number of device transmissions ({len(transmissions)}) does not match number of registered devices ({self.num_devices}"
            )

        outputs = [d.generate_output(t, s) for d, s, t in zip(self.devices, _states, transmissions)]
        return outputs

    def transmit_devices(
        self, states: Sequence[DST | None] | None = None, notify: bool = True
    ) -> Sequence[DeviceTransmission]:
        """Generated information transmitted by all registered devices.

        Args:

            states:
                States of the transmitting devices.
                If not specified, the current device states will be queried by calling :meth:`Device.state<hermespy.core.device.Device.state>`.

            notify:
                Notify the transmit DSP layer's callbacks about the transmission results.
                Enabled by default.

        Returns: List of generated information transmitted by each device.
        """

        _states = [None] * self.num_devices if states is None else states
        transmissions = [d.transmit(s, notify) for d, s in zip(self.devices, _states)]
        return transmissions

    def process_inputs(
        self,
        impinging_signals: Sequence[DeviceInput] | Sequence[Signal] | Sequence[Sequence[Signal]],
        states: Sequence[DST | None] | None = None,
    ) -> Sequence[ProcessedDeviceInput]:
        """Process input signals impinging onto the scenario's devices.

        Args:
            impinging_signals: List of signals impinging onto the devices.
            states:
                States of the transmitting devices.
                If not specified, the current device states will be queried by calling :meth:`hermespy.core.device.Device.state`.

        Returns: List of the processed device input information.

        Raises:

            ValueError: If the number of `impinging_signals` does not match the number of registered devices.
        """

        _states = [None] * self.num_devices if states is None else states

        if len(impinging_signals) != self.num_devices:
            raise ValueError(
                f"Number of impinging signals ({len(impinging_signals)}) does not match the number if registered devices ({self.num_devices}) within this scenario"
            )

        # Call the process input method for each device
        processed_inputs = [d.process_input(i, s) for d, i, s in zip(self.devices, impinging_signals, _states)]  # type: ignore

        return processed_inputs

    def receive_operators(
        self,
        operator_inputs: Sequence[ProcessedDeviceInput] | Sequence[Sequence[Signal]],
        states: Sequence[DST | None] | None = None,
        notify: bool = True,
    ) -> Sequence[Sequence[Reception]]:
        """Receive over the registered operators.

        Args:

            operator_inputs:
                Signal models to be processed by the receive DSP algorithms.
                Two-dimensional sequence where the first dimension corresponds to the devices and the second to the operators.

            states:
                States of the transmitting devices.
                If not specified, the current device states will be queried by calling :meth:`hermespy.core.device.Device.state`.

            notify:
                Notify the receive DSP layer's callbacks about the reception results.
                Enabled by default.

        Returns: list of information generated by receiving over the device's operators.

        Raises:

            ValueError: If the number of operator inputs does not match the number of receive devices.
        """

        _states = [None] * self.num_devices if states is None else states

        if len(operator_inputs) != self.num_devices:
            raise ValueError(
                f"Number of operator inputs ({len(operator_inputs)}) does not match the number of registered scenario devices ({self.num_devices})"
            )

        # Generate receptions
        receptions = [d.receive_operators(i, s, notify) for d, i, s in zip(self.devices, operator_inputs, _states)]  # type: ignore
        return receptions

    def receive_devices(
        self,
        impinging_signals: Sequence[DeviceInput] | Sequence[Signal] | Sequence[Sequence[Signal]],
        states: Sequence[DST | None] | None = None,
        notify: bool = True,
    ) -> Sequence[DeviceReception]:
        """Receive over all scenario devices.

        Internally calls :meth:`Scenario.process_inputs` and :meth:`Scenario.receive_operators`.

        Args:

            impinging_signals:
                list of signals impinging onto the devices.

            states:
                States of the transmitting devices.
                If not specified, the current device states will be queried by calling :meth:`Device.state<hermespy.core.device.Device.state>`.

            notify:
                Notify the receiving DSP layer's callbacks about the reception results.
                Enabled by default.

        Returns: list of the processed device input information.

        Raises:

            ValueError: If the number of `impinging_signals` does not match the number of registered devices.
        """

        _states = [d.state() for d in self.devices] if states is None else states

        # Generate inputs
        processed_inputs = self.process_inputs(impinging_signals, _states)

        # Generate operator receptions
        operator_receptions = self.receive_operators(
            [i.operator_inputs for i in processed_inputs], _states, notify
        )

        # Generate device receptions
        device_receptions = [
            DeviceReception.From_ProcessedDeviceInput(i, r)
            for i, r in zip(processed_inputs, operator_receptions)
        ]
        return device_receptions

    @property
    def num_drops(self) -> int:
        """Number of drops within the scenario.

        If the scenario is in replay mode, this property represents the
        recorded number of drops

        If the scenario is in record mode, this property represents the
        current number of recorded drops.

        Returns: Number of drops. Zero if the scenario is in default mode.
        """

        # Drops are not supposed to be counted in default mode
        # So return zero in this case
        if self.__mode == ScenarioMode.DEFAULT:
            return 0

        return self.__drop_counter

    @classmethod
    @abstractmethod
    def _device_type(cls) -> Type[DeviceType]:
        """Type of device used by the scenario.

        Wrapped by the scenario base class' :meth:`.new_device` method.

        Returns:
            Type of the device.
        """
        ...  # pragma: no cover

    @classmethod
    @abstractmethod
    def _drop_type(cls) -> Type[DropType]:
        """Type of drop object used by the scenario.

        Wrapped by the scenario base class' :meth:`.drop` method.

        Returns:
            Type of the drop object.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _drop(self) -> DropType:
        """Generate a single scenario drop.

        Wrapped by the scenario base class' :meth:`.drop` method.

        Returns:
            The drop object containing all information.
        """
        ...  # pragma: no cover

    def drop(self) -> DropType:
        """Generate a single data drop from all scenario devices.

        Return: The generated drop information.
        """

        if self.mode == ScenarioMode.REPLAY:
            # Ensure a valid deserialization process is available
            if self.__deserialization_process is None:
                raise RuntimeError("Replay scenario mode requires a valid deserialization process")

            # Recall the latest drop, identified by the campaign and the drop counter
            drop: DropType = self.__deserialization_process.deserialize_object_sequence(
                "drops", self._drop_type(), self.__drop_counter, 1 + self.__drop_counter
            )[0]
            self.__drop_counter += 1

            # Notify the transmit callbacks about the replayed transmissions
            for device, replayed_transmission in zip(self.devices, drop.device_transmissions):
                for operator, operator_transmission in zip(
                    device.transmitters, replayed_transmission.operator_transmissions
                ):
                    operator.notify_transmit_callbacks(operator_transmission)

            # Replay device operator receptions
            # This will simulatenously notify the receive operator callbacks
            _ = self.receive_operators(drop.operator_inputs)

        else:
            # Generate a new drop
            drop = self._drop()

            # Serialize the drop to HDF if in record mode
            if self.mode == ScenarioMode.RECORD:
                self.__serialization_process.serialize_object_sequence([drop], "drops", True, True)
                self.__drop_counter += 1

        return drop


ScenarioType = TypeVar("ScenarioType", bound="Scenario")
"""Type of scenario."""


class ReplayScenario(Scenario[Device, DeviceState, Drop]):
    """Scenario which is unable to generate drops."""

    @override
    def _drop(self) -> Drop:
        raise RuntimeError("Replay scenarios may not generate data drops")

    @classmethod
    @override
    def _device_type(cls) -> Type[Device]:
        return Device

    @classmethod
    @override
    def _drop_type(cls) -> Type[Drop]:
        return Drop
