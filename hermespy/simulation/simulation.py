# -*- coding: utf-8 -*-
"""
==========
Simulation
==========
"""

from __future__ import annotations
from collections.abc import Sequence
from sys import maxsize
from time import time
from typing import Any, Callable, Dict, List, Mapping, Optional, overload, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from h5py import Group
from os import path
from ray import remote
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode, Node
from rich.console import Console

from hermespy.core import (
    DeviceInput,
    Drop,
    Serializable,
    Pipeline,
    Verbosity,
    Operator,
    ConsoleMode,
    Evaluator,
    register,
    MonteCarloActor,
    MonteCarlo,
    MonteCarloResult,
    Scenario,
    Signal,
    DeviceOutput,
    SNRType,
)
from hermespy.channel import (
    Channel,
    ChannelPropagation,
    ChannelRealization,
    DirectiveChannelRealization,
    IdealChannel,
    QuadrigaInterface,
)
from .simulated_device import (
    TriggerModel,
    TriggerRealization,
    ProcessedSimulatedDeviceInput,
    SimulatedDevice,
    SimulatedDeviceTransmission,
    SimulatedDeviceReception,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulatedDrop(Drop):
    """Drop containing all information generated during a simulated wireless scenario transmission,
    channel propagation and reception."""

    __channel_realizations: Sequence[Sequence[DirectiveChannelRealization]]

    def __init__(
        self,
        timestamp: float,
        device_transmissions: Sequence[SimulatedDeviceTransmission],
        channel_realizations: Sequence[Sequence[DirectiveChannelRealization]],
        device_receptions: Sequence[SimulatedDeviceReception],
    ) -> None:
        """
        Args:

            timestamp (float):
                Time at which the drop was generated.

            device_transmissions (Sequence[DeviceTransmission]):
                Transmitted device information.

            channel_realizations (Sequence[Sequence[DirectiveChannelRealization]]):
                Realizations of the wireless channels over which the simualation propagated device transmissions.

            device_receptions (Sequence[ProcessedSimulatedDeviceReception]):
                Received device information.
        """

        self.__channel_realizations = channel_realizations
        Drop.__init__(self, timestamp, device_transmissions, device_receptions)

    @property
    def channel_realizations(self) -> Sequence[Sequence[DirectiveChannelRealization]]:
        """Channel realizations over which signals were propagated.

        Returns: Two-dimensional numpy matrix with each entry corresponding to the respective device link's channel.
        """

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

        for d_out in range(num_devices):
            for d_in in range(d_out + 1):
                realization_group = self._create_group(
                    group, f"channel_realization_{d_out:02d}_{d_in:02d}"
                )
                self.channel_realizations[d_out][d_in].realization.to_HDF(realization_group)

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

        channel_realizations = np.empty((num_devices, num_devices), dtype=np.object_)
        for device_beta_idx in range(num_devices):
            beta_device = scenario.devices[device_beta_idx]

            for device_alpha_idx in range(device_beta_idx + 1):
                alpha_device = scenario.devices[device_alpha_idx]

                # Recall the channel realization
                channel = scenario.channels[device_beta_idx, device_alpha_idx]
                realization = channel.recall_realization(
                    group[f"channel_realization_{device_beta_idx:02d}_{device_alpha_idx:02d}"]
                )

                # Place the channel realization into the channel realization matrix
                channel_realizations[
                    device_beta_idx, device_alpha_idx
                ] = DirectiveChannelRealization(alpha_device, beta_device, realization)
                channel_realizations[
                    device_alpha_idx, device_beta_idx
                ] = DirectiveChannelRealization(beta_device, alpha_device, realization)

        return SimulatedDrop(timestamp, transmissions, channel_realizations.tolist(), receptions)


class SimulationScenario(Scenario[SimulatedDevice]):
    yaml_tag = "SimulationScenario"

    __channels: np.ndarray  # Channel matrix linking devices
    __snr: Optional[float]  # Signal to noise ratio at the receiver-side
    __snr_type: SNRType  # Global global type of signal to noise ratio.

    def __init__(
        self, snr: float = float("inf"), snr_type: str | SNRType = SNRType.PN0, *args, **kwargs
    ) -> None:
        """
        Args:

            snr (float, optional):
                The assumed linear signal to noise ratio.
                Infinite by default, i.e. no added noise during reception.

            snr_type (Union[str, SNRType], optional):
                The signal to noise ratio metric to be used.
                By default, signal power to noise power is assumed.
        """

        Scenario.__init__(self, *args, **kwargs)
        self.snr = snr
        self.snr_type = snr_type
        self.__channels = np.ndarray((0, 0), dtype=object)

    def new_device(self, *args, **kwargs) -> SimulatedDevice:
        """Add a new device to the simulation scenario.

        Returns:
            SimulatedDevice: Newly added simulated device.
        """

        device = SimulatedDevice(*args, **kwargs)
        self.add_device(device)

        return device

    def add_device(self, device: SimulatedDevice) -> None:
        # Add the device to the scenario
        Scenario.add_device(self, device)
        device.scenario = self

        if self.num_devices == 1:
            self.__channels = np.array([[IdealChannel(device, device)]], dtype=object)

        else:
            # Create new channels from each existing device to the newly added device
            new_channels = np.array([[IdealChannel(device, rx)] for rx in self.devices])

            # Complete channel matrix by the newly created channels
            self.__channels = np.append(self.__channels, new_channels[:-1], axis=1)
            self.__channels = np.append(self.__channels, new_channels.T, axis=0)

    @property
    def channels(self) -> np.ndarray:
        """Channel matrix between devices.

        Returns:
            np.ndarray:
                An `MxM` matrix of channels between devices.
        """

        return self.__channels

    def channel(self, transmitter: SimulatedDevice, receiver: SimulatedDevice) -> Channel:
        """Access a specific channel between two devices.

        Args:

            transmitter (SimulatedDevice):
                The device transmitting into the channel.

            receiver (SimulatedDevice):
                the device receiving from the channel

        Returns:
            Channel:
                Channel between `transmitter` and `receiver`.

        Raises:
            ValueError:
                Should `transmitter` or `receiver` not be registered with this scenario.
        """

        devices = self.devices

        if transmitter not in devices:
            raise ValueError("Provided transmitter is not registered with this scenario")

        if receiver not in devices:
            raise ValueError("Provided receiver is not registered with this scenario")

        index_transmitter = devices.index(transmitter)
        index_receiver = devices.index(receiver)

        return self.__channels[index_transmitter, index_receiver]

    def departing_channels(
        self, transmitter: SimulatedDevice, active_only: bool = False
    ) -> List[Channel]:
        """Collect all channels departing from a transmitting device.

        Args:

            transmitter (SimulatedDevice):
                The transmitting device.

            active_only (bool, optional):
                Consider only active channels.
                A channel is considered active if its gain is greater than zero.

        Returns: A list of departing channels.

        Raises:

            ValueError: Should `transmitter` not be registered with this scenario.
        """

        devices = self.devices

        if transmitter not in devices:
            raise ValueError("The provided transmitter is not registered with this scenario.")

        transmitter_index = devices.index(transmitter)
        channels: List[Channel] = self.__channels[:, transmitter_index].tolist()

        if active_only:
            channels = [channel for channel in channels if channel.gain > 0.0]

        return channels

    def arriving_channels(
        self, receiver: SimulatedDevice, active_only: bool = False
    ) -> List[Channel]:
        """Collect all channels arriving at a device.

        Args:
            receiver (Receiver):
                The receiving modem.

            active_only (bool, optional):
                Consider only active channels.
                A channel is considered active if its gain is greater than zero.

        Returns: A list of arriving channels.

        Raises:

            ValueError: Should `receiver` not be registered with this scenario.
        """

        devices = self.devices

        if receiver not in devices:
            raise ValueError("The provided transmitter is not registered with this scenario.")

        receiver_index = devices.index(receiver)
        channels: List[Channel] = self.__channels[receiver_index,].tolist()

        if active_only:
            channels = [channel for channel in channels if channel.gain > 0.0]

        return channels

    def set_channel(
        self,
        beta_device: int | SimulatedDevice,
        alpha_device: int | SimulatedDevice,
        channel: Channel | None,
    ) -> None:
        """Specify a channel within the channel matrix.

        Args:

            beta_device (int | SimulatedDevice):
                Index of the receiver within the channel matrix.

            alpha_device (int | SimulatedDevice):
                Index of the transmitter within the channel matrix.

            channel (Channel | None):
                The channel instance to be set at position (`transmitter_index`, `receiver_index`).

        Raises:
            ValueError:
                If `transmitter_index` or `receiver_index` are greater than the channel matrix dimensions.
        """

        if isinstance(beta_device, SimulatedDevice):
            beta_device = self.devices.index(beta_device)

        if isinstance(alpha_device, SimulatedDevice):
            alpha_device = self.devices.index(alpha_device)

        if self.__channels.shape[0] <= alpha_device or 0 > alpha_device:
            raise ValueError("Alpha device index greater than channel matrix dimension")

        if self.__channels.shape[1] <= beta_device or 0 > beta_device:
            raise ValueError("Beta Device index greater than channel matrix dimension")

        # Update channel field within the matrix
        self.__channels[alpha_device, beta_device] = channel
        self.__channels[beta_device, alpha_device] = channel

        if channel is not None:
            # Set proper receiver and transmitter fields
            channel.alpha_device = self.devices[alpha_device]
            channel.beta_device = self.devices[beta_device]
            channel.scenario = self

    @register(first_impact="receive_devices", title="SNR")  # type: ignore[misc]
    @property
    def snr(self) -> Optional[float]:
        """Ratio of signal energy to noise power at the receiver-side.

        Returns:
            Optional[float]:
                Linear signal energy to noise power ratio.
                `None` if not specified.

        Raises:
            ValueError: On ratios smaller or equal to zero.
        """

        return self.__snr

    @snr.setter
    def snr(self, value: Optional[float]) -> None:
        if value is None:
            self.__snr = None

        else:
            if value <= 0.0:
                raise ValueError("Signal to noise ratio must be greater than zero")

            self.__snr = value

    @register(first_impact="receive_devices", title="SNR Type")  # type: ignore
    @property
    def snr_type(self) -> SNRType:
        """Type of signal-to-noise ratio.

        Returns:
            SNRType: The SNR type.
        """

        return self.__snr_type

    @snr_type.setter
    def snr_type(self, snr_type: Union[str, int, SNRType]) -> None:
        """Modify the type of signal-to-noise ratio.

        Args:
            snr_type (Union[str, int, SNRType]):
                The new type of signal to noise ratio, string or enum representation.
        """

        if isinstance(snr_type, str):
            snr_type = SNRType[snr_type]

        elif isinstance(snr_type, int):
            snr_type = SNRType(snr_type)

        self.__snr_type = snr_type

    def transmit_devices(self, cache: bool = True) -> Sequence[SimulatedDeviceTransmission]:
        """Generate simulated device transmissions of all registered devices.

        Devices sharing trigger models will be triggered simultaneously.

        Args:

            cache (bool, optional):
                Cache the generated transmissions at the respective devices.
                Enabled by default.

        Returns:
            Sequence of simulated simulated device transmissions.
        """

        # Collect unique triggers
        triggers: List[TriggerModel] = []
        trigger_realizations: List[TriggerRealization] = []
        transmissions: List[SimulatedDeviceTransmission] = []

        for device in self.devices:
            trigger_realization: TriggerRealization

            if device.trigger_model not in triggers:
                trigger_realization = device.trigger_model.realize()

                triggers.append(device.trigger_model)
                trigger_realizations.append(trigger_realization)

            else:
                trigger_realization = trigger_realizations[triggers.index(device.trigger_model)]

            transmission = device.transmit(cache=cache, trigger_realization=trigger_realization)
            transmissions.append(transmission)

        return transmissions

    def propagate(self, transmissions: Sequence[DeviceOutput]) -> List[List[ChannelPropagation]]:
        """Propagate device transmissions over the scenario's channel instances.

        Args:

            transmissions (Sequence[DeviceOutput])
                Sequence of device transmissisons.

        Returns: Propagation matrix.

        Raises:

            ValueError: If the length of `transmissions` does not match the number of registered devices.
        """

        if len(transmissions) != self.num_devices:
            raise ValueError(
                f"Number of transmit signals ({len(transmissions)}) does not match "
                f"the number of registered devices ({self.num_devices})"
            )

        # Initialize the propagated signals
        propagation_matrix = np.empty((self.num_devices, self.num_devices), dtype=object)

        # Loop over each channel within the channel matrix and propagate the signals over the respective channel model
        for device_alpha_idx, alpha_device in enumerate(self.devices):
            for device_beta_idx, beta_device in enumerate(self.devices[: 1 + device_alpha_idx]):
                # Select and realize the channel linking device alpha and device beta
                channel: Channel = self.channels[device_alpha_idx, device_beta_idx]
                channel_realization: ChannelRealization = channel.realize()

                # Propagate signal emitted from device alpha to device beta over the linking channel
                alpha_propagation = channel_realization.propagate(
                    transmissions[device_alpha_idx], alpha_device, beta_device
                )

                # Propagate signal emitted from device beta to device alpha over the linking channel
                beta_propagation = channel_realization.propagate(
                    transmissions[device_beta_idx], beta_device, alpha_device
                )

                # Store propagtions in their respective coordinates within the propagation matrix
                propagation_matrix[device_alpha_idx, device_beta_idx] = beta_propagation
                propagation_matrix[device_beta_idx, device_alpha_idx] = alpha_propagation

        return propagation_matrix.tolist()

    @overload
    def process_inputs(
        self,
        impinging_signals: Sequence[DeviceInput],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> List[ProcessedSimulatedDeviceInput]:
        ...  # pragma: no cover

    @overload
    def process_inputs(
        self,
        impinging_signals: Sequence[Signal],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> List[ProcessedSimulatedDeviceInput]:
        ...  # pragma: no cover

    @overload
    def process_inputs(
        self,
        impinging_signals: Sequence[Sequence[Signal]],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> List[ProcessedSimulatedDeviceInput]:
        ...  # pragma: no cover

    @overload
    def process_inputs(
        self,
        impinging_signals: Sequence[Sequence[ChannelPropagation]],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> List[ProcessedSimulatedDeviceInput]:
        ...  # pragma: no cover

    def process_inputs(
        self,
        impinging_signals: Sequence[DeviceInput]
        | Sequence[Signal]
        | Sequence[Sequence[Signal]]
        | Sequence[Sequence[ChannelPropagation]],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> List[ProcessedSimulatedDeviceInput]:
        """Process input signals impinging onto the scenario's devices.

        Args:

            impinging_signals (Sequence[DeviceInput | Signal | Sequence[Signal]] | Sequence[Sequence[Signal]] | Sequence[Sequence[ChannelPropagation]]):
                List of signals impinging onto the devices.

            cache (bool, optional):
                Cache the operator inputs at the registered receive operators for further processing.
                Enabled by default.

            trigger_realizations (Sequence[TriggerRealization], optional):
                Sequence of trigger realizations.
                If not specified, ideal triggerings are assumed for all devices.

        Returns: List of the processed device input information.

        Raises:

            ValueError: If the number of `impinging_signals` does not match the number of registered devices.
        """

        if trigger_realizations is None:
            # ToDo: Currently trigger realizations are not shared when directly running process_inputs
            trigger_realizations = [
                TriggerRealization(0, device.sampling_rate) for device in self.devices
            ]

        if len(impinging_signals) != self.num_devices:
            raise ValueError(
                f"Number of impinging signals ({len(impinging_signals)}) does not match the number if registered devices ({self.num_devices}) within this scenario"
            )

        if len(trigger_realizations) != self.num_devices:
            raise ValueError(
                f"Number of trigger realizations ({len(trigger_realizations)}) does not match the number if registered devices ({self.num_devices}) within this scenario"
            )

        # Call the process input method for each device
        processed_inputs = [d.process_input(i, cache=cache, trigger_realization=t) for d, i, t in zip(self.devices, impinging_signals, trigger_realizations)]  # type: ignore

        return processed_inputs

    @overload
    def receive_devices(
        self,
        impinging_signals: Sequence[DeviceInput],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> Sequence[SimulatedDeviceReception]:
        ...  # pragma: no cover

    @overload
    def receive_devices(
        self,
        impinging_signals: Sequence[Signal],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> Sequence[SimulatedDeviceReception]:
        ...  # pragma: no cover

    @overload
    def receive_devices(
        self,
        impinging_signals: Sequence[Sequence[Signal]],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> Sequence[SimulatedDeviceReception]:
        ...  # pragma: no cover

    @overload
    def receive_devices(
        self,
        impinging_signals: Sequence[Sequence[ChannelPropagation]],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> Sequence[SimulatedDeviceReception]:
        ...  # pragma: no cover

    def receive_devices(
        self,
        impinging_signals: Sequence[DeviceInput]
        | Sequence[Signal]
        | Sequence[Sequence[Signal]]
        | Sequence[Sequence[ChannelPropagation]],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> Sequence[SimulatedDeviceReception]:
        """Receive over all simulated scenario devices.

        Internally calls :meth:`SimulationScenario.process_inputs` and :meth:`Scenario.receive_operators`.

        Args:

            impinging_signals (List[Union[DeviceInput, Signal, Iterable[Signal]]]):
                List of signals impinging onto the devices.

            cache (bool, optional):
                Cache the operator inputs at the registered receive operators for further processing.
                Enabled by default.

            trigger_realizations (Sequence[TriggerRealization], optional):
                Sequence of trigger realizations.
                If not specified, ideal triggerings are assumed for all devices.

        Returns: List of the processed device input information.

        Raises:

            ValueError: If the number of `impinging_signals` does not match the number of registered devices.
        """

        # Generate inputs
        processed_inputs = self.process_inputs(
            impinging_signals, cache=cache, trigger_realizations=trigger_realizations
        )

        # Generate operator receptions
        operator_receptions = self.receive_operators(
            [i.operator_inputs for i in processed_inputs], cache=cache
        )

        # Generate device receptions
        device_receptions = [SimulatedDeviceReception.From_ProcessedSimulatedDeviceInput(i, r) for i, r in zip(processed_inputs, operator_receptions)]  # type: ignore
        return device_receptions

    def _drop(self) -> SimulatedDrop:
        # Generate drop timestamp
        timestamp = time()

        # Generate device transmissions
        device_transmissions = self.transmit_devices()

        # Simulate channel propagation
        channel_propagations = self.propagate(device_transmissions)
        channel_realizations = [[tx.realization for tx in rx] for rx in channel_propagations]

        # Process receptions
        trigger_realizations = [t.trigger_realization for t in device_transmissions]
        device_receptions = self.receive_devices(
            channel_propagations, trigger_realizations=trigger_realizations
        )

        # Return finished drop
        return SimulatedDrop(
            timestamp, device_transmissions, channel_realizations, device_receptions
        )


class SimulationRunner(object):
    """Runner remote thread deployed by Monte Carlo routines"""

    __scenario: SimulationScenario  # Scenario to be run
    __propagation: Sequence[Sequence[ChannelPropagation]] | None
    __processed_inputs: Sequence[ProcessedSimulatedDeviceInput]

    def __init__(self, scenario: SimulationScenario) -> None:
        """
        Args:

            scenario(SimulationScenario):
                Scenario to be run.
        """

        self.__scenario = scenario
        self.__propagation = None
        self.__processed_inputs = []

    def transmit_operators(self) -> None:
        """Generate base-band signal models emitted by all registered transmitting operators.

        Internaly resolves to the scenario's transmit operators routine :meth:`SimulationScenario.transmit_operators`.
        """

        # Resolve to the scenario transmit operators routine
        _ = self.__scenario.transmit_operators()

    def generate_outputs(self) -> None:
        """Generate radio-frequency band signal models emitted by devices.

        Internally resolves to the scenario's generate outputs routine :meth:`SimulationScenario.generate_outputs`.
        """

        # Resolve to the scenario output generation routine
        _ = self.__scenario.generate_outputs()

    def propagate(self) -> None:
        """Propagate the signals generated by registered transmitters over the channel model.

        Signals receiving at each receive modem are a superposition of all transmit signals impinging
        onto the receive modem over activated channels.

        The signal stream matrices contain the number of antennas on the first dimension and the number of
        signal samples on the second dimension.

        Raises:

            RuntimeError: If the propagation stage is called without prior device transmission.
            RuntimeError: If the number of transmit signals does not match the number of registered devices.
        """

        device_outputs = [device.output for device in self.__scenario.devices]
        if any([t is None for t in device_outputs]):
            raise RuntimeError(
                "Propagation simulation stage called without prior device transmission"
            )

        # Propagate device outputs
        self.__propagation = self.__scenario.propagate(device_outputs)

    def process_inputs(self) -> None:
        """Process device inputs after channel propgation.

        Raises:

            RuntimeError: If the propagation stage is called without prior channel propagation.
            RuntimeError: If the number of arriving signals does not match the number of registered devices.
        """

        propagation_matrix = self.__propagation

        if propagation_matrix is None:
            raise RuntimeError(
                "Receive device simulation stage called without prior channel propagation"
            )

        if len(propagation_matrix) != self.__scenario.num_devices:
            raise RuntimeError(
                f"Number of arriving signals ({len(propagation_matrix)}) does not match "
                f"the number of receiving devices ({self.__scenario.num_devices})"
            )

        self.__processed_inputs: Sequence[ProcessedSimulatedDeviceInput] = []
        for device, impinging_signals in zip(self.__scenario.devices, propagation_matrix):
            self.__processed_inputs.append(
                device.process_input(
                    impinging_signals=impinging_signals,
                    snr=self.__scenario.snr,
                    snr_type=self.__scenario.snr_type,
                )
            )

    def receive_operators(self) -> None:
        """Demodulate base-band signal models received by all registered receiving operators.

        Internally resolves to the scenario's receive operators routine :meth:`SimulationScenario.receive_operators`.
        """

        # Resolve to the scenario's operator receive routine
        _ = self.__scenario.receive_operators()


@remote(num_cpus=1)
class SimulationActor(MonteCarloActor[SimulationScenario], SimulationRunner):
    """Remote ray actor generated from the simulation runner class."""

    def __init__(self, argument_tuple: Any, index: int, catch_exceptions: bool = True) -> None:
        """
        Args:

            argument_tuple (Any):
                MonteCarloActor initialization arguments.
        """

        MonteCarloActor.__init__(self, argument_tuple, index, catch_exceptions)
        SimulationRunner.__init__(self, self._investigated_object)

        # Update the internal random seed pseudo-deterministically for each actor instance
        seed = self._investigated_object._rng.integers(0, maxsize)
        individual_seed = seed + index * 12345678
        self._investigated_object.seed = individual_seed

    @staticmethod
    def stage_identifiers() -> List[str]:
        return [
            "transmit_operators",
            "generate_outputs",
            "propagate",
            "process_inputs",
            "receive_operators",
        ]

    def stage_executors(self) -> List[Callable]:
        return [
            self.transmit_operators,
            self.generate_outputs,
            self.propagate,
            self.process_inputs,
            self.receive_operators,
        ]


class Simulation(
    Serializable, Pipeline[SimulationScenario, SimulatedDevice], MonteCarlo[SimulationScenario]
):
    """HermesPy simulation configuration."""

    yaml_tag = "Simulation"
    property_blacklist = {"console", "console_mode", "scenario"}

    plot_results: bool
    """Plot results after simulation runs"""

    dump_results: bool
    """Dump results to files after simulation runs."""

    def __init__(
        self,
        scenario: SimulationScenario | None = None,
        num_samples: int = 100,
        drop_duration: float = 0.0,
        plot_results: bool = False,
        dump_results: bool = True,
        console_mode: ConsoleMode = ConsoleMode.INTERACTIVE,
        ray_address: str | None = None,
        results_dir: str | None = None,
        verbosity: str | Verbosity = Verbosity.INFO,
        seed: int | None = None,
        num_actors: int | None = None,
    ) -> None:
        """Args:

        scenario (SimulationScenario, optional):
            The simulated scenario.
            If none is provided, an empty one will be initialized.

        num_samples (int, optional):
            Number of drops generated per sweeping grid section.
            100 by default.

        drop_duration(float, optional):
            Duration of simulation drops in seconds.

        plot_results (bool, optional):
            Plot results after simulation runs.
            Disabled by default.

        dump_results (bool, optional):
            Dump results to files after simulation runs.
            Enabled by default.

        ray_address (str, optional):
            The address of the ray head node.
            If None is provided, the head node will be launched in this machine.

        results_dir (str, optional):
            Directory in which all simulation artifacts will be dropped.

        verbosity (Union[str, Verbosity], optional):
            Information output behaviour during execution.

        seed (int, optional):
            Random seed used to initialize the pseudo-random number generator.
        """

        scenario = SimulationScenario() if scenario is None else scenario

        if seed is not None:
            scenario.seed = seed

        # Initialize base classes
        Pipeline.__init__(
            self, scenario, results_dir=results_dir, verbosity=verbosity, console_mode=console_mode
        )
        MonteCarlo.__init__(
            self,
            self.scenario,
            num_samples,
            console=self.console,
            console_mode=console_mode,
            ray_address=ray_address,
            num_actors=num_actors,
        )

        self.plot_results = plot_results
        self.dump_results = dump_results
        self.drop_duration = drop_duration
        self.num_drops = num_samples

    @property
    def num_samples(self) -> int:
        return self.num_drops

    @num_samples.setter
    def num_samples(self, value: int) -> None:
        self.num_drops = value

    @Pipeline.console_mode.setter  # type: ignore
    def console_mode(self, value: ConsoleMode) -> None:  # type: ignore
        Pipeline.console_mode.fset(self, value)  # type: ignore
        MonteCarlo.console_mode.fset(self, value)  # type: ignore

    @Pipeline.console.setter  # type: ignore
    def console(self, value: Console) -> None:  # type: ignore
        Pipeline.console.fset(self, value)  # type: ignore
        MonteCarlo.console.fset(self, value)  # type: ignore

    def run(self) -> MonteCarloResult:
        # Print indicator that the simulation is starting
        if self.console_mode != ConsoleMode.SILENT:
            self.console.print()  # Just an empty line
            self.console.rule("Simulation Campaign")
            self.console.print()  # Just an empty line

        # Generate simulation result
        result = self.simulate(SimulationActor)

        # Visualize results if the flag respective is enabled
        figures: List[plt.FigureBase] = []
        if self.plot_results:
            with self.style_context():
                figures = result.plot()

        # Dump results if the respective flag is enabled
        if self.dump_results and self.results_dir is not None:
            # Save figures to png files
            for figure_idx, base_figure in enumerate(figures):
                figure_instace = base_figure.get_figure()
                if figure_instace is not None:
                    figure_instace.savefig(
                        path.join(self.results_dir, f"figure_{figure_idx}.png"), format="png"
                    )

            # Save results to matlab file
            result.save_to_matlab(path.join(self.results_dir, "results.mat"))

        # Show plots if the flag is enabled
        # if self.plot_results:
        #    plt.show()

        # Return result object
        return result

    def set_channel(
        self, alpha: int | SimulatedDevice, beta: int | SimulatedDevice, channel: Channel | None
    ) -> None:
        """Specify a channel within the channel matrix.

        Convenience method resolving to :meth:`.SimulationScenario.set_channel`.

        Args:

            receiver (int | SimulatedDevice):
                Index of the receiver within the channel matrix.

            transmitter (int | SimulatedDevice):
                Index of the transmitter within the channel matrix.

            channel (Channel | None):
                The channel instance to be set at position (`transmitter_index`, `receiver_index`).
        """

        self.scenario.set_channel(alpha, beta, channel)

    @classmethod
    def to_yaml(
        cls: Type[Simulation], representer: SafeRepresenter, node: Simulation
    ) -> MappingNode:
        """Serialize an `Simulation` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Simulation):
                The `Simulation` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        # Prepare dimensions
        dimension_fields: List[Mapping[str, Any]] = []
        for dimension in node.dimensions:
            dimension_map = {
                "property": dimension.dimension,
                "points": [p.value for p in dimension.sample_points],
                "title": dimension.title,
            }

            considered_objects = dimension.considered_objects
            if considered_objects != (node.scenario,):
                dimension_map["objects"] = considered_objects

            dimension_fields.append(dimension_map)

        additional_fields = {
            "snr_type": node.scenario.snr_type,
            "verbosity": node.verbosity.name,
            "Devices": node.scenario.devices,
            "Operators": node.scenario.operators,
            "Evaluators": node.evaluators,
            "Dimensions": dimension_fields,
            "Channels": node.scenario.channels.flatten().tolist(),
        }

        return node._mapping_serialization_wrapper(representer, additional_fields=additional_fields)

    @classmethod
    def from_yaml(cls: Type[Simulation], constructor: SafeConstructor, node: Node) -> Simulation:
        """Recall a new `Simulation` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Simulation` serialization.

        Returns:
            Simulation:
                Newly created `Simulation` instance.
        """

        state: dict = constructor.construct_mapping(node, deep=True)

        # Launch a global quadriga instance
        quadriga_interface: Optional[QuadrigaInterface] = state.pop(
            QuadrigaInterface.yaml_tag, None
        )
        if quadriga_interface is not None:  # pragma: no cover
            QuadrigaInterface.SetGlobalInstance(quadriga_interface)

        # Pop configuration sections for "special" treatment
        devices: List[SimulatedDevice] = state.pop("Devices", [])
        channels: List[Channel] = state.pop("Channels", [])
        _: List[Operator] = state.pop("Operators", [])
        evaluators: List[Evaluator] = state.pop("Evaluators", [])
        dimensions: Dict[str, Any] | List[Mapping[str, Any]] = state.pop("Dimensions", {})

        # Initialize simulation
        state["scenario"] = SimulationScenario(
            snr=state.pop("snr", float("inf")), snr_type=state.pop("snr_type", SNRType.EBN0)
        )
        simulation: Simulation = cls.InitializationWrapper(state)

        # Add devices to the simulation
        for device in devices:
            simulation.scenario.add_device(device)

        # Assign channel models
        for channel in channels:
            # If the scenario features just a single device, we can infer the transmitter and receiver easily
            if channel.alpha_device is None or channel.beta_device is None:
                if simulation.scenario.num_devices > 1:
                    raise RuntimeError(
                        "Please specifiy the transmitting and receiving device of each channel in a multi-device scenario"
                    )

                channel.alpha_device = simulation.scenario.devices[0]
                channel.beta_device = simulation.scenario.devices[0]

            simulation.scenario.set_channel(channel.alpha_device, channel.beta_device, channel)

        # Register evaluators
        for evaluator in evaluators:
            simulation.add_evaluator(evaluator)

        # Add simulation dimensions
        if isinstance(dimensions, list):
            for dimension in dimensions:
                considered_objects = dimension.get("objects", (simulation.scenario,))
                new_dim = simulation.new_dimension(
                    dimension["property"], dimension["points"], *considered_objects
                )

                title = dimension.get("title", None)
                if title is not None:
                    new_dim.title = title

        else:
            for property_name, property_values in dimensions.items():
                simulation.new_dimension(property_name, property_values, simulation.scenario)

        # Return simulation instance recovered from the serialization
        return simulation

    @staticmethod
    def _pip_packages() -> List[str]:
        return MonteCarlo._pip_packages() + ["sparse", "protobuf", "numba"]
