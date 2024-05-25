# -*- coding: utf-8 -*-

from __future__ import annotations
from time import time
from typing import List, Sequence, Tuple, overload

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D  # type: ignore
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # type: ignore

from hermespy.channel import (
    Channel,
    ChannelRealization,
    ChannelSample,
    IdealChannel,
    InterpolationMode,
)
from hermespy.core import (
    DeviceInput,
    DeviceOutput,
    register,
    Scenario,
    Signal,
    Transmission,
    VAT,
    VisualizableAttribute,
    Visualization,
)
from .drop import SimulatedDrop
from .noise import NoiseLevel, NoiseModel
from .simulated_device import (
    ProcessedSimulatedDeviceInput,
    SimulatedDevice,
    SimulatedDeviceOutput,
    SimulatedDeviceReception,
    SimulatedDeviceTransmission,
    TriggerModel,
    TriggerRealization,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ScenarioVisualization(Visualization):

    def __init__(
        self,
        figure: plt.Figure | None,
        axes: VAT,
        device_frames: List[Line3DCollection],
        device_frame_scale: float,
    ) -> None:

        # Initialize base class
        Visualization.__init__(self, figure, axes)

        # Store class attributes
        self.device_frames = device_frames
        self.device_frame_scale = device_frame_scale


class _ScenarioVisualizer(VisualizableAttribute[ScenarioVisualization]):
    """Visualize the scenario's spatial configuration."""

    __scenario: SimulationScenario

    def __init__(self, scenario: SimulationScenario) -> None:
        self.__scenario = scenario

    @property
    def title(self) -> str:
        return "Simulation Scenario"

    def create_figure(self, **kwargs) -> Tuple[plt.FigureBase, VAT]:
        return plt.subplots(
            *self._axes_dimensions(**kwargs), squeeze=False, subplot_kw={"projection": "3d"}
        )

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> ScenarioVisualization:

        _ax: Axes3D = axes[0, 0]

        # Plot the trajectories of all devices
        trajectories = [d.trajectory for d in self.__scenario.devices]
        max_time = max([t.max_timestamp for t in trajectories])

        t = np.linspace(0, max_time, 101, endpoint=True)
        color_cycle = self._get_color_cycle()
        x_limits = (0.0, 0.0)
        y_limits = (0.0, 0.0)
        z_limits = (0.0, 0.0)
        for d, trajectory in enumerate(trajectories):

            # Compile the trajectory samples to be visualized
            positions = np.array([trajectory.sample(time).pose.translation for time in t])

            # Update the limits of the plot
            min_points = np.min(positions, axis=0)
            max_points = np.max(positions, axis=0)
            x_limits = min(x_limits[0], min_points[0]), max(x_limits[1], max_points[0])
            y_limits = min(y_limits[0], min_points[1]), max(y_limits[1], max_points[1])
            z_limits = min(z_limits[0], min_points[2]), max(z_limits[1], max_points[2])

            linewidth = min(5, max(1.0, 0.5 * (max(max_points) - min(min_points))))
            _ax.plot(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                color=color_cycle[d],
                linewidth=linewidth,
                label=f"Device {d}",
            )

        # Set the appropriate limits for the plot
        minimal_limit = min(x_limits[0], y_limits[0], z_limits[0])
        maximal_limit = max(x_limits[1], y_limits[1], z_limits[1])
        _ax.set_xlim3d(minimal_limit, maximal_limit)
        _ax.set_ylim3d(minimal_limit, maximal_limit)
        _ax.set_zlim3d(minimal_limit, maximal_limit)
        device_frame_scale = 0.1 * (maximal_limit - minimal_limit)

        device_frames: List[Line3DCollection] = []
        for _ in self.__scenario.devices:

            # Draw wire coordinate frames
            frame_collection = Line3DCollection(
                [
                    [
                        np.zeros(3, dtype=np.float_),
                        np.array([device_frame_scale, 0, 0], dtype=np.float_),
                    ],
                    [
                        np.zeros(3, dtype=np.float_),
                        np.array([0, device_frame_scale, 0], dtype=np.float_),
                    ],
                    [
                        np.zeros(3, dtype=np.float_),
                        np.array([0, 0, device_frame_scale], dtype=np.float_),
                    ],
                ],
                colors=["r", "g", "b"],
            )
            _ax.add_collection3d(frame_collection)
            device_frames.append(frame_collection)

        _ax.set_xlabel("X [m]")
        _ax.set_ylabel("Y [m]")
        _ax.set_zlabel("Z [m]")
        _ax.legend()
        for axis in [_ax.xaxis, _ax.yaxis, _ax.zaxis]:
            axis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        return ScenarioVisualization(figure, axes, device_frames, device_frame_scale)

    def _update_visualization(
        self, visualization: ScenarioVisualization, time: float = 0.0, **kwargs
    ) -> None:
        # Update the device frames according to the device poses at the given time
        s = visualization.device_frame_scale
        for device, frame in zip(self.__scenario.devices, visualization.device_frames):
            pose = device.trajectory.sample(time).pose
            line_vectors = pose @ np.array([[s, 0, 0], [0, s, 0], [0, 0, s], [1, 1, 1]])
            frame.set_segments(
                [
                    [pose[:3, 3], line_vectors[:3, 0]],
                    [pose[:3, 3], line_vectors[:3, 1]],
                    [pose[:3, 3], line_vectors[:3, 2]],
                ]
            )


class SimulationScenario(Scenario[SimulatedDevice]):
    """Description of a physical layer wireless communication scenario."""

    yaml_tag = "SimulationScenario"

    __channels: np.ndarray  # Channel matrix linking devices
    __noise_level: NoiseLevel | None  # Global noise level of the scenario
    __noise_model: NoiseModel | None  # Global noise model of the scenario

    def __init__(
        self,
        noise_level: NoiseLevel | None = None,
        noise_model: NoiseModel | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:

            noise_level (NoiseLevel, optional):
                Global noise level of the scenario assumed for all devices.
                If not specified, the noise configuration is device-specific.

            noise_model (NoiseModel, optional):
                Global noise model of the scenario assumed for all devices.
                If not specified, the noise configuration is device-specific.
        """

        # Initialize base class
        Scenario.__init__(self, *args, **kwargs)

        # Initialize class attributes
        self.noise_level = noise_level
        self.noise_model = noise_model
        self.__channels = np.ndarray((0, 0), dtype=object)
        self.__visualizer = _ScenarioVisualizer(self)

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

    @register(first_impact="receive_devices", title="Scenario Noise Level")  # type: ignore[misc]
    @property
    def noise_level(self) -> NoiseLevel | None:
        """Global noise level of the scenario.

        If not set, i.e. `None`, the noise level is device-specific.
        """

        return self.__noise_level

    @noise_level.setter
    def noise_level(self, value: NoiseLevel | None) -> None:
        self.__noise_level = value

    @register(first_impact="receive_devices", title="Noise Model")  # type: ignore[misc]
    @property
    def noise_model(self) -> NoiseModel | None:
        """Global noise model of the scenario.

        If not set, i.e. `None`, the noise model is device-specific.
        """

        return self.__noise_model

    @noise_model.setter
    def noise_model(self, value: NoiseModel | None) -> None:
        self.__noise_model = value

        if value is not None:
            self.__noise_model.random_mother = self

    def realize_triggers(self) -> Sequence[TriggerRealization]:
        """Realize the trigger models of all registered devices.

        Devices sharing trigger models will be triggered simulatenously.

        Returns: A sequence of trigger model realizations.
        """

        # Collect unique triggers
        triggers: List[TriggerModel] = []
        unique_realizations: List[TriggerRealization] = []
        device_realizations: List[TriggerRealization] = []

        for device in self.devices:
            device_realization: TriggerRealization

            if device.trigger_model not in triggers:
                device_realization = device.trigger_model.realize(self._rng)

                triggers.append(device.trigger_model)
                unique_realizations.append(device_realization)

            else:
                device_realization = unique_realizations[triggers.index(device.trigger_model)]

            device_realizations.append(device_realization)

        return device_realizations

    def generate_outputs(
        self,
        transmissions: List[List[Transmission]] | None = None,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> Sequence[SimulatedDeviceOutput]:
        # Assume cached operator transmissions if none were provided
        _transmissions: List[None] | List[List[Transmission]] = (
            [None] * self.num_devices if not transmissions else transmissions
        )

        if len(_transmissions) != self.num_devices:
            raise ValueError(
                f"Number of device transmissions ({len(_transmissions)}) does not match number of registered devices ({self.num_devices}"
            )

        _trigger_realizations = (
            self.realize_triggers() if trigger_realizations is None else trigger_realizations
        )

        if len(_trigger_realizations) != self.num_devices:
            raise ValueError(
                f"Number of trigger realizations ({len(_trigger_realizations)}) does not match number of registered devices ({self.num_devices}"
            )

        outputs = [
            d.generate_output(t, True, tr)
            for d, t, tr in zip(self.devices, _transmissions, _trigger_realizations)
        ]
        return outputs

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

        # Realize triggers
        trigger_realizations = self.realize_triggers()

        # Transmit devices
        transmissions: List[SimulatedDeviceTransmission] = [
            d.transmit(cache=cache, trigger_realization=t)
            for d, t in zip(self.devices, trigger_realizations)
        ]
        return transmissions

    def propagate(
        self, transmissions: Sequence[DeviceOutput]
    ) -> Tuple[List[List[Signal]], List[ChannelRealization]]:
        """Propagate device transmissions over the scenario's channel instances.

        Args:

            transmissions (Sequence[DeviceOutput])
                Sequence of device transmissisons.

        Returns:
            - Matrix of signal propagations between devices.
            - List of lists of unique channel realizations linking the devices.

        Raises:

            ValueError: If the length of `transmissions` does not match the number of registered devices.
        """

        if len(transmissions) != self.num_devices:
            raise ValueError(
                f"Number of transmit signals ({len(transmissions)}) does not match "
                f"the number of registered devices ({self.num_devices})"
            )

        # Initialize the propagated signals
        propagation_matrix = np.empty((self.num_devices, self.num_devices), dtype=np.object_)

        # Loop over each channel within the channel matrix and propagate the signals over the respective channel model
        channel_realizations: List[ChannelRealization] = []
        for device_alpha_idx, alpha_device in enumerate(self.devices):
            for device_beta_idx, beta_device in enumerate(self.devices[: 1 + device_alpha_idx]):

                # Select and realize the channel linking device alpha and device beta
                channel: Channel[ChannelRealization, ChannelSample] = self.channels[
                    device_alpha_idx, device_beta_idx
                ]
                channel_realization: ChannelRealization[ChannelSample] = channel.realize()
                channel_realizations.append(channel_realization)

                # Sample the channel realization for a propagation from device alpha to device beta
                alpha_beta_sample = channel_realization.sample(alpha_device, beta_device)

                # Sample the reciprocal channel realization for a propagation from device beta to device alpha
                beta_alpha_sample = channel_realization.reciprocal_sample(
                    alpha_beta_sample, beta_device, alpha_device
                )

                # Propagate signal emitted from device alpha to device beta over the linking channel
                alpha_propagation = alpha_beta_sample.propagate(
                    transmissions[device_alpha_idx], InterpolationMode.NEAREST
                )

                # Propagate signal emitted from device beta to device alpha over the linking channel
                beta_propagation = beta_alpha_sample.propagate(
                    transmissions[device_beta_idx], InterpolationMode.NEAREST
                )

                # Store propagtions in their respective coordinates within the propagation matrix
                propagation_matrix[device_alpha_idx, device_beta_idx] = beta_propagation
                propagation_matrix[device_beta_idx, device_alpha_idx] = alpha_propagation

        return propagation_matrix.tolist(), channel_realizations

    @overload
    def process_inputs(
        self,
        impinging_signals: Sequence[DeviceInput],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> List[ProcessedSimulatedDeviceInput]: ...  # pragma: no cover

    @overload
    def process_inputs(
        self,
        impinging_signals: Sequence[Signal],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> List[ProcessedSimulatedDeviceInput]: ...  # pragma: no cover

    @overload
    def process_inputs(
        self,
        impinging_signals: Sequence[Sequence[Signal]],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> List[ProcessedSimulatedDeviceInput]: ...  # pragma: no cover

    def process_inputs(
        self,
        impinging_signals: Sequence[DeviceInput] | Sequence[Signal] | Sequence[Sequence[Signal]],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> List[ProcessedSimulatedDeviceInput]:
        """Process input signals impinging onto the scenario's devices.

        Args:

            impinging_signals (Sequence[DeviceInput | Signal | Sequence[Signal]] | Sequence[Sequence[Signal]]):
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
        processed_inputs = [d.process_input(i, cache, t, self.noise_level, self.noise_model) for d, i, t in zip(self.devices, impinging_signals, trigger_realizations)]  # type: ignore

        return processed_inputs

    @overload
    def receive_devices(
        self,
        impinging_signals: Sequence[DeviceInput],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> Sequence[SimulatedDeviceReception]: ...  # pragma: no cover

    @overload
    def receive_devices(
        self,
        impinging_signals: Sequence[Signal],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> Sequence[SimulatedDeviceReception]: ...  # pragma: no cover

    @overload
    def receive_devices(
        self,
        impinging_signals: Sequence[Sequence[Signal]],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> Sequence[SimulatedDeviceReception]: ...  # pragma: no cover

    def receive_devices(
        self,
        impinging_signals: Sequence[DeviceInput] | Sequence[Signal] | Sequence[Sequence[Signal]],
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
        channel_propagations, channel_realizations = self.propagate(device_transmissions)

        # Process receptions
        trigger_realizations = [t.trigger_realization for t in device_transmissions]
        device_receptions = self.receive_devices(
            channel_propagations, trigger_realizations=trigger_realizations
        )

        # Return finished drop
        return SimulatedDrop(
            timestamp, device_transmissions, channel_realizations, device_receptions
        )

    @property
    def visualize(self) -> _ScenarioVisualizer:
        return self.__visualizer
