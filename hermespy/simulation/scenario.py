# -*- coding: utf-8 -*-

from __future__ import annotations
from time import time
from typing import Sequence, Tuple, overload

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
        device_frames: list[Line3DCollection],
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

        device_frames: list[Line3DCollection] = []
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

    __default_channel: Channel  # Initial channel to be assumed for device links
    __channels: set[Channel]  # Set of unique channel model instances
    __links: dict[frozenset[SimulatedDevice], Channel]
    __noise_level: NoiseLevel | None  # Global noise level of the scenario
    __noise_model: NoiseModel | None  # Global noise model of the scenario

    def __init__(
        self,
        default_channel: Channel | None = None,
        noise_level: NoiseLevel | None = None,
        noise_model: NoiseModel | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:

            default_channel (Channel, optional):
                Default channel model to be assumed for all device links.
                If not specified, the `default_channel` is set to an ideal distortionless channel model.

            noise_level (NoiseLevel, optional):
                Global noise level of the scenario assumed for all devices.
                If not specified, the noise configuration is device-specific.

            noise_model (NoiseModel, optional):
                Global noise model of the scenario assumed for all devices.
                If not specified, the noise configuration is device-specific.
        """

        # Prepare channel matrices for device links
        self.__default_channel = default_channel if default_channel is not None else IdealChannel()
        self.__channels = {self.__default_channel}
        self.__links = dict()

        # Initialize base class
        Scenario.__init__(self, *args, **kwargs)

        # Initialize class attributes
        self.noise_level = noise_level
        self.noise_model = noise_model
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

    @property
    def channels(self) -> set[Channel]:
        """Unique channel model instances interconnecting devices within this scenario."""

        return self.__channels

    def channel(self, alpha_device: SimulatedDevice, beta_device: SimulatedDevice) -> Channel:
        """Access a specific channel between two devices.

        Args:

            alpha_device (SimulatedDevice):
                First device linked by the requested channel.

            beta_device (SimulatedDevice):
                Second device linked by the requested channel.

        Returns:
            Channel:
                Channel between `transmitter` and `receiver`.

        Raises:
            ValueError:
                Should `alpha_device` or `beta_device` not be registered with this scenario.
        """

        if alpha_device not in self.devices:
            raise ValueError("Provided alpha device is not registered with this scenario")

        if beta_device not in self.devices:
            raise ValueError("Provided beta device is not registered with this scenario")

        return self.__links.get(frozenset((alpha_device, beta_device)), self.__default_channel)

    def device_channels(self, device: SimulatedDevice, active_only: bool = False) -> set[Channel]:
        """Collect all channels to which a specific device is linked.

        Args:

            device (SimulatedDevice):
                The device in question.

            active_only (bool, optional):
                Consider only active channels.
                A channel is considered active if its gain is greater than zero.
                Disabled by default, so all channels are considered.

        Returns: A set of unique channel instances.

        Raises:

            ValueError: Should `device` is not registered within this scenario.
        """

        if device not in self.devices:
            raise ValueError("Provided device is not registered with this scenario")

        device_channels: set[Channel] = set()
        link_entries = 0
        for linked_devices, channel in self.__links.items():
            if device in linked_devices:
                if not active_only or channel.gain > 0:
                    device_channels.add(channel)
                    link_entries += 1

        # Append the default channel if required
        if link_entries < self.num_devices:
            device_channels.add(self.__default_channel)

        return device_channels

    def set_channel(
        self, alpha_device: SimulatedDevice, beta_device: SimulatedDevice, channel: Channel
    ) -> None:
        """Specify a channel within the channel matrix.

        Args:

            alpha_device (SimulatedDevice):
                First device to be linked by `channel`.

            beta_device (SimulatedDevice):
                Second device to be linked by `channel`.

            channel (Channel):
                The channel instance to link `alpha_device` and `beta_device`.

        Raises:
            ValueError: If `alpha_device` or `beta_device` are not registered with this scenario.
        """

        if alpha_device not in self.devices:
            raise ValueError("Alpha device is not registered with this scenario")

        if beta_device not in self.devices:
            raise ValueError("Beta device is not registered with this scenario")

        # Update the link
        link_key = frozenset((alpha_device, beta_device))
        old_channel = self.__links.get(link_key, None)
        self.__links[frozenset((alpha_device, beta_device))] = channel

        # Remove the old channel from the set of device channels and unique channel instances
        # if it is not linked to any other device
        if old_channel is not None:
            if old_channel not in self.__links.values():
                self.__channels.remove(old_channel)

        # Update the set of unique channel instances
        self.__channels.add(channel)
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
        triggers: list[TriggerModel] = []
        unique_realizations: list[TriggerRealization] = []
        device_realizations: list[TriggerRealization] = []

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
        transmissions: list[list[Transmission]] | None = None,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> Sequence[SimulatedDeviceOutput]:
        # Assume cached operator transmissions if none were provided
        _transmissions: list[None] | list[list[Transmission]] = (
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
        transmissions: list[SimulatedDeviceTransmission] = [
            d.transmit(cache=cache, trigger_realization=t)
            for d, t in zip(self.devices, trigger_realizations)
        ]
        return transmissions

    def propagate(
        self,
        transmissions: Sequence[DeviceOutput],
        interpolation_mode: InterpolationMode = InterpolationMode.NEAREST,
    ) -> Tuple[list[list[Signal]], list[ChannelRealization]]:
        """Propagate device transmissions over the scenario's channel instances.

        Args:

            transmissions (Sequence[DeviceOutput])
                Sequence of device transmissisons.

            interpolation_mode (InterpolationMode, optional):
                Interpolation mode for the channel samples.
                Defaults to `InterpolationMode.NEAREST`.

        Returns:
            - Matrix of signal propagations between devices.
            - list of lists of unique channel realizations linking the devices.

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

        # Realize all channel instances
        channel_realizations: dict[Channel, ChannelRealization] = {
            c: c.realize() for c in self.channels
        }

        # Propagate signals over all linking channels
        for device_alpha_idx, alpha_device in enumerate(self.devices):
            for device_beta_idx, beta_device in enumerate(self.devices[: 1 + device_alpha_idx]):

                # Find the correct channel realization for the propagation between device alpha and device beta
                linking_channel = self.channel(alpha_device, beta_device)
                channel_realization = channel_realizations[linking_channel]

                # Sample the channel realization for a propagation from device alpha to device beta
                alpha_beta_sample: ChannelSample = channel_realization.sample(
                    alpha_device, beta_device
                )

                # Propagate signal emitted from device alpha to device beta over the linking channel
                propagation_matrix[device_beta_idx, device_alpha_idx] = alpha_beta_sample.propagate(
                    transmissions[device_alpha_idx], interpolation_mode
                )

                # Abort if we're on the self-interference diagonal to avoid redundant calculations
                if device_alpha_idx == device_beta_idx:
                    continue

                # Sample the reciprocal channel realization for a propagation from device beta to device alpha
                beta_alpha_sample: ChannelSample = channel_realization.reciprocal_sample(
                    alpha_beta_sample, beta_device, alpha_device
                )

                # Propagate signal emitted from device beta to device alpha over the linking channel
                propagation_matrix[device_alpha_idx, device_beta_idx] = beta_alpha_sample.propagate(
                    transmissions[device_beta_idx], interpolation_mode
                )

        return propagation_matrix.tolist(), list(channel_realizations.values())

    @overload
    def process_inputs(
        self,
        impinging_signals: Sequence[DeviceInput],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> list[ProcessedSimulatedDeviceInput]: ...  # pragma: no cover

    @overload
    def process_inputs(
        self,
        impinging_signals: Sequence[Signal],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> list[ProcessedSimulatedDeviceInput]: ...  # pragma: no cover

    @overload
    def process_inputs(
        self,
        impinging_signals: Sequence[Sequence[Signal]],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> list[ProcessedSimulatedDeviceInput]: ...  # pragma: no cover

    def process_inputs(
        self,
        impinging_signals: Sequence[DeviceInput] | Sequence[Signal] | Sequence[Sequence[Signal]],
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> list[ProcessedSimulatedDeviceInput]:
        """Process input signals impinging onto the scenario's devices.

        Args:

            impinging_signals (Sequence[DeviceInput | Signal | Sequence[Signal]] | Sequence[Sequence[Signal]]):
                list of signals impinging onto the devices.

            cache (bool, optional):
                Cache the operator inputs at the registered receive operators for further processing.
                Enabled by default.

            trigger_realizations (Sequence[TriggerRealization], optional):
                Sequence of trigger realizations.
                If not specified, ideal triggerings are assumed for all devices.

        Returns: list of the processed device input information.

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

            impinging_signals (list[Union[DeviceInput, Signal, Iterable[Signal]]]):
                list of signals impinging onto the devices.

            cache (bool, optional):
                Cache the operator inputs at the registered receive operators for further processing.
                Enabled by default.

            trigger_realizations (Sequence[TriggerRealization], optional):
                Sequence of trigger realizations.
                If not specified, ideal triggerings are assumed for all devices.

        Returns: list of the processed device input information.

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
