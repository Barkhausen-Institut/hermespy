# -*- coding: utf-8 -*-

from __future__ import annotations
from collections.abc import Sequence
from itertools import product
from sys import maxsize
from typing import Any, Callable, Dict, List, Mapping, Type

import numpy as np
from os import path
from ray import remote
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode, Node
from rich.console import Console

from hermespy.core import (
    Serializable,
    Pipeline,
    Verbosity,
    Operator,
    ConsoleMode,
    Evaluator,
    MonteCarloActor,
    MonteCarlo,
    MonteCarloResult,
    Signal,
    Visualization,
)
from hermespy.channel import Channel, ChannelRealization
from .scenario import SimulationScenario
from .simulated_device import (
    DeviceState,
    TriggerRealization,
    ProcessedSimulatedDeviceInput,
    SimulatedDevice,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulationRunner(object):
    """Runner remote thread deployed by Monte Carlo routines"""

    __scenario: SimulationScenario  # Scenario to be run
    __leaking_signals: Sequence[Sequence[Signal]] | None
    __trigger_realizations: Sequence[TriggerRealization]
    __propagation: Sequence[Sequence[Signal]] | None
    __processed_inputs: Sequence[ProcessedSimulatedDeviceInput]
    __device_states: Sequence[DeviceState] | None
    __channel_realizations: Sequence[ChannelRealization] | None

    def __init__(self, scenario: SimulationScenario) -> None:
        """
        Args:

            scenario(SimulationScenario):
                Scenario to be run.
        """

        self.__scenario = scenario
        self.__leaking_signals = None
        self.__channel_realizations = None
        self.__device_states = None
        self.__trigger_realizations = None
        self.__propagation = None
        self.__processed_inputs = []

    def realize_channels(self) -> None:
        self.__channel_realizations = self.__scenario.realize_channels()

    def sample_trajectories(self, timestamp: float = 0.0) -> None:
        self.__device_states = [d.state(timestamp) for d in self.__scenario.devices]

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

        self.__trigger_realizations = self.__scenario.realize_triggers()
        self.__leaking_signals = [
            o.leaking_signals
            for o in self.__scenario.generate_outputs(None, self.__trigger_realizations)
        ]

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

        if self.__device_states is None or self.__channel_realizations is None:
            raise RuntimeError(
                "Propagation simulation stage called without prior channel or device realization"
            )

        device_outputs = [device.output for device in self.__scenario.devices]
        if any([t is None for t in device_outputs]):
            raise RuntimeError(
                "Propagation simulation stage called without prior device transmission"
            )

        # Propagate device outputs
        self.__propagation, _ = self.__scenario.propagate(
            device_outputs, self.__device_states, self.__channel_realizations
        )

    def process_inputs(self) -> None:
        """Process device inputs after channel propgation.

        Raises:

            RuntimeError: If the propagation stage is called without prior channel propagation.
            RuntimeError: If the number of arriving signals does not match the number of registered devices.
        """

        propagation_matrix = self.__propagation

        if self.__trigger_realizations is None or self.__leaking_signals is None:
            raise RuntimeError(
                "Process inputs simulation stage without prior call to generate outputs"
            )

        if len(self.__trigger_realizations) != self.__scenario.num_devices:
            raise RuntimeError(
                "Number of trigger realizations does not match the number of registered devices"
            )

        if len(self.__leaking_signals) != self.__scenario.num_devices:
            raise RuntimeError(
                "Number of leaking signals does not match the number of registered devices"
            )

        if propagation_matrix is None:
            raise RuntimeError(
                "Process inputs simulation stage called without prior channel propagation"
            )

        if len(propagation_matrix) != self.__scenario.num_devices:
            raise RuntimeError(
                f"Number of arriving signals ({len(propagation_matrix)}) does not match "
                f"the number of receiving devices ({self.__scenario.num_devices})"
            )

        self.__processed_inputs: Sequence[ProcessedSimulatedDeviceInput] = []
        for device, leaking_signals, impinging_signals, trigger_realization in zip(
            self.__scenario.devices,
            self.__leaking_signals,
            propagation_matrix,
            self.__trigger_realizations,
        ):
            self.__processed_inputs.append(
                device.process_input(
                    impinging_signals=impinging_signals,
                    noise_level=self.__scenario.noise_level,  # type: ignore[operator]
                    noise_model=self.__scenario.noise_model,  # type: ignore[operator]
                    trigger_realization=trigger_realization,
                    leaking_signals=leaking_signals,
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

    def __init__(
        self,
        argument_tuple: Any,
        index: int,
        stage_arguments: Mapping[str, Sequence[tuple]] | None = None,
        catch_exceptions: bool = True,
    ) -> None:
        """
        Args:

            argument_tuple:
                Object to be investigated during the simulation runtime.
                Dimensions over which the simulation will iterate.
                Evaluators used to process the investigated object sample state.

            index (int):
                Global index of the actor.

            stage_arguments (Mapping[str, Sequence[Tuple]], optional):
                Arguments for the simulation stages.

            catch_exceptions (bool, optional):
                Catch exceptions during run.
                Enabled by default.
        """

        # Initialize base classes
        MonteCarloActor.__init__(self, argument_tuple, index, stage_arguments, catch_exceptions)
        SimulationRunner.__init__(self, self._investigated_object)

        # Update the internal random seed pseudo-deterministically for each actor instance
        seed = self._investigated_object._rng.integers(0, maxsize)
        individual_seed = seed + index * 12345678
        self._investigated_object.seed = individual_seed

    @staticmethod
    def stage_identifiers() -> List[str]:
        return [
            "realize_channels",
            "sample_trajectories",
            "transmit_operators",
            "generate_outputs",
            "propagate",
            "process_inputs",
            "receive_operators",
        ]

    def stage_executors(self) -> List[Callable]:
        return [
            self.realize_channels,
            self.sample_trajectories,
            self.transmit_operators,
            self.generate_outputs,
            self.propagate,
            self.process_inputs,
            self.receive_operators,
        ]


class Simulation(
    Serializable, Pipeline[SimulationScenario, SimulatedDevice], MonteCarlo[SimulationScenario]
):
    """Executable HermesPy simulation configuration."""

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
        drop_interval: float = float("inf"),
        plot_results: bool = False,
        dump_results: bool = True,
        console_mode: ConsoleMode = ConsoleMode.INTERACTIVE,
        ray_address: str | None = None,
        results_dir: str | None = None,
        verbosity: str | Verbosity = Verbosity.INFO,
        seed: int | None = None,
        num_actors: int | None = None,
    ) -> None:
        """
        Args:

            scenario (SimulationScenario, optional):
                The simulated scenario.
                If none is provided, an empty one will be initialized.

            num_samples (int, optional):
                Number of drops generated per sweeping grid section.
                100 by default.

            drop_duration(float, optional):
                Duration of simulation drops in seconds.

            drop_interval(float, optional):
                Interval at which drops are being generated in seconds.
                If not specified, only a single drop is generated at the beginning of the simulation.

            plot_results (bool, optional):
                Plot results after simulation runs.
                Disabled by default.

            dump_results (bool, optional):
                Dump results to files after simulation runs.
                Enabled by default.

            console_mode (ConsoleMode, optional):
                Console output behaviour during execution.

            ray_address (str, optional):
                The address of the ray head node.
                If None is provided, the head node will be launched in this machine.

            results_dir (str, optional):
                Directory in which all simulation artifacts will be dropped.

            verbosity (Union[str, Verbosity], optional):
                Information output behaviour during execution.

            seed (int, optional):
                Random seed used to initialize the pseudo-random number generator.

            num_actors (int, optional):
                Number of actors to be deployed for parallel execution.
                If None is provided, the number of actors will be set to the number of available CPU cores.
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

        # Initialize class attributes
        self.plot_results = plot_results
        self.dump_results = dump_results
        self.drop_duration = drop_duration
        self.num_drops = num_samples
        self.drop_interval = drop_interval

    @property
    def num_samples(self) -> int:
        return self.num_drops

    @num_samples.setter
    def num_samples(self, value: int) -> None:
        self.num_drops = value

    @property
    def drop_interval(self) -> float:
        """Interval at which drops are being generated in seconds.

        Raises:
            ValueError for values smaller or equal to zero.
        """

        return self.__drop_interval

    @drop_interval.setter
    def drop_interval(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Drop interval must be greater than zero.")

        self.__drop_interval = value

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

        # Generate timestamps at which drops are generated
        max_timestamp = max(d.trajectory.max_timestamp for d in self.scenario.devices)
        timestamps = (
            np.arange(0, max_timestamp, self.drop_interval, np.float_)
            if max_timestamp > 0.0
            else np.zeros(1, np.float_)
        )
        stage_arguments = dict()
        if timestamps[-1] > 0.0:
            stage_arguments["sample_trajectories"] = [(t,) for t in timestamps]

        if self.console_mode != ConsoleMode.SILENT:
            self.console.log(
                f"Generating {len(timestamps)} drops at an interval of {self.drop_interval} seconds along the trajectories of moveable objects"
            )

        # Generate simulation result
        result = self.simulate(SimulationActor)

        # Visualize results if the flag respective is enabled
        visualizations: Sequence[Visualization] = []
        if self.plot_results:
            with self.style_context():
                visualizations = result.plot()

        # Dump results if the respective flag is enabled
        if self.dump_results and self.results_dir is not None:
            # Save figures to png files
            for figure_idx, visualization in enumerate(visualizations):
                if visualization.figure is not None:
                    visualization.figure.savefig(
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
        self, alpha: SimulatedDevice, beta: SimulatedDevice, channel: Channel | None
    ) -> None:
        """Specify a channel within the channel matrix.

        Convenience method resolving to the :meth:`set_channel<SimulationScenario.set_channel>` method
        of the managed :class:`SimulationScenario` instance,
        which can be accessed via the :attr:`scenario<hermespy.core.monte_carlo.MonteCarlo.scenario>` property.

        Args:

            receiver (SimulatedDevice):
                Index of the receiver within the channel matrix.

            transmitter (SimulatedDevice):
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

        # Collection channel models
        channels = []
        for device_alpha, device_beta in product(node.scenario.devices, node.scenario.devices):
            channel = node.scenario.channel(device_alpha, device_beta)
            if channel is not None:
                channels.append((device_alpha, device_beta, channel))

        additional_fields = {
            "noise_model": node.scenario.noise_model,  # type: ignore[operator]
            "noise_level": node.scenario.noise_level,  # type: ignore[operator]
            "verbosity": node.verbosity.name,
            "Devices": node.scenario.devices,
            "Operators": node.scenario.operators,
            "Evaluators": node.evaluators,
            "Dimensions": dimension_fields,
            "Channels": channels,
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

        # Pop configuration sections for "special" treatment
        devices: List[SimulatedDevice] = state.pop("Devices", [])
        channels: list[tuple[SimulatedDevice, SimulatedDevice, Channel]] = state.pop("Channels", [])
        _: List[Operator] = state.pop("Operators", [])
        evaluators: List[Evaluator] = state.pop("Evaluators", [])
        dimensions: Dict[str, Any] | List[Mapping[str, Any]] = state.pop("Dimensions", {})

        # Initialize simulation
        state["scenario"] = SimulationScenario(
            noise_level=state.pop("noise_level", None), noise_model=state.pop("noise_model", None)
        )
        simulation: Simulation = cls.InitializationWrapper(state)

        # Add devices to the simulation
        for device in devices:
            simulation.scenario.add_device(device)

        # Assign channel models
        for device_alpha, device_beta, channel in channels:
            simulation.scenario.set_channel(device_alpha, device_beta, channel)

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
