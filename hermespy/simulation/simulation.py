# -*- coding: utf-8 -*-

from __future__ import annotations
from collections.abc import Sequence
from sys import maxsize
from typing import Any, Callable, List, Mapping

import numpy as np
from os import path
from ray import remote
from rich.console import Console

from hermespy.core import (
    Pipeline,
    Verbosity,
    ConsoleMode,
    MonteCarloActor,
    MonteCarlo,
    MonteCarloResult,
    Signal,
    Transmission,
    Visualization,
)
from hermespy.channel import Channel, ChannelRealization
from .drop import SimulatedDrop
from .scenario import SimulationScenario
from .simulated_device import (
    ProcessedSimulatedDeviceInput,
    SimulatedDevice,
    SimulatedDeviceOutput,
    SimulatedDeviceState,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulationRunner(object):
    """Runner remote thread deployed by Monte Carlo routines"""

    __scenario: SimulationScenario  # Scenario to be run
    __propagation: Sequence[Sequence[Signal]] | None
    __processed_inputs: Sequence[ProcessedSimulatedDeviceInput]
    __device_states: Sequence[SimulatedDeviceState] | None
    __operator_transmissions: Sequence[Sequence[Transmission]] | None
    __device_outputs: Sequence[SimulatedDeviceOutput] | None
    __channel_realizations: Sequence[ChannelRealization] | None

    def __init__(self, scenario: SimulationScenario) -> None:
        """
        Args:

            scenario: Scenario to be run.
        """

        self.__scenario = scenario
        self.__channel_realizations = None
        self.__device_states = None
        self.__operator_transmissions = None
        self.__device_outputs = None
        self.__propagation = None
        self.__processed_inputs = []

    def realize_channels(self) -> None:
        self.__channel_realizations = self.__scenario.realize_channels()

    def sample_states(self, timestamp: float = 0.0) -> None:
        self.__device_states = [d.state(timestamp) for d in self.__scenario.devices]

    def transmit_operators(self) -> None:
        """Generate base-band signal models emitted by all registered transmitting operators.

        Internaly resolves to the scenario's transmit operators routine :meth:`SimulationScenario.transmit_operators`.
        """

        # Resolve to the scenario transmit operators routine
        self.__operator_transmissions = self.__scenario.transmit_operators(self.__device_states)

    def generate_outputs(self) -> None:
        """Generate radio-frequency band signal models emitted by devices.

        Internally resolves to the scenario's generate outputs routine :meth:`SimulationScenario.generate_outputs`.
        """

        self.__device_outputs = self.__scenario.generate_outputs(
            self.__operator_transmissions, self.__device_states
        )

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

        if self.__device_outputs is None:
            raise RuntimeError(
                "Propagation simulation stage called without prior device transmission"
            )

        # Propagate device outputs
        self.__propagation, _ = self.__scenario.propagate(
            self.__device_outputs, self.__device_states, self.__channel_realizations
        )

    def process_inputs(self) -> None:
        """Process device inputs after channel propgation.

        Raises:

            RuntimeError: If the propagation stage is called without prior channel propagation.
            RuntimeError: If the number of arriving signals does not match the number of registered devices.
        """

        propagation_matrix = self.__propagation

        if self.__device_outputs is None:
            raise RuntimeError(
                "Process inputs simulation stage without prior call to generate outputs"
            )

        if len(self.__device_outputs) != self.__scenario.num_devices:
            raise RuntimeError(
                "Number of device outputs does not match the number of registered devices"
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
        device: SimulatedDevice
        state: SimulatedDeviceState
        output: SimulatedDeviceOutput
        for device, state, output, impinging_signals in zip(
            self.__scenario.devices, self.__device_states, self.__device_outputs, propagation_matrix
        ):
            self.__processed_inputs.append(
                device.process_input(
                    impinging_signals,
                    state,
                    output.trigger_realization,
                    self.__scenario.noise_level,  # type: ignore[operator]
                    self.__scenario.noise_model,  # type: ignore[operator]
                    output.leaking_signals,
                )
            )

    def receive_operators(self) -> None:
        """Demodulate base-band signal models received by all registered receiving operators.

        Internally resolves to the scenario's receive operators routine :meth:`SimulationScenario.receive_operators`.
        """

        # Resolve to the scenario's operator receive routine
        _ = self.__scenario.receive_operators(self.__processed_inputs, self.__device_states, True)


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

            index:
                Global index of the actor.

            stage_arguments:
                Arguments for the simulation stages.

            catch_exceptions:
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
            "sample_states",
            "transmit_operators",
            "generate_outputs",
            "propagate",
            "process_inputs",
            "receive_operators",
        ]

    def stage_executors(self) -> List[Callable]:
        return [
            self.realize_channels,
            self.sample_states,
            self.transmit_operators,
            self.generate_outputs,
            self.propagate,
            self.process_inputs,
            self.receive_operators,
        ]


class Simulation(Pipeline[SimulationScenario, SimulatedDevice], MonteCarlo[SimulationScenario]):
    """Executable HermesPy simulation configuration."""

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

            scenario:
                The simulated scenario.
                If none is provided, an empty one will be initialized.

            num_samples:
                Number of drops generated per sweeping grid section.
                100 by default.

            drop_duration:
                Duration of simulation drops in seconds.

            drop_interval:
                Interval at which drops are being generated in seconds.
                If not specified, only a single drop is generated at the beginning of the simulation.

            plot_results:
                Plot results after simulation runs.
                Disabled by default.

            dump_results:
                Dump results to files after simulation runs.
                Enabled by default.

            console_mode:
                Console output behaviour during execution.

            ray_address:
                The address of the ray head node.
                If None is provided, the head node will be launched in this machine.

            results_dir:
                Directory in which all simulation artifacts will be dropped.

            verbosity:
                Information output behaviour during execution.

            seed:
                Random seed used to initialize the pseudo-random number generator.

            num_actors:
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
            ValueError: For values smaller or equal to zero.
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

    @property
    def devices(self) -> Sequence[SimulatedDevice]:
        """Sequence of all devices registered in the simulation scenario"""

        return self.scenario.devices

    def drop(self, timestamp: float = 0.0) -> SimulatedDrop:
        """Generate a drop at a specific timestamp.

        Args:

            timestamp:
                Timestamp at which the drop is generated.
                Defaults to 0.0.
        """

        # Generate a drop at the specified timestamp
        return self.scenario.drop(timestamp)

    def run(self) -> MonteCarloResult:
        # Print indicator that the simulation is starting
        if self.console_mode != ConsoleMode.SILENT:
            self.console.print()  # Just an empty line
            self.console.rule("Simulation Campaign")
            self.console.print()  # Just an empty line

        # Generate timestamps at which drops are generated
        max_timestamp = max(d.trajectory.max_timestamp for d in self.scenario.devices)
        timestamps = (
            np.arange(0, max_timestamp, self.drop_interval, np.float64)
            if max_timestamp > 0.0
            else np.zeros(1, np.float64)
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

        Convenience method to setting the channel of the managed of the managed :class:`SimulationScenario<hermespy.simulation.scenario.SimulationScenario>` instance,
        which can be accessed via the :attr:`scenario<hermespy.core.pipeline.Pipeline.scenario>` property.

        Args:

            receiver:
                Index of the receiver within the channel matrix.

            transmitter:
                Index of the transmitter within the channel matrix.

            channel:
                The channel instance to be set at position (`transmitter_index`, `receiver_index`).
        """

        self.scenario.set_channel(alpha, beta, channel)

    @staticmethod
    def _pip_packages() -> List[str]:
        return MonteCarlo._pip_packages() + ["sparse", "protobuf", "numba"]
