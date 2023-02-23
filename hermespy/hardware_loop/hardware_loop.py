# -*- coding: utf-8 -*-
"""
=============
Hardware Loop
=============
"""

from __future__ import annotations
from contextlib import nullcontext
from os import path
from typing import Any, Dict, Generic, List, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.prompt import Confirm
from ruamel.yaml import SafeConstructor, Node

from hermespy.core import ConsoleMode, Drop, Evaluation, Evaluator, MonteCarloResult, Pipeline, Serializable, Verbosity
from hermespy.core.monte_carlo import GridDimension, SampleGrid, GridSection, MonteCarloSample
from hermespy.tools import tile_figures
from .physical_device import PhysicalDevice, PhysicalDeviceType
from .scenario import PhysicalScenarioType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class HardwareLoop(Serializable, Generic[PhysicalScenarioType, PhysicalDeviceType], Pipeline[PhysicalScenarioType, PhysicalDeviceType]):
    """Hermespy hardware loop configuration."""

    yaml_tag = "HardwareLoop"
    """YAML serialization tag"""

    manual_triggering: bool
    """Require a user input to trigger each drop manually"""

    plot_information: bool
    """Plot information during loop runtime"""

    __dimensions: List[GridDimension]  # Parameter dimension over which a run sweeps
    __evaluators: List[Evaluator]  # Evaluators further processing drop information

    def __init__(self, system: PhysicalScenarioType, manual_triggering: bool = False, plot_information: bool = True, **kwargs) -> None:
        """
        Args:

            system (PhysicalScenarioType):
                The physical scenario being controlled by the hardware loop.

            manual_triggering (bool, optional):
                Require a keyboard user input to trigger each drop manually.
                Disabled by default.

            plot_information (bool, optional):
                Plot information during loop runtime.
                Enabled by default.
        """

        Pipeline.__init__(self, scenario=system, **kwargs)

        self.manual_triggering = manual_triggering
        self.plot_information = plot_information
        self.__dimensions = []
        self.__evaluators = []

    def new_dimension(self, dimension: str, sample_points: List[Any], *args: Tuple[Any]) -> GridDimension:
        """Add a dimension to the sweep grid.

        Must be a property of the :meth:`HardwareLoop.scenario`.

        Args:

            dimension (str):
                String representation of dimension location relative to the investigated object.

            sample_points (List[Any]):
                List points at which the dimension will be sampled into a grid.
                The type of points must be identical to the grid arguments / type.

            *args (Tuple[Any], optional):
                References to the object the imension belongs to.
                Resolved to the investigated object by default,
                but may be an attribute or sub-attribute of the investigated object.

        Returns: The newly created dimension object.
        """

        considered_objects = (self.__scenario) if len(args) < 1 else args
        grid_dimension = GridDimension(considered_objects, dimension, sample_points)
        self.add_dimension(grid_dimension)

        return grid_dimension

    def add_dimension(self, dimension: GridDimension) -> None:
        """Add a new dimension to the simulation grid.

        Args:

            dimension:
                Dimension to be added.

        Raises:

            ValueError:
                If the `dimension` already exists within the grid.
        """

        if dimension in self.__dimensions:
            raise ValueError("Dimension instance already registered within the grid")

        self.__dimensions.append(dimension)

    def add_evaluator(self, evaluator: Evaluator) -> None:
        """Add new evaluator to the hardware loop.

        Args:

            evaluator (Evaluator):
                The evaluator to be added.
        """

        self.__evaluators.append(evaluator)

    @property
    def num_evaluators(self) -> int:
        """Number of registered evaluators.

        Returns: The number of evaluators.
        """

        return len(self.__evaluators)

    def __plot_drop(self, drop: Drop, evaluations: List[Evaluation], device_figures: List[plt.Figure], evaluator_figures: List[plt.Figure]) -> None:

        for device_tx, device_rx, figure in zip(drop.device_transmissions, drop.device_receptions, device_figures):

            figure[1][0].clear()
            figure[1][1].clear()

            device_tx.mixed_signal.plot(axes=figure[1][0], space="time", legend=False)
            device_rx.impinging_signals[0].plot(axes=figure[1][1], space="time", legend=False)

            figure[0].canvas.draw()
            figure[0].canvas.flush_events()

        for evaluation, figure in zip(evaluations, evaluator_figures):

            figure[1].clear()
            evaluation.plot(figure[1])

            figure[0].canvas.draw()
            figure[0].canvas.flush_events()

    def run(self, overwrite=True, campaign: str = "default") -> None:
        """Run the hardware loop configuration.

        Args:

            overwrite (bool, optional):
                Allow the replacement of an already existing savefile.

            campaing (str, optional):
                Name of the measurement campaign.
        """

        # Prepare the results file
        if self.results_dir:
            file_location = path.join(self.results_dir, "drops.h5")
            self.scenario.record(file_location, overwrite=overwrite, campaign=campaign)

        # Run internally
        self.__run()

        # Save results and close file streams
        self.scenario.stop()

    def replay(self, file_location: str) -> None:
        """Replay a stored pipeline run.

        Args:

            file_location (str):
                File system location of the replay.
        """

        # Start the scenario replay
        self.scenario.replay(file_location)

        # Run internally
        self.__run()

        # Stop the scenario replay
        self.scenario.stop()

    def __run(self) -> None:
        """Internal run method executing drops"""

        # Prepare figures if plotting is enabled.
        if self.plot_information:

            plt.ion()

            device_figures = []
            evaluator_figures = []

            with self.style_context():

                for device_idx in range(self.scenario.num_devices):

                    sub = plt.subplots(1, 2)
                    sub[0].suptitle(f"Device #{device_idx}")

                    device_figures.append(sub)

                for e, evaluator in enumerate(self.__evaluators):

                    sub = plt.subplots()
                    sub[0].suptitle(f"Evaluator #{e}: {evaluator.title}")

                    evaluator_figures.append(sub)

            # Tile the created figures
            tile_figures(2, 4)

        # Prepare container for evaluation artifacts
        samples = SampleGrid(self.__dimensions, self.__evaluators)

        # Print indicator that the simulation is starting
        if self.console_mode != ConsoleMode.SILENT:

            self.console.print()  # Just an empty line
            self.console.rule("Hardware Loop")
            self.console.print()  # Just an empty line

        # Prepare the console
        progress = Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), transient=True, console=self.console)
        confirm = Confirm(console=self.console)

        num_total_drops = self.num_drops
        grid_tasks = []

        for grid_dimension in self.__dimensions:

            grid_tasks.append(progress.add_task("[cyan]" + grid_dimension.title, total=grid_dimension.num_sample_points))
            num_total_drops *= grid_dimension.num_sample_points

        loop_drop_progress = progress.add_task("[green]Drops", total=self.num_drops)
        total_progress = progress.add_task("[red]Progress", total=num_total_drops)

        with progress if self.console_mode == ConsoleMode.INTERACTIVE else nullcontext():  # type: ignore

            # Start counting the total number of completed drops
            total = 0

            # Generate the number of required drops
            grid = (self.num_drops, *[d.num_sample_points for d in self.__dimensions])
            for indices in np.ndindex(grid):

                # Configure the parameters according to the grid indces
                for dim, p in zip(self.__dimensions, indices[1:]):
                    dim.configure_point(p)

                # Ask for a trigger input if manual mode is enabled
                # Abort execution if user denies
                if self.manual_triggering:

                    progress.live.stop()

                    if not confirm.ask(f"Trigger next drop ({total+1})?"):
                        break

                    progress.live.start()

                # Generate the next drop
                try:

                    drop = self.scenario.drop()

                    # Extract evaluations
                    evaluations = [e.evaluate() for e in self.__evaluators]

                    # Compute artifacts
                    artifacts = [e.artifact() for e in evaluations]

                    # Store artifacts
                    grid_section: GridSection = samples[indices[1:]]
                    grid_section.add_samples(MonteCarloSample(indices[1:], indices[0], artifacts), self.__evaluators)

                    # Update the plotting figures
                    if self.plot_information:
                        self.__plot_drop(drop, evaluations, device_figures, evaluator_figures)

                    # Print results
                    if self.console is not ConsoleMode.SILENT and self.verbosity.value <= Verbosity.INFO.value:

                        result_str = f"# {total:<5}"

                        for dimesion, i in zip(self.__dimensions, indices[1:]):
                            result_str += f" {dimesion.title[-20:]} = {dimesion.sample_points[i]:<5}"

                        for evaluator, artifact in zip(self.__evaluators, artifacts):
                            result_str += f" {evaluator.abbreviation}: {str(artifact):5}"

                        if self.console_mode == ConsoleMode.INTERACTIVE:
                            progress.log(result_str)

                        else:
                            self.console.log(result_str)

                except Exception:
                    self._handle_exception()

                # Update progress
                total += 1

                for task, completed in zip(grid_tasks, indices[1:]):
                    progress.update(task, completed=completed)

                progress.update(loop_drop_progress, completed=indices[0])
                progress.update(total_progress, completed=total)

            if self.plot_information:
                plt.ioff()

        # Compute the evaluation results
        result: MonteCarloResult = MonteCarloResult(self.__dimensions, self.__evaluators, samples, 0.0)

        # Generate result plots
        result_figures = result.plot()

        # Save results if a directory was provided
        if self.results_dir:

            result.save_to_matlab(path.join(self.results_dir, "results.mat"))

            for idx, (figure, evaluator) in enumerate(zip(result_figures, self.__evaluators)):
                figure.savefig(path.join(self.results_dir, f"result_{idx}_{evaluator.abbreviation}.png"), format="png")

        plt.show()

    @classmethod
    def from_yaml(cls: Type[HardwareLoop], constructor: SafeConstructor, node: Node) -> HardwareLoop:

        state = constructor.construct_mapping(node, deep=True)

        devices: List[PhysicalDevice] = state.pop("Devices", [])
        state.pop("Operators", [])
        evaluators: List[Evaluator] = state.pop("Evaluators", [])
        dimensions: Dict[str, Any] = state.pop("Dimensions", {})

        # Initialize the hardware loop
        hardware_loop: HardwareLoop = cls.InitializationWrapper(state)

        # Add devices to the system
        for device in devices:
            hardware_loop.scenario.add_device(device)

        # Register evaluators
        for evaluator in evaluators:
            hardware_loop.add_evaluator(evaluator)

        # Add simulation dimensions
        for dimension_key, dimension_values in dimensions.items():
            hardware_loop.new_dimension(dimension_key, dimension_values)

        # Return fully configured hardware loop
        return hardware_loop
