# -*- coding: utf-8 -*-
"""
=============
Hardware Loop
=============
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from threading import Event, Thread
from contextlib import AbstractContextManager, ExitStack
from os import path
from signal import signal, SIGINT
from types import TracebackType
from typing import Any, Generic, List, Mapping, Sequence, Tuple, Type
from warnings import catch_warnings, simplefilter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.prompt import Confirm
from ruamel.yaml import SafeConstructor, Node, SafeRepresenter, MappingNode

from hermespy.core import (
    Artifact,
    ConsoleMode,
    Drop,
    Evaluation,
    EvaluationResult,
    Evaluator,
    MonteCarloResult,
    Pipeline,
    Serializable,
    SerializableEnum,
    VAT,
    Verbosity,
)
from hermespy.core.monte_carlo import GridDimension, SampleGrid, MonteCarloSample, VT
from hermespy.tools import tile_figures
from .physical_device import PDT
from .physical_device_dummy import PhysicalScenarioDummy
from .scenario import PhysicalScenarioType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class IterationPriority(SerializableEnum):
    """Iteration priority of the hardware loop.

    Used by the :meth:`HardwareLoop.run` method.
    """

    DROPS = 0
    """Iterate over drops first before iterating over the parameter grid."""

    GRID = 1
    """Iterate over the parameter grid first before iterating over drops."""


class EvaluatorPlotMode(SerializableEnum):
    """Evalution plot mode during hardware loop runtime."""

    HIDE = 0
    """Do not plot evaluation results during hardware loop runtime."""

    EVALUATION = 1
    """Plot the evaluation during hardware loop runtime."""

    ARTIFACTS = 2
    """Plot the series of generated scalar artifacts during hardware loop runtime."""


class EvaluatorRegistration(Evaluator):
    """Evaluator registration for the hardware loop.

    Created by the :meth:`HardwareLoop.add_evaluator` method.
    """

    __evaluator: Evaluator
    __plot_mode: EvaluatorPlotMode

    def __init__(self, evaluator: Evaluator, plot_mode: EvaluatorPlotMode) -> None:
        """
        Args:

            evaluator (Evaluator):
                Registered evaluator.

            plot_mode (EvaluatorPlotMode):
                Plot mode of the registered evaluator.
        """

        # Initialize class attributes
        self.__evaluator = evaluator
        self.__plot_mode = plot_mode

    @property
    def evaluator(self) -> Evaluator:
        """Registered evaluator."""

        return self.__evaluator

    @property
    def plot_mode(self) -> EvaluatorPlotMode:
        """Plot mode of the registered evaluator."""

        return self.__plot_mode

    def evaluate(self) -> Evaluation:
        return self.__evaluator.evaluate()

    @property
    def abbreviation(self) -> str:
        return self.__evaluator.abbreviation

    @property
    def title(self) -> str:
        return self.__evaluator.title

    @property
    def confidence(self) -> float:
        return self.__evaluator.confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        self.__evaluator.confidence = value

    @property
    def tolerance(self) -> float:
        return self.__evaluator.tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        self.__evaluator.tolerance = value

    def generate_result(
        self, grid: Sequence[GridDimension], artifacts: np.ndarray
    ) -> EvaluationResult:
        return self.__evaluator.generate_result(grid, artifacts)


class HardwareLoopSample(object):
    """Sample of the hardware loop.

    Generated during :meth:`HardwareLoop.run`.
    """

    __drop: Drop
    __evaluations: Sequence[Evaluation]
    __artifacts: Sequence[Artifact]

    def __init__(
        self, drop: Drop, evaluations: Sequence[Evaluation], artifacts: Sequence[Artifact]
    ) -> None:
        # Initialize class attributes
        self.__drop = drop
        self.__evaluations = evaluations
        self.__artifacts = artifacts

    @property
    def drop(self) -> Drop:
        """Drop of the hardware loop sample."""

        return self.__drop

    @property
    def evaluations(self) -> Sequence[Evaluation]:
        """Evaluations of the hardware loop sample."""

        return self.__evaluations

    @property
    def artifacts(self) -> Sequence[Artifact]:
        """Artifacts of the hardware loop sample."""

        return self.__artifacts


class HardwareLoopPlot(ABC, Generic[VT]):
    """Base class for all plots visualized during hardware loop runtime."""

    __hardware_loop: HardwareLoop | None
    __title: str
    __figure: plt.Figure | None
    __axes: VAT | None
    __visualization: VT | None

    def __init__(self, title: str | None = None) -> None:
        """
        Args:

            title (str, optional):
                Title of the hardware loop plot.
                If not specified, resorts to the default title of the plot.
        """

        # Initialize class attributes
        self.__hardware_loop = None
        self.__title = title
        self.__figure = None
        self.__axes = None
        self.__visualization = None

    @property
    def hardware_loop(self) -> HardwareLoop | None:
        """Hardware loop this plot is attached to."""

        return self.__hardware_loop

    @hardware_loop.setter
    def hardware_loop(self, value: HardwareLoop) -> None:
        """Hardware loop this plot is attached to."""

        if self.__hardware_loop is not None:
            raise RuntimeError("Plot already assigned to a hardware loop.")

        self.__hardware_loop = value

    @property
    @abstractmethod
    def _default_title(self) -> str:
        """Default title of the hardware loop plot."""
        ...  # pragma: no cover

    @property
    def title(self) -> str:
        """Title of the hardware loop plot."""

        return self.__title if self.__title else self._default_title

    def prepare_plot(self) -> Tuple[plt.Figure, VAT]:
        """Prepare the figure for the hardware loop plot.

        Returns: The prepared figure and axes to be plotted into.
        """

        self.__figure, self.__axes = self._prepare_plot()
        self.__figure.suptitle(self.title)

        return self.__figure, self.__axes

    @abstractmethod
    def _prepare_plot(self) -> Tuple[plt.Figure, VAT]:
        """Prepare the figure for the hardware loop plot.

        Returns: The prepared figure and axes to be plotted into.
        """
        ...  # pragma: no cover

    def update_plot(self, sample: HardwareLoopSample) -> None:
        """Update the hardware loop plot.

        Internally calls the abstract :meth:`_update_plot` method.

        Args:

            sample (HardwareLoopSample):
                Hardware loop sample to be plotted.

        Raises:

            RuntimeError: If the hardware loop is not set.
        """

        # Assert correct state
        if self.hardware_loop is None:
            raise RuntimeError("Unable plot if hardware loop is not set")

        # Prepare the plot if not done yet
        _axes: VAT
        if self.__axes is None or self.__figure is None:
            figure, _axes = self.prepare_plot()
        else:
            figure = self.__figure
            _axes = self.__axes

        # If the visualizable has not been plotted yet, prepare the plot
        if self.__visualization is None:
            self.__visualization = self._initial_plot(sample, _axes)

        # Otherwise, update the plot data
        else:
            self._update_plot(sample, self.__visualization)

        # Re-draw the plot
        figure.canvas.draw()
        figure.canvas.flush_events()

    @abstractmethod
    def _initial_plot(self, sample: HardwareLoopSample, axes: VAT) -> VT:
        """Initial plot of the hardware loop plot.

        Abstract subroutine of :meth:`HardwareLoopPlot.update_plot`.

        Args:

            sample (HardwareLoopSample): Hardware loop sample to be plotted.
            axes (VAT): The visualization to be plotted into.

        Returns: The plotted information including axes and lines.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _update_plot(self, sample: HardwareLoopSample, visualization: VT) -> None:
        """Update the hardware loop plot.

        Abstract subroutine of :meth:`HardwareLoopPlot.update_plot`.

        Args:

            sample (HardwareLoopSample):
                Hardware loop sample to be plotted.

            visualization (VT): The visualization to be updated.
        """
        ...  # pragma: no cover


class PlotThread(Thread):
    """Thread for parallel plotting during hardware loop runtime."""

    __plot: HardwareLoopPlot
    __sample: HardwareLoopSample | None = None
    __update_plot: Event
    __alive: bool
    __rc_params: mpl.RcParams

    def __init__(self, rc_params: mpl.RcParams, plot: HardwareLoopPlot, **kwargs) -> None:
        """
        Args:

            rc_params (mpl.RcParams): Matplotlib style parameters.
            plot (HardwareLoopPlot): Plot to be visualized by the thread.
            \**kwargs: Additional keyword arguments to be passed to the base class.
        """

        # Initialize class attributes
        self.__plot = plot
        self.__sample = None
        self.__update_plot = Event()
        self.__alive = True
        self.__rc_params = rc_params

        # Initialize base class
        Thread.__init__(self, **kwargs)

    def run(self) -> None:
        with mpl.rc_context(self.__rc_params), plt.ion():
            # Prepare the plot
            with catch_warnings():
                simplefilter("ignore")
                figure, _ = self.__plot.prepare_plot()

            while self.__alive:
                if self.__update_plot.wait(0.1):
                    self.__plot.update_plot(self.__sample)
                    self.__update_plot.clear()

        # Close the plot
        # This is required as to not confuse the matplotlib backend which thread to use
        # in upcoming plots
        plt.close(figure)

    def update_plot(self, sample: HardwareLoopSample) -> None:
        """Update the plot with a new sample.

        Args:

            sample (HardwareLoopSample): Sample to be plotted.
        """

        self.__sample = sample
        self.__update_plot.set()

    def stop(self) -> None:
        self.__alive = False


class ThreadContextManager(AbstractContextManager):
    """Context manager for managing threads.

    Entering the context manager starts the threads, exiting stops them.
    """

    __threads: List[PlotThread]  # Threads managed by this context manager

    def __init__(self, threads: List[PlotThread]) -> None:
        """
        Args:

            threads (List[PlotThread]): Threads to be managed by this context manager.
        """

        # Initialize base class
        super().__init__()

        # Initialize class attributes
        self.__threads = threads

    def __enter__(self) -> Any:
        for thread in self.__threads:
            if not thread.is_alive():
                thread.start()

        super().__enter__()

    def __exit__(
        self,
        __exc_type: Type[BaseException] | None,
        __exc_value: BaseException | None,
        __traceback: TracebackType | None,
    ) -> bool | None:
        # Stop the threads
        for thread in self.__threads:
            thread.stop()

        # Wait for the threads to finish
        for thread in self.__threads:
            thread.join(timeout=1.0)

        return super().__exit__(__exc_type, __exc_value, __traceback)


class HardwareLoop(
    Serializable, Generic[PhysicalScenarioType, PDT], Pipeline[PhysicalScenarioType, PDT]
):
    """Hermespy hardware loop configuration."""

    yaml_tag = "HardwareLoop"
    """YAML serialization tag"""

    property_blacklist = {"console"}
    serialized_attributes = {"scenario", "manual_triggering", "plot_information", "record_drops"}

    manual_triggering: bool
    """Require a user input to trigger each drop manually"""

    plot_information: bool
    """Plot information during loop runtime"""

    record_drops: bool
    """Record drops during loop runtime"""

    __dimensions: List[GridDimension]  # Parameter dimension over which a run sweeps
    __evaluators: List[Evaluator]  # Evaluators further processing drop information
    __plots: List[HardwareLoopPlot]
    __iteration_priority: IterationPriority
    __interrupt_run: bool

    def __init__(
        self,
        scenario: PhysicalScenarioType,
        manual_triggering: bool = False,
        plot_information: bool = True,
        iteration_priority: IterationPriority = IterationPriority.DROPS,
        record_drops: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:

            scenario (PhysicalScenarioType):
                The physical scenario being controlled by the hardware loop.

            manual_triggering (bool, optional):
                Require a keyboard user input to trigger each drop manually.
                Disabled by default.

            plot_information (bool, optional):
                Plot information during loop runtime.
                Enabled by default.

            iteration_priority (IterationPriority, optional):
                Which dimension to iterate over first.
                Defaults to :attr:`IterationPriority.DROPS`.

            record_drops (bool, optional):
                Record drops during loop runtime.
                Enabled by default.
        """

        # Initialize base classes
        Pipeline.__init__(self, scenario=scenario, **kwargs)

        # Initialize class attributes
        self.manual_triggering = manual_triggering
        self.plot_information = plot_information
        self.iteration_priority = iteration_priority
        self.record_drops = record_drops
        self.__dimensions = []
        self.__evaluators = []
        self.__plots = []
        self.__interrupt_run = False

    def new_dimension(
        self, dimension: str, sample_points: List[Any], *args: Tuple[Any]
    ) -> GridDimension:
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

            dimension (GridDimension):
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

    def add_plot(self, plot: HardwareLoopPlot) -> None:
        """Add a new plot to be  visualized by the hardware loop during runtime.

        Args:

            plot (HardwareLoopPlot):
                The plot to be added.
        """

        plot.hardware_loop = self
        self.__plots.append(plot)

    @property
    def evaluators(self) -> List[Evaluator]:
        """List of registered evaluators."""

        return self.__evaluators

    @property
    def num_evaluators(self) -> int:
        """Number of registered evaluators.

        Returns: The number of evaluators.
        """

        return len(self.__evaluators)

    def evaluator_index(self, evaluator: Evaluator) -> int:
        """Index of the given evaluator.

        Args:

            evaluator (Evaluator):
                The evaluator to be searched for.

        Returns: The index of the evaluator.
        """

        return self.__evaluators.index(evaluator)

    @property
    def iteration_priority(self) -> IterationPriority:
        """Iteration priority of the hardware loop."""

        return self.__iteration_priority

    @iteration_priority.setter
    def iteration_priority(self, value: IterationPriority) -> None:
        self.__iteration_priority = value

    def run(self, overwrite=True, campaign: str = "default") -> None:
        """Run the hardware loop configuration.

        Args:

            overwrite (bool, optional):
                Allow the replacement of an already existing savefile.

            campaing (str, optional):
                Name of the measurement campaign.
        """

        # Prepare the results file
        # Only required if the results directory is set and drops are recorded
        if self.results_dir:
            if self.record_drops:
                file_location = path.join(self.results_dir, "drops.h5")

                # Workaround for scenarios which might not be serializable:
                # Create a dummy scenario and patch the operators for recording
                if not isinstance(self.scenario, Serializable):
                    state = PhysicalScenarioDummy()
                    for device in self.scenario.devices:
                        state.new_device(
                            carrier_frequency=device.carrier_frequency,
                            sampling_rate=device.sampling_rate,
                        )

                    # Patch operators for recording
                    operator_devices = [
                        (operator.device, self.scenario.device_index(operator.device))
                        for operator in self.scenario.operators
                    ]

                    for (_, device_idx), operator in zip(operator_devices, self.scenario.operators):
                        operator.device = state.devices[device_idx]  # type: ignore

                    self.scenario.record(
                        file_location, overwrite=overwrite, campaign=campaign, state=state
                    )

                    for (_, device_idx), operator in zip(operator_devices, state.operators):
                        operator.device = self.scenario.devices[device_idx]  # type: ignore

                else:
                    self.scenario.record(file_location, overwrite=overwrite, campaign=campaign)

            else:
                if (
                    self.console_mode is not ConsoleMode.SILENT
                    and self.verbosity.value >= Verbosity.INFO.value
                ):
                    self.console.print("Skipping drop recording", style="bright_yellow")

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

    def __generate_sample(
        self, section_indices: Tuple[int, ...], sample_index: int
    ) -> HardwareLoopSample:
        """Generate a sample from the grid.

        Args:

            section_indices (Tuple[int, ...]):
                The indices of the section within the sample grid.

            sample_index (int):
                Index of the sample within the section.

        Returns: The generated sample.
        """

        # Configure the parameters according to the grid indces
        for dim, p in zip(self.__dimensions, section_indices):
            dim.configure_point(p)

        # Generate a new drop (triggers the hardware)
        drop = self.scenario.drop()

        # Generate evaluations and artifacts
        evaluations = [e.evaluate() for e in self.__evaluators]
        artifacts = [e.artifact() for e in evaluations]

        # Return sample
        return HardwareLoopSample(drop, evaluations, artifacts)

    def __sigint_handler(self, signum: int, frame: Any) -> None:
        """Signal handler for SIGINT."""

        # Print a message
        if self.console_mode is not ConsoleMode.SILENT:
            self.console.log("Received SIGINT, stopping hardware loop", style="bright_red")

        self.__interrupt_run = True

    def __run(self) -> None:
        """Internal run method executing drops"""

        # Reset variables
        self.__interrupt_run = False

        # Register sigint handler to this instance
        signal(SIGINT, self.__sigint_handler)

        # Initialize the sample grid
        sample_grid = SampleGrid(self.__dimensions, self.__evaluators)

        # Print indicator that the simulation is starting
        if (
            self.console_mode is not ConsoleMode.SILENT
            and self.verbosity.value >= Verbosity.INFO.value
        ):  # pragma: no cover
            self.console.print()  # Just an empty line
            self.console.rule("Hardware Loop")
            self.console.print()  # Just an empty line

        # Prepare the console
        progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            transient=True,
            console=self.console,
        )
        confirm = Confirm(console=self.console)

        num_total_drops = self.num_drops
        grid_tasks = []

        for grid_dimension in self.__dimensions:
            grid_tasks.append(
                progress.add_task(
                    "[cyan]" + grid_dimension.title, total=grid_dimension.num_sample_points
                )
            )
            num_total_drops *= grid_dimension.num_sample_points

        loop_drop_progress = progress.add_task("[green]Drops", total=self.num_drops)
        total_progress = progress.add_task("[red]Progress", total=num_total_drops)

        with ExitStack() as stack:
            # Initialize plots
            plot_threads: List[PlotThread] = []
            if self.plot_information:
                with self.style_context():
                    rc_params = mpl.rcParams.copy()
                    plot_threads = [
                        PlotThread(rc_params, plot, name=f"HardwareLoop-Plot-{p}")
                        for p, plot in enumerate(self.__plots)
                    ]

                # Tile the generated figures
                tile_figures(2, 4)

            # Add all threads to the context stack
            stack.enter_context(ThreadContextManager(plot_threads))

            # Add the progress bar to the context stack
            if self.console_mode == ConsoleMode.INTERACTIVE:
                stack.enter_context(progress)

            # Start counting the total number of completed drops
            total = 0

            # Generate the number of required drops
            index_grid: Tuple[int, ...] = tuple()

            if self.iteration_priority is IterationPriority.GRID:
                index_grid = (self.num_drops, *[d.num_sample_points for d in self.__dimensions])
                grid_selector = slice(1, 1 + len(self.__dimensions))
                drop_selector = 0

            elif self.iteration_priority is IterationPriority.DROPS:
                index_grid = (*[d.num_sample_points for d in self.__dimensions], self.num_drops)
                grid_selector = slice(0, len(self.__dimensions))
                drop_selector = len(self.__dimensions)

            else:
                # This should never happen
                raise RuntimeError(f"Invalid iteration priority: {self.iteration_priority}")

            for indices in np.ndindex(index_grid):
                # Abort if sigint is received
                if self.__interrupt_run:
                    break

                sample_index = indices[drop_selector]
                section_indices = indices[grid_selector]

                # Ask for a trigger input if manual mode is enabled
                # Abort execution if user denies
                if self.manual_triggering:  # pragma: no cover
                    progress.live.stop()

                    if not confirm.ask(f"Trigger next drop ({total+1})?"):
                        break

                    progress.live.start()

                # Generate the next drop
                try:
                    # Generate a new samples
                    loop_sample = self.__generate_sample(section_indices, sample_index)
                    grid_sample = MonteCarloSample(
                        section_indices, sample_index, loop_sample.artifacts
                    )

                    # Save sample
                    sample_grid[section_indices].add_samples(grid_sample, self.__evaluators)

                    # Print results
                    if (
                        self.console_mode is not ConsoleMode.SILENT
                        and self.verbosity.value <= Verbosity.INFO.value
                    ):
                        result_str = f"# {total:<5}"

                        for dimesion, i in zip(self.__dimensions, indices[grid_selector]):
                            result_str += f" {dimesion.title} = {dimesion.sample_points[i].title}"

                        for evaluator, artifact in zip(self.__evaluators, loop_sample.artifacts):
                            result_str += f" {evaluator.abbreviation}: {str(artifact):5}"

                        if self.console_mode == ConsoleMode.INTERACTIVE:
                            progress.log(result_str)

                        elif self.console_mode != ConsoleMode.SILENT:
                            self.console.log(result_str)

                    if self.plot_information:
                        for thread in plot_threads:
                            thread.update_plot(loop_sample)

                except Exception as e:
                    self._handle_exception(e, confirm=False)

                # Update progress
                total += 1

                for task, completed in zip(grid_tasks, indices[grid_selector]):
                    progress.update(task, completed=completed)

                progress.update(loop_drop_progress, completed=indices[drop_selector])
                progress.update(total_progress, completed=total)

        # Compute the evaluation results
        result: MonteCarloResult = MonteCarloResult(
            self.__dimensions, self.__evaluators, sample_grid, 0.0
        )

        # Generate result plots
        visualizations = result.plot()

        # Save results if a directory was provided
        if self.results_dir:
            result.save_to_matlab(path.join(self.results_dir, "results.mat"))

            for idx, (visualization, evaluator) in enumerate(
                zip(visualizations, self.__evaluators)
            ):
                if visualization.figure is not None:
                    visualization.figure.savefig(
                        path.join(self.results_dir, f"result_{idx}_{evaluator.abbreviation}.png"),
                        format="png",
                    )

        if self.plot_information:
            plt.show()

    @classmethod
    def to_yaml(
        cls: Type[HardwareLoop], representer: SafeRepresenter, node: HardwareLoop
    ) -> MappingNode:
        # Prepare dimensions
        dimension_fields: List[Mapping[str, Any]] = []
        for dimension in node.__dimensions:
            dimension_fields.append(
                {
                    "objects": dimension.considered_objects,
                    "property": dimension.dimension,
                    "points": [p.value for p in dimension.sample_points],
                    "title": dimension.title,
                }
            )

        additional_fields = {"Evaluators": node.__evaluators, "Dimensions": dimension_fields}

        return node._mapping_serialization_wrapper(representer, additional_fields=additional_fields)

    @classmethod
    def from_yaml(
        cls: Type[HardwareLoop], constructor: SafeConstructor, node: Node
    ) -> HardwareLoop:
        state = constructor.construct_mapping(node, deep=True)

        state.pop("Operators", [])
        evaluators: List[Evaluator] = state.pop("Evaluators", [])
        dimensions: List[Mapping[str, Any]] = state.pop("Dimensions", [])

        # Initialize the hardware loop
        hardware_loop: HardwareLoop = cls.InitializationWrapper(state)

        # Register evaluators
        for evaluator in evaluators:
            hardware_loop.add_evaluator(evaluator)

        # Add sweeping dimensions
        for dimension in dimensions:
            new_dim = hardware_loop.new_dimension(
                dimension["property"], dimension["points"], *dimension["objects"]
            )

            title = dimension.get("title", None)
            if title is not None:
                new_dim.title = title

        # Return fully configured hardware loop
        return hardware_loop
