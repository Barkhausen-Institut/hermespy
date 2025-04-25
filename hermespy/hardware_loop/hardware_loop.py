# -*- coding: utf-8 -*-
"""
=============
Hardware Loop
=============
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from threading import Thread, Condition, Lock
from contextlib import AbstractContextManager, ExitStack
from os import path
from signal import signal, SIGINT
from types import TracebackType
from typing import Any, Callable, Generic, List, Sequence, Tuple, Type
from warnings import catch_warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.prompt import Confirm

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
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
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

            evaluator:
                Registered evaluator.

            plot_mode:
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
    __canvas: plt.FigureCanvasBase | None
    __axes: VAT | None
    __figure: plt.FigureBase | None
    __visualization: VT | None

    def __init__(
        self,
        title: str | None = None,
        canvas: plt.FigureCanvasBase | None = None,
        axes: VAT | None = None,
    ) -> None:
        """
        Args:

            title:
                Title of the hardware loop plot.
                If not specified, resorts to the default title of the plot.

            canvas:
                Canvas to plot into.
                If not specified, a new canvas is created.

            axes:
                Visualization axes to plot into.
                If not specified, a new figure is created.
        """

        # Initialize class attributes
        self.__hardware_loop = None
        self.__title = title
        self.__canvas = canvas
        self.__axes = axes
        self.__figure = None
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

    def prepare_plot(self) -> Tuple[plt.FigureCanvasBase, VAT, plt.FigureBase]:
        """Prepare the figure for the hardware loop plot.

        Returns: The prepared figure canvas, axes and fiugure to be plotted into.
        """

        if self.__canvas is not None and self.__axes is not None:
            return self.__canvas, self.__axes, self.__figure

        self.__figure, self.__axes = self._prepare_plot()
        self.__figure.suptitle(self.title)
        self.__canvas = self.__figure.canvas

        return self.__canvas, self.__axes, self.__figure

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

            sample: Hardware loop sample to be plotted.
        """

        # Prepare the plot if not done yet
        if self.__axes is None or self.__canvas is None:
            self.__canvas, self.__axes, self.__figure = self.prepare_plot()

        # If the visualizable has not been plotted yet, prepare the plot
        if self.__visualization is None:
            self.__visualization = self._initial_plot(sample, self.__axes)

        # Otherwise, update the plot data
        else:
            self._update_plot(sample, self.__visualization)

        # Re-draw the plot
        self.__canvas.draw()

        # Flush the events
        if self.__figure is not None:
            self.__canvas.flush_events()

    def close_plot(self) -> None:
        """Close the hardware loop plot."""

        if self.__figure is not None:
            plt.close(self.__figure)  # type: ignore

        self.__axes = None
        self.__canvas = None
        self.__figure = None
        self.__visualization = None

    @abstractmethod
    def _initial_plot(self, sample: HardwareLoopSample, axes: VAT) -> VT:
        """Initial plot of the hardware loop plot.

        Abstract subroutine of :meth:`HardwareLoopPlot.update_plot`.

        Args:

            sample: Hardware loop sample to be plotted.
            axes: The visualization to be plotted into.

        Returns: The plotted information including axes and lines.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _update_plot(self, sample: HardwareLoopSample, visualization: VT) -> None:
        """Update the hardware loop plot.

        Abstract subroutine of :meth:`HardwareLoopPlot.update_plot`.

        Args:

            sample: Hardware loop sample to be plotted.
            visualization: The visualization to be updated.
        """
        ...  # pragma: no cover


class PlotThread(Thread):
    """Thread for parallel plotting during hardware loop runtime."""

    __plots: Sequence[HardwareLoopPlot]
    __sample: HardwareLoopSample | None = None
    __new_sample: Condition
    __sample_lock: Lock
    __alive: bool
    __rc_params: mpl.RcParams

    def __init__(
        self,
        rc_params: mpl.RcParams,
        plots: HardwareLoopPlot | Sequence[HardwareLoopPlot],
        **kwargs,
    ) -> None:
        """
        Args:
            rc_params: Matplotlib style parameters.
            plot: Plot to be visualized by the thread.
            \*\*kwargs: Additional keyword arguments to be passed to the base class.
        """

        # Initialize class attributes
        self.__plots = [plots] if isinstance(plots, HardwareLoopPlot) else plots
        self.__sample = None
        self.__alive = True
        self.__rc_params = rc_params
        self.__new_sample = Condition()
        self.__sample_lock = Lock()

        # Initialize base class
        Thread.__init__(self, **kwargs)

    def run(self, interactive: bool = True) -> None:
        """Run the plot thread.

        Args:

            interactive:
                Run the plot in interactive mode.
                This will make the thread responsible for drawing.
                Enabled by default.
        """

        # Initialize figures and plots managed by this thread
        with ExitStack() as stack:

            # Set the correct matplotlib style parameters
            stack.enter_context(mpl.rc_context(rc=self.__rc_params))

            # Ignore the matlotlib user warnings for starting GUIs outside the main thread
            stack.enter_context(catch_warnings(
                category=UserWarning,
                action="ignore",
            ))

            if interactive:
                stack.enter_context(plt.ion())

            for plot in self.__plots:
                plot.prepare_plot()

            stack.enter_context(self.__new_sample)

            while self.__alive:
                while not self.__new_sample.wait(1.0) and self.__alive:
                    pass

                with self.__sample_lock:
                    for plot in self.__plots:
                        plot.update_plot(self.__sample)

        # Close the plots
        # This is required as to not confuse the matplotlib backend which thread to use
        # in upcoming plots
        for plot in self.__plots:
            plot.close_plot()

    def update_plot(self, sample: HardwareLoopSample) -> None:
        """Update the plot with a new sample.

        Args:

            sample (HardwareLoopSample): Sample to be plotted.
        """

        with self.__sample_lock:
            self.__sample = sample

        with self.__new_sample:
            self.__new_sample.notify()

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

            threads: Threads to be managed by this context manager.
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
            thread.join(None)  # Might require a timeout

        return super().__exit__(__exc_type, __exc_value, __traceback)


class HardwareLoop(Generic[PhysicalScenarioType, PDT], Pipeline[PhysicalScenarioType, PDT]):
    """Hermespy hardware loop configuration."""

    manual_triggering: bool
    """Require a user input to trigger each drop manually"""

    plot_information: bool
    """Plot information during loop runtime"""

    record_drops: bool
    """Record drops during loop runtime"""

    __dimensions: list[GridDimension]  # Parameter dimension over which a run sweeps
    __evaluators: list[Evaluator]  # Evaluators further processing drop information
    __plots: list[HardwareLoopPlot]
    __pre_drop_hooks: list[Callable[[PhysicalScenarioType, Console], None]]
    __post_drop_hooks: list[Callable[[PhysicalScenarioType, Console], None]]
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

            scenario:
                The physical scenario being controlled by the hardware loop.

            manual_triggering:
                Require a keyboard user input to trigger each drop manually.
                Disabled by default.

            plot_information:
                Plot information during loop runtime.
                Enabled by default.

            iteration_priority:
                Which dimension to iterate over first.
                Defaults to :attr:`IterationPriority.DROPS`.

            record_drops:
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
        self.__pre_drop_hooks = []
        self.__post_drop_hooks = []
        self.__interrupt_run = False

    def new_dimension(
        self, dimension: str, sample_points: List[Any], *args: Tuple[Any], **kwargs
    ) -> GridDimension:
        """Add a dimension to the sweep grid.

        Must be a property of the managed scenario.

        Args:

            dimension:
                String representation of dimension location relative to the investigated object.

            sample_points:
                List points at which the dimension will be sampled into a grid.
                The type of points must be identical to the grid arguments / type.

            \*args:
                References to the object the imension belongs to.
                Resolved to the investigated object by default,
                but may be an attribute or sub-attribute of the investigated object.

            \*\*kwargs:
                Additional keyword arguments to be passed to the dimension.
                See :class:`GridDimension<hermespy.core.monte_carlo.GridDimension>` for more information.

        Returns: The newly created dimension object.
        """

        considered_objects = (self.__scenario) if len(args) < 1 else args
        grid_dimension = GridDimension(considered_objects, dimension, sample_points, **kwargs)
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

            evaluator:
                The evaluator to be added.
        """

        self.__evaluators.append(evaluator)

    def add_plot(self, plot: HardwareLoopPlot) -> None:
        """Add a new plot to be  visualized by the hardware loop during runtime.

        Args:

            plot:
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

            evaluator:
                The evaluator to be searched for.

        Returns: The index of the evaluator.
        """

        return self.__evaluators.index(evaluator)

    @property
    def pre_drop_hooks(self) -> list[Callable[[PhysicalScenarioType, Console], None]]:
        """List of pre-drop hooks.

        Pre-drop hooks are called before each drop is generated and can be used to
        add additional actions not natively supported by the hardware loop.
        """

        return self.__pre_drop_hooks

    def add_pre_drop_hook(self, hook: Callable[[PhysicalScenarioType, Console], None]) -> None:
        """Add a pre-drop hook.

        Args:

            hook:
                The hook to be added.
                The hook must accept the scenario and console as arguments.

        Raises:

            ValueError: If the hook is already registered.
        """

        if hook in self.pre_drop_hooks:
            raise ValueError("Hook already registered")

        self.__pre_drop_hooks.append(hook)

    @property
    def post_drop_hooks(self) -> list[Callable[[PhysicalScenarioType, Console], None]]:
        """List of post-drop hooks.

        Post-drop hooks are called after each drop is generated and can be used to
        add additional actions not natively supported by the hardware loop.
        """

        return self.__post_drop_hooks

    def add_post_drop_hook(self, hook: Callable[[PhysicalScenarioType, Console], None]) -> None:
        """Add a post-drop hook.

        Args:

            hook:
                The hook to be added.
                The hook must accept the scenario and console as arguments.

        Raises:

            ValueError: If the hook is already registered.
        """

        if hook in self.post_drop_hooks:
            raise ValueError("Hook already registered")

        self.__post_drop_hooks.append(hook)

    @property
    def iteration_priority(self) -> IterationPriority:
        """Iteration priority of the hardware loop."""

        return self.__iteration_priority

    @iteration_priority.setter
    def iteration_priority(self, value: IterationPriority) -> None:
        self.__iteration_priority = value

    def run(
        self, overwrite=True, campaign: str | None = None, serialize_state: bool = True
    ) -> MonteCarloResult:
        """Run the hardware loop configuration.

        Args:

            overwrite:
                Allow the replacement of an already existing savefile.

            campaing:
                Name of the measurement campaign.

            serialize_state:
                Serialize the state of the scenario to the results file.
                Enabled by default.

        Returns: The result of the hardware loop.
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
                        for transmitter in device.transmitters:
                            device.transmitters.add(transmitter)
                        for receiver in device.receivers:
                            device.receivers.add(receiver)
                    self.scenario.record(
                        file_location,
                        overwrite=overwrite,
                        campaign=campaign,
                        state=state,
                        serialize_state=serialize_state,
                    )
                else:
                    self.scenario.record(
                        file_location,
                        overwrite=overwrite,
                        campaign=campaign,
                        serialize_state=serialize_state,
                    )

            else:
                if (
                    self.console_mode is not ConsoleMode.SILENT
                    and self.verbosity.value >= Verbosity.INFO.value
                ):
                    self.console.print("Skipping drop recording", style="bright_yellow")

        # Run internally
        result = self.__run()

        # Save results and close file streams
        self.scenario.stop()

        # Save results if a directory was provided
        if self.results_dir:

            # Save the results to a file
            result.save_to_matlab(path.join(self.results_dir, "results.mat"))

            # Generate result plots
            result_visualizations = result.plot()

            for idx, (visualization, evaluator) in enumerate(
                zip(result_visualizations, self.__evaluators)
            ):
                if visualization.figure is not None:
                    visualization.figure.savefig(
                        path.join(self.results_dir, f"result_{idx}_{evaluator.abbreviation}.png"),
                        format="png",
                    )

        # Return the result
        return result

    def replay(self, file_location: str) -> None:
        """Replay a stored pipeline run.

        Args:

            file_location:
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

            section_indices:
                The indices of the section within the sample grid.

            sample_index:
                Index of the sample within the section.

        Returns: The generated sample.
        """

        # Configure the parameters according to the grid indces
        for dim, p in zip(self.__dimensions, section_indices):
            dim.configure_point(p)

        # Generate a new drop (triggers the hardware)
        drop = self.scenario.drop()

        # Generate evaluations and artifacts
        evaluations = [e.evaluate() for e in self.evaluators]
        artifacts = [e.artifact() for e in evaluations]

        # Return sample
        return HardwareLoopSample(drop, evaluations, artifacts)

    def __sigint_handler(self, signum: int, frame: Any) -> None:
        """Signal handler for SIGINT."""

        # Print a message
        if self.console_mode is not ConsoleMode.SILENT:
            self.console.log("Received SIGINT, stopping hardware loop", style="bright_red")

        self.__interrupt_run = True

    def __run(self) -> MonteCarloResult:
        """Internal run method executing drops.

        Returns: The result of the hardware loop.
        """

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
                    # Execute pre-drop hooks
                    for hook in self.__pre_drop_hooks:
                        hook(self.scenario, self.console)

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

                    # Execute post-drop hooks
                    for hook in self.__post_drop_hooks:
                        hook(self.scenario, self.console)

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

        return result
