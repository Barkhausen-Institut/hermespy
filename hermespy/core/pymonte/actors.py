from __future__ import annotations
from abc import abstractmethod
from typing import Callable, Generic, Sequence

import numpy as np
from ray import get, put, ObjectRef, wait

from .definitions import MO, UnmatchableException
from .evaluation import Evaluator, EvaluationResult
from .grid import GridDimension, GridDimensionInfo
from .artifact import Artifact, MonteCarloSample

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MonteCarloQueueManager(object):
    """Queue management actor for monte carlo simulations.

    The queue manager is responsible for distributing section indices of the
    simulation grid to individual actors.
    """

    __num_grid_sections: int
    __num_samples: int
    __batch_size: int
    __max_num_samples: int
    __grid_task_count: np.ndarray
    __active_section_flags: np.ndarray

    def __init__(
        self, grid: Sequence[GridDimensionInfo], num_samples: int, batch_size: int | None = None
    ) -> None:
        """
        Args:
            grid: The grid to be simulated.
            num_samples: The number of samples to collect for each section.
            batch_size: The number of individual section index tuples to return when calling :meth:`next_batch`.
        """

        self.__num_samples = num_samples

        if len(grid) > 0:
            self.__num_grid_sections = np.prod([d.num_sample_points for d in grid], dtype=int)

            self.__batch_size = self.__num_grid_sections if batch_size is None else batch_size
            self.__max_num_samples = self.__num_grid_sections * num_samples

            # Generate section sample containers and meta-information
            self.__grid_task_count = np.zeros([d.num_sample_points for d in grid], dtype=int)
            self.__active_section_flags = np.ones_like(self.__grid_task_count, dtype=bool)
            self.__active_section_coordinates = np.empty(
                [self.__num_grid_sections, len(grid)], dtype=int
            )
            for c, coordinates in enumerate(np.ndindex(*[d.num_sample_points for d in grid])):
                # Store the section coordinates
                self.__active_section_coordinates[c, :] = coordinates

        # Treat the special case of an empty grid
        # The simulation should still collect samples without reconfiguring the investigated object
        else:
            self.__num_grid_sections = 1
            self.__batch_size = 1
            self.__max_num_samples = num_samples
            self.__grid_task_count = np.zeros((1), dtype=int)
            self.__active_section_flags = np.array([True], dtype=bool)
            self.__active_section_coordinates = np.empty((1, 0), dtype=int)

        self.__num_active_sections = self.__num_grid_sections

    def next_batch(self) -> list[tuple[int, ...]]:
        """Get the next batch of simulation grid sections to run.

        This function gets called remotely by actors once they are ready to run a new batch of simulation tasks.

        Returns: A list of tuples containing the coordinates of the sections to run in this batch.
        """

        # Skip if no active sections available
        if self.__num_active_sections < 1:
            return []

        batch: list[tuple[int, ...]] = []
        skip_offset = 0
        for t in range(self.__batch_size):
            active_section_index = (t - skip_offset) % self.__num_active_sections

            # Add the section to the program
            section_coordinates = tuple(self.__active_section_coordinates[active_section_index])
            batch.append(section_coordinates)

            # Increment the task count for the section
            task_count = self.__grid_task_count[section_coordinates] + 1
            self.__grid_task_count[section_coordinates] = task_count

            # Disable the section if the number of tasks is expected to be met
            if task_count >= self.__num_samples:
                self.disable_section(section_coordinates)
                skip_offset += 1
                if self.__num_active_sections < 1:
                    break

        return batch

    def disable_section(self, coordinates: tuple[int, ...]) -> None:
        """Disable a section from being processed in the future.

        Once a section is disabled, its respective coordinates will not be returned by the :meth:`next_batch` method anymore.

        Args:
            coordinates: Coordinates of the section to disable.
        """

        # Find the index of the section to disable
        section_index = np.where((self.__active_section_coordinates == coordinates).all(axis=1))[0]

        # Abort if the section indices do not exist
        # This might happend if the section was already disabled
        if len(section_index) < 1:
            return

        # Remove the section from the list of currently active sections
        self.__active_section_coordinates = np.delete(
            self.__active_section_coordinates, section_index[0], axis=0
        )
        self.__active_section_flags[coordinates] = False
        self.__num_active_sections -= 1

    def query_progress(self) -> tuple[float, np.ndarray]:
        """Query the current absolute progress of the managed simulation.

        Returns: A floating point value between zero and one indicating the progress of the simulation as well as a boolean array indicating which sections are still actively being processed.
        """

        # No active sections left, return 100% progress
        if self.__num_active_sections < 1:
            return 1.0, self.__active_section_flags

        return (
            (self.__num_grid_sections - self.__num_active_sections) * self.__num_samples
            + np.sum(self.__grid_task_count[self.__active_section_coordinates])
        ) / self.__max_num_samples, self.__active_section_flags


class MonteCarloCollector(object):
    """Collects samples from actors during simulation runtime."""

    __queue_manager: MonteCarloQueueManager
    __actors: list[MonteCarloActor]
    __results: list[EvaluationResult]
    __active: bool = True  # Whether the collector is still running

    def __init__(
        self,
        queue_manager: MonteCarloQueueManager,
        actors: list[MonteCarloActor],
        grid: Sequence[GridDimensionInfo],
        evaluators: list[Evaluator],
    ) -> None:
        """
        Args:
            queue_manager: Reference to the queue management actor.
            actors: References to invidual monte carlo simulation actors.
            grid: The grid to be simulated.
            evaluators: The evaluators to be used for collecting processing the simulation results.
        """

        self.__queue_manager = queue_manager
        self.__actors = actors
        self.__results = [e.initialize_result(grid) for e in evaluators]
        self.__active = False
        self.__future_map: dict[ObjectRef, MonteCarloActor] = {}

    def run(self) -> None:
        """Start up the collector, collecting samples from all actors.

        This method will block and keep running until :meth:`fetch_results` is called.
        """

        self.__future_map = {a.fetch_results.remote(): a for a in self.__actors}  # type: ignore[attr-defined]
        self.__active = True

        while self.__active:
            # Wait for the next sample to be available
            ready_future = wait(list(self.__future_map.keys()), num_returns=1)[0][0]

            # Process the samples received from the actor
            self.__process_samples(ready_future, True)

            # Re-queue the request for the next result
            actor = self.__future_map.pop(ready_future)
            self.__future_map[actor.fetch_results.remote()] = actor  # type: ignore[attr-defined]

    def __process_samples(
        self,
        ready_future: ObjectRef[list[ObjectRef[list[MonteCarloSample]]]],
        compute_confidence: bool,
    ) -> None:

        sample_references: list[ObjectRef[list[MonteCarloSample]]] = get(ready_future)

        for sample_reference in sample_references:
            # Get the sample from the actor
            samples: list[MonteCarloSample] = get(sample_reference)

            for sample in samples:

                # Get the grid section coordinates from the sample
                section_coordinates = sample.grid_section

                # Update all evaluation results with their respective sample artifacts
                for evaluation_result, artifact in zip(self.__results, sample.artifacts):
                    confident = evaluation_result.add_artifact(
                        section_coordinates, artifact, compute_confidence
                    )

                    # If the confidence threshold is reached, disable the section
                    if compute_confidence and confident:
                        self.__queue_manager.disable_section.remote(section_coordinates)  # type: ignore[attr-defined]

    def query_estimates(self) -> list[None | np.ndarray]:
        """Query intermediate estimates during simulation runtime.

        Returns: A list with the length of the number of evaluators, each containing the runtime estimates for the respective evaluator.
        If no estimates are available, the entry will be :py:obj:`None`.
        """

        return [r.runtime_estimates() for r in self.__results]

    def fetch_results(self) -> list[EvaluationResult]:
        """Fetch the results of the collector run.

        Returns: list of evaluation results from the collector run.
        """

        # Stop the collector
        self.__active = False

        # Process the last results within the queue
        query_futures = list(self.__future_map.keys())
        futures = wait(query_futures, num_returns=len(query_futures))[0]

        for future in futures:
            self.__process_samples(future, False)

        return self.__results


class MonteCarloActor(Generic[MO]):
    """Monte Carlo Simulation Actor.

    Actors are essentially workers running in a private process executing simulation tasks.
    The result of each individual simulation task is a simulation sample.
    """

    __queue_manager: MonteCarloQueueManager
    __results: list[ObjectRef[list[MonteCarloSample]]]
    catch_exceptions: bool
    __investigated_object: MO
    __grid: Sequence[GridDimension]
    __evaluators: Sequence[Evaluator]
    __stage_arguments: dict[str, Sequence[tuple]]

    def __init__(
        self,
        queue_manager: MonteCarloQueueManager,
        argument_tuple: tuple[MO, Sequence[GridDimension], Sequence[Evaluator]],
        index: int,
        stage_arguments: dict[str, Sequence[tuple]] | None = None,
        catch_exceptions: bool = True,
    ) -> None:
        """
        Args:
            argument_tuple:
                Object to be investigated during the simulation runtime.
                Dimensions over which the simulation will iterate.
                Evaluators used to process the investigated object sample state.
            index: Global index of the actor.
            stage_arguments: Arguments for the simulation stages.
            catch_exceptions: Catch exceptions during run. Enabled by default.
        """

        # Assert that stage arguments are valid
        self.__stage_arguments = dict() if stage_arguments is None else stage_arguments
        for stage_key in self.__stage_arguments:
            if stage_key not in self.stage_identifiers():
                raise ValueError(f"Invalid stage identifier in stage arguments {stage_key}")

        investigated_object = argument_tuple[0]
        grid = argument_tuple[1]
        evaluators = argument_tuple[2]

        self.index = index
        self.__queue_manager = queue_manager
        self.__results = []
        self.catch_exceptions = catch_exceptions
        self.__investigated_object = investigated_object
        self.__grid = grid
        self.__evaluators = evaluators
        self.__stage_identifiers = self.stage_identifiers()
        self.__stage_executors = self.stage_executors()
        self.__num_stages = len(self.__stage_executors)

    @property
    def _investigated_object(self) -> MO:
        """State of the Investigated Object."""

        return self.__investigated_object  # pragma: no cover

    def __execute_stages(self, start: int, stop: int, artifacts: list[list[Artifact]]) -> None:
        """Recursive subroutine of run, collecting artifacts from the simulation stage parametrizations.

        Args:
            start: Index of the first stage to be executed
            stop: Index of the last stage to be executed
        """

        # Abort and collect artifacts if the end of the stage list is reached
        if start > stop:
            artifacts.append([evaluator.evaluate().artifact() for evaluator in self.__evaluators])
            return

        # Execute the next stage
        stage_identifier = self.__stage_identifiers[start]
        stage_executor = self.__stage_executors[start]
        stage_arguments = self.__stage_arguments.get(stage_identifier, [tuple()])
        for arguments in stage_arguments:
            # Execute the stage with the provided arguments
            stage_executor(*arguments)

            # Proceed to the next stage
            self.__execute_stages(start + 1, stop, artifacts)

    def run(self) -> None:

        while True:

            # Get the next batch of sections to run
            batch = get(self.__queue_manager.next_batch.remote())  # type: ignore[attr-defined]

            # If the batch is empty, i.e. no more sections to run, break the loop
            if len(batch) < 1:
                break

            # Run the batch and stash the result
            result = self.__run_batch(batch)

            if len(result) > 0:
                self.__results.append(put(result))

    def fetch_results(self) -> list[ObjectRef[list[MonteCarloSample]]]:
        """Fetch the results of the actor run.

        Returns: list of results from the actor run.
        """

        results = self.__results.copy()
        self.__results.clear()  # Clear the results after fetching
        return results

    def __run_batch(self, batch: list[tuple[int, ...]]) -> list[MonteCarloSample]:
        """Run the simulation actor.

        Args:
            program: A list of simulation grid section indices for which to collect samples.

        Returns: The resulting samples from the simulation run.
        """

        # Catch any exception during actor running
        try:
            # Intially, re-configure the full grid
            recent_section_indices = np.array(batch[0], dtype=int)
            for d, i in enumerate(recent_section_indices):
                self.__grid[d].configure_point(i)

            # Initialize the result dictionary
            samples: list[MonteCarloSample] = []

            # Run through the program steps
            for section_indices in batch:

                # Detect the grid dimensions where sections changed, i.e. which require re-configuration
                section_index_array = np.asarray(section_indices, dtype=int)
                reconfigured_dimensions = np.argwhere(
                    section_index_array != recent_section_indices
                ).flatten()

                # Reconfigure the dimensions
                # Note that for the first grid_section this has already been done
                for d in reconfigured_dimensions:
                    self.__grid[d].configure_point(section_index_array[d])

                # Detect the first and last impacted simulation stage depending on the reconfigured dimensions
                first_impact = self.__num_stages
                last_impact = 0
                for d in reconfigured_dimensions:
                    grid_dimension = self.__grid[d]

                    if grid_dimension.first_impact is None:
                        first_impact = 0

                    elif grid_dimension.first_impact in self.__stage_identifiers:
                        first_impact = min(
                            first_impact,
                            self.__stage_identifiers.index(grid_dimension.first_impact),
                        )

                    if grid_dimension.last_impact is None:
                        last_impact = self.__num_stages - 1

                    elif grid_dimension.last_impact in self.__stage_identifiers:
                        last_impact = max(
                            last_impact, self.__stage_identifiers.index(grid_dimension.last_impact)
                        )

                if first_impact >= self.__num_stages:
                    first_impact = 0

                if last_impact <= 0:
                    last_impact = self.__num_stages - 1

                artifacts: list[list[Artifact]] = []
                self.__execute_stages(first_impact, last_impact, artifacts)

                samples.extend(
                    MonteCarloSample(section_indices, a, artifact)
                    for a, artifact in enumerate(artifacts)
                )

                # Update the recent section for the next iteration
                recent_section_indices = section_index_array

            # Convert the samples dictionary to ray object references
            return samples

        # Catch any exception during actor running
        except Exception as e:
            if self.catch_exceptions:
                print(e)
            else:
                raise UnmatchableException(
                    f"Actor #{self.index} encountered an error during run: {e}"
                ) from e

        return []

    @staticmethod
    @abstractmethod
    def stage_identifiers() -> list[str]:
        """list of simulation stage identifiers.

        Simulations stages will be executed in the order specified here.

        Returns:
            list of function identifiers for simulation stage execution routines.
        """
        ...  # pragma: no cover

    @abstractmethod
    def stage_executors(self) -> list[Callable]:
        """list of simulation stage execution callbacks.

        Simulations stages will be executed in the order specified here.

        Returns:
            list of function callbacks for simulation stage execution routines.
        """
        ...  # pragma: no cover
