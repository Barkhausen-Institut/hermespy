# -*- coding: utf-8 -*-
"""
=======
PyMonte
=======

PyMonte is a stand-alone core module of HermesPy,
enabling efficient and flexible MonteCarlo simulations over arbitrary configuration parameter combinations.
"""

from __future__ import annotations

from abc import abstractmethod
from functools import reduce
from itertools import product
from math import exp, sqrt
from time import perf_counter
from typing import Any, Callable, Generic, List, Optional, Set, Type, TypeVar, Tuple, Union
from warnings import catch_warnings, simplefilter

import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.util import ActorPool
from rich.console import Console, Group
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.live import Live
from rich.table import Table
from scipy.constants import pi
from scipy.io import savemat
from scipy.stats import norm

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


MO = TypeVar('MO')
"""Type of Monte Carlo object under investigation."""

AT = TypeVar('AT')
"""Type of Monte Carlo evaluation artifact."""


class Artifact(object):
    """Result of an evaluator evaluating a single sample."""

    @abstractmethod
    def __str__(self) -> str:
        """String representation of this artifact.

        Will be used to display the artifact in the console output.

        Returns:
            str: String representation.
        """

        return "?"

    @abstractmethod
    def to_scalar(self) -> Optional[float]:
        """Conversion of the artifact to a scalar.

        Used to evaluate premature stopping criteria for the underlying evaluation.

        Returns:
            Optional[float]:
                Scalar floating-point representation.
                `None` if a conversion to scalar is impossible.
        """

        return None


class ArtifactTemplate(Generic[AT], Artifact):
    """Integer artifact resulting from a Monte Carlo sample evaluation."""

    __artifact: AT      # Artifact

    def __init__(self,
                 artifact: AT) -> None:
        """
        Args:

            artifact (AT):
                Artifact value.
        """

        self.__artifact = artifact

    @property
    def artifact(self) -> AT:
        """Evaluation artifact.

        Returns:
            AT: Copy of the artifact.
        """

        return self.__artifact

    def __str__(self) -> str:
        return f"{self.to_scalar():.3f}"

    def to_scalar(self) -> float:

        return self.artifact


class Evaluator(Generic[MO]):
    """Monte Carlo Sample Evaluator.

    Once a simulation sample has been generated, its properties of interest must be extracted.
    This is done by evaluators.
    """

    # Berry-Esseen constants ToDo: Check the proper selection here
    __C_0: float = .4785
    __C_1: float = 30.2338

    __confidence: float
    __tolerance: float
    __plot_scale: str       # Plot axis scaling

    def __init__(self) -> None:

        self.confidence = 1.
        self.tolerance = 0.
        self.plot_scale = 'linear'

    @abstractmethod
    def evaluate(self, investigated_object: MO) -> Artifact:
        """Evaluate a sampled state of the investigated object.

        Args:

            investigated_object (MO):
                Investigated object.

        Returns:

            Artifact:
                Artifact resulting from the evaluation.
        """
        ...

    @property
    @abstractmethod
    def abbreviation(self) -> str:
        """Short string representation of this evaluator.

        Used as a label for console output and plot axes annotations.

        Returns:
            str: String representation
        """
        ...

    @property
    @abstractmethod
    def title(self) -> str:
        """Long string representation of this evaluator.

        Used as plot title.

        Returns:
            str: String representation
        """
        ...

    @property
    def confidence(self) -> float:
        """Confidence required for premature simulation abortion.

        Returns:
            float: Confidence between zero and one.

        Raises:
            ValueError: If confidence is lower than zero or greater than one.
        """

        return self.__confidence

    @confidence.setter
    def confidence(self, value: float) -> None:

        if value < 0. or value > 1.:
            raise ValueError("Confidence level must be in the interval between zero and one")

        self.__confidence = value

    @property
    def tolerance(self) -> float:
        """Tolerance for premature simulation abortion.

        Returns:
            float: Non-negative tolerance.

        Raises:
            ValueError: If tolerance is negative.
        """

        return self.__tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Tolerance must be greater or equal to zero")

        self.__tolerance = value

    def __str__(self) -> str:
        """String object representation.

        Returns:
            str: String representation.
        """

        return self.abbreviation

    @staticmethod
    def _scalar_cdf(scalar: float) -> float:
        """Assumed cumulative probability of the scalar representation.

        Args:

            scalar (float):
                The scalar value.

        Returns:

            float: Cumulative probability between zero and one.
        """

        return norm.cdf(scalar)

    @property
    def plot_scale(self) -> str:
        """Scale of the scalar evaluation plot.

        Refer to the `Matplotlib`_ documentation for a list of a accepted values.

        Returns:
            str: The  scale identifier string.

        .. _Matplotlib: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_yscale.html
        """

        return self.__plot_scale

    @plot_scale.setter
    def plot_scale(self, value: str) -> None:

        self.__plot_scale = value

    def confidence_level(self,
                         scalars: np.ndarray) -> float:
        """Compute the confidence level in a given set of scalars.

        Refer to :footcite:t:`2014:bayer` for a detailed derivation of the implement equations.

        Args:
            scalars (np.ndarray): Numpy vector of scalar representations.

        Raises:
            ValueError: If `scalars` is not a vector.
        """

        # Raise a value error if the scalars argument is not a vector
        if scalars.ndim != 1:
            raise ValueError("Scalars must be a vector (on-dimensional array)")

        n = len(scalars)

        # Compute unbiased samples
        sample_mean = np.mean(scalars)
        unbiased_samples = scalars - sample_mean

        # Compute moments
        sigma_moment = sqrt(np.sum(unbiased_samples ** 2) / n)

        # A sigma moment of 0 indicates zero variance within the samples, therefore maximum confidence
        if sigma_moment == 0.:
            return 1.0

        beta_bar_moment = np.sum(np.abs(unbiased_samples) ** 3) / (n * sigma_moment ** 3)
        beta_hat_moment = np.sum(unbiased_samples ** 3) / (n * sigma_moment ** 3)
        kappa_moment = np.sum(unbiased_samples ** 4) / (n * sigma_moment ** 4) - 3

        # Estimate the confidence
        sample_sqrt = sqrt(n)
        sigma_tolerance = sample_sqrt * self.tolerance / sigma_moment
        sigma_tolerance_squared = sigma_tolerance ** 2
        kappa_term = 4 * (2 / (n - 1) + kappa_moment / n)

        confidence_bound = 2 * ((1 - self._scalar_cdf(sigma_tolerance)) +
                                min(self.__C_0, self.__C_1 * (1 + abs(sigma_tolerance)) ** -3)
                                * beta_bar_moment / sample_sqrt * min(1., kappa_term))

        if kappa_term < 1. and sigma_tolerance_squared < 1418.:

            confidence_bound += ((1 - kappa_term) * abs(sigma_tolerance_squared - 1) * abs(beta_hat_moment) /
                                 (exp(.5 * sigma_tolerance_squared) * 3 * sqrt(2 * pi * n) * sigma_moment ** 3))

        return 1. - confidence_bound


class MonteCarloSample(object):
    """Single sample of a Monte Carlo simulation."""

    __sample_index: int                 # Index of the sample
    __grid_section: Tuple[int, ...]     # Grid section from which the sample was generated
    __artifacts: List[Artifact]         # Artifacts of evaluation

    def __init__(self,
                 grid_section: Tuple[int, ...],
                 sample_index: int,
                 artifacts: List[Artifact]) -> None:
        """
        Args:
        
            grid_section (Tuple[int, ...]):
                Grid section from which the sample was generated.

            sample_index (int):
                Index of the sample.
                In other words this object represents the `sample_index`th sample of the selected `grid_section`.

            artifacts (List[Artifact]):
                Artifacts of evaluation
        """

        self.__grid_section = grid_section
        self.__sample_index = sample_index
        self.__artifacts = artifacts

    @property
    def grid_section(self) -> Tuple[int, ...]:
        """Grid section from which this sample was generated.

        Returns:
            Tuple[int, ...]: Tuple of grid section indices.
        """

        return self.__grid_section

    @property
    def sample_index(self) -> int:
        """Index of the sample this object represents.

        Returns:
            int: Sample index.
        """

        return self.__sample_index

    @property
    def artifacts(self) -> List[Artifact]:
        """Artifacts resulting from the sample's evaluations.

        Returns:
            List[Artifact]: List of artifacts.
        """

        return self.__artifacts

    @property
    def num_artifacts(self) -> int:
        """Number of contained artifact objects.

        Returns:
            int: Number of artifacts.
        """

        return len(self.__artifacts)

    @property
    def artifact_scalars(self) -> np.ndarray:
        """Collect scalar artifact representations.

        Returns:
            np.ndarray: Vector of scalar artifact representations.
        """

        return np.array([artifact.to_scalar() for artifact in self.artifacts], dtype=float)


class GridSection(object):

    __coordinates: Tuple[int, ...]          # Section indices within the simulation grid
    __samples: List[MonteCarloSample]       # Set of generated samples
    __evaluators: List[Evaluator]           # Number of artifacts per sample
    __scalars: np.ndarray
    __evaluator_confidences: np.ndarray     # Confidence level for each evaluator

    def __init__(self,
                 coordinates: Tuple[int, ...],
                 evaluators: List[Evaluator]) -> None:
        """
        Args:

            coordinates (Tuple[int, ...]):
                Section indices within the simulation grid.

            evaluators (List[Evaluator]):
                Configured evaluators.
        """

        self.__coordinates = coordinates
        self.__samples = []
        self.__evaluators = evaluators
        self.__scalars = np.empty((self.num_evaluators, 0), dtype=float)
        self.__evaluator_confidences = np.zeros(self.num_evaluators, dtype=bool)

    @property
    def coordinates(self) -> Tuple[int, ...]:
        """Grid section coordinates within the simulation grid.

        Returns:
            Tuple[int, ...]: Section coordinates.
        """

        return self.__coordinates

    @property
    def num_samples(self) -> int:
        """Number of already generated samples within this section.

        Returns:
            int: Number of samples.
        """

        return len(self.__samples)

    @property
    def num_evaluators(self) -> int:
        """Number of configured evaluators.

        Returns:
            int: Number of evaluators.
        """

        return len(self.__evaluators)

    def add_samples(self, samples: Union[MonteCarloSample, List[MonteCarloSample]]) -> None:
        """Add a new sample to this grid section collection.

        Args:

            samples (Union[MonteCarloSample, List[MonteCarloSample])):
                Samples to be added to this section.

        Raises:
            ValueError: If the number of artifacts in `sample` does not match the initialization.
        """

        if not isinstance(samples, list):
            samples = [samples]

        for sample in samples:

            # Make sure the provided number of artifacts is correct
            if sample.num_artifacts != self.num_evaluators:
                raise ValueError(f"Number of sample artifacts ({sample.num_artifacts}) does not match the "
                                 f"configured number of evaluators ({self.num_evaluators})")

            # Append sample to the stored list
            self.__samples.append(sample)

            # Update scalar storage
            scalars = sample.artifact_scalars
            self.__scalars = np.append(self.__scalars, scalars[::, np.newaxis], axis=1)

        # Update scalar confidences
        for evaluator_idx, evaluator in enumerate(self.__evaluators):

            # Retrieve cached scalars for the respective evaluator
            evaluator_scalars = self.__scalars[evaluator_idx, :]

            # Compute confidence level
            confidence_level = evaluator.confidence_level(evaluator_scalars)
            self.__evaluator_confidences[evaluator_idx] = confidence_level >= evaluator.confidence

    @property
    def confidences(self) -> np.ndarray:
        """Confidence in the scalar evaluations.

        Returns:
            np.ndarray: Boolean array indicating confidence.
        """

        return self.__evaluator_confidences

    @property
    def scalars(self) -> np.ndarray:
        """Access the scalar evaluator representations in this grid section.

        Returns:
            np.ndarray:
                Matrix of scalar representations.
                First dimension indicates the evaluator index, second dimension the sample.
        """

        return self.__scalars.copy()


class MonteCarloActor(Generic[MO]):
    """Monte Carlo Simulation Actor.

    Actors are essentially workers running in a private process executing simulation tasks.
    The result of each individual simulation task is a simulation sample.
    """

    __investigated_object: MO                       # Copy of the object to be investigated
    __dimension_parameters: List[List[Any]]         # Parameter samples along each simulation grid dimension
    __configuration_lambdas: List[Callable]         # Configuration lambdas to configure grid parameters
    __evaluators: List[Evaluator[MO]]               # Evaluators used to process the investigated object sample state
    __section_block_size: int = 10                  # Number of samples per section block

    def __init__(self,
                 argument_tuple: Tuple[MO, dict[str, List[Any]], List[Evaluator[MO]]],
                 section_block_size: int = 10) -> None:
        """
        Args:

            argument_tuple:
                Object to be investigated during the simulation runtime.
                Dimensions over which the simulation will iterate.
                Evaluators used to process the investigated object sample state.

            section_block_size (int):
                Number of samples generated per section block.
        """

        investigated_object = argument_tuple[0]
        dimensions = argument_tuple[1]
        evaluators = argument_tuple[2]
        self.__section_block_size = section_block_size

        self.__investigated_object = investigated_object  # deepcopy(investigated_object)
        self.__dimension_parameters = [dimension_parameters for dimension_parameters in dimensions.values()]
        self.__evaluators = evaluators

        # Generate configuration lambdas
        self.__configuration_lambdas = [self.__setter_lambda(dimension) for dimension in dimensions.keys()]

    @property
    def _investigated_object(self) -> MO:
        """State of the Investigated Object.

        Returns:
            Mo: Investigated object.
        """

        return self.__investigated_object

    def run(self,
            grid_section: Tuple[int, ...]) -> List[MonteCarloSample]:
        """Run the simulation actor.

        Args:

            grid_section (Tuple[int, ...]):
                Sample point index of each grid dimension.

        Returns:
            MonteCarloSample:
                The generated sample object.
        """

        samples: List[MonteCarloSample] = []
        for sample_index in range(self.__section_block_size):

            # Configure the object under investigation
            for (configuration_lambda, parameters, parameter_idx) in zip(self.__configuration_lambdas,
                                                                         self.__dimension_parameters,
                                                                         grid_section):
                configuration_lambda(parameters[parameter_idx])

            # Sample the investigated object
            sampled_object = self.sample()

            # Evaluate the sample
            evaluations = [evaluator.evaluate(sampled_object) for evaluator in self.__evaluators]

            # Save generated sample
            samples.append(MonteCarloSample(grid_section, sample_index, evaluations))

        # Return results
        return samples

    @abstractmethod
    def sample(self) -> MO:
        """Generate a sample of the investigated object.

        Returns:
            MO: The resulting sample.
        """
        ...

    def __setter_lambda(self, dimension: str) -> Callable:
        """Generate a setter lambda callback for a selected grid dimension.

        dimension (str):
            String representation of dimension location relative to the investigated object.

        Returns:
            Callable: The setter lambda.
        """

        stages = dimension.split('.')
        object_reference = reduce(lambda obj, attr: getattr(obj, attr), stages[:-1], self.__investigated_object)

        # Return a lambda to the function if the reference is callable
        function_reference = getattr(object_reference, stages[-1])
        if callable(function_reference):
            return lambda args: function_reference(args)

        # Return a setting lambda if the reference is not callable
        # Implicitly we assume that every non-callable reference is an attribute
        return lambda args: setattr(object_reference, stages[-1], args)


class MonteCarloResult(Generic[MO]):

    __dimensions: dict[str, List[Any]]
    __evaluators: List[Evaluator[MO]]
    __sections: np.ndarray
    __performance_time: float           # Time required to compute the simulation.

    def __init__(self,
                 dimensions: dict[str, List[Any]],
                 evaluators: List[Evaluator],
                 sections: np.ndarray,
                 performance_time: float) -> None:
        """
        Args:

            dimensions (dict[str, List[Any]]):
                Dimensions over which the simulation has swept.

            evaluators (List[Evaluator]):
                Evaluators used to evaluated the simulation artifacts.

            sections (np.ndarray):
                Evaluation results.

            performance_time (float):
                Time required to compute the simulation.

        Raises:
            ValueError:
                If the dimensions of `samples` do not match the supplied sweeping dimensions and evaluators.
        """

        self.__dimensions = dimensions
        self.__evaluators = evaluators
        self.__sections = sections
        self.__performance_time = performance_time

    def plot(self) -> List[plt.Figure]:
        """Plot evaluation figures for all contained evaluator artifacts.

        Returns:
            List[plt.Figure]:
                List of handles to all created Matplotlib figures.
        """

        dimension_strs = list(self.__dimensions.keys())
        dimension_values = list(self.__dimensions.values())

        visualized_slice = 0

        figures: List[plt.Figure] = []

        # Prepare artifacts
        section: GridSection
        graph_artifacts = np.array([np.mean(section.scalars, axis=1)
                                    for section in self.__sections.flatten()], dtype=float)

        for evaluator_idx, (evaluator, scalar_means) in enumerate(zip(self.__evaluators, graph_artifacts.T)):

            figure, axes = plt.subplots()
            figure.suptitle(evaluator.title)
            axes.plot(dimension_values[visualized_slice], scalar_means)

            # Configure axes labels
            axes.set_xlabel(dimension_strs[visualized_slice])
            axes.set_ylabel(evaluator.abbreviation)

            # Configure axes scales
            axes.set_yscale(evaluator.plot_scale)

            # Save figure to result list
            figures.append(figure)

        # Return list of resulting figures
        return figures

    def save_to_matlab(self, file: str) -> None:
        """Save simulation results to a matlab file.

        Args:

            file (str):
                File location to which the results should be saved.
        """

        # Prepare artifacts
        mean_scalar_artifacts = np.empty([*self.__sections.shape, len(self.__evaluators)], dtype=float)
        flat_iter = self.__sections.flat
        for section in flat_iter:
            mean_scalar_artifacts[np.array(flat_iter.coords)-1, :] = np.mean(section.scalars, axis=1)

        mat_dict = {
            "dimensions": np.array(list(self.__dimensions.keys()), dtype=str),
            "dimension_sections": np.array(list(product(self.__dimensions.values()))),
            "evaluators": [evaluator.abbreviation for evaluator in self.__evaluators],
            "evaluations": mean_scalar_artifacts,
            "performance_time": self.__performance_time,
        }

        """mat_dict = {      
            "snr_type": self.snr_type.name,
            "snr_vector": self.snr_loop,
            "ber_mean": self.average_bit_error_rate,
            "fer_mean": self.average_block_error_rate,
            "ber_lower": self.bit_error_min,
            "ber_upper": self.bit_error_max,
            "fer_lower": self.block_error_min,
            "fer_upper": self.block_error_max,
        }

        if self.__calc_transmit_spectrum:
            for idx, (periodogram, frequency) in enumerate(zip(self._periodogram_tx, self._frequency_range_tx)):
                if periodogram is not None and frequency is not None:
                    mat_dict["frequency_tx_" + str(idx)] = fft.fftshift(frequency)
                    mat_dict["power_spectral_density_tx_" + str(idx)] = fft.fftshift(periodogram) / np.amax(periodogram)

        if self.__calc_transmit_stft:
            for idx, (time, freq, power) in enumerate(self._stft_tx):
                if time is not None and freq is not None and power is not None:
                    mat_dict["stft_time_tx_" + str(idx)] = time
                    mat_dict["stft_frequency_tx" + str(idx)] = freq
                    mat_dict["stft_power_tx" + str(idx)] = power

        if self.__calc_receive_spectrum:
            for idx, (periodogram, frequency) in enumerate(zip(self._periodogram_rx, self._frequency_range_rx)):

                mat_dict["frequency_rx_" + str(idx)] = fft.fftshift(frequency)
                mat_dict["power_spectral_density_rx_" + str(idx)] = fft.fftshift(periodogram) / np.amax(periodogram)

        if self.__calc_receive_stft:
            for idx, (time, freq, power) in enumerate(self._stft_rx):
                if time is not None and freq is not None and power is not None:
                    mat_dict["stft_time_rx_" + str(idx)] = time
                    mat_dict["stft_frequency_rx_" + str(idx)] = freq
                    mat_dict["stft_power_rx_" + str(idx)] = power

        ber_theory = np.nan * np.ones((self.__scenario.num_transmitters,
                                      self.__scenario.num_receivers,
                                      self.__num_snr_loops), dtype=float)
        fer_theory = np.nan * np.ones((self.__scenario.num_transmitters,
                                      self.__scenario.num_receivers,
                                      self.__num_snr_loops), dtype=float)
        theory_notes = [[np.nan for _ in self.__scenario.receivers] for _ in self.__scenario.transmitters]

        if self.theoretical_results is not None:

            for tx_idx, rx_idx in zip(range(self.__scenario.num_transmitters), range(self.__scenario.num_receivers)):

                link_theory = self.theoretical_results[tx_idx, rx_idx]
                if link_theory is not None:

                    if 'ber' in link_theory:
                        ber_theory[tx_idx, rx_idx, :] = link_theory['ber']

                    if 'fer' in link_theory:
                        fer_theory[tx_idx, rx_idx, :] = link_theory['fer']

                    if 'notes' in link_theory:
                        theory_notes[tx_idx][rx_idx] = link_theory['notes']

            mat_dict["ber_theory"] = ber_theory
            mat_dict["fer_theory"] = fer_theory
            mat_dict["theory_notes"] = theory_notes"""

        # Save results in matlab file
        savemat(file, mat_dict)


class MonteCarlo(Generic[MO]):
    """Grid of parameters over which to iterate the simulation."""

    __num_samples: int                          # Maximum number of samples per grid element
    __min_num_samples: int                      # Minimum number of samples per grid element
    __num_actors: int                           # Number of dedicated actors spawned during simulation
    __investigated_object: MO                   # The object to be investigated
    __dimensions: dict[str, List[Any]]          # Simulation dimensions which make up the grid
    __evaluators: List[Evaluator[MO]]           # Evaluators used to process the investigated object sample state
    __console: Console                          # Console the simulation writes to
    __section_block_size: int                   # Number of samples per section block

    def __init__(self,
                 investigated_object: MO,
                 num_samples: int,
                 evaluators: Optional[List[Evaluator[MO]]] = None,
                 min_num_samples: int = -1,
                 num_actors: int = 0,
                 console: Optional[Console] = None,
                 section_block_size: int = 10) -> None:
        """
        Args:
            investigated_object (MO):
                Object to be investigated during the simulation runtime.

            num_samples (int):
                Number of generated samples per grid element.

            evaluators (Set[MonteCarloEvaluators[MO]]):
                Evaluators used to process the investigated object sample state.

            min_num_samples (int, optional):
                Minimum number of generated samples per grid element.

            num_actors (int, optional):
                Number of dedicated actors spawned during simulation.
                By default, the number of actors will be the number of available CPU cores.

            console (Console, optional):
                Console the simulation writes to.

            section_block_size (int, optional):
                Number of samples per section block.
                10 by default, although this number is somewhat arbitrary.
        """

        # Initialize ray if it hasn't been initialized yet. Required to query ideal number of actors
        if not ray.is_initialized():

            with catch_warnings():

                simplefilter("ignore")
                ray.init()

        self.__dimensions = {}
        self.__investigated_object = investigated_object
        self.__evaluators = [] if evaluators is None else evaluators
        self.num_samples = num_samples
        self.min_num_samples = min_num_samples if min_num_samples >= 0 else int(.5 * num_samples)
        self.num_actors = int(ray.available_resources()['CPU']) if num_actors <= 0 else num_actors
        self.__console = Console() if console is None else console
        self.section_block_size = section_block_size

    def simulate(self,
                 actor: Type[MonteCarloActor]) -> MonteCarloResult[MO]:
        """Launch the Monte Carlo simulation.

        Args:

            actor (Type[MonteCarloActor]):
                The actor from which to generate the simulation samples.

        Returns:
            np.ndarray: Generated samples.
        """

        # Generate start timestamp
        start_time = perf_counter()

        # Print meta-information and greeting
        self.console.log(f"Launched simulation campaign with {self.__num_actors} dedicated actors")

        max_num_samples = self.num_samples
        dimension_str = f"{max_num_samples}"
        for dimension in self.__dimensions.values():

            num_sections = len(dimension)
            max_num_samples *= num_sections
            dimension_str += f" x {num_sections}"

        self.console.log(f"Generating a maximum of {max_num_samples} = " + dimension_str +
                         f" samples inspected by {len(self.__evaluators)} evaluators\n")

        # Render simulation grid table
        dimension_table = Table(title="Simulation Grid", title_justify="left")
        dimension_table.add_column("Dimension", style="cyan")
        dimension_table.add_column("Sections", style="green")

        for dimension, sections in self.__dimensions.items():

            section_str = ""
            for section in sections:
                section_str += f"{section:.2f} "

            dimension_table.add_row(dimension, section_str)

        self.console.print(dimension_table)
        self.console.print()

        # Launch actors and queue the first tasks
        with self.console.status("Launching Actor Pool...", spinner='dots'):

            # Generate the actor pool
            actor_pool = ActorPool([actor.remote((self.__investigated_object, self.__dimensions, self.__evaluators))
                                    for _ in range(self.__num_actors)])

            # Generate section sample containers and meta-information
            grid_task_count = np.zeros([len(dimension) for dimension in self.__dimensions.values()], dtype=int)
            grid_active_mask = np.ones([len(dimension) for dimension in self.__dimensions.values()], dtype=bool)
            grid = np.empty([len(dimension) for dimension in self.__dimensions.values()], dtype=object)

            grid_iter = grid.flat
            for _ in grid_iter:
                cords = tuple(np.array(grid_iter.coords) - 1)
                grid[cords] = GridSection(cords, self.__evaluators)

            # Submit initial actor tasks
            task_overhead = 2  # A little overhead in task submission might speed things up? Not clear atm.
            for _ in range(self.__num_actors + task_overhead):
                _ = self.__queue_next(actor_pool, grid, grid_active_mask, grid_task_count)

        # Initialize results table
        num_result_rows = 5
        results: List[List[str]] = [[] for _ in range(num_result_rows)]

        # Initialize progress bar
        progress = Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), transient=True)
        task1 = progress.add_task("Computing", total=max_num_samples)

        # Display results in a live table
        status_group = Group(progress, '')
        with Live(status_group, console=self.console):

            # Keep executing until all samples are computed
            result_index = 0
            while actor_pool.has_next():

                # Retrieve result from pool
                samples: List[MonteCarloSample] = actor_pool.get_next_unordered(timeout=None)

                # Decrease task counter
                grid_task_count[samples[0].grid_section] -= 1

                # Queue next task and retrieve progress
                num_samples = self.__queue_next(actor_pool, grid, grid_active_mask, grid_task_count)

                # Save result
                grid_section: GridSection = grid[samples[0].grid_section]
                grid_section.add_samples(samples)

                # Check for stopping criteria
                if grid_section.num_samples >= self.min_num_samples:

                    confident: bool = sum(np.invert(grid_section.confidences)) == 0.
                    if confident:
                        grid_active_mask[samples[0].grid_section] = False

                # Print sample information by updating the table
                for sample in samples:

                    results_row: List[str] = []

                    for dimension, section_idx in zip(self.__dimensions.values(), sample.grid_section):
                        results_row.append(f"{dimension[section_idx]:.2f}")

                    results_row.append(str(result_index))

                    for artifact in sample.artifacts:
                        results_row.append(str(artifact))

                    results[result_index % num_result_rows] = results_row
                    result_index += 1

                # Render results table
                results_table = Table(min_width=self.console.measure(progress).minimum)

                for dimension in self.__dimensions:
                    results_table.add_column(dimension, style="cyan")

                results_table.add_column("#", style="blue")

                for evaluator in self.__evaluators:
                    results_table.add_column(evaluator.abbreviation, style="green")

                for result in results:
                    results_table.add_row(*result)

                status_group.renderables[1] = results_table
                progress.update(task1, completed=num_samples)

        # Measure elapsed time
        stop_time = perf_counter()
        performance_time = stop_time - start_time

        # Print finish notifier
        self.console.print()
        self.console.log(f"Simulation finished after {performance_time:.2f} seconds")

        return MonteCarloResult[MO](self.__dimensions, self.__evaluators, grid, performance_time)

    def __next_section(self,
                       grid: np.ndarray,
                       grid_active_mask: np.ndarray,
                       grid_task_count: np.ndarray) -> Tuple[Optional[Tuple[int, ...]], float]:
        """Calculate grid section and sample indices given a global sample index.

        Args:

            grid (np.ndarray):
                Simulation result grid.

            grid_active_mask (np.ndarray):
                Activity mask.

            grid_task_count (np.ndarray):
                Count of already submitted tasks within the sections.

        Returns:

            Optional[Tuple[int, ...]]:
                Coordinates of the next section to be queued for sampling.
                `None` if we are done!
        """

        num_processed_samples = 0
        flat_grid = grid.flatten()
        grid_section: GridSection
        section_coordinates = None
        for grid_section in flat_grid:

            section_coordinates = grid_section.coordinates

            # The grid section has already been marked inactive? Skip!
            # The grid section has already the maximum amount of tasks queued? Skip!
            if (not grid_active_mask[section_coordinates] or
                    grid_section.num_samples + grid_task_count[section_coordinates] >= self.num_samples):

                num_processed_samples += (self.num_samples -
                                          grid_task_count[section_coordinates] * self.section_block_size)
                section_coordinates = None
                continue

            num_processed_samples += grid_section.num_samples
            break

        return section_coordinates, num_processed_samples

    def __queue_next(self,
                     pool: ActorPool,
                     grid: np.ndarray,
                     grid_active_mask: np.ndarray,
                     grid_task_count: np.ndarray) -> float:

        next_section, progress = self.__next_section(grid, grid_active_mask, grid_task_count)

        if next_section is not None:

            pool.submit(lambda a, s: a.run.remote(s), next_section)
            grid_task_count[next_section] += 1

        return progress

    @property
    def investigated_object(self) -> Any:
        """The object to be investigated during the simulation runtime."""

        return self.__investigated_object

    def add_dimension(self, dimension: str, sample_points: List[Any]) -> None:
        """Add a dimension to the simulation grid.

        Must be a property of the investigated object.

        Args:
            dimension (str):
                String representation of dimension location relative to the investigated object.

            sample_points (List[Any]):
                List points at which the dimension will be sampled into a grid.
                The type of points must be identical to the grid arguments / type.

        Raises:
            ValueError: If the selected `dimension` does not exist within the investigated object.
        """

        # Make sure the dimension exists
        try:
            _ = reduce(lambda obj, attr: getattr(obj, attr), dimension.split('.'), self.__investigated_object)

        except AttributeError:
            raise ValueError("Dimension '" + dimension + "' does not exist within the investigated object")

        if len(sample_points) < 1:
            raise ValueError("A simulation grid dimension must have at least one sample point")

        self.__dimensions[dimension] = sample_points

    def add_evaluator(self, evaluator: Evaluator[MO]) -> None:
        """Add new evaluator to the Monte Carlo simulation.

        Args:

            evaluator (Evaluator[MO]):
                The evaluator to be added.
        """

        self.__evaluators.append(evaluator)

    @property
    def num_samples(self) -> int:
        """Number of samples per simulation grid element.

        Returns:
            int: Number of samples

        Raises:
            ValueError: If number of samples is smaller than one.
        """

        return self.__num_samples

    @num_samples.setter
    def num_samples(self, value: int) -> None:
        """Set number of samples per simulation grid element."""

        if value < 1:
            raise ValueError("Number of samples must be greater than zero")

        self .__num_samples = value
        
    @property
    def min_num_samples(self) -> int:
        """Minimum number of samples per simulation grid element.

        Returns:
            int: Number of samples

        Raises:
            ValueError: If number of samples is smaller than zero.
        """

        return self.__min_num_samples

    @min_num_samples.setter
    def min_num_samples(self, value: int) -> None:
        """Set minimum number of samples per simulation grid element."""

        if value < 0.:
            raise ValueError("Number of samples must be greater or equal to zero")

        self .__min_num_samples = value

    @property
    def max_num_samples(self) -> int:
        """Maximum number of samples over the whole simulation.

        Returns:
            int: Number of samples.
        """

        num_samples = self.num_samples
        for grid_sections in self.__dimensions.values():
            num_samples *= len(grid_sections)

        return num_samples

    @property
    def num_actors(self) -> int:
        """Number of dedicated actors spawned during simulation runs.

        Returns:
            int: Number of actors.

        Raises:
            ValueError: If the number of actors is smaller than zero.
        """

        # Return the number of available CPU cores as default value
        return self.__num_actors

    @num_actors.setter
    def num_actors(self, value: int) -> None:
        """Set number of dedicated actors spawned during simulation runs."""

        if value <= 0:
            raise ValueError("Number of actors must be greater or equal to zero")

        self.__num_actors = value

    @property
    def console(self) -> Console:
        """Console the Simulation writes to.

        Returns:
            Console: Handle to the console.
        """

        return self.__console

    @console.setter
    def console(self, value: Console) -> None:

        self.__console = value

    @property
    def section_block_size(self) -> int:
        """Number of generated samples per section block.

        Returns:
            int: Number of samples per block.

        Raises:
            ValueError:
                If the block size is smaller than one.
        """

        return self.__section_block_size

    @section_block_size.setter
    def section_block_size(self, value: int) -> None:

        if value < 1:
            raise ValueError("Section block size must be greater or equal to one")

        self.__section_block_size = value
