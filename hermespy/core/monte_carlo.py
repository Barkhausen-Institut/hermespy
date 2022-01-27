# -*- coding: utf-8 -*-
"""Monte Carlo Simulation on Python Ray."""

from __future__ import annotations

from abc import abstractmethod
from math import ceil
from functools import reduce
from shutil import get_terminal_size
from typing import Any, Callable, Generic, List, Optional, Set, Type, TypeVar, Tuple
from warnings import catch_warnings, simplefilter

import matplotlib.pyplot as plt
import numpy as np
import ray
from ray.util import ActorPool
from scipy.stats import bayes_mvs

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
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

        return str(self.to_scalar())

    def to_scalar(self) -> float:

        return self.artifact


class Evaluator(Generic[MO]):
    """Monte Carlo Sample Evaluator.

    Once a simulation sample has been generated, its properties of interest must be extracted.
    This is done by evaluators.
    """

    __confidence_level: float
    __confidence_margin: float

    def __init__(self) -> None:

        self.confidence_level = 1.
        self.confidence_margin = 0.

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
    def confidence_level(self) -> float:
        """Confidence required for premature simulation abortion.

        Returns:
            float: Confidence between zero and one.

        Raises:
            ValueError: If confidence is lower than zero or greater than one.
        """

        return self.__confidence_level

    @confidence_level.setter
    def confidence_level(self, value: float) -> None:

        if value < 0. or value > 1.:
            raise ValueError("Confidence level must be in the interval between zero and one")

        self.__confidence_level = value

    @property
    def confidence_margin(self) -> float:
        """Confidence margin required for premature simulation abortion.

        Returns:
            float: Non-negative confidence margin

        Raises:
            ValueError: If margin is negative.
        """

        return self.__confidence_margin

    @confidence_margin.setter
    def confidence_margin(self, value: float) -> None:

        if value < 0.:
            raise ValueError("Confidence margin must be greater than zero")

        self.__confidence_margin = value

    def __str__(self) -> str:
        """String object representation.

        Returns:
            str: String representation.
        """

        return self.abbreviation


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

    __indices: Tuple[int, ...]          # Section indices within the simulation grid
    __samples: List[MonteCarloSample]   # Set of generated samples
    __evaluators: List[Evaluator]       # Number of artifacts per sample
    __scalars: np.ndarray
    __scalar_means: np.ndarray
    __scalar_margins: np.ndarray

    def __init__(self,
                 indices: Tuple[int, ...],
                 evaluators: List[Evaluator]) -> None:
        """
        Args:

            indices (Tuple[int, ...]):
                Section indices within the simulation grid.

            evaluators (List[Evaluator]):
                Configured evaluators.
        """

        self.__indices = indices
        self.__samples = []
        self.__evaluators = evaluators
        self.__scalars = np.empty((self.num_evaluators, 0), dtype=float)
        self.__scalar_means = np.zeros(self.num_evaluators, dtype=float)
        self.__scalar_margins = np.empty((self.num_evaluators, 2), dtype=float)
        self.__scalar_margins[:, 0] = float('inf')
        self.__scalar_margins[:, 1] = -float('inf')

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

    def add_sample(self, sample: MonteCarloSample) -> None:
        """Add a new sample to this grid section collection.

        Args:
            sample (MonteCarloSample):
                The sample to be added.

        Raises:
            ValueError: If the number of artifacts in `sample` does not match the initialization.
        """

        # Make sure the provided number of artifacts is correct
        if sample.num_artifacts != self.num_evaluators:
            raise ValueError(f"Number of sample artifacts ({sample.num_artifacts}) does not match the "
                             f"configured number of evaluators ({self.num_evaluators})")

        # Append sample to the stored list
        self.__samples.append(sample)
        num_samples = self.num_samples
        num_samples_old = num_samples - 1

        # Update scalar storage
        scalars = sample.artifact_scalars
        self.__scalars = np.append(self.__scalars, scalars[np.newaxis, ::], axis=1)

        # Update mean scalar artifact representations
        self.__scalar_means = num_samples_old / num_samples * self.__scalar_means + scalars * num_samples ** -1

        # Update scalar confidences
        for evaluator_idx, (evaluator, scalar_mean) in enumerate(zip(self.__evaluators, self.__scalar_means)):

            evaluator_scalars = self.__scalars[evaluator_idx, :]

            # Make sure there is enough sample diversity in order to compute bayesian margins
            if evaluator_scalars.min() != evaluator_scalars.max():

                mean, var, std = bayes_mvs(evaluator_scalars, evaluator.confidence_level)
                lower_bound = mean[1][0]
                upper_bound = mean[1][1]

                self.__scalar_margins[evaluator_idx, :] = (lower_bound, upper_bound)

            else:
                self.__scalar_margins[evaluator_idx, :] = (-float('inf'), float('inf'))

    @property
    def confidence_margins(self) -> np.ndarray:
        """Margins of the confidences.

        Returns:
            np.ndarray: Numpy array with lower and upper margins on the second dimension.
        """

        return self.__scalar_margins

    @property
    def confidences(self) -> np.ndarray:
        """Confidence in the scalar evaluations.

        Returns:
            np.ndarray: Boolean array indicating confidence.
        """

        # We are confident if the true mean lies within the computed margins
        return self.__scalar_margins[:, 0] <= self.__scalar_means <= self.__scalar_margins[:, 1]


class MonteCarloActor(Generic[MO]):
    """Monte Carlo Simulation Actor.

    Actors are essentially workers running in a private process executing simulation tasks.
    The result of each individual simulation task is a simulation sample.
    """

    __investigated_object: MO                       # Copy of the object to be investigated
    __dimension_parameters: List[List[Any]]         # Parameter samples along each simulation grid dimension
    __configuration_lambdas: List[Callable]         # Configuration lambdas to configure grid parameters
    __evaluators: List[Evaluator[MO]]               # Evaluators used to process the investigated object sample state

    def __init__(self,
                 x) -> None:
                 # investigated_object: MO,
                 # dimensions: dict[str, List[Any]],
                 # evaluators: Set[Evaluator[MO]]) -> None:
        """
        Args:

            investigated_object (MO):
                Object to be investigated during the simulation runtime.

            dimensions (dict[str, List[Any]]):
                Dimensions over which the simulation will iterate.

            evaluators (Set[Evaluator[MO]]):
                Evaluators used to process the investigated object sample state.
        """

        investigated_object = x[0]
        dimensions = x[1]
        evaluators = x[2]

        self.__investigated_object = investigated_object  # deepcopy(investigated_object)
        self.__dimension_parameters = [dimension_parameters for dimension_parameters in dimensions.values()]
        self.__evaluators = evaluators

        # Generate configuration lambdas
        self.__configuration_lambdas = [self.__setter_lambda(dimension) for dimension in dimensions.keys()]

    def run(self,
            grid_section: Tuple[int, ...],
            sample_index: int) -> MonteCarloSample:
        """Run the simulation actor.

        Args:

            grid_section (Tuple[int, ...]):
                Sample point index of each grid dimension.

            sample_index (int):
                Sample index of the grid section.

        Returns:
            MonteCarloSample:
                The generated sample object.
        """

        # Configure the object under investigation
        for (configuration_lambda, parameters, parameter_idx) in zip(self.__configuration_lambdas,
                                                                     self.__dimension_parameters,
                                                                     grid_section):
            configuration_lambda(parameters[parameter_idx])

        # Sample the investigated object
        sampled_object = self.sample(self.__investigated_object)

        # Evaluate the sample
        evaluations = [evaluator.evaluate(sampled_object) for evaluator in self.__evaluators]

        # Return results
        return MonteCarloSample(grid_section, sample_index, evaluations)

    @abstractmethod
    def sample(self, investigated_object: MO) -> MO:
        """Generate a sample of the investigated object.

        Args:
            investigated_object (MO):
                The object to be investigated.
                It will be already configured to the grid parameters of the current sample.

        Returns:
            object: The resulting sample.
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
    __samples: np.ndarray

    def __init__(self,
                 dimensions: dict[str, List[Any]],
                 evaluators: List[Evaluator],
                 samples: np.ndarray) -> None:
        """
        Args:

            dimensions (dict[str, List[Any]]):
                Dimensions over which the simulation has swept.

            evaluators (List[Evaluator]):
                Evaluators used to evaluated the simulation artifacts.

            samples (np.ndarray):
                Evaluation results.

        Raises:
            ValueError:
                If the dimensions of `samples` do not match the supplied sweeping dimensions and evaluators.
        """

        self.__dimensions = dimensions
        self.__evaluators = evaluators
        self.__samples = samples

    def plot(self) -> List[plt.Figure]:

        dimension_strs = list(self.__dimensions.keys())
        dimension_values = list(self.__dimensions.values())

        visualized_slice = 0

        figures: List[plt.Figure] = []

        for evaluator_idx, evaluator in enumerate(self.__evaluators):

            # Collect artifacts

            graph_artifacts = np.array([sample.artifacts[evaluator_idx] for sample in self.__samples.flatten()])
            scalar_representation = np.array([artifact.to_scalar() for artifact in graph_artifacts])

            figure, axes = plt.subplots()
            figure.suptitle(evaluator.title)
            axes.plot(dimension_values[visualized_slice], scalar_representation)
            axes.set_xlabel(dimension_strs[visualized_slice])
            axes.set_ylabel(evaluator.abbreviation)

            figures.append(figure)

        return figures


class MonteCarlo(Generic[MO]):
    """Grid of parameters over which to iterate the simulation."""

    __num_samples: int                          # Maximum number of samples per grid element
    __min_num_samples: int                      # Minimum number of samples per grid element
    __num_actors: int                           # Number of dedicated actors spawned during simulation
    __investigated_object: MO                   # The object to be investigated
    __dimensions: dict[str, List[Any]]          # Simulation dimensions which make up the grid
    __evaluators: List[Evaluator[MO]]  # Evaluators used to process the investigated object sample state

    def __init__(self,
                 investigated_object: MO,
                 num_samples: int,
                 evaluators: Optional[List[Evaluator[MO]]] = None,
                 min_num_samples: int = -1,
                 num_actors: int = 0) -> None:
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

    def simulate(self, actor: Type[MonteCarloActor]) -> MonteCarloResult[MO]:
        """Launch the Monte Carlo simulation.

        Args:

            actor (Type[MonteCarloActor]):
                The actor from which to generate the simulation samples.

        Returns:
            np.ndarray: Generated samples.
        """

        # Print meta-information and greeting
        print(f"Launched simulation campaign with {self.__num_actors} dedicated actors")

        max_num_samples = self.num_samples
        dimension_str = f"{max_num_samples}"
        for dimension in self.__dimensions.values():

            num_sections = len(dimension)
            max_num_samples *= num_sections
            dimension_str += f" x {num_sections}"

        print(f"Generating a maximum of {max_num_samples} = " + dimension_str +
              f" samples inspected by {len(self.__evaluators)} evaluators\n")

        # Print grid information
        if len(self.__dimensions) > 0:

            print("Simulation Grid\nDimension    Sections")
            for dimension, sections in self.__dimensions.items():

                line = f"{dimension:<10}   "
                for section in sections:
                    line += f"{section:<6}  "
                print(line)

            print("\r")

        # Query terminal dimensions
        terminal_size = get_terminal_size()
        progress_percent_size = 5
        progress_bar_size = terminal_size.columns - progress_percent_size - 10
        progress_bar_elements = progress_bar_size - 2

        # Print axis information
        header = ""

        for dimension in self.__dimensions:
            header += f"{dimension:<13}"

        header += f"{'Sample':<13}"

        for evaluator in self.__evaluators:
            header += f"{evaluator.abbreviation:<13}"

        num_header_sections = 1 + len(self.__dimensions) + len(self.__evaluators)
        num_separator_elements = 13 * num_header_sections

        print(header)
        print("="*num_separator_elements)

        # Generate the actor pool
        actor_pool = ActorPool([actor.remote((self.__investigated_object, self.__dimensions, self.__evaluators))
                                for _ in range(self.__num_actors)])

        # Generate section sample containers and meta-information
        grid = np.empty([len(dimension) for dimension in self.__dimensions.values()], dtype=object)

        grid_iter = grid.flat
        for _ in grid_iter:
            cords = np.array(grid_iter.coords) - 1
            grid[cords] = GridSection(cords, self.__evaluators)

        # Global sample counter
        num_samples = 0

        # Submit initial actor tasks
        for _ in range(self.__num_actors + 2):

            actor_pool.submit(lambda a, s: a.run.remote(*s),
                              self.__section(num_samples))
            num_samples += 1

        sample_info_queue = ['' for _ in range(8)]

        # Keep executing until all samples are computed
        while actor_pool.has_next():

            # Retrieve result from pool
            sample: MonteCarloSample = actor_pool.get_next_unordered(timeout=None)

            # Print sample information
            sample_info = ''
            for dimension, section_idx in zip(self.__dimensions.values(), sample.grid_section):
                sample_info += f"{dimension[section_idx]:<13}"

            sample_info += f"{sample.sample_index:<13}"

            for artifact in sample.artifacts:
                sample_info += f"{str(artifact):<13}"

            progress = num_samples / max_num_samples
            progress_percent = int(100 * progress)
            progress_num_elements = int(ceil(progress_bar_elements * progress))
            progress_elements = '█' * progress_num_elements + '░' * (progress_bar_elements - progress_num_elements)

            sample_info_queue.pop(0)
            sample_info_queue.append(sample_info)

            for sample_info in reversed(sample_info_queue):
                print(sample_info, flush=False)

            print(f"Progress: [{progress_elements}]{progress_percent:>4}%", flush=True)

            # Save result
            grid_section: GridSection = grid[sample.grid_section]
            grid_section.add_sample(sample)

            # Check if we are already confident in the results
            if grid_section.num_samples >= self.min_num_samples:

                confident: bool = sum(np.invert(grid_section.confidences)) == 0.
                if confident:
                    print("confidence")

            # Push next job to actor pool if there are still jobs to be done
            if num_samples < max_num_samples:

                actor_pool.submit(lambda a, s: a.run.remote(*s),
                                  self.__section(num_samples))
                num_samples += 1

        return MonteCarloResult[MO](self.__dimensions, self.__evaluators, grid)

    def __section(self, index: int) -> Tuple[Tuple[int, ...], int]:
        """Calculate grid section and sample indices given a global sample index.

        Args:

            index (int):
                Global sample index over all grid sections.

        Returns:

            Tuple[np.ndarray, int]:

                section_indices (Tuple[int, ...],):
                    Indices of the grid section associated with the `index`.

                sample_index (np.ndarray):
                    Index of the sample within the respective section.
        """

        section_indices = tuple()
        num_dimension_sections = np.array([len(v) for v in self.__dimensions.values()], dtype=int)

        for dimension_idx, num_sections in enumerate(num_dimension_sections):

            section_indices += (int(index / np.prod(num_dimension_sections[1+dimension_idx:])
                                    / self.num_samples) % num_sections),

        return section_indices, 0

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
