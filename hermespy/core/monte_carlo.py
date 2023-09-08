# -*- coding: utf-8 -*-
"""
=======
PyMonte
=======

PyMonte is a stand-alone core module of HermesPy,
enabling efficient and flexible Monte Carlo simulations over arbitrary configuration parameter combinations.
By wrapping the core of the `Ray`_ project,
any object serializable by the `pickle`_ standard module can become a system model for a Monte Carlo style simulation
campaign.

.. mermaid::

   %%{init: {'theme': 'dark'}}%%
   flowchart LR

   subgraph gridsection[Grid Section]

      parameter_a(Parameter)
      parameter_b(Parameter)
   end

   object((Investigated Object))
   evaluator_a{{Evaluator}}
   evaluator_b{{Evaluator}}
   evaluator_c{{Evaluator}}

   subgraph sample[Sample]

       artifact_a[(Artifact)]
       artifact_b[(Artifact)]
       artifact_c[(Artifact)]
   end

   parameter_a --> object
   parameter_b --> object
   object ---> evaluator_a ---> artifact_a
   object ---> evaluator_b ---> artifact_b
   object ---> evaluator_c ---> artifact_c


Monte Carlo simulations usually sweep over multiple combinations of multiple parameters settings,
configuring the underlying system model and generating simulation samples from independent realizations
of the model state.
PyMonte refers to a single parameter combination as :class:`.GridSection`,
with the set of all parameter combinations making up the simulation grid.
Each settable property of the investigated object is treated as a potential simulation parameter within the grid,
i.e. each settable property can be represented by an axis within the multidimensional simulation grid.

:class:`.Evaluator` instances extract performance indicators from each investigated object realization, referred to as :class:`.Artifact`.
A set of artifacts drawn from the same investigated object realization make up a single :class:`.MonteCarloSample`.
During the execution of PyMonte simulations between :math:`M_\\mathrm{min}` and :math:`M_\\mathrm{max}`
are generated from investigated object realizations for each grid section.
The sample generation for each grid section may be aborted prematurely if all evaluators have reached a configured
confidence threshold
Refer to :footcite:t:`2014:bayer` for a detailed description of the implemented algorithm.

.. mermaid::

   %%{init: {'theme': 'dark'}}%%
   flowchart LR

   controller{Simulation Controller}

   gridsection_a[Grid Section]
   gridsection_b[Grid Section]

   sample_a[Sample]
   sample_b[Sample]

   subgraph actor_a[Actor #1]

       object_a((Investigated Object))
   end

   subgraph actor_b[Actor #N]

       object_b((Investigated Object))
   end

   controller --> gridsection_a --> actor_a --> sample_a
   controller --> gridsection_b --> actor_b --> sample_b


The actual simulation workload distribution is visualized in the previous flowchart.
Using `Ray`_, PyMonte spawns a number of :class:`.MonteCarloActor` containers,
with the number of actors depending on the available resources (i.e. number of CPU cores) detected.
A central simulation controller schedules the workload by assigning :class:`.GridSection` indices as tasks
to the actors, which return the resulting simulation Samples after the simulation iteration is completed.

.. _Ray: https://www.ray.io/
.. _pickle: https://docs.python.org/3/library/pickle.html
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext
from functools import reduce
from math import exp, sqrt
from time import perf_counter
from typing import Any, Callable, Generic, List, Optional, Type, TypeVar, Tuple, Union, SupportsFloat
from warnings import catch_warnings, simplefilter

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.axes as axes
import numpy as np
import ray
import transaction  # type: ignore
from os import path
from persistent import Persistent  # type: ignore
from persistent.mapping import PersistentMapping  # type: ignore
from persistent.list import PersistentList  # type: ignore
from ray.util import ActorPool
from rich.console import Console, Group
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.live import Live
from rich.table import Table
from scipy.constants import pi
from scipy.io import savemat
from scipy.stats import norm
from tempfile import NamedTemporaryFile
from ZODB.FileStorage import FileStorage  # type: ignore
from ZODB import DB  # type: ignore
from BTrees.OOBTree import OOBTree  # type: ignore

from hermespy.tools import lin2db
from .definitions import ConsoleMode
from .logarithmic import LogarithmicSequence, ValueType
from .visualize import Visualizable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


MO = TypeVar("MO")
"""Type of Monte Carlo object under investigation.

:meta private:
"""

AT = TypeVar("AT")
"""Type of Monte Carlo evaluation artifact.

:meta private:
"""

FAT = TypeVar("FAT", bound=SupportsFloat)
"""Type of floating point Monte Carlo evaluation artifact.

:meta private:
"""

ET = TypeVar("ET")
"""Type of Monte Carlo evaluation.

:meta private:
"""

SERT = TypeVar("SERT", bound="ScalarEvaluationResult")
"""Type of scalar Monte Carlo evaluation result.

:meta private:
"""

EV = TypeVar("EV", bound="Evaluator")
"""Type of Monte Carlo evalutor.

:meta private:
"""


class Artifact(Persistent):
    """Result of an investigated object evaluation.

    Generated by :class:`.Evaluator` instances operating on investigated object states.
    In other words, :meth:`.Evaluator.evaluate` is expected to return an instance derived of this base class.

    Artifacts may, in general represent any sort of object.
    However, it is encouraged to provide a scalar floating-point representation for data visualization by implementing
    the :meth:`.to_scalar` method.
    """

    @abstractmethod
    def __str__(self) -> str:
        """String representation of this artifact.

        Will be used to visualize the artifact's content in console outputs.

        Returns:
            str: String representation.
        """
        ...  # pragma: no cover

    @abstractmethod
    def to_scalar(self) -> Optional[float]:
        """Scalar representation of this artifact's content.

        Used to evaluate premature stopping criteria for the underlying evaluation.

        Returns:
            Optional[float]:
                Scalar floating-point representation.
                `None` if a conversion to scalar is impossible.
        """
        ...  # pragma: no cover


class ArtifactTemplate(Generic[FAT], Artifact):
    """Scalar numerical result of an investigated object evaluation.

    Implements the common case of an :class:`.Artifact` representing a scalar numerical value.
    """

    __artifact: FAT  # Artifact

    def __init__(self, artifact: FAT) -> None:
        """
        Args:

            artifact (AT):
                Artifact value.
        """

        self.__artifact = artifact

    @property
    def artifact(self) -> FAT:
        """Evaluation artifact.

        Provides direct access to the represented artifact.

        Returns:
            AT: Copy of the artifact.
        """

        return self.__artifact

    def __str__(self) -> str:
        return f"{self.to_scalar():.3f}"

    def to_scalar(self) -> float:
        return float(self.artifact)


class Evaluation(Visualizable):
    """Evaluation of a single simulation sample.

    Evaluations are generated by :class:`Evaluators <Evaluator>`
    during :meth:`Evaluator.evaluate`.
    """

    @abstractmethod
    def artifact(self) -> Artifact:
        """Generate an artifact from this evaluation.

        Returns: The evaluation artifact."""
        ...  # pragma: no cover


class EvaluationTemplate(Generic[ET], Evaluation, ABC):
    evaluation: ET

    def __init__(self, evaluation: ET) -> None:
        self.evaluation = evaluation


class EvaluationResult(Visualizable, ABC):
    """Result of an evaluation routine iterating over a parameter grid.

    Evaluation results are generated by :class:`Evaluator Instances <.Evaluator>` as a final
    step within the evaluation routine.
    """

    @abstractmethod
    def to_array(self) -> np.ndarray:
        """Convert the evaluation result raw data to an array representation.

        Used to store the results in arbitrary binary file formats after simulation execution.

        Returns:

            The array result representation.
        """
        ...  # pragma: no cover

    @staticmethod
    def __configure_axis(axis: axes.Axes, tick_format: ValueType = ValueType.LIN) -> None:
        """Subroutine to configure an axis.

        Args:

            axis:
                The matplotlib axis to be configured.

            tick_format (ValueType):
                The tick format.
        """

        if tick_format is ValueType.DB:
            axis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{lin2db(x):.2g}dB"))

    @staticmethod
    def _plot_linear(grid: List[GridDimension], sample_points: Sequence[np.float_], scalar_data: np.ndarray, evaluator: Evaluator, axes: plt.Axes) -> None:
        """Visualize result as a linear plot.

        Args:

            grid (List[GridDimension]):
                Simulation grid.

            sample_points (np.ndarray):
                Grid coordinates to visualize.

            scalar_data (np.ndarray):
                Results at the respective coordinates.

            evaluator (Evaluator):
                The evaluator generating the results.

            axes (plt.Axes):
                Axes to plot in.
                Must be initialized as a 2D axes.
        """

        # Configure axes labels
        axes.set_xlabel(grid[0].title)
        axes.set_ylabel(evaluator.abbreviation)

        # Configure axes scales
        axes.set_xscale(grid[0].plot_scale)
        axes.set_yscale(evaluator.plot_scale)

        EvaluationResult.__configure_axis(axes.xaxis, grid[0].tick_format)
        EvaluationResult.__configure_axis(axes.yaxis, evaluator.tick_format)

        # Plot final result
        axes.plot(sample_points, scalar_data)

    @staticmethod
    def _plot_surface(grid: List[GridDimension], sample_points: Sequence[np.float_], data: np.ndarray, evaluator: Evaluator, axes: plt.Axes) -> None:
        """Visualize result as a surface plot.

        Args:

            grid (List[GridDimension]):
                Simulation grid.

            sample_points (Sequence[np.float_]):
                Grid coordinates to visualize.

            data (np.ndarray):
                Results at the respective coordinates.

            evaluator (Evaluator):
                The evaluator generating the results.

            axes (plt.Axes):
                Axes to plot in.
                Must be initialized as a 3D axes.
        """

        # Configure axes labels and scales
        axes.set(xlabel=grid[0].title, ylabel=grid[1].title, zlabel=evaluator.abbreviation)

        EvaluationResult.__configure_axis(axes.xaxis, grid[0].tick_format)
        EvaluationResult.__configure_axis(axes.yaxis, grid[1].tick_format)
        EvaluationResult.__configure_axis(axes.zaxis, evaluator.tick_format)

        y, x = np.meshgrid(np.asarray(grid[1].sample_points), np.asarray(sample_points))
        axes.plot_surface(x.astype(float), y.astype(float), data)

    @staticmethod
    def _plot_multidim(grid: List[GridDimension], scalar_data: np.ndarray, x_dimension: int, y_label: str, x_scale: str, y_scale: str, axes: plt.Axes) -> None:
        """Plot multidimensional simulation results into a two-dimensional axes system.

        Args:
            grid (List[GridDimension]):
                Simulation grid.

            scalar_data (np.ndarray):
                Scalar data to be plotted.

            x_dimension (int):
                The dimension index featured on the plot's x-axis.

            y_label (str):
                The plot's y-axis label

            x_scale (str):
                The plot's x-axis scale.

            y_scale (str):
                The plot's y-axis scale.

            axes (plt.Axes):
                Axes to plot in.
                Must be initialized as a 2D axes.
        """

        # Configure axes labels
        axes.set_xlabel(grid[x_dimension].title)
        axes.set_ylabel(y_label)

        # Configure axes scales
        axes.set_yscale(y_scale)
        axes.set_xscale(x_scale)

        subgrid = grid.copy()
        del subgrid[x_dimension]

        section_magnitudes = tuple(s.num_sample_points for s in subgrid)
        for section_indices in np.ndindex(section_magnitudes):
            # Generate the graph line label
            line_label = ""
            for i, v in enumerate(section_indices):
                line_label += f"{grid[i+1].title} = {grid[i+1].sample_points[v]}, "
            line_label = line_label[:-2]

            if len(line_label) < 1:
                line_label = None

            # Select the graph line scalars
            line_scalars = scalar_data[(..., *section_indices)]  # type: ignore

            # Plot the graph line
            axes.plot(grid[x_dimension].sample_points, line_scalars, label=line_label)

        if len(section_magnitudes) > 0 and section_magnitudes[0] > 1:
            axes.legend()

    @staticmethod
    def _plot_empty(axes: plt.Axes) -> None:
        # Create single axes
        axes.text(0.5, 0.5, "NO DATA AVAILABLE", horizontalalignment="center")


class ScalarEvaluationResult(EvaluationResult):
    """Base class for scalar evaluation results."""

    plot_surface: bool

    __grid: List[GridDimension]
    __scalar_results: np.ndarray
    __evaluator: Evaluator
    __base_dimension_index: int

    def __init__(self, grid: Sequence[GridDimension], scalar_results: np.ndarray, evaluator: Evaluator, plot_surface: bool = True) -> None:
        """
        Args:

            grid (Sequence[GridDimension]):
                Simulation grid.

            scalar_results (np.ndarray):
                Scalar results generated from collecting samples over the simulation grid.

            evaluator (Evaluator):
                The evaluator generating the results.

            plot_surface (bool, optional):
                Enable surface plotting for two-dimensional grids.
                Enabled by default.
        """

        self.__grid = list(grid)
        self.__scalar_results = scalar_results
        self.__evaluator = evaluator
        self.plot_surface = plot_surface
        self.__base_dimension_index = 0

    @property
    def title(self) -> str:
        # The plotting title should resolve to the represented evaluator's title
        return self.__evaluator.title

    def _new_axes(self) -> Tuple[plt.Figure, plt.Axes]:
        # Generate 3D axes with surface plotting enabled
        if len(self.__grid) == 2 and self.plot_surface:
            fig = plt.figure()
            axes = fig.add_subplot(111, projection="3d")

        # Generate 2D axes otherwise
        else:
            fig, axes = plt.subplots()

        return fig, axes

    def _plot(self, axes: plt.Axes) -> None:
        # If the grid contains no data, dont plot anything
        if len(self.__grid) < 1:
            self._plot_empty(axes)
            return

        # Shuffle grid and respective scalar results so that the selected base dimension is always the first entry
        grid = self.__grid.copy()
        grid.insert(0, grid.pop(self.__base_dimension_index))
        scalars = np.moveaxis(self.__scalar_results, self.__base_dimension_index, 0)
        sample_points = self.__grid[0].sample_points

        # A single axis plot, the simple case
        if len(grid) < 2:
            self._plot_linear(grid, sample_points, scalars, self.__evaluator, axes)

        # Two dimensions, with surface plotting enabled
        elif len(grid) == 2 and self.plot_surface:
            self._plot_surface(grid, sample_points, scalars, self.__evaluator, axes)

        # Multiple dimensions, resort to legend-based multiplots
        else:
            self._plot_multidim(grid, scalars, 0, self.__evaluator.abbreviation, grid[0].plot_scale, self.__evaluator.plot_scale, axes)

    def to_array(self) -> np.ndarray:
        return self.__scalar_results

    @classmethod
    def From_Artifacts(cls: Type[SERT], grid: Sequence[GridDimension], artifacts: np.ndarray, evaluator: Evaluator, plot_surface: bool = True) -> SERT:
        """Generate a scalar evaluation result from a set of artifacts.

        Args:

            grid (Sequence[GridDimension]):
                The simulation grid.

            artifacts (np.ndarray):
                Numpy object array whose dimensions represent grid dimensions.

            evaluator (Evaluator):
                The evaluator generating the artifacts.

            plot_surface (bool):
                Whether to plot the result as a surface plot.

        Returns:

            The scalar evaluation result.
        """

        scalar_results = np.empty(artifacts.shape, dtype=float)
        for section_coords in np.ndindex(artifacts.shape):
            scalar_results[section_coords] = np.mean([artifact.to_scalar() for artifact in artifacts[section_coords]])

        return cls(grid, scalar_results, evaluator, plot_surface)


class Evaluator(ABC):
    """Evaluation routine for investigated object states, extracting performance indicators of interest.

    Evaluators represent the process of extracting arbitrary performance indicator samples :math:`X_m` in the form of
    :class:`.Artifact` instances from investigated object states.
    Once a :class:`.MonteCarloActor` has set its investigated object to a new random state,
    it calls the :func:`.evaluate` routines of all configured evaluators,
    collecting the resulting respective :class:`.Artifact` instances.

    For a given set of :class:`.Artifact` instances,
    evaluators are expected to report a :meth:`.confidence_level` which may result in a premature abortion of the
    sample collection routine for a single :class:`.GridSection`.
    By default, the routine suggested by :footcite:t:`2014:bayer` is applied:
    Considering a tolerance :math:`\\mathrm{TOL} \\in \\mathbb{R}_{++}` the confidence in the mean performance indicator

    .. math::

        \\bar{X}_M = \\frac{1}{M} \\sum_{m = 1}^{M} X_m

    is considered  sufficient if a threshold :math:`\\delta \\in \\mathbb{R}_{++}`, defined as

    .. math::

        \\mathrm{P}\\left(\\left\\| \\bar{X}_M - \\mathrm{E}\\left[ X \\right] \\right\\| > \\mathrm{TOL} \\right) \\leq \\delta

    has been crossed.
    The number of collected actually collected artifacts per :class:`.GridSection` :math:`M \\in [M_{\\mathrm{min}}, M_{\\mathrm{max}}]`
    is between a minimum number of required samples :math:`M_{\\mathrm{min}} \\in \\mathbb{R}_{+}` and an upper limit of
    :math:`M_{\\mathrm{max}} \\in \\mathbb{R}_{++}`.
    """

    tick_format: ValueType
    __confidence: float
    __tolerance: float
    __plot_scale: str  # Plot axis scaling

    def __init__(self) -> None:
        self.confidence = 1.0
        self.tolerance = 0.0
        self.plot_scale = "linear"
        self.tick_format = ValueType.LIN

    @abstractmethod
    def evaluate(self) -> Evaluation:
        """Evaluate the state of an investigated object.

        Implements the process of extracting an arbitrary performance indicator, represented by
        the returned :class:`.Artifact` :math:`X_m`.

        Returns: Artifact :math:`X_m` resulting from the evaluation.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def abbreviation(self) -> str:
        """Short string representation of this evaluator.

        Used as a label for console output and plot axes annotations.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def title(self) -> str:
        """Long string representation of this evaluator.

        Used as plot title.
        """
        ...  # pragma: no cover

    @property
    def confidence(self) -> float:
        """Confidence threshold required for premature simulation abortion.

        The confidence threshold :math:`\\delta \\in [0, 1]` is the upper bound to the
        confidence level

        .. math::

            \\mathrm{P}\\left(\\left\\| \\bar{X}_M - \\mathrm{E}\\left[ X \\right] \\right\\| > \\mathrm{TOL} \\right)

        at which the sample collection for a single :class:`.GridSection` may be prematurely aborted :footcite:p:`2014:bayer`.

        Raises:
            ValueError: If confidence is lower than zero or greater than one.
        """

        return self.__confidence

    @confidence.setter
    def confidence(self, value: float) -> None:
        if value < 0.0 or value > 1.0:
            raise ValueError("Confidence level must be in the interval between zero and one")

        self.__confidence = value

    @property
    def tolerance(self) -> float:
        """Tolerance level required for premature simulation abortion.

        The tolerance :math:`\\mathrm{TOL} \\in \\mathbb{R}_{++}` is the upper bound to the interval

        .. math::

           \\left\\| \\bar{X}_M - \\mathrm{E}\\left[ X \\right] \\right\\|

        by which the performance indicator estimation :math:`\\bar{X}_M` may diverge from the actual expected
        value :math:`\\mathrm{E}\\left[ X \\right]`.

        Returns:
            float: Non-negative tolerance :math:`\\mathrm{TOL}`.

        Raises:
            ValueError: If tolerance is negative.
        """

        return self.__tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        if value < 0.0:
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

    @abstractmethod
    def generate_result(self, grid: Sequence[GridDimension], artifacts: np.ndarray) -> EvaluationResult:
        """Generates an evaluation result from the artifacts collected over the whole simulation grid.

        Args:

            grid (Sequence[GridDimension]):
                The Simulation grid.

            artifacts (np.ndarray):
                Numpy object array whose dimensions represent grid dimensions.

        Returns:

            The evaluation result.
        """
        ...  # pragma: no cover


class MonteCarloSample(Persistent):
    """Single sample of a Monte Carlo simulation."""

    __sample_index: int  # Index of the sample
    # Grid section from which the sample was generated
    __grid_section: Tuple[int, ...]
    __artifacts: PersistentList[Artifact]  # Artifacts of evaluation

    def __init__(self, grid_section: Tuple[int, ...], sample_index: int, artifacts: Sequence[Artifact]) -> None:
        """
        Args:

            grid_section (Tuple[int, ...]):
                Grid section from which the sample was generated.

            sample_index (int):
                Index of the sample.
                In other words this object represents the `sample_index`th sample of the selected `grid_section`.

            artifacts (Sequence[Artifact]):
                Artifacts of evaluation
        """

        self.__grid_section = grid_section
        self.__sample_index = sample_index
        self.__artifacts = PersistentList(artifacts)

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


class GridSection(Persistent):
    # Berry-Esseen constants ToDo: Check the proper selection here
    __C_0: float = 0.4785
    __C_1: float = 30.2338

    # Section indices within the simulation grid
    __coordinates: Tuple[int, ...]
    __samples: PersistentList[MonteCarloSample]  # Set of generated samples
    # Confidence level for each evaluator
    __evaluator_confidences: np.ndarray

    def __init__(self, coordinates: Tuple[int, ...], num_evaluators: int) -> None:
        """
        Args:

            coordinates (Tuple[int, ...]):
                Section indices within the simulation grid.
        """

        self.__coordinates = coordinates
        self.__samples = PersistentList()
        self.__num_evaluators = num_evaluators

        self.__evaluator_confidences = np.zeros(num_evaluators, dtype=float)
        self.__means = np.zeros(num_evaluators, dtype=float)
        self.__second_moments = np.zeros(num_evaluators, dtype=float)
        self.__third_moments = np.zeros(num_evaluators, dtype=float)
        self.__third_moments_abs = np.zeros(num_evaluators, dtype=float)
        self.__fourth_moments = np.zeros(num_evaluators, dtype=float)

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
    def samples(self) -> List[MonteCarloSample]:
        """The collected evaluation samples within this grid section.

        Returns: List of samples.
        """

        return self.__samples

    def add_samples(self, samples: MonteCarloSample | Sequence[MonteCarloSample], evaluators: Sequence[Evaluator]) -> None:
        """Add a new sample to this grid section collection.

        Args:

            samples (Union[MonteCarloSample, Sequence[MonteCarloSample])):
                Samples to be added to this section.

            evaluators (Sequence[Evaluator]):
                References to the evaluators generating the sample artifacts.

        Raises:
            ValueError: If the number of artifacts in `sample` does not match the initialization.
        """

        _samples = samples if isinstance(samples, Sequence) else [samples]

        for sample in _samples:
            # Make sure the provided number of artifacts is correct
            if sample.num_artifacts != self.__num_evaluators:
                raise ValueError(f"Number of sample artifacts ({sample.num_artifacts}) does not match the " f"configured number of evaluators ({self.__num_evaluators})")

            # Update confidences
            self.__update_confidences(sample.artifact_scalars, evaluators)

            # Append sample to the stored list
            self.__samples.append(sample)

    @property
    def confidences(self) -> np.ndarray:
        """Confidence in the estimated evaluations.

        Returns: Array indicating probabilities for each evaluator
        """

        return self.__evaluator_confidences

    def confidence_status(self, evaluators: Sequence[Evaluator]) -> bool:
        """Check if each evaluator has reached its required confidence thershold.

        Conidence indicates that the simulation for the parameter combination this grid section represents
        may be aborted, i.e. no more samples are required.

        Args:

            evaluators (Sequence[Evaluator]):
                Evaluators giving feedback about their cofidence status.

        Returns: Confidence indicator.
        """

        for evaluator, confidence in zip(evaluators, self.__evaluator_confidences):
            if confidence < evaluator.confidence:
                return False

        return True

    def __update_confidences(self, artifact_scalars: np.ndarray, evaluators: Sequence[Evaluator]) -> None:
        """Update the confidence estimates for this grid section.

        Args:

            artifact_scalars (np.ndarray):
                Scalar representations of the artifacts generated by the sample.

            evaluators (Sequence[Evaluator]):
                Evaluators generating the artifacts.

        Raises:

            ValueError: If the artifact scalars argument is not a vector.
            ValueError: If the number of artifact scalars does not match the number of configured evaluators.
        """

        # Raise a value error if the scalars argument is not a vector
        if artifact_scalars.ndim != 1:
            raise ValueError("Artifact scalars must be a vector (on-dimensional array)")

        if len(artifact_scalars) != self.__num_evaluators:
            raise ValueError("Number of artifact scalars does not match the number of configured evaluators")

        for e, (scalar, evaluator, mean, second_moment, third_moment, third_moment_abs, fourth_moment) in enumerate(zip(artifact_scalars, evaluators, self.__means, self.__second_moments, self.__third_moments, self.__third_moments_abs, self.__fourth_moments)):
            # Return zero if the evaluator tolerance is set to zero
            if evaluator.tolerance == 0.0:
                self.__evaluator_confidences[e] = 0.0
                continue

            n = self.num_samples  # Current number of samples
            nn = 1 + self.num_samples  # New number of samples

            # Update the moment estimates
            delta = scalar - mean
            updated_mean = (n * mean) / nn
            updated_second_moment = second_moment + delta**2 * (n) / nn
            updated_third_moment = third_moment + delta**3 * (n**2 - n) / nn**2 - 3 * second_moment * delta / nn
            updated_third_moment_abs = third_moment_abs + abs(delta**3 * (n**2 - n) / nn**2 - 3 * second_moment * delta / nn)
            updated_fourth_moment = fourth_moment + delta**4 * (n**3 - n**2 + n) / nn**3 + 6 * delta**2 * second_moment / nn**2 - 4 * delta * third_moment / nn

            # Store the updated moment estimates for the next update
            self.__means[e] = updated_mean
            self.__second_moments[e] = updated_second_moment
            self.__third_moments[e] = updated_third_moment
            self.__third_moments_abs[e] = updated_third_moment_abs
            self.__fourth_moments[e] = updated_fourth_moment

            # Abort if the second moment indicates no variance or not enough samples have been collected
            if updated_second_moment == 0.0 or nn < 2:
                self.__evaluator_confidences[e] = 1.0
                continue

            # Compute moments (equation 4.1)
            deviance = sqrt(updated_second_moment / nn)
            beta_bar_moment = third_moment / (nn * deviance**3)
            beta_hat_moment = third_moment_abs / (nn * deviance**3)
            kappa_moment = fourth_moment / (nn * deviance**4) - 3

            # Estimate the confidence
            sample_sqrt = sqrt(nn)
            sigma_tolerance = sample_sqrt * evaluator.tolerance / deviance
            sigma_tolerance_squared = sigma_tolerance**2
            kappa_term = 4 * (2 / (nn - 1) + kappa_moment / nn)

            confidence_bound = 2 * (1 - evaluator._scalar_cdf(sigma_tolerance)) + 2 * min(self.__C_0, self.__C_1 / (1 + abs(sigma_tolerance)) ** 3) * beta_bar_moment / sample_sqrt * min(1.0, kappa_term)

            # This is necessary to preventa math overflow from the exponential denominator
            if kappa_term < 1.0 and sigma_tolerance_squared < 1418.0:
                confidence_bound += (1 - kappa_term) * abs(sigma_tolerance_squared - 1) * abs(beta_hat_moment) / (exp(0.5 * sigma_tolerance_squared) * 3 * sqrt(2 * pi * n) * deviance**3)

            # Store the current confidence estimate
            self.__evaluator_confidences[e] = 1.0 - min(1.0, confidence_bound)


class ActorRunResult(object):
    """Result object returned by generating a single sample from a simulation runner."""

    samples: List[MonteCarloSample] | None
    message: str | None

    def __init__(self, samples: List[MonteCarloSample] | None = None, message: str | None = None) -> None:
        """
        Args:

            samples (List[MonteCarloSample], optional):
                Samples generated by the remote actor run.

            message (str, optional):
                Message return from the remote actor run.
        """

        self.samples = [] if not samples else samples
        self.message = message


class SampleGrid(Persistent):
    """Grid of simulation samples.

    Persisent data structure containing the simulation samples generated by a :class:`.MonteCarloRunner`.
    """

    __sections: OOBTree

    def __init__(self, grid_configuration: List[GridDimension], evaluators: Sequence[Evaluator]) -> None:
        """
        Args:

            grid_configuration (List[GridDimension]):
                The simulation grid configuration.

            evaluators (Sequence[Evaluator]):
                The evaluators generating the artifacts.
        """

        self.__sections = OOBTree()
        num_evaluators = len(evaluators)

        for coordinates in np.ndindex(*[dimension.num_sample_points for dimension in grid_configuration]):
            coordinate_tuple = tuple(coordinates)
            self.__sections[coordinate_tuple] = GridSection(coordinate_tuple, num_evaluators)

    def __getitem__(self, coordinates: Tuple[int, ...]) -> GridSection:
        return self.__sections[coordinates]

    def __iter__(self):
        """Iterating over the sample grid is equivalent to iterating over the sections tree"""

        return iter(self.__sections.values())


class MonteCarloActor(Generic[MO]):
    """Monte Carlo Simulation Actor.

    Actors are essentially workers running in a private process executing simulation tasks.
    The result of each individual simulation task is a simulation sample.
    """

    catch_exceptions: bool  # Catch exceptions during run.
    __investigated_object: MO  # Copy of the object to be investigated
    __grid: Sequence[GridDimension]  # Simulation grid dimensions
    __evaluators: Sequence[Evaluator]  # Evaluators used to process the investigated object sample state

    def __init__(self, argument_tuple: Tuple[MO, Sequence[GridDimension], Sequence[Evaluator]], index: int, catch_exceptions: bool = True) -> None:
        """
        Args:

            argument_tuple:
                Object to be investigated during the simulation runtime.
                Dimensions over which the simulation will iterate.
                Evaluators used to process the investigated object sample state.

            index (int):
                Global index of the actor.

            catch_exceptions (bool, optional):
                Catch exceptions during run.
                Enabled by default.
        """

        investigated_object = argument_tuple[0]
        grid = argument_tuple[1]
        evaluators = argument_tuple[2]

        self.index = index
        self.catch_exceptions = catch_exceptions
        self.__investigated_object = investigated_object
        self.__grid = grid
        self.__evaluators = evaluators
        self.__stage_identifiers = self.stage_identifiers()
        self.__stage_executors = self.stage_executors()
        self.__num_stages = len(self.__stage_executors)

    @property
    def _investigated_object(self) -> MO:
        """State of the Investigated Object.

        Returns:
            Mo: Investigated object.
        """

        return self.__investigated_object  # pragma: no cover

    def run(self, program: List[Tuple[int, ...]]) -> ActorRunResult:
        """Run the simulation actor.

        Args:

            program (List[Tuple[int, ...]]):
                A list of simulation grid section indices for which to collect samples.

        Returns:
            A list of generated :class:`MonteCarloSample`s.
            Contains the same number of entries as `program`.
        """

        result = ActorRunResult()

        # Catch any exception during actor running
        try:
            # Intially, re-configure the full grid
            recent_section_indices = np.array(program[0], dtype=int)
            for d, i in enumerate(recent_section_indices):
                self.__grid[d].configure_point(i)

            # Run through the program steps
            for section_indices in program:
                # Detect the grid dimensions where sections changed, i.e. which require re-configuration
                section_index_array = np.asarray(section_indices, dtype=int)
                reconfigured_dimensions = np.argwhere(section_index_array != recent_section_indices).flatten()

                # Reconfigure the dimensions
                # Not that for the first grid_section this has already been done
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
                        first_impact = min(first_impact, self.__stage_identifiers.index(grid_dimension.first_impact))

                    if grid_dimension.last_impact is None:
                        last_impact = self.__num_stages - 1

                    elif grid_dimension.last_impact in self.__stage_identifiers:
                        last_impact = max(last_impact, self.__stage_identifiers.index(grid_dimension.last_impact))

                if first_impact >= self.__num_stages:
                    first_impact = 0

                if last_impact <= 0:
                    last_impact = self.__num_stages - 1

                # Execute impacted simulation stages
                # Note that for the first grid_section all stages are executed
                for stage in self.__stage_executors[first_impact : 1 + last_impact]:
                    stage()

                # Collect evaluation artifacts
                artifacts = [evaluator.evaluate().artifact() for evaluator in self.__evaluators]

                # Save the samples
                result.samples.append(MonteCarloSample(section_indices, 0, artifacts))

                # Update the recent section for the next iteration
                recent_section_indices = section_index_array

        except Exception as e:
            if self.catch_exceptions:
                result.message = str(e)

            else:
                raise e  # pragma: no cover

        return result

    @staticmethod
    @abstractmethod
    def stage_identifiers() -> List[str]:
        """List of simulation stage identifiers.

        Simulations stages will be executed in the order specified here.

        Returns:

            List of function identifiers for simulation stage execution routines.
        """
        ...  # pragma: no cover

    @abstractmethod
    def stage_executors(self) -> List[Callable]:
        """List of simulation stage execution callbacks.

        Simulations stages will be executed in the order specified here.

        Returns:

            List of function callbacks for simulation stage execution routines.
        """
        ...  # pragma: no cover


class MonteCarloResult(object):
    """Result of a Monte Carlo simulation."""

    __grid: Sequence[GridDimension]
    __evaluators: Sequence[Evaluator]
    __performance_time: float  # Absolute time required to compute the simulation
    __results: Sequence[EvaluationResult]

    def __init__(self, grid: Sequence[GridDimension], evaluators: Sequence[Evaluator], sample_grid: SampleGrid, performance_time: float) -> None:
        """
        Args:

            grid (Sequence[GridDimension]):
                Dimensions over which the simulation has swept.

            evaluators (Sequence[Evaluator]):
                Evaluators used to evaluated the simulation artifacts.

            sample_grid (SampleGrid):
                Grid containing evaluation artifacts collected over the grid iterations.

            performance_time (float):
                Time required to compute the simulation.

        Raises:
            ValueError:
                If the dimensions of `samples` do not match the supplied sweeping dimensions and evaluators.
        """

        self.__grid = grid
        self.__evaluators = evaluators
        self.__performance_time = performance_time

        self.__results = []
        for evaluator_idx, evaluator in enumerate(evaluators):
            # Collect artifacts for the respective evaluator
            evaluator_artifacts = np.empty(tuple([d.num_sample_points for d in grid]), dtype=object)

            section: GridSection
            for section in sample_grid:
                evaluator_artifacts[section.coordinates] = [sample.artifacts[evaluator_idx] for sample in section.samples]

            self.__results.append(evaluator.generate_result(grid, evaluator_artifacts))

    @property
    def performance_time(self) -> float:
        """Simulation runtime.

        Returns:
            Simulation runtime in seconds.
        """

        return self.__performance_time

    def plot(self) -> List[plt.Figure]:
        """Plot evaluation figures for all contained evaluator artifacts.

        Returns:
            List[plt.Figure]:
                List of handles to all created Matplotlib figures.
        """
        return [result.plot() for result in self.__results]

    def save_to_matlab(self, file: str) -> None:
        """Save simulation results to a matlab file.

        Args:

            file (str):
                File location to which the results should be saved.
        """

        mat_dict = {"dimensions": [d.title for d in self.__grid], "evaluators": [evaluator.abbreviation for evaluator in self.__evaluators], "performance_time": self.__performance_time}

        # Append evaluation array representions
        for r, result in enumerate(self.__results):
            mat_dict[f"result_{r}"] = result.to_array()

        # Append evaluation array representions
        for r, result in enumerate(self.__results):
            mat_dict[f"result_{r}"] = result.to_array()

        for dimension in self.__grid:
            mat_dict[dimension.title] = dimension.sample_points

        # Save results in matlab file
        savemat(file, mat_dict)

    @property
    def evaluation_results(self) -> Sequence[EvaluationResult]:
        """Access individual evaluation results.

        Returns: List of evaluation results.
        """

        return self.__results


class GridDimension(object):
    """Single axis within the simulation grid."""

    __considered_objects: Tuple[Any, ...]
    __dimension: str
    __sample_points: List[Any]
    __title: str | None
    __setter_lambdas: Tuple[Callable, ...]
    __plot_scale: str
    __tick_format: ValueType
    __first_impact: str | None
    __last_impact: str | None

    def __init__(self, considered_objects: Any | Tuple[Any, ...], dimension: str, sample_points: List[Any], title: str | None = None, plot_scale: str | None = None, tick_format: Optional[ValueType] = None) -> None:
        """
        Args:

            considered_objects (Union[Any, Tuple[Any, ...]]):
                The considered objects of this grid section.

            dimension (str):
                Path to the attribute.

            sample_points (List[Any]):
                Sections the grid is sampled at.

            title (str, optional):
                Title of this dimension.
                If not specified, the attribute string is assumed.

            plot_scale (str, optional):
                Scale of the axis within plots.

            tick_format (ValueType, optional):
                Format of the tick labels.
                Linear by default.

        Raises:

            ValueError:
                If the selected `dimension` does not exist within the `considered_object`.
        """

        _considered_objects = considered_objects if isinstance(considered_objects, tuple) else (considered_objects,)
        self.__considered_objects = tuple()

        property_path = dimension.split(".")
        object_path = property_path[:-1]
        property_name = property_path[-1]

        # Infer plot scale of the x-axis
        if plot_scale is None:
            if isinstance(sample_points, LogarithmicSequence):
                self.plot_scale = "log"

            else:
                self.plot_scale = "linear"

        else:
            self.plot_scale = plot_scale

        # Infer tick format of the x-axis
        if tick_format is None:
            if isinstance(sample_points, LogarithmicSequence):
                self.tick_format = ValueType.DB

            else:
                self.tick_format = ValueType.LIN

        else:
            self.tick_format = tick_format

        self.__setter_lambdas = tuple()
        self.__dimension = dimension
        self.__sample_points = sample_points
        self.__title = title
        self.__first_impact = None
        self.__last_impact = None

        for considered_object in _considered_objects:
            # Make sure the dimension exists
            try:
                dimension_object = reduce(lambda obj, attr: getattr(obj, attr), object_path, considered_object)
                dimension_class = type(dimension_object)
                dimension_property: RegisteredDimension = getattr(dimension_class, property_name)

            except AttributeError:
                raise ValueError("Dimension '" + dimension + "' does not exist within the investigated object")

            if len(sample_points) < 1:
                raise ValueError("A simulation grid dimension must have at least one sample point")

            # Update impacts if the dimension is registered as a PyMonte simulation dimension
            if RegisteredDimension.is_registered(dimension_property):
                first_impact = dimension_property.first_impact
                last_impact = dimension_property.last_impact

                if self.__first_impact and first_impact != self.__first_impact:
                    raise ValueError("Diverging impacts on multi-object grid dimension initialization")

                if self.__last_impact and last_impact != self.__last_impact:
                    raise ValueError("Diverging impacts on multi-object grid dimension initialization")

                self.__first_impact = first_impact
                self.__last_impact = last_impact

                # Updated the depicted title if the dimension offers an option and it wasn't exactly specified
                if title is None and dimension_property.title is not None:
                    self.__title = dimension_property.title

            self.__considered_objects += (considered_object,)
            self.__setter_lambdas += (self.__create_setter_lambda(considered_object, dimension),)

    @property
    def considered_objects(self) -> Tuple[Any, ...]:
        """Considered objects of this grid section."""

        return self.__considered_objects

    @property
    def dimension(self) -> str:
        """Dimension property name."""

        return self.__dimension

    @property
    def sample_points(self) -> List[Any]:
        """Points at which this grid dimension is sampled.

        Returns:

            List[Any]:
                List of sample points.
        """

        return self.__sample_points

    @property
    def num_sample_points(self) -> int:
        """Number of dimension sample points.

        Returns:

            int:
                Number of sample points.
        """

        return len(self.__sample_points)

    def configure_point(self, point_index: int) -> None:
        """Configure a specific sample point.

        Args:

            point_index (int):
                Index of the sample point to configure.

        Raises:

            ValueError:
                For invalid indexes.
        """

        if point_index < 0 or point_index >= len(self.__sample_points):
            raise ValueError(f"Index {point_index} is out of the range for grid dimension '{self.title}'")

        for setter_lambda in self.__setter_lambdas:
            setter_lambda(self.__sample_points[point_index])

    @property
    def first_impact(self) -> str | None:
        """Index of the first impacted simulation pipeline stage.

        Returns:

            Pipeline stage index.
            `None`, if the stage is unknown.
        """

        return self.__first_impact

    @property
    def last_impact(self) -> str | None:
        """Index of the last impacted simulation pipeline stage.

        Returns:

            Pipeline stage index.
            `None`, if the stage is unknown.
        """

        return self.__last_impact

    @property
    def title(self) -> str:
        """Title of the dimension.

        Returns:
            The title string.
        """

        return self.__dimension if self.__title is None else self.__title

    @title.setter
    def title(self, value: str) -> None:
        if value is None or len(value) == 0:
            self.__title = None

        else:
            self.__title = value

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

    @property
    def tick_format(self) -> ValueType:
        """Axis tick format of the scalar evaluation plot.

        Returns: The tick format.
        """

        return self.__tick_format

    @tick_format.setter
    def tick_format(self, value: ValueType) -> None:
        self.__tick_format = value

    @staticmethod
    def __create_setter_lambda(considered_object: Any, dimension: str) -> Callable:
        """Generate a setter lambda callback for a selected grid dimension.

        Args:

            considered_object (Any):
                The considered object root.

            dimension (str):
                String representation of dimension location relative to the investigated object.

        Returns:
            Callable: The setter lambda.
        """

        stages = dimension.split(".")
        object_reference = reduce(lambda obj, attr: getattr(obj, attr), stages[:-1], considered_object)

        # Return a lambda to the function if the reference is callable
        function_reference = getattr(object_reference, stages[-1])
        if callable(function_reference):
            return lambda args: function_reference(args)

        # Return a setting lambda if the reference is not callable
        # Implicitly we assume that every non-callable reference is an attribute
        return lambda args: setattr(object_reference, stages[-1], args)


class RegisteredDimension(property):
    """Register a class property getter as a PyMonte simulation dimension.

    Registered properties may specify their simulation stage impacts and therefore significantly
    increase simulation runtime in cases where computationally demanding section re-calculations
    can be reduced.
    """

    __first_impact: str | None
    __last_impact: str | None
    __title: str | None

    def __init__(self, _property: property, first_impact: str | None = None, last_impact: str | None = None, title: str | None = None) -> None:
        """
        Args:

            first_impact (str, optional):
                Name of the first simulation stage within the simulation pipeline
                which is impacted by manipulating this property.
                If not specified, the initial stage is assumed.

            last_impact (str, optional):
                Name of the last simulation stage within the simulation pipeline
                which is impacted by manipulating this property.
                If not specified, the final stage is assumed.

            title (str, optional):
                Displayed title of the dimension.
                If not specified, the dimension's name will be assumed.
        """

        self.__first_impact = first_impact
        self.__last_impact = last_impact
        self.__title = title

        property.__init__(self, getattr(_property, "fget", None), getattr(_property, "fset", None), getattr(_property, "fdel", None), getattr(_property, "doc", None))

    @classmethod
    def is_registered(cls, object: Any) -> bool:
        """Check if any object is a registered PyMonte simulation dimension.

        Args:

            object (Any):
                The object in question.

        Returns:

            A boolean indicator.
        """

        return isinstance(object, cls)

    @property
    def first_impact(self) -> str | None:
        """First impact of the dimension within the simulation pipeline."""

        return self.__first_impact

    @property
    def last_impact(self) -> str | None:
        """Last impact of the dimension within the simulation pipeline."""
        return self.__last_impact

    @property
    def title(self) -> str | None:
        """Displayed title of the dimension."""

        return self.__title

    def getter(self, fget: Callable[[Any], Any]) -> RegisteredDimension:
        updated_property = property.getter(self, fget)
        return RegisteredDimension(updated_property, self.first_impact, self.last_impact, self.title)

    def setter(self, fset: Callable[[Any, Any], None]) -> RegisteredDimension:
        updated_property = property(self.fget, fset, self.fdel, self.__doc__)
        return RegisteredDimension(updated_property, self.first_impact, self.last_impact, self.title)

    def deleter(self, fdel: Callable[[Any], None]) -> RegisteredDimension:
        updated_property = property.deleter(self, fdel)
        return RegisteredDimension(updated_property, self.first_impact, self.last_impact, self.title)


def register(*args, **kwargs) -> Callable[[Any], RegisteredDimension]:
    """Shorthand to register a property as a MonteCarlo dimension.

    Args:

        _property (property): The property to be registered.
    """

    return lambda _property: RegisteredDimension(_property, *args, **kwargs)


class MonteCarlo(Generic[MO]):
    """Grid of parameters over which to iterate the simulation."""

    # Interval between result logs in seconds
    __progress_log_interval: float

    # Maximum number of samples per grid element
    __num_samples: int
    # Minimum number of samples per grid element
    __min_num_samples: int
    # Number of dedicated actors spawned during simulation
    __num_actors: int
    __investigated_object: MO  # The object to be investigated
    # Simulation grid dimensions which make up the grid
    __dimensions: List[GridDimension]
    # Evaluators used to process the investigated object sample state
    __evaluators: List[Evaluator]
    __console: Console  # Console the simulation writes to
    # Printing behaviour of the simulation during runtime
    __console_mode: ConsoleMode
    # Number of samples per section block
    __section_block_size: int | None
    # Cache simulation results in a database during runtime
    __database_caching: bool
    # Number of CPUs reserved for a single actor
    __cpus_per_actor: int
    runtime_env: bool
    # Catch exceptions occuring during simulation runtime
    catch_exceptions: bool

    def __init__(
        self,
        investigated_object: MO,
        num_samples: int,
        evaluators: Sequence[Evaluator] | None = None,
        min_num_samples: int = -1,
        num_actors: int | None = None,
        console: Console | None = None,
        console_mode: ConsoleMode = ConsoleMode.INTERACTIVE,
        section_block_size: Optional[int] = None,
        database_caching: bool = False,
        ray_address: str | None = None,
        cpus_per_actor: int = 1,
        runtime_env: bool = False,
        catch_exceptions: bool = True,
        progress_log_interval: float = 1.0,
    ) -> None:
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

            console_mode (ConsoleMode, optional):
                The printing behaviour of the simulation during runtime.

            section_block_size (int, optional):
                Number of samples per section block.
                By default, the size of the simulation grid is selected.

            database_caching (bool, optional):
                Cache simulation results in a hard drive ZODB database during runtime.
                Disabled by default.

            ray_address (str, optional):
                The address of the ray head node.
                If None is provided, the head node will be launched in this machine.

            cpus_per_actor (int, optional):
                Number of CPU cores reserved per actor.
                One by default.

            runtime_env (bool, optional):
                Create a virtual environment on each host.
                Disabled by default.

            catch_exceptions (bool, optional):
                Catch exceptions occuring during simulation runtime.
                Enabled by default.

            progress_log_interval (float, optional):
                Interval between result logs in seconds.
                1 second by default.
        """

        self.runtime_env = runtime_env

        # Initialize ray if it hasn't been initialized yet. Required to query ideal number of actors
        if not ray.is_initialized():
            runtime_env_info = {"py_modules": self._py_modules(), "pip": self._pip_packages()}

            with catch_warnings():
                simplefilter("ignore")
                ray.init(address=ray_address, runtime_env=runtime_env_info if self.runtime_env else None, logging_level=logging.ERROR)

        self.__dimensions = []
        self.__investigated_object = investigated_object
        self.__evaluators = [] if evaluators is None else list(evaluators)
        self.num_samples = num_samples
        self.min_num_samples = min_num_samples if min_num_samples >= 0 else int(0.5 * num_samples)
        self.__console = Console() if console is None else console
        self.__console_mode = console_mode
        self.section_block_size = section_block_size
        self.__database_caching = database_caching
        self.cpus_per_actor = cpus_per_actor
        self.num_actors = num_actors
        self.catch_exceptions = catch_exceptions
        self.__progress_log_interval = progress_log_interval

    @staticmethod
    def __sample_point_to_str(sample_point: Any) -> str:
        # Check if the sample point has a custom string cast
        if type(sample_point).__str__ is not object.__str__:
            return str(sample_point)

        if isinstance(sample_point, float):
            return f"{sample_point:.2g}"

        if isinstance(sample_point, int):
            return f"{sample_point}"

        return sample_point.__class__.__name__

    def simulate(self, actor: Type[MonteCarloActor]) -> MonteCarloResult:
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
        if self.__console_mode != ConsoleMode.SILENT:
            self.console.log(f"Launched simulation campaign with {self.num_actors} dedicated actors")

        # Sort dimensions after impact in descending order
        def sort(dimension: GridDimension) -> int:
            if dimension.first_impact not in actor.stage_identifiers():
                return 0

            return actor.stage_identifiers().index(dimension.first_impact)

        self.__dimensions.sort(key=sort)

        max_num_samples = self.num_samples
        dimension_str = f"{max_num_samples}"
        for dimension in self.__dimensions:
            max_num_samples *= dimension.num_sample_points
            dimension_str += f" x {dimension.num_sample_points}"

        if self.__console_mode != ConsoleMode.SILENT:
            self.console.log(f"Generating a maximum of {max_num_samples} = " + dimension_str + f" samples inspected by {len(self.__evaluators)} evaluators\n")

        # Render simulation grid table
        if self.__console_mode != ConsoleMode.SILENT and len(self.__dimensions) > 0:
            dimension_table = Table(title="Simulation Grid", title_justify="left")
            dimension_table.add_column("Dimension", style="cyan")
            dimension_table.add_column("Sections", style="green")

            for dimension in self.__dimensions:
                section_str = ""
                for sample_point in dimension.sample_points:
                    section_str += self.__sample_point_to_str(sample_point) + " "

                dimension_table.add_row(dimension.title, section_str)

            self.console.print(dimension_table)
            self.console.print()

        # Prepare the database connection if the flag is enabled
        if self.__database_caching:
            with self.console.status("Initializing Database...", spinner="dots") if self.__console_mode == ConsoleMode.INTERACTIVE else nullcontext():  # type: ignore
                # Create ZODB database to store artifacts during runtime
                database_file = NamedTemporaryFile(delete=False)
                database_storage = FileStorage(database_file.name)
                database = DB(database_storage, historical_cache_size=0)
                database_connection = database.open()

                # Create the simulation grid within the databse
                database_connection
                database_root: PersistentMapping = database_connection.root()

                if "sample_grid" not in database_root:
                    database_root["sample_grid"] = SampleGrid(self.__dimensions, self.__evaluators)

                sample_grid: SampleGrid = database_root["sample_grid"]

        else:
            sample_grid = SampleGrid(self.__dimensions, self.__evaluators)  # pragma: no cover

        # Launch actors and queue the first tasks
        with self.console.status("Launching Actor Pool...", spinner="dots") if self.__console_mode == ConsoleMode.INTERACTIVE else nullcontext():  # type: ignore
            # Generate the actor pool
            actor_pool = ActorPool([actor.options(num_cpus=self.cpus_per_actor).remote((self.__investigated_object, self.__dimensions, self.__evaluators), a, self.catch_exceptions) for a in range(self.num_actors)])  # type: ignore

            # Generate section sample containers and meta-information
            grid_task_count = np.zeros([dimension.num_sample_points for dimension in self.__dimensions], dtype=int)
            grid_active_mask = np.ones([dimension.num_sample_points for dimension in self.__dimensions], dtype=bool)

            # Submit initial actor tasks
            # 2  # A little overhead in task submission might speed things up? Not clear atm.
            task_overhead = 0
            for _ in range(self.num_actors + task_overhead):
                _ = self.__queue_next(actor_pool, sample_grid, grid_active_mask, grid_task_count)

        # Initialize progress bar
        progress = Progress(SpinnerColumn(), *Progress.get_default_columns(), TimeElapsedColumn(), transient=True)
        task1 = progress.add_task("Computing", total=max_num_samples)
        last_progress_plot_time = 0.0
        num_result_rows = min(10, self.section_block_size)

        # Display results in a live table
        status_group = Group("", progress)
        with Live(status_group, console=self.console) if self.__console_mode == ConsoleMode.INTERACTIVE else nullcontext():  # type: ignore
            # Keep executing until all samples are computed
            while actor_pool.has_next():
                # Receive samples from the queue
                samples = self.__receive_next(actor_pool, sample_grid, grid_active_mask, grid_task_count)

                # Queue next task and compute retrieve progress
                self.__queue_next(actor_pool, sample_grid, grid_active_mask, grid_task_count)

                # Update result log if enough time has passed
                progress_plot_time = perf_counter()
                if progress_plot_time - last_progress_plot_time > self.__progress_log_interval:
                    last_progress_plot_time = progress_plot_time

                    # Compute absolute progress
                    absolute_progress = 0
                    for section in sample_grid:
                        absolute_progress += section.num_samples

                    # Update progress bar visualization
                    if self.__console_mode == ConsoleMode.INTERACTIVE:
                        progress.update(task1, completed=absolute_progress)

                        result_rows: List[List[str]] = []
                        for sample in samples[:num_result_rows]:
                            results_row: List[str] = []

                            for dimension, section_idx in zip(self.__dimensions, sample.grid_section):
                                results_row.append(self.__sample_point_to_str(dimension.sample_points[section_idx]))

                            results_row.append(str(sample_grid[sample.grid_section].num_samples))

                            for artifact in sample.artifacts:
                                results_row.append(str(artifact))

                            result_rows.append(results_row)

                        # Render results table
                        results_table = Table(min_width=self.console.measure(progress).minimum)

                        for dimension in self.__dimensions:
                            results_table.add_column(dimension.title, style="cyan")

                        results_table.add_column("#", style="blue")

                        for evaluator in self.__evaluators:
                            results_table.add_column(evaluator.abbreviation, style="green")

                        for result_row in result_rows:
                            results_table.add_row(*result_row)

                        status_group.renderables[0] = results_table

                    elif self.__console_mode == ConsoleMode.LINEAR:
                        self.console.log(f"Progress: {100*absolute_progress/max_num_samples:.3f}")

                # Abort exectuion loop prematurely if all sections are flagged inactive
                # Some results might be lost, but who cares? Speed! Speed! Speed!
                # if absolute_progress >= self.max_num_samples:
                #    break

        # Measure elapsed time
        stop_time = perf_counter()
        performance_time = stop_time - start_time

        # Print finish notifier
        if self.__console_mode != ConsoleMode.SILENT:
            self.console.print()
            self.console.log(f"Simulation finished after {performance_time:.2f} seconds")

        # Compute the result
        result = MonteCarloResult(self.__dimensions, self.__evaluators, sample_grid, performance_time)

        # Close database connection, must occur AFTER result computation
        if self.__database_caching:
            database_connection.close()

        return result

    def __receive_next(self, pool: ActorPool, grid: SampleGrid, grid_active_mask: np.ndarray, grid_task_count: np.ndarray) -> List[MonteCarloSample]:
        # Retrieve result from pool
        runResult: ActorRunResult = pool.get_next_unordered(timeout=None)

        # Display run message if the result
        if runResult.message:
            self.console.log(runResult.message)

        # Save result
        for sample in runResult.samples:
            # Retrieve the respective grid section and add sample
            grid_section: GridSection = grid[sample.grid_section]
            grid_section.add_samples(sample, self.__evaluators)

            # Update task counter
            task_count = max(0, grid_task_count[grid_section.coordinates] - 1)
            grid_task_count[grid_section.coordinates] = task_count

            # Check for stopping criteria
            if grid_active_mask[grid_section.coordinates]:
                # Abort section if the number of samples is expected to be met
                if grid_section.num_samples + task_count >= self.num_samples:
                    grid_active_mask[grid_section.coordinates] = False  # pragma no cover

                # Abort section if the confidence threshold has been reached
                elif grid_section.num_samples >= self.min_num_samples:
                    grid_active_mask[grid_section.coordinates] = not grid_section.confidence_status(self.__evaluators)

        # Commit transaction of added samples
        if self.__database_caching:
            transaction.commit()

        return runResult.samples

    def __queue_next(self, pool: ActorPool, grid: SampleGrid, grid_active_mask: np.ndarray, grid_task_count: np.ndarray):
        # Query active sections and respective task counts
        active_sections = np.argwhere(grid_active_mask)
        active_sections_task_count = grid_task_count[grid_active_mask]

        program: List[Tuple[int, ...]] = []
        for section_coordinates, task_count in zip(active_sections[: self.section_block_size, :], active_sections_task_count.flat[: self.section_block_size]):
            section_coordinates = tuple(section_coordinates)
            program.append(section_coordinates)

            task_count += 1
            grid_task_count[section_coordinates] = task_count

            if task_count + grid[section_coordinates].num_samples >= self.num_samples:
                grid_active_mask[section_coordinates] = False
                break

            # ToDo: Enhance routine to always submit section_block_size amount of indices per program

        if len(program) > 0:
            pool.submit(lambda a, p: a.run.remote(p), program)

    @property
    def investigated_object(self) -> Any:
        """The object to be investigated during the simulation runtime."""

        return self.__investigated_object

    def new_dimension(self, dimension: str, sample_points: List[Any], *args: Any, **kwargs) -> GridDimension:
        """Add a dimension to the simulation grid.

        Must be a property of the investigated object.

        Args:

            dimension (str):
                String representation of dimension location relative to the investigated object.

            sample_points (List[Any]):
                List points at which the dimension will be sampled into a grid.
                The type of points must be identical to the grid arguments / type.

            *args (Tuple[Any], optional):
                References to the object the dimension belongs to.
                Resolved to the investigated object by default,
                but may be an attribute or sub-attribute of the investigated object.

            **kwargs:
                Additional initialization arguments passed to :class:`GridDimension`.

        Returns:
            The newly created dimension object.
        """

        considered_objects = (self.__investigated_object,) if len(args) < 1 else args
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

    @property
    def dimensions(self) -> List[GridDimension]:
        """Simulation grid dimensions which make up the grid."""

        return self.__dimensions.copy()

    def add_evaluator(self, evaluator: Evaluator) -> None:
        """Add new evaluator to the Monte Carlo simulation.

        Args:

            evaluator (Evaluator[MO]):
                The evaluator to be added.
        """

        self.__evaluators.append(evaluator)

    @property
    def evaluators(self) -> List[Evaluator]:
        """Evaluators used to process the investigated object sample state."""

        return self.__evaluators.copy()

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

        self.__num_samples = value

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

        if value < 0.0:
            raise ValueError("Number of samples must be greater or equal to zero")

        self.__min_num_samples = value

    @property
    def max_num_samples(self) -> int:
        """Maximum number of samples over the whole simulation.

        Returns:
            int: Number of samples.
        """

        num_samples = self.num_samples
        for dimension in self.__dimensions:
            num_samples *= dimension.num_sample_points

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
        if self.__num_actors is not None:
            return self.__num_actors

        # Otherwise, the number of actors depends on the number of available CPUs and
        # the cpu requirements per actor
        return max(1, int(ray.available_resources().get("CPU", 1) / self.cpus_per_actor))

    @num_actors.setter
    def num_actors(self, value: Optional[int]) -> None:
        """Set number of dedicated actors spawned during simulation runs."""

        if value is None:
            self.__num_actors = None

        elif value < 1:
            raise ValueError("Number of actors must be greater or equal to one")

        else:
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

        if self.__section_block_size is not None:
            return self.__section_block_size

        size = 1
        for dimension in self.__dimensions:
            size *= dimension.num_sample_points

        return size

    @section_block_size.setter
    def section_block_size(self, value: Optional[int]) -> None:
        if value is not None and value < 1:
            raise ValueError("Section block size must be greater or equal to one")

        self.__section_block_size = value

    @property
    def cpus_per_actor(self) -> int:
        """Number of CPU cores reserved for each actor.

        Returns:

            Number of cores.

        Raises:

            ValueError: If the number of cores is smaller than one
        """

        return self.__cpus_per_actor

    @cpus_per_actor.setter
    def cpus_per_actor(self, num: int) -> None:
        if num < 1:
            raise ValueError("Number if CPU cores per actor must be greater or equal to one")

        self.__cpus_per_actor = num

    @property
    def console_mode(self) -> ConsoleMode:
        """Console mode during simulation runtime.

        Returms: The current console mode.
        """

        return self.__console_mode

    @console_mode.setter
    def console_mode(self, value: Union[ConsoleMode, str]) -> None:
        # Convert string arguments to iterable
        if isinstance(value, str):
            value = ConsoleMode[value]

        self.__console_mode = value

    @staticmethod
    def _py_modules() -> List[str]:
        """List of python modules required by remote workers.

        In order to deploy Ray to computing clusters, dependencies listed here
        will be installed on remote nodes.

        Returns:
            List of module names.
        """

        return [path.join(path.dirname(path.realpath(__file__)), "..")]

    @staticmethod
    def _pip_packages() -> List[str]:
        """List of python packages required by remote workers.

        In order to deploy Ray to computing clusters, dependencies listed here
        will be installed on remote nodes.

        Returns:
            List of package names.
        """

        return ["ray", "numpy", "scipy", "ZODB", "matplotlib", "rich"]
