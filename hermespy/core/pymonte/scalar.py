# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, TypeVar
from typing_extensions import override
from warnings import catch_warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # type: ignoreimport numpy as np
import numpy as np
from scipy.stats import norm


from ..logarithmic import ValueType
from ..visualize import PlotVisualization, Visualizable, VAT
from .artifact import Artifact
from .evaluation import Evaluator, EvaluationResult
from .grid import GridDimensionInfo

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


SERT = TypeVar("SERT", bound="ScalarEvaluationResult")
"""Type of scalar Monte Carlo evaluation result."""


class ScalarEvaluationResult(EvaluationResult):
    """Base class for scalar evaluation results."""

    __accuracy: float
    __confidence: float
    __artifact_sums: np.ndarray
    __artifact_count: np.ndarray
    plot_surface: bool

    def __init__(
        self,
        grid: Sequence[GridDimensionInfo],
        evaluator: Evaluator,
        accuracy: float = 0.0,
        confidence: float = 1.0,
        min_num_samples: int = 1024,
        plot_surface: bool = True,
        base_dimension_index: int = 0,
    ) -> None:
        """
        Args:
            grid: Simulation grid.
            evaluator: Evaluator associated with this result.
            accuracy: Acceptable bound around the mean value of the estimated scalar performance indicator.
            tolerance: Required confidence level for the given `accuracy`.
            min_num_samples: Minimum number of samples required to compute the confidence bound.
            plot_surface:
                Enable surface plotting for two-dimensional grids.
                Enabled by default.
            base_dimension_index: Index of the base dimension used for plotting.
        """

        # Ensure that the confidence bound is between zero and one
        if confidence > 1.0 or confidence < 0.0:
            raise ValueError("Coinfidence requirement must be between zero and one")
        
        # Ensure the confidence bound is non-negative
        if accuracy < 0.0:
            raise ValueError("Confidence bound must be non-negative")
        
        if min_num_samples < 2:
            raise ValueError("Minimum number of samples must be at least two")

        # Initialize base class
        EvaluationResult.__init__(self, grid, evaluator, base_dimension_index)
        
        # Initialize confidence parameters
        grid_dimensions = [d.num_sample_points for d in grid]
        
        # Initialize class attributes
        self.__accuracy = accuracy
        self.__confidence = confidence
        self.__min_num_samples = min_num_samples
        self.__artifact_sums = np.zeros(grid_dimensions, dtype=float)
        self.__artifact_squared_sums = np.zeros(grid_dimensions, dtype=float)
        self.__artifact_count = np.zeros(grid_dimensions, dtype=int)
        self.plot_surface = plot_surface

    @property
    @override
    def title(self) -> str:
        # The plotting title should resolve to the represented evaluator's title
        return self.evaluator.title

    @override
    def add_artifact(self, coordinates: tuple[int, ...], artifact: Artifact, compute_confidence: bool = True) -> bool:
        
        confident = False
        scalar = artifact.to_scalar()
        
        if scalar is not None:
            sum = self.__artifact_sums[coordinates] + scalar
            squared_sum = self.__artifact_squared_sums[coordinates] + scalar**2
            count = self.__artifact_count[coordinates] + 1
            
            self.__artifact_sums[coordinates] = sum
            self.__artifact_squared_sums[coordinates] = squared_sum
            self.__artifact_count[coordinates] = count

            if compute_confidence and (count % self.__min_num_samples) == 0:
                # The confidence is an implementation of Algorithm 1
                # from ON NON-ASYMPTOTIC OPTIMAL STOPPING CRITERIA IN MONTE CARLO SIMULATIONS by Bayer et al.
                std = ((squared_sum - (sum**2 / count)) / (count - 1))**.5
                if std > 0.0:
                    confidence = 2 * (1 - self._scalar_cdf(count**.5 * self.__accuracy / std))
                    confident = confidence < self.__confidence

        return confident

    @staticmethod
    def _scalar_cdf(scalar: float) -> float:
        """Assumed cumulative probability of the scalar representation.

        Args:
            scalar: The scalar value.

        Returns: Cumulative probability between zero and one.
        """

        return norm.cdf(scalar)

    @override
    def runtime_estimates(self) -> None | np.ndarray:
        return self.to_array()

    @override
    def create_figure(self, **kwargs) -> tuple[plt.FigureBase, VAT]:
        if len(self.grid) == 2 and self.plot_surface:
            return plt.subplots(1, 1, squeeze=False, subplot_kw={"projection": "3d"})

        return Visualizable.create_figure(self, **kwargs)

    @override
    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> PlotVisualization:
        if len(self.grid) == 2 and self.plot_surface:
            lines = self._prepare_surface_visualization(axes[0, 0])

        else:
            lines = self._prepare_multidim_visualization(axes[0, 0])

        line_array = np.empty_like(axes, dtype=np.object_)
        line_array[0, 0] = lines
        return PlotVisualization(figure, axes, line_array)

    @override
    def _update_visualization(self, visualization: PlotVisualization, **kwargs) -> None:
        # If the grid contains no data, dont plot anything
        if len(self.grid) < 1:
            return self._plot_empty(visualization)

        # Shuffle grid and respective scalar results so that the selected base dimension is always the first entry
        grid = list(self.grid).copy()
        grid.insert(0, grid.pop(self.base_dimension_index))
        scalars = np.moveaxis(self.to_array(), self.base_dimension_index, 0)

        if len(grid) == 2 and self.plot_surface:
            return self._update_surface_visualization(scalars, visualization)

        # Multiple dimensions, resort to legend-based multiplots
        else:
            self._update_multidim_visualization(scalars, visualization)

    @override
    def to_array(self) -> np.ndarray:
        with catch_warnings(record=False):
            return self.__artifact_sums / self.__artifact_count

    def _prepare_surface_visualization(self, ax: Axes3D) -> list[Line2D]:
        # Configure axes labels and scales
        ax.set(
            xlabel=self.grid[0].title, ylabel=self.grid[1].title, zlabel=self.evaluator.abbreviation
        )

        self._configure_axis(ax.xaxis, self.grid[0].tick_format)
        self._configure_axis(ax.yaxis, self.grid[1].tick_format)
        if self.evaluator is not None:
            self._configure_axis(ax.zaxis, self.evaluator.tick_format)

        # x_points = np.asarray([s.value for s in self.grid[0].sample_points])
        # y_points = np.asarray([s.value for s in self.grid[1].sample_points])
        # y, x = np.meshgrid(y_points, x_points)

        # ax.plot_surface(x.astype(float), y.astype(float), np.zeros_like(y, dtype=np.float64))
        return []  # 3D plotting returns a poly3d collection that is not supported

    def _update_surface_visualization(
        self, data: np.ndarray, visualization: PlotVisualization
    ) -> None:
        """Plot two-dimensional simulation results into a three-dimensional axes system."""

        x_points = np.asarray([s.value for s in self.grid[0].sample_points])
        y_points = np.asarray([s.value for s in self.grid[1].sample_points])
        y, x = np.meshgrid(y_points, x_points)

        visualization.axes[0, 0].plot_surface(x.astype(float), y.astype(float), data)

    
class ScalarEvaluator(Evaluator):
    """Evaluation routine for investigated object states, extracting scalar performance indicators of interest.

    Evaluators represent the process of extracting arbitrary performance indicator samples :math:`X_m` in the form of
    :class:`.Artifact` instances from investigated object states.
    Once a :class:`.MonteCarloActor` has set its investigated object to a new random state,
    it calls the :meth:`evaluate<hermespy.core.monte_carlo.Evaluator.evaluate>` routines of all configured evaluators,
    collecting the resulting respective :class:`.Artifact` instances.

    For a given set of :class:`.Artifact` instances,
    evaluators are expected to report a :meth:`.confidence` which may result in a premature abortion of the
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

    __confidence: float
    __tolerance: float
    __min_num_samples: int
    __plot_surface: bool

    def __init__(
        self,
        confidence: float = 1.0,
        tolerance: float = 0.0,
        min_num_samples: int = 1024,
        plot_scale: str = "linear",
        tick_format: ValueType = ValueType.LIN,
        plot_surface: bool = True,
    ) -> None:
        """
        Args:
            confidence: Required confidence level for the given `tolerance` between zero and one.
            tolerance: Acceptable non-negative bound around the mean value of the estimated scalar performance indicator.
            min_num_samples: Minimum number of samples required to compute the confidence bound.
            plot_scale: Scale of the plot. Can be ``'linear'`` or ``'log'``.
            tick_format: Tick format of the plot.
            plot_surface: Enable surface plotting for two-dimensional grids. Enabled by default.
        """

        # Initialize base class
        Evaluator.__init__(self, plot_scale, tick_format)

        # Initialize attributes
        self.confidence = confidence
        self.tolerance = tolerance
        self.min_num_samples = min_num_samples
        self.plot_surface = plot_surface

    @override
    def initialize_result(self, grid: Sequence[GridDimensionInfo]) -> ScalarEvaluationResult:
        return ScalarEvaluationResult(grid, self, self.tolerance, self.confidence, self.min_num_samples, self.plot_surface)

    @override
    def generate_result(self, grid: Sequence[GridDimensionInfo], artifacts: np.ndarray) -> ScalarEvaluationResult:
        result = self.initialize_result(grid)
        for coordinates, artifact_list in np.ndenumerate(artifacts):
            for artifact in artifact_list:
                result.add_artifact(coordinates, artifact, False)
        return result

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

        Returns: Non-negative tolerance :math:`\\mathrm{TOL}`.

        Raises:
            ValueError: If tolerance is negative.
        """

        return self.__tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Tolerance must be greater or equal to zero")

        self.__tolerance = value

    @property
    def min_num_samples(self) -> int:
        """Minimum number of samples required to compute the confidence bound.

        The minimum number of samples :math:`M_{\\mathrm{min}} \\in \\mathbb{R}_{+}` is the minimum number of
        samples required to compute the confidence bound for a single :class:`.GridSection`.

        Returns: Minimum number of samples :math:`M_{\\mathrm{min}}`.

        Raises:
            ValueError: If minimum number of samples is less than two.
        """
        return self.__min_num_samples

    @min_num_samples.setter
    def min_num_samples(self, value: int) -> None:

        if value < 2:
            raise ValueError("Minimum number of samples must be at least two")

        self.__min_num_samples = value

    @property
    def plot_surface(self) -> bool:
        """Enable surface plotting for two-dimensional grids."""
        return self.__plot_surface

    @plot_surface.setter
    def plot_surface(self, value: bool) -> None:
        self.__plot_surface = value


class ScalarDimension(ABC):
    """Base class for objects that can be configured by scalar values.

    When a property of type :class:`ScalarDimension` is defined as a simulation parameter :class:`GridDimension`,
    the simulation will automatically configure the object with the scalar value of the sample point
    during simulation runtime.

    The configuration operation is represented by the lshift operator `<<`.
    """

    @abstractmethod
    def __lshift__(self, scalar: float) -> None:
        """Configure the object with a scalar value.

        Args:
            scalar: Scalar value to configure the object with.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def title(self) -> str:
        """Title of the scalar dimension.

        Displayed in plots and tables during simulation runtime.
        """
        ...  # pragma: no cover
