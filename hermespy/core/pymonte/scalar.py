# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, Type, TypeVar
from typing_extensions import override
from warnings import catch_warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from ..visualize import PlotVisualization, Visualizable, VAT
from .artifact import Artifact
from .evaluation import Evaluator, EvaluationResult
from .grid import GridDimension

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

    __accuracy_tolerance: float
    __confidence: float
    __artifact_sums: np.ndarray
    __artifact_count: np.ndarray
    plot_surface: bool
    __base_dimension_index: int

    def __init__(
        self,
        grid: Sequence[GridDimension],
        evaluator: Evaluator,
        accuracy_tolerance: float = 0.0,
        confidence: float = 1.0,
        min_num_samples: int = 32,
        plot_surface: bool = True,
    ) -> None:
        """
        Args:
            grid: Simulation grid.
            scalar_results: Scalar results generated from collecting samples over the simulation grid.
            tolerance:
            plot_surface:
                Enable surface plotting for two-dimensional grids.
                Enabled by default.
        """

        # Ensure that the confidence bound is between zero and one
        if confidence > 1.0 or confidence < 0.0:
            raise ValueError("Coinfidence requirement must be between zero and one")
        
        # Ensure the confidence bound is non-negative
        if accuracy_tolerance < 0.0:
            raise ValueError("Confidence bound must be non-negative")
        
        if min_num_samples < 2:
            raise ValueError("Minimum number of samples must be at least two")

        # Initialize base class
        EvaluationResult.__init__(self, grid, evaluator)
        
        # Initialize confidence parameters
        grid_dimensions = [d.num_sample_points for d in grid]
        
        # Initialize class attributes
        self.__accuracy_tolerance = accuracy_tolerance
        self.__confidence = confidence
        self.__min_num_samples = min_num_samples
        self.__artifact_sums = np.zeros(grid_dimensions, dtype=float)
        self.__artifact_squared_sums = np.zeros(grid_dimensions, dtype=float)
        self.__artifact_count = np.zeros(grid_dimensions, dtype=int)
        self.plot_surface = plot_surface
        self.__base_dimension_index = 0

    @property
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
                std = ((squared_sum - (sum**2 / count)) / (count - 1))**.5
                if std > 0.0:
                    confidence = 2 * (1 - self._scalar_cdf(count**.5 * self.__accuracy_tolerance / std))
                    confident = confidence <= self.__confidence

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

    def create_figure(self, **kwargs) -> tuple[plt.FigureBase, VAT]:
        if len(self.grid) == 2 and self.plot_surface:
            return plt.subplots(1, 1, squeeze=False, subplot_kw={"projection": "3d"})

        return Visualizable.create_figure(self, **kwargs)

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

    def _update_visualization(self, visualization: PlotVisualization, **kwargs) -> None:
        # If the grid contains no data, dont plot anything
        if len(self.grid) < 1:
            return self._plot_empty(visualization)

        # Shuffle grid and respective scalar results so that the selected base dimension is always the first entry
        grid = list(self.grid).copy()
        grid.insert(0, grid.pop(self.__base_dimension_index))
        scalars = np.moveaxis(self.to_array(), self.__base_dimension_index, 0)

        if len(grid) == 2 and self.plot_surface:
            return self._update_surface_visualization(scalars, visualization)

        # Multiple dimensions, resort to legend-based multiplots
        else:
            self._update_multidim_visualization(scalars, visualization)

    def to_array(self) -> np.ndarray:
        with catch_warnings(action='ignore'):
            return self.__artifact_sums / self.__artifact_count
        
    @classmethod
    def From_Artifacts(
        cls: Type[SERT],
        grid: Sequence[GridDimension],
        artifacts: np.ndarray,
        evaluator: Evaluator,
        plot_surface: bool = True,
    ) -> SERT:
        """Generate a scalar evaluation result from a set of artifacts.

        Args:
            grid: The simulation grid.
            artifacts: Numpy object array whose dimensions represent grid dimensions.
            evaluator: The evaluator generating the artifacts.
            plot_surface: Whether to plot the result as a surface plot.

        Returns: The scalar evaluation result.
        """

        scalar_results = np.empty(artifacts.shape, dtype=float)
        for section_coords in np.ndindex(artifacts.shape):
            scalar_results[section_coords] = np.mean(
                [artifact.to_scalar() for artifact in artifacts[section_coords]]
            )

        return cls(grid, scalar_results, evaluator, plot_surface)
    

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