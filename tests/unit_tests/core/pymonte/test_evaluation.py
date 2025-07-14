# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from rich.console import Console

from hermespy.core import ValueType, PlotVisualization, VAT, Visualization
from hermespy.core.pymonte.artifact import ArtifactTemplate
from hermespy.core.pymonte.evaluation import Evaluator, EvaluationResult, EvaluationTemplate
from hermespy.core.pymonte.grid import GridDimension
from hermespy.core.pymonte.scalar import ScalarEvaluationResult
from ...utils import SimulationTestContext
from .object import TestObjectMock
from .test_evaluation import EvaluatorMock, EvaluationMock

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SumEvaluator(Evaluator):
    """An evaluator summing up object properties"""

    __investigated_object: TestObjectMock

    def __init__(self, investigated_object: TestObjectMock) -> None:
        self.__investigated_object = investigated_object
        Evaluator.__init__(self)

    def evaluate(self) -> EvaluationMock:
        summed = self.__investigated_object.property_a + self.__investigated_object.property_b + self.__investigated_object.property_c
        return EvaluationMock(summed)

    @property
    def abbreviation(self) -> str:
        return "SUM"

    @property
    def title(self) -> str:
        return "Sum Evaluator"

    def generate_result(self, grid: list[GridDimension], artifacts: np.ndarray) -> EvaluationResult:
        return ScalarEvaluationResult.From_Artifacts(grid, artifacts, self)


class ProductEvaluator(Evaluator):
    """An evaluator multiplying object properties"""

    __investigated_object: TestObjectMock

    def __init__(self, investigated_object: TestObjectMock) -> None:
        self.__investigated_object = investigated_object
        Evaluator.__init__(self)

    def evaluate(self) -> EvaluationMock:
        product = self.__investigated_object.property_a * self.__investigated_object.property_b * self.__investigated_object.property_c
        return EvaluationMock(product)

    @property
    def abbreviation(self) -> str:
        return "Product"

    @property
    def title(self) -> str:
        return "Product Evaluator"

    def generate_result(self, grid: list[GridDimension], artifacts: np.ndarray) -> EvaluationResult:
        return ScalarEvaluationResult.From_Artifacts(grid, artifacts, self)


class TestEvaluator(TestCase):
    """Test base class for all evaluators"""

    def setUp(self) -> None:
        self.rng = default_rng(42)
        self.evaluator = EvaluatorMock()
        self.evaluator.tolerance = 0.2

    def test_init(self) -> None:
        """Initialization should set the proper default attributes"""

        self.assertEqual(1.0, self.evaluator.confidence)
        self.assertEqual(0.2, self.evaluator.tolerance)

    def test_confidence_setget(self) -> None:
        """Confidence property getter should return setter argument"""

        confidence = 0.5
        self.evaluator.confidence = confidence

        self.assertEqual(confidence, self.evaluator.confidence)

    def test_confidence_validation(self) -> None:
        """Confidence property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.evaluator.confidence = -1.0

        with self.assertRaises(ValueError):
            self.evaluator.confidence = 1.5

        try:
            self.evaluator.confidence = 0.0
            self.evaluator.confidence = 1.0

        except ValueError:
            self.fail()

    def test_tolerance_setget(self) -> None:
        """Tolerance property getter should return setter argument"""

        tolerance = 0.5
        self.evaluator.tolerance = tolerance

        self.assertEqual(tolerance, self.evaluator.tolerance)

    def test_tolerance_validation(self) -> None:
        """Confidence margin property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.evaluator.tolerance = -1.0

        try:
            self.evaluator.tolerance = 0.0
            self.evaluator.tolerance = 1.0

        except ValueError:
            self.fail()

    def test_str(self) -> None:
        """Evaluator string representation should return a string"""

        self.assertEqual(self.evaluator.abbreviation, self.evaluator.__str__())

    def test_scalar_cdf(self) -> None:
        """Scalar cumulitive distribution function should return the cumulative probability"""

        cdf_low = self.evaluator._scalar_cdf(0.0)
        cdf_high = self.evaluator._scalar_cdf(1.0)

        self.assertTrue(cdf_low < cdf_high)

    def test_plot_scale_setget(self) -> None:
        """Plot scale property getter should return setter argument"""

        plot_scale = "abce"
        self.evaluator.plot_scale = plot_scale

        self.assertEqual(plot_scale, self.evaluator.plot_scale)


class EvaluationMock(EvaluationTemplate[float, Visualization]):
    def artifact(self) -> ArtifactTemplate[float]:
        return ArtifactTemplate[float](self.evaluation)

    def _prepare_visualization(self, figure: plt.Figure | None, axes: VAT, **kwargs) -> Visualization:
        return MagicMock(spec=Visualization)

    def _update_visualization(self, visualization: Visualization, **kwargs) -> None:
        pass


class EvaluationResultMock(EvaluationResult):
    """Mock of an evaluation result"""

    def to_array(self) -> np.ndarray:
        return np.random.standard_normal([d.num_sample_points for d in self.grid])

    def _prepare_visualization(self, figure: plt.Figure | None, axes: VAT, **kwargs) -> PlotVisualization:
        return MagicMock(spec=PlotVisualization)

    def _update_visualization(self, visualization: Visualization, **kwargs) -> None:
        pass


class TestEvaluationResult(TestCase):
    """Test evaluation result base class"""

    def setUp(self) -> None:
        self.grid = [GridDimension(TestObjectMock(), "property_b", np.arange(10), tick_format=ValueType.DB)]
        self.evaluator = EvaluatorMock()
        self.result = EvaluationResultMock(self.grid, self.evaluator)

    def test_print(self) -> None:
        """Printing should call the correct printing routine"""

        console = Mock(spec=Console)

        with patch.object(self.result, 'to_array') as to_array_patch:

            # Check default string conversion
            to_array_patch.return_value = np.array([["x"] for _ in range(10)])
            self.result.print(console)
            console.print.assert_called()
            console.print.reset_mock()

            # Check decibel conversion
            self.evaluator.tick_format = ValueType.DB
            to_array_patch.return_value = np.array([[i] for i in range(10)])
            self.result.print(console)
            console.print.assert_called()
            console.print.reset_mock()

            # Check linear conversion
            self.evaluator.tick_format = ValueType.LIN
            self.evaluator.plot_scale = "lin"
            to_array_patch.return_value = np.array([[i] for i in range(10)])
            self.result.print(console)
            console.print.assert_called()
            console.print.reset_mock()

            # Check multiple results
            to_array_patch.return_value = np.array([[i, 2*i, 3*i] for i in range(10)])
            self.result.print(console)
            console.print.assert_called()
            console.print.reset_mock()

            # Check unknown format
            to_array_patch.return_value = np.array([object() for i in range(10)])
            self.result.print(console)
            console.print.assert_called()
            console.print.reset_mock()

    def test_surface_plotting(self) -> None:
        """Surface plotting should call the correct plotting routine"""

        grid = [GridDimension(TestObjectMock(), "property_b", np.arange(10)) for _ in range(2)]
        self.result = EvaluationResultMock(grid, self.evaluator)

        scalar_data = np.random.uniform(size=(10, 10))

        with SimulationTestContext():
            figure, axes = plt.subplots(1, 1, squeeze=False)

            # Prepare plot
            lines = self.result._prepare_surface_visualization(axes[0, 0])

            lines_array = np.empty((1, 1), dtype=np.object_)
            lines_array[0, 0] = lines
            visualization = PlotVisualization(figure, axes, lines_array)

            # Update plot
            self.result._update_surface_visualization(scalar_data, visualization)
            axes[0, 0].plot_surface.assert_called()

    def test_multidim_plotting(self) -> None:
        """Multidimensional plotting should call the correct plotting routine"""

        scalar_data = np.random.uniform(size=(10))

        with SimulationTestContext():
            figure, axes = plt.subplots(1, 1, squeeze=False)

            # Prepare plot
            lines = self.result._prepare_multidim_visualization(axes[0, 0])
            axes[0, 0].plot.assert_called()

            lines_array = np.empty((1, 1), dtype=np.object_)
            lines_array[0, 0] = lines
            visualization = PlotVisualization(figure, axes, lines_array)

            # Update plot
            self.result._update_multidim_visualization(scalar_data, visualization)
            lines[0].set_ydata.assert_called()

    def test_empty_plotting(self) -> None:
        """Empty plotting should call the correct plotting routine"""

        visualization = MagicMock(spec=PlotVisualization)
        axes = Mock()
        axes_collection = np.array([[axes]], dtype=np.object_)
        visualization.axes = axes_collection

        self.result._plot_empty(visualization)
        axes.text.assert_called_once()
