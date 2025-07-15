# -*- coding: utf-8 -*-

from typing_extensions import override
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
from hermespy.core.pymonte.scalar import ScalarEvaluationResult, ScalarEvaluator
from ...utils import SimulationTestContext
from .object import TestObjectMock

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


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


class SumEvaluator(ScalarEvaluator):
    """An evaluator summing up object properties"""

    __investigated_object: TestObjectMock

    def __init__(
        self,
        investigated_object: TestObjectMock,
    ) -> None:
        self.__investigated_object = investigated_object
        ScalarEvaluator.__init__(self)

    @override
    def evaluate(self) -> EvaluationMock:
        summed = self.__investigated_object.property_a + self.__investigated_object.property_b + self.__investigated_object.property_c
        return EvaluationMock(summed)

    @property
    @override
    def abbreviation(self) -> str:
        return "SUM"

    @property
    @override
    def title(self) -> str:
        return "Sum Evaluator"
    
    @override
    def initialize_result(self, grid: list[GridDimension]) -> ScalarEvaluationResult:
        return ScalarEvaluationResult(grid, self, self.tolerance, self.confidence)

    def generate_result(self, grid: list[GridDimension], artifacts: np.ndarray) -> EvaluationResult:
        return ScalarEvaluationResult.From_Artifacts(grid, artifacts, self)


class ProductEvaluator(ScalarEvaluator):
    """An evaluator multiplying object properties"""

    __investigated_object: TestObjectMock

    def __init__(self, investigated_object: TestObjectMock) -> None:
        self.__investigated_object = investigated_object
        ScalarEvaluator.__init__(self)

    @override
    def evaluate(self) -> EvaluationMock:
        product = self.__investigated_object.property_a * self.__investigated_object.property_b * self.__investigated_object.property_c
        return EvaluationMock(product)

    @property
    @override
    def abbreviation(self) -> str:
        return "Product"

    @property
    @override
    def title(self) -> str:
        return "Product Evaluator"
    
    @override
    def initialize_result(self, grid: list[GridDimension]) -> ScalarEvaluationResult:
        return ScalarEvaluationResult(grid, self, self.tolerance, self.confidence)


    @override
    def generate_result(self, grid: list[GridDimension], artifacts: np.ndarray) -> EvaluationResult:
        return ScalarEvaluationResult.From_Artifacts(grid, artifacts, self)


class TestEvaluationResult(TestCase):
    """Test evaluation result base class"""

    def setUp(self) -> None:
        self.grid = [GridDimension(TestObjectMock(), "property_b", np.arange(10), tick_format=ValueType.DB)]
        self.object = TestObjectMock()
        self.evaluator = ProductEvaluator(self.object)
        self.result = self.evaluator.initialize_result(self.grid)
        
    def test_properties(self) -> None:
        """Test the properties of the evaluation result"""

        self.assertIs(self.grid, self.result.grid)
        self.assertIs(self.evaluator, self.result.evaluator)

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


class TestEvaluator(TestCase):
    """Test base class for all evaluators"""

    def setUp(self) -> None:
        self.evaluator = ProductEvaluator(TestObjectMock())

    def test_init(self) -> None:
        """Initialization should set the proper default attributes"""

        self.assertEqual("linear", self.evaluator.plot_scale)
        self.assertEqual(ValueType.LIN, self.evaluator.tick_format)
