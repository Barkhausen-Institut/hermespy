# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.core.pymonte.grid import GridDimension
from hermespy.core.pymonte.scalar import ScalarEvaluationResult
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


class TestScalarEvaluationResult(TestCase):
    """Test processed scalar evaluation result class"""

    def test_surface_plotting(self) -> None:
        """Surface plotting should call the correct plotting routine"""

        grid = [GridDimension(TestObjectMock(), "property_b", np.arange(10)) for _ in range(2)]
        scalar_data = np.random.uniform(size=(10, 10))
        evaluator = Mock()

        result = ScalarEvaluationResult(grid, scalar_data, evaluator)

        with SimulationTestContext():
            visualization = result.visualize()
            visualization.axes[0, 0].plot_surface.assert_called()

    def test_multidim_plotting(self) -> None:
        """Multidimensional plotting should call the correct plotting routine"""

        grid = [GridDimension(TestObjectMock(), "property_b", np.arange(10)) for _ in range(3)]
        scalar_data = np.random.uniform(size=(10, 10))
        evaluator = Mock()

        result = ScalarEvaluationResult(grid, scalar_data, evaluator)

        with SimulationTestContext():
            visualization = result.visualize()
            visualization.axes[0, 0].plot.assert_called()

    def test_plot_no_data(self) -> None:
        """Even without grid dimensions an empty figure should be generated"""

        result = ScalarEvaluationResult([], np.empty(0, dtype=object), EvaluatorMock())

        with SimulationTestContext():
            visualization = result.visualize()
            visualization.axes[0, 0].text.assert_called()

    def test_to_array(self) -> None:
        """Array conversion should return the correct array"""

        grid = [GridDimension(TestObjectMock(), "property_b", np.arange(10)) for _ in range(1)]
        scalar_data = np.random.uniform(size=(10, 10))
        evaluator = Mock()

        result = ScalarEvaluationResult(grid, scalar_data, evaluator)

        assert_array_equal(scalar_data, result.to_array())
