# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_array_equal

from hermespy.core import PlotVisualization, ValueType
from hermespy.core.pymonte import GridDimensionInfo, ScalarEvaluationResult, ArtifactTemplate, SamplePoint
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

        grid = [GridDimensionInfo([SamplePoint(a) for a in range(10)], "property_b", "linear", ValueType.LIN) for _ in range(2)]
  
        result = ScalarEvaluationResult(grid, Mock())
        for i in range(10):
            result.add_artifact((0), ArtifactTemplate(i), False)
            result.add_artifact((1), ArtifactTemplate(i) ,False)

        with SimulationTestContext():
            visualization = result.visualize()
            visualization.axes[0, 0].plot_surface.assert_called()

    def test_multidim_plotting(self) -> None:
        """Multidimensional plotting should call the correct plotting routine"""

        grid = [GridDimensionInfo([SamplePoint(a) for a in range(10)], "property_b", "linear", ValueType.LIN) for _ in range(3)]
  
        result = ScalarEvaluationResult(grid, Mock())
        for i in range(10):
            for j in range(3):
                result.add_artifact((j), ArtifactTemplate(i), False)

        with SimulationTestContext():
            visualization = result.visualize()
            visualization.axes[0, 0].plot.assert_called()
