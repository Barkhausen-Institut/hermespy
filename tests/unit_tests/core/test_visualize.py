# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Generic
from unittest import TestCase
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import StemContainer
from matplotlib.collections import PathCollection

from hermespy.core import ImageVisualization, Visualizable, VisualizableAttribute, Visualization, VAT, VT, PlotVisualization, StemVisualization, ScatterVisualization

from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class VisualizableMock(Visualizable[Visualization]):
    """Mock class for the visualizable base class for testing purposes"""

    def _prepare_visualization(self, figure: plt.Figure | None, axes: VAT, **kwargs) -> Visualization:
        return MagicMock(spec=Visualization)

    def _update_visualization(self, visualization: Visualization, **kwargs) -> None:
        pass


class VisualizableAttributeMock(VisualizableAttribute[Visualization]):
    """Mock class for testing visualizable attributes"""

    def _prepare_visualization(self, figure: plt.Figure | None, axes: VAT, **kwargs) -> Visualization:
        return MagicMock(spec=Visualization)

    def _update_visualization(self, visualization: Visualization, **kwargs) -> None:
        pass


class _TestVisualization(Generic[VT], TestCase):

    _figure: plt.Figure
    _ax: plt.Axes
    _axes: VAT
    _visualization: VT

    def setUp(self) -> None:

        self._figure = MagicMock(spec=plt.Figure)
        self._ax = MagicMock(spec=plt.Axes)
        self._axes = np.array([[self._ax]], dtype=np.object_)

    def test_properties(self) -> None:
        """Test the properties of the visualization base class"""

        self.assertIs(self._visualization.figure, self._figure)
        self.assertIs(self._visualization.axes, self._axes)


class TestPlotVisualization(_TestVisualization[PlotVisualization]):
    """Test the plot visualization class"""

    def setUp(self) -> None:
        super().setUp()

        self._lines = [MagicMock(spec=plt.Line2D)]
        self._lines_array = np.array([self._lines], dtype=np.object_)
        self._lines_array[0, 0] = self._lines[0]
        self._visualization = PlotVisualization(self._figure, self._axes, self._lines_array)

    def test_init_validation(self) -> None:
        """Initialization should raise ValueError for invalid arguments"""

        with self.assertRaises(ValueError):
            _ = PlotVisualization(self._figure, np.empty((0, 0), dtype=np.object_), self._lines_array)

    def test_properties(self) -> None:
        """Test the properties of the plot visualization class"""

        super().test_properties()
        self.assertIs(self._visualization.lines, self._lines_array)


class TestStemVisualization(_TestVisualization[StemVisualization]):
    """Test the stem visualization class"""

    def setUp(self) -> None:
        super().setUp()

        self._container = MagicMock(spec=StemContainer)
        self._visualization = StemVisualization(self._figure, self._axes, self._container)

    def test_properties(self) -> None:
        """Test the properties of the stem visualization class"""

        super().test_properties()
        self.assertIs(self._visualization.container, self._container)


class TestScatterVisualization(_TestVisualization[ScatterVisualization]):
    """Test the scatter visualization class"""

    def setUp(self) -> None:
        super().setUp()

        self._paths = MagicMock(spec=PathCollection)
        self._visualization = ScatterVisualization(self._figure, self._axes, self._paths)

    def test_properties(self) -> None:
        """Test the properties of the scatter visualization class"""

        super().test_properties()
        self.assertIs(self._visualization.paths, self._paths)


class TestImageVisualization(_TestVisualization[ImageVisualization]):
    """Test the image visualization class"""

    def setUp(self) -> None:
        super().setUp()

        self._image = MagicMock()
        self._visualization = ImageVisualization(self._figure, self._axes, self._image)

    def test_properties(self) -> None:
        """Test the properties of the image visualization class"""

        super().test_properties()
        self.assertIs(self._visualization.image, self._image)


class TestVisualizable(TestCase):
    """Test the base classe for visualizable objects"""

    def setUp(self) -> None:
        self.visualizable = VisualizableMock()
        
    def test_visualization_caching(self) -> None:
        
        self.assertIsNone(self.visualizable.visualization)
        
        with SimulationTestContext(patch_plot=True):
            visualization = self.visualizable.visualize()
            self.assertIs(self.visualizable.visualization, visualization)

    def test_visualize_new_figure(self) -> None:
        """Test visualize routine creating a new figure"""
        
        with SimulationTestContext(patch_plot=True):
            
            visualization = self.visualizable.visualize()
            self.assertIsInstance(visualization, Visualization)
            
    def test_visualize_existing_figure(self) -> None:
        """Test visualize routine using an existing figure"""
        
        with SimulationTestContext(patch_plot=True):
            
            figure, axes = self.visualizable.create_figure()
            visualization = self.visualizable.visualize(axes)
            self.assertIsInstance(visualization, Visualization)
            
    def test_update_visualization_validation(self) -> None:
        """Test the update visualization routine validation"""
        
        with self.assertRaises(RuntimeError):
            self.visualizable.update_visualization()
            
    def test_update_visualization(self) -> None:
        """Test the update visualization routine"""
        
        with SimulationTestContext(patch_plot=True):
            
            visualization = self.visualizable.visualize()
            self.visualizable.update_visualization(visualization)
            self.visualizable.update_visualization()
            
            
class TestVisualizableAttribute(TestCase):
    """Test the base classe for visualizable attributes"""
    
    def setUp(self) -> None:
        
        self.base_class = MagicMock()
        self.visualizable = VisualizableAttributeMock()
        self.base_class.plot = self.visualizable
        
    def test_call(self) -> None:
        """Test the call routine"""
        
        with SimulationTestContext(patch_plot=True):
            visualization = self.base_class.plot()
        
        self.assertIsInstance(visualization, Visualization)


del _TestVisualization
