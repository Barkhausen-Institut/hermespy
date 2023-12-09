# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import Visualizable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestVisualizable(TestCase):
    def setUp(self) -> None:
        self.visualizable = Visualizable()

    def test_plot_new_figure(self) -> None:
        """Test plotting to a new figure"""

        with patch("matplotlib.pyplot.figure") as figure_patch:
            _ = self.visualizable.plot()
            figure_patch.assert_called()

    def test_plot_validation(self) -> None:
        """Plotting with a provided axes should raise an error if the axes are empty"""

        axes = np.empty((0, 0), dtype=np.object_)
        with self.assertRaises(ValueError):
            _ = self.visualizable.plot(axes)

        axes_mock = Mock(spec=plt.Axes)
        axes_mock.get_figure.return_value = None
        axes = np.array([[axes_mock]], dtype=np.object_)
        with self.assertRaises(RuntimeError):
            _ = self.visualizable.plot(axes)

    def test_plot_existing_figure(self) -> None:
        """Test plotting into an existing figure"""

        axes_mock = Mock(spec=plt.Axes)
        axes_collection = np.array([[axes_mock]], dtype=np.object_)
        figure = self.visualizable.plot(axes_collection)
