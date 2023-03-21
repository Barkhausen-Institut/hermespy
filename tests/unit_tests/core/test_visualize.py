# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch

from hermespy.core import Visualizable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestVisualizable(TestCase):

    def setUp(self) -> None:
        
        self.visualizable = Visualizable()

    def test_plot_new_figure(self) -> None:
        """Test plotting to a new figure"""

        with patch('matplotlib.pyplot.figure') as figure_patch:

            _ = self.visualizable.plot()
            figure_patch.assert_called()

    def test_plot_existing_figure(self) -> None:
        """Test plotting into an existing figure"""

        axes = Mock()
        result = self.visualizable.plot(axes)

        self.assertIsInstance(result, Mock)
