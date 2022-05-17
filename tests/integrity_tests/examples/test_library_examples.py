# -*- coding: utf-8 -*-

from io import StringIO

from unittest import TestCase
from unittest.mock import patch


class TestLibraryExamples(TestCase):
    """Test library example execution without exceptions"""

    def setUp(self) -> None:
        ...

    @patch('matplotlib.pyplot.figure')
    def test_getting_started_link(self, mock_figure) -> None:
        """Test getting started library link example execution"""

        import _examples.library.getting_started_link
        mock_figure.assert_called()

    @patch('matplotlib.pyplot.figure')
    @patch('sys.stdout')
    def test_getting_started_simulation_multidim(self, mock_stdout, mock_figure) -> None:
        """Test getting started library multidimensional simulation example execution"""

        import _examples.library.getting_started_simulation_multidim
        mock_figure.assert_called()

    @patch('matplotlib.pyplot.figure')
    @patch('sys.stdout')
    def test_getting_started_simulation(self, mock_stdout, mock_figure) -> None:
        """Test getting started library simulation example execution"""

        import _examples.library.getting_started_simulation
        mock_figure.assert_called()

    @patch('matplotlib.pyplot.figure')
    def test_getting_started(self, mock_figure) -> None:
        """Test getting started library example execution"""

        import _examples.library.getting_started
        mock_figure.assert_called()
