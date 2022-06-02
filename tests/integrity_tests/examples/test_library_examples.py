# -*- coding: utf-8 -*-

from io import StringIO
from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock
from warnings import catch_warnings, simplefilter

import ray as ray

from hermespy.core import MonteCarlo

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestLibraryExamples(TestCase):
    """Test library example execution without exceptions"""

    def setUp(self) -> None:
        ...

    @classmethod
    def setUpClass(cls) -> None:

        # Run ray in local mode
        with catch_warnings():

            simplefilter("ignore")
            ray.init(local_mode=True)

    @classmethod
    def tearDownClass(cls) -> None:

        # Shut down ray 
        ray.shutdown()

    @patch('matplotlib.pyplot.figure')
    def test_getting_started_link(self, mock_figure) -> None:
        """Test getting started library link example execution"""

        import _examples.library.getting_started_link
        mock_figure.assert_called()

    @patch('matplotlib.pyplot.figure')
    @patch('sys.stdout')
    def test_getting_started_simulation_multidim(self, mock_stdout, mock_figure) -> None:
        """Test getting started library multidimensional simulation example execution"""

        with patch('hermespy.simulation.Simulation.num_samples', new_callable=PropertyMock) as num_samples:

            num_samples.return_value = 1
            import _examples.library.getting_started_simulation_multidim

        mock_figure.assert_called()

    @patch('matplotlib.pyplot.figure')
    @patch('sys.stdout')
    def test_getting_started_simulation(self, mock_stdout, mock_figure) -> None:
        """Test getting started library simulation example execution"""

        with patch('hermespy.simulation.Simulation.num_samples', new_callable=PropertyMock) as num_samples:

            num_samples.return_value = 1
            import _examples.library.getting_started_simulation

        mock_figure.assert_called()

    @patch('matplotlib.pyplot.figure')
    def test_getting_started(self, mock_figure) -> None:
        """Test getting started library example execution"""

        import _examples.library.getting_started
        mock_figure.assert_called()
