# -*- coding: utf-8 -*-

from os import path as os_path
from sys import path as sys_path
from unittest import TestCase
from unittest.mock import patch, PropertyMock
from warnings import catch_warnings, simplefilter

import ray as ray

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestLibraryExamples(TestCase):
    """Test library example execution without exceptions"""

    def setUp(self) -> None:
        
        library_dir = os_path.abspath(os_path.join(os_path.dirname(__file__), '..', '..', '..', '_examples', 'library'))
        sys_path.append(library_dir)

    @classmethod
    def setUpClass(cls) -> None:
        
        if not ray.is_initialized():

            # Run ray in local mode
            with catch_warnings():

                simplefilter("ignore")
                ray.init(local_mode=True)

    @classmethod
    def tearDownClass(cls) -> None:

        ...
        # Shut down ray 
        #ray.shutdown()

    @patch('matplotlib.pyplot.figure')
    def test_getting_started_link(self, mock_figure) -> None:
        """Test getting started library link example execution"""

        import getting_started_link
        mock_figure.assert_called()
        
    @patch('matplotlib.pyplot.figure')
    def test_getting_started_ofdm_link(self, mock_figure) -> None:
        """Test getting started library OFDM link example execution"""

        import getting_started_ofdm_link
        mock_figure.assert_called()

    @patch('matplotlib.pyplot.figure')
    @patch('sys.stdout')
    def test_getting_started_simulation_multidim(self, mock_stdout, mock_figure) -> None:
        """Test getting started library multidimensional simulation example execution"""

        with patch('hermespy.simulation.Simulation.num_samples', new_callable=PropertyMock) as num_samples:

            num_samples.return_value = 1
            import getting_started_simulation_multidim

        mock_figure.assert_called()

    @patch('matplotlib.pyplot.figure')
    @patch('sys.stdout')
    def test_getting_started_simulation(self, mock_stdout, mock_figure) -> None:
        """Test getting started library simulation example execution"""

        with patch('hermespy.simulation.Simulation.num_samples', new_callable=PropertyMock) as num_samples:

            num_samples.return_value = 1
            import getting_started_simulation

        mock_figure.assert_called()

    @patch('matplotlib.pyplot.figure')
    def test_getting_started(self, mock_figure) -> None:
        """Test getting started library example execution"""

        import getting_started
        mock_figure.assert_called()

    @patch('sys.stdout')
    def test_usrp_loop(self, _) -> None:
        """Test USRP loop example execution"""
        
        from hermespy.hardware_loop import PhysicalDeviceDummy

        with patch('hermespy.hardware_loop.uhd.system.UsrpSystem.new_device') as new_device_patch, \
            patch('hermespy.hardware_loop.hardware_loop.HardwareLoop.new_dimension'):
            
                new_device_patch.side_effect = lambda *args, **kwargs: PhysicalDeviceDummy()
                import usrp_loop
