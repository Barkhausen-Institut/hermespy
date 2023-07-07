# -*- coding: utf-8 -*-

import logging
from contextlib import ExitStack
from os import path as os_path
from sys import gettrace, path as sys_path
from unittest import TestCase
from unittest.mock import patch, PropertyMock

import ray as ray

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
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
        
        ray.init(local_mode=True, num_cpus=1, ignore_reinit_error=True, logging_level=logging.ERROR)

    @classmethod
    def tearDownClass(cls) -> None:

        ray.shutdown()

    @patch('matplotlib.pyplot.figure')
    def test_getting_started_link(self, mock_figure) -> None:
        """Test getting started library link example execution"""

        import getting_started_link  # type: ignore
        mock_figure.assert_called()
        
    @patch('matplotlib.pyplot.figure')
    def test_getting_started_ofdm_link(self, mock_figure) -> None:
        """Test getting started library OFDM link example execution"""

        import getting_started_ofdm_link  # type: ignore
        mock_figure.assert_called()

    @patch('matplotlib.pyplot.figure')
    @patch('sys.stdout')
    def test_getting_started_simulation_multidim(self, mock_stdout, mock_figure) -> None:
        """Test getting started library multidimensional simulation example execution"""

        with patch('hermespy.simulation.Simulation.num_samples', new_callable=PropertyMock) as num_samples:

            num_samples.return_value = 1
            import getting_started_simulation_multidim  # type: ignore

        mock_figure.assert_called()

    @patch('matplotlib.pyplot.figure')
    @patch('sys.stdout')
    def test_getting_started_simulation(self, mock_stdout, mock_figure) -> None:
        """Test getting started library simulation example execution"""

        with patch('hermespy.simulation.Simulation.num_samples', new_callable=PropertyMock) as num_samples:

            num_samples.return_value = 1
            import getting_started_simulation  # type: ignore

        mock_figure.assert_called()

    @patch('matplotlib.pyplot.figure')
    def test_getting_started(self, mock_figure) -> None:
        """Test getting started library example execution"""

        import getting_started  # type: ignore
        mock_figure.assert_called()
        
    def test_getting_started_radarlink(self) -> None:
        """Test getting started radar link example"""
        
        with ExitStack() as stack:
            
            if gettrace() is None:
                stack.enter_context(patch('matplotlib.pyplot.figure'))
                
            import getting_started_radarlink  # type: ignore

    def test_usrp_loop(self) -> None:
        """Test USRP loop example execution"""
        
        with ExitStack() as stack:
            
            if gettrace() is None:
                stack.enter_context(patch('sys.stdout'))
                stack.enter_context(patch('matplotlib.pyplot.figure'))
                
            from hermespy.hardware_loop import PhysicalScenarioDummy, PhysicalDeviceDummy

            new_device = PhysicalDeviceDummy()
            def new_device_callback(self, *args, **kwargs):
                
                if new_device not in self.devices:
                    self.add_device(new_device)
                return new_device
                         
            new_device_patch = stack.enter_context(patch.object(PhysicalScenarioDummy, 'new_device', autospec=True))
            new_device_patch.side_effect = new_device_callback

            stack.enter_context(patch('hermespy.hardware_loop.UsrpSystem', PhysicalScenarioDummy))
            stack.enter_context(patch('hermespy.hardware_loop.HardwareLoop.new_dimension'))

            import usrp_loop  # type: ignore
