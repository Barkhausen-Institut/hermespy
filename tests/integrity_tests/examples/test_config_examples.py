# -*- coding: utf-8 -*-

from io import StringIO
from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock
from tempfile import TemporaryDirectory
from typing import Any, List, Optional
from warnings import catch_warnings, simplefilter

import ray as ray

from hermespy.bin.hermes import hermes
from hermespy.core.monte_carlo import MonteCarlo, GridDimension

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


def new_dimension_mock(cls: MonteCarlo,
                       dimension: str,
                       sample_points: List[Any],
                       considered_object: Optional[Any] = None) -> GridDimension:

    # Only take a single sample point into account to speed up simulations
    dimension = GridDimension(cls.investigated_object, dimension, [sample_points[0]])
    cls.add_dimension(dimension)

    return dimension


class TestConfigurationExamples(TestCase):
    """Test configuration example execution without exceptions"""

    def setUp(self) -> None:
        
        # Create temporary directory to store simulation artifacts
        self.tempdir = TemporaryDirectory()

    def tearDown(self) -> None:

        # Clear temporary directory and remove all simulation artifacts
        self.tempdir.cleanup()

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

    def __run_yaml(self, path: str) -> None:
        """Run a yaml simulation and test for proper execution.

        Args:

            path (str):
                Path to the yaml configuration file.
        """

        with patch('hermespy.simulation.Simulation.num_samples', new_callable=PropertyMock) as num_samples, patch('sys.stdout') as stdout, patch('matplotlib.pyplot.figure') as figure, patch.object(MonteCarlo, 'new_dimension', new=new_dimension_mock):

            num_samples.return_value = 1
            args = ['-p', path, '-o', self.tempdir.name]
            hermes(args)
    
    def test_chirp_fsk_lora(self) -> None:
        """Test example settings for chirp FSK modulation"""

        self.__run_yaml("_examples/settings/chirp_fsk_lora")

    def test_chirp_qam(self) -> None:
        """Test example settings for chirp QAM modulation"""

        self.__run_yaml("_examples/settings/chirp_qam")

    def test_interference_ofdm_sc(self) -> None:
        """Test example settings for single carrier OFDM interference"""

        self.__run_yaml("_examples/settings/interference_ofdm_single_carrier")

    def test_ofdm_5g(self) -> None:
        """Test example settings for 5G OFDM modulation"""

        self.__run_yaml("_examples/settings/ofdm_5g")

    def test_ofdm_single_carrier(self) -> None:
        """Test example settings for single carrier OFDM modulation"""

        self.__run_yaml("_examples/settings/ofdm_single_carrier")

    def test_operator_separation(self) -> None:
        """Test example settings for operator separation"""

        self.__run_yaml("_examples/settings/operator_separation")