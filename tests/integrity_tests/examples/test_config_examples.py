# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, patch, PropertyMock
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
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


monte_carlo_init = MonteCarlo.__init__

def init_mock(cls: MonteCarlo, *args, **kwargs) -> None:
    
    args = list(args)
    args[1] = 1
    kwargs['num_actors'] = 1            # Only spawn a single actor
    kwargs['catch_exceptions'] = False  # Don't catch exceptions during runtime
    
    monte_carlo_init(cls, *args, **kwargs)


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
        
        if not ray.is_initialized():
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

        with patch('sys.stdout'), patch.object(MonteCarlo, '__init__', new=init_mock), patch.object(MonteCarlo, 'new_dimension', new=new_dimension_mock), patch('matplotlib.pyplot.figure'):

            hermes([path, '-o', self.tempdir.name])
    
    def test_chirp_fsk_lora(self) -> None:
        """Test example settings for chirp FSK modulation"""

        self.__run_yaml("_examples/settings/chirp_fsk_lora.yml")

    def test_chirp_qam(self) -> None:
        """Test example settings for chirp QAM modulation"""

        self.__run_yaml("_examples/settings/chirp_qam.yml")
        
    def test_hardware_model(self) -> None:
        """Test example settings for hardware simulation"""

        self.__run_yaml("_examples/settings/hardware_model.yml")

    def test_interference_ofdm_sc(self) -> None:
        """Test example settings for single carrier OFDM interference"""

        # Currently disabled due to suspicious high runtime
        # self.__run_yaml("_examples/settings/interference_ofdm_single_carrier.yml")

    def test_jcas(self) -> None:
        """Test example settings for joint communications and sensing"""

        self.__run_yaml("_examples/settings/jcas.yml")

    def test_ofdm_5g(self) -> None:
        """Test example settings for 5G OFDM modulation"""

        self.__run_yaml("_examples/settings/ofdm_5g.yml")

    def test_ofdm_single_carrier(self) -> None:
        """Test example settings for single carrier OFDM modulation"""

        self.__run_yaml("_examples/settings/ofdm_single_carrier.yml")

    def test_operator_separation(self) -> None:
        """Test example settings for operator separation"""

        self.__run_yaml("_examples/settings/operator_separation.yml")
