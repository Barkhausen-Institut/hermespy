# -*- coding: utf-8 -*-

import logging
from os import getenv
from unittest import TestCase
from unittest.mock import patch
from sys import gettrace
from tempfile import TemporaryDirectory
from typing import Any, List, Optional

import ray as ray

from hermespy.bin.hermes import hermes_simulation
from hermespy.core import ConsoleMode, MonteCarlo, GridDimension, Verbosity
from hermespy.simulation import Simulation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


monte_carlo_init = MonteCarlo.__init__
simulation_init = Simulation.__init__


def monte_carlo_init_mock(cls: MonteCarlo, *args, **kwargs) -> None:
    
    args = list(args)
    args[1] = 1
    kwargs['num_actors'] = 1            # Only spawn a single actor
    kwargs['catch_exceptions'] = False  # Don't catch exceptions during runtime
    kwargs['min_num_samples'] = 1       # Only generate a single sample

    monte_carlo_init(cls, *args, **kwargs)


def simulation_init_mock(self: Simulation, scenario, num_samples: int = 100, drop_duration: float = 0.0, plot_results: bool = False, dump_results: bool = True, console_mode: ConsoleMode = ConsoleMode.INTERACTIVE, ray_address = None, results_dir = None, verbosity = Verbosity.INFO, seed = None, num_actors = None) -> None:

    num_samples = 1
    simulation_init(self, scenario, num_samples, drop_duration, plot_results, dump_results, console_mode, ray_address, results_dir, verbosity, seed, num_actors)


def new_dimension_mock(cls: MonteCarlo,
                       dimension: str,
                       sample_points: List[Any],
                       considered_object: Optional[Any] = None) -> GridDimension:

    _considered_object = cls.investigated_object if considered_object is None else considered_object

    # Only take a single sample point into account to speed up simulations
    dimension = GridDimension(_considered_object, dimension, [sample_points[0]])
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
        
        ray.init(local_mode=True, num_cpus=1, ignore_reinit_error=True, logging_level=logging.ERROR)

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

        # Make sure we're not in debug mode
        if getenv('HERMES_TEST_PLOT', 'False').lower() != 'true':
            with patch('sys.stdout'), patch.object(MonteCarlo, '__init__', new=monte_carlo_init_mock), patch.object(Simulation, '__init__', new=simulation_init_mock), patch.object(MonteCarlo, 'new_dimension', new=new_dimension_mock), patch('matplotlib.pyplot.figure'):

                hermes_simulation([path, '-o', self.tempdir.name])
                
        else:
            with patch.object(MonteCarlo, '__init__', new=monte_carlo_init_mock), patch.object(Simulation, '__init__', new=simulation_init_mock):
                hermes_simulation([path, '-o', self.tempdir.name])
    
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

        # Currently disabled due to suspiciously high runtime
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

        # ToDo: Re-implement operator separation properly
        # self.__run_yaml("_examples/settings/operator_separation.yml")
