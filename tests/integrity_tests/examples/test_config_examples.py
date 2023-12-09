# -*- coding: utf-8 -*-

import logging
from unittest import TestCase
from tempfile import TemporaryDirectory

import ray as ray

from hermespy.bin.hermes import hermes_simulation
from ..helpers import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


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

        with SimulationTestContext():
            hermes_simulation([path, "-o", self.tempdir.name])

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
