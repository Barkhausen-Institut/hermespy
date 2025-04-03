# -*- coding: utf-8 -*-

import logging
import os.path as path
from unittest import TestCase

import ray as ray

from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestDocumentationExamples(TestCase):
    def setUp(self) -> None:
        self.test_context = SimulationTestContext()
        self.base_path = path.abspath(path.join(path.dirname(path.abspath(__file__)), "..", "..", "..", "docssource", "scripts", "plots"))

    @classmethod
    def setUpClass(cls) -> None:
        ray.init(local_mode=True, num_cpus=1, ignore_reinit_error=True, logging_level=logging.ERROR)

    @classmethod
    def tearDownClass(cls) -> None:
        # Shut down ray
        ray.shutdown()

    def __run_plot(self, script: str) -> None:
        """Run a python script from the documentation's plots directory.

        Args:

            script (str):
                Path to the python script relative to the plots directory.
        """

        script_path = path.join(self.base_path, script)

        with self.test_context:
            exec(open(script_path).read())

    def test_fmcw_bandwidth(self) -> None:
        """Test plot snippet for FMCW radar's bandwidth property"""

        self.__run_plot("radar_fmcw_bandwidth.py")

    def test_fmcw_chirp_duration(self) -> None:
        """Test plot snippet for FMCW radar's chirp duration property"""

        self.__run_plot("radar_fmcw_chirp_duration.py")

    def test_fmcw_num_chirps(self) -> None:
        """Test plot snippet for FMCW radar's num chirps property"""

        self.__run_plot("radar_fmcw_num_chirps.py")

    def test_fmcw_pulse_rep_interal(self) -> None:
        """Test plot snippet for FMCW radar's pulse rep interval property"""

        self.__run_plot("radar_fmcw_pulse_rep_interval.py")
