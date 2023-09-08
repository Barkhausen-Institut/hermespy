# -*- coding: utf-8 -*-

import logging
import os.path as path
from unittest import TestCase

import ray as ray

from ..helpers import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestDocumentationExamples(TestCase):
    
    def setUp(self) -> None:
        
        self.test_context = SimulationTestContext()
        self.base_path = path.abspath(path.join(path.dirname(path.abspath(__file__)), '..', '..', '..', 'docssource', 'scripts', 'examples'))

    @classmethod
    def setUpClass(cls) -> None:
        
        ray.init(local_mode=True, num_cpus=1, ignore_reinit_error=True, logging_level=logging.ERROR)

    @classmethod
    def tearDownClass(cls) -> None:

        # Shut down ray
        ray.shutdown()
        
    def __run_example(self, script: str) -> None:
        """Run a python script from the documentation's examples directory.
        
        Args:
        
            script (str):
                Path to the python script relative to the examples directory.
        """
        
        script_path = path.join(self.base_path, script)
        
        with self.test_context:
            exec(open(script_path).read())
        
    def test_DetectionProbEvaluation(self) -> None:
        """Test example snippet for detection probability evaluation"""
        
        self.__run_example('radar_evaluators_DetectionProbEvaluator.py')

    def test_ReceiverOperatingCharacteristic(self) -> None:
        """Test example snippet for receiver oeperating characteristic evaluation"""
        
        self.__run_example('radar_evaluators_ReceiverOperatingCharacteristic.py')

    def test_RootMeanSquareError(self) -> None:
        """Test example snippet for root mean square error evaluation"""
        
        self.__run_example('radar_evaluators_RootMeanSquareError.py')

    def test_FMCW(self) -> None:
        """Test example snippet for FMCW waveforms"""
        
        self.__run_example('radar_fmcw_FMCW.py')
