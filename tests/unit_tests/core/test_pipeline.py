# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

from hermespy.core.pipeline import Pipeline

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockPipeline(Pipeline[Mock]):
    """Mock pipeline for testing only."""
    
    def run(self) -> None:
        
        return

class TestPipeline(TestCase):
    """Test the base simulation pipeline"""
    
    def setUp(self) -> None:
        
        self.scenario = Mock()
        self.pipeline = MockPipeline(scenario=self.scenario)

    def test_scenario_get(self) -> None:
        """The scenrio property should return the correct scenario handle"""
        
        self.assertIs(self.scenario, self.pipeline.scenario)
        
    def test_num_drops_setget(self) -> None:
        """Number of drops property getter should return setter argument"""
        
        expected_num_drops = 123
        self.pipeline.num_drops = expected_num_drops
        
        self.assertEqual(expected_num_drops, self.pipeline.num_drops)
    
    def test_num_drops_validation(self) -> None:
        """Number of drops property setter should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.pipeline.num_drops = 0
            
        with self.assertRaises(ValueError):
            self.pipeline.num_drops = -1
    