# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

from hermespy.core.pipeline import Pipeline

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockPipeline(Pipeline[Mock, Mock]):
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
            
    def test_add_device(self) -> None:
        """The add_device method should add the device to the pipeline"""
        
        device = Mock()
        self.pipeline.add_device(device)
        
        self.scenario.add_device.assert_called_once_with(device)

    def test_new_device(self) -> None:
        """The new_device method should create a new device and add it to the pipeline"""
        
        _ = self.pipeline.new_device()
        self.scenario.new_device.assert_called_once()

    def test_device_index(self) -> None:
        """The device_index method should return the correct index"""
        
        device = Mock()
        self.scenario.device_index.return_value = 123
        self.assertEqual(123, self.pipeline.device_index(device))
        self.scenario.device_index.assert_called_once_with(device)
