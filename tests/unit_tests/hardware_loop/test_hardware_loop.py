# -*- coding: utf-8 -*-
"""Test HermesPy physical device module."""

from unittest import TestCase
from unittest.mock import Mock

from hermespy.hardware_loop import HardwareLoop, PhysicalScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockPhysicalScenario(PhysicalScenario[Mock]):
    """Mock physical scenario for testing only."""
    
    def _trigger(self) -> None:
        return

class TestPhysicalDevice(TestCase):
    """Test the physical device base class."""

    def setUp(self) -> None:

        self.scenario = MockPhysicalScenario()
        self.hardware_loop = HardwareLoop[MockPhysicalScenario](self.scenario, manual_triggering=True)

    def test_init(self) -> None:
        """Physical device class should be properly initialized"""
        
        self.assertIs(self.scenario, self.hardware_loop.scenario)
        self.assertTrue(self.hardware_loop.manual_triggering)
