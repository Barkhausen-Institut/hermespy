# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch

from core import Signal

from hermespy.hardware_loop import PhysicalScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalScenarioMock(PhysicalScenario[Mock]):
    """Physical scenario base class mock for unit testing"""

    def _trigger(self) -> None:
        return
    
    def _trigger_direct(self, transmissions: list[Signal], devices: list[Mock], calibrate: bool = True) -> list[Signal]:
        return []


class TestPhysicalScenario(TestCase):
    def setUp(self) -> None:
        self.scenario = PhysicalScenarioMock()
        self.device = Mock()
        self.device.transmitters = []
        self.device.receivers = []
        self.scenario.add_device(self.device)

    def test_receive_devices(self) -> None:
        """Test extended reception routine"""

        receptions = self.scenario.receive_devices()

        self.device.process_input.assert_called_once()
        self.device.receive_operators.assert_called_once()
        self.assertEqual(1, len(receptions))

    @patch.object(PhysicalScenarioMock, "_trigger")
    def test_drop(self, _trigger: MagicMock) -> None:
        """Test the physical scenario drop geneartion"""

        _ = self.scenario._drop()

        self.device.transmit.assert_called_once()
        self.device.process_input.assert_called_once()
        self.device.receive_operators.assert_called_once()
        _trigger.assert_called_once()
