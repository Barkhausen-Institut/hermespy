# -*- coding: utf-8 -*-

from unittest import TestCase

from hermespy.hardware_loop.audio import AudioDevice, AudioScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestAudioScenario(TestCase):
    """Test the audio scenario class."""

    def setUp(self) -> None:
        self.scenario = AudioScenario()

    def test_new_device(self) -> None:
        """Test the new device routine"""

        device = self.scenario.new_device(6, 4, [1], [1])
        self.assertIsInstance(device, AudioDevice)

        self.assertEqual(1, len(self.scenario.devices))
        self.assertIs(device, self.scenario.devices[0])

        device2 = self.scenario.new_device(6, 4, [1], [1])

        self.assertEqual(2, len(self.scenario.devices))
        self.assertIs(device2, self.scenario.devices[1])

    def test_trigger(self) -> None:
        """Test the trigger routine"""

        # Trigger of the audio scenario is not implemented
        _ = self.scenario.drop()
