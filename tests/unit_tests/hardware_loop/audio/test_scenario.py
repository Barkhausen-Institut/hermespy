# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import patch

import numpy as np

from hermespy.core import Signal
from hermespy.hardware_loop.audio import AudioDevice, AudioScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestAudioScenario(TestCase):
    """Test the audio scenario class."""

    def setUp(self) -> None:
        self.scenario = AudioScenario()
        self.rng = np.random.default_rng(42)

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

    def test_trigger_direct(self) -> None:
        """Test the trigger direct routine"""

        device_alpha = self.scenario.new_device(0, 0, [1], [1])        
        alpha_transmission = Signal.Create(self.rng.standard_normal((1, 10)), device_alpha.sampling_rate, device_alpha.carrier_frequency)
        
        with patch.object(device_alpha, "trigger_direct") as trigger_direct_mock:
            _ = self.scenario.trigger_direct([alpha_transmission], [device_alpha])
            trigger_direct_mock.assert_called_once_with(alpha_transmission)
