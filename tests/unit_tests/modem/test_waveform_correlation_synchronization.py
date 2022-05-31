# -*- coding: utf-8 -*-
"""Test general pilot-based correlation synchronization"""

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.core import ChannelStateInformation, Signal
from hermespy.modem.waveform_correlation_synchronization import CorrelationSynchronization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestCorellationSynchronization(TestCase):
    """Correlation synchronization class testing"""

    def setUp(self) -> None:

        self.threshold = .91
        self.guard_ratio = .81
        self.synchronization = CorrelationSynchronization(threshold=self.threshold, guard_ratio=self.guard_ratio)
    
    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertEqual(self.threshold, self.synchronization.threshold)
        self.assertEqual(self.guard_ratio, self.synchronization.guard_ratio)

    def test_threshold_setget(self) -> None:
        """Threshold property getter should return setter argument"""

        expected_threshold = .1
        self.synchronization.threshold = expected_threshold

        self.assertEqual(expected_threshold, self.synchronization.threshold)

    def test_threshold_validation(self) -> None:
        """Threshold property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.synchronization.threshold = -.1

        with self.assertRaises(ValueError):
            self.synchronization.threshold = 1.1

    def test_guard_ratio_setget(self) -> None:
        """Guard ratio property getter should return setter argument"""

        expected_guard_ratio = .1
        self.synchronization.guard_ratio = expected_guard_ratio

        self.assertEqual(expected_guard_ratio, self.synchronization.guard_ratio)

    def test_guard_ratio_validation(self) -> None:
        """Guard ratio property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.synchronization.guard_ratio = -.1

        with self.assertRaises(ValueError):
            self.synchronization.guard_ratio = 1.1

    def test_synchronize(self) -> None:
        """Synchronization should properly order pilot sections into frames"""

        pilot_sequence = Signal(np.ones(20, dtype=complex), 1.)

        waveform_generator = Mock()
        waveform_generator.pilot_signal = pilot_sequence
        waveform_generator.samples_in_frame = 20
        self.synchronization.waveform_generator = waveform_generator

        shifted_sequence = np.append(np.zeros((1, 10), dtype=complex), pilot_sequence.samples, axis=1)
        shifted_csi = ChannelStateInformation.Ideal(30)

        frames = self.synchronization.synchronize(shifted_sequence, shifted_csi)

        self.assertEqual(1, len(frames))
        assert_array_equal(pilot_sequence.samples, frames[0][0])

    def test_to_yaml(self) -> None:
        """YAML serialization should result in a proper state representation"""

        representer = Mock()
        node = CorrelationSynchronization.to_yaml(representer, self.synchronization)

        representer.represent_mapping.assert_called()
