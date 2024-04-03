# -*- coding: utf-8 -*-
"""Test general pilot-based correlation synchronization"""

from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from hermespy.core import Signal
from hermespy.modem.waveform_correlation_synchronization import CorrelationSynchronization
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestCorellationSynchronization(TestCase):
    """Correlation synchronization class testing"""

    def setUp(self) -> None:
        self.threshold = 0.91
        self.guard_ratio = 0.81
        self.synchronization = CorrelationSynchronization(threshold=self.threshold, guard_ratio=self.guard_ratio)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertEqual(self.threshold, self.synchronization.threshold)
        self.assertEqual(self.guard_ratio, self.synchronization.guard_ratio)

    def test_threshold_setget(self) -> None:
        """Threshold property getter should return setter argument"""

        expected_threshold = 0.1
        self.synchronization.threshold = expected_threshold

        self.assertEqual(expected_threshold, self.synchronization.threshold)

    def test_threshold_validation(self) -> None:
        """Threshold property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.synchronization.threshold = -0.1

        with self.assertRaises(ValueError):
            self.synchronization.threshold = 1.1

    def test_guard_ratio_setget(self) -> None:
        """Guard ratio property getter should return setter argument"""

        expected_guard_ratio = 0.1
        self.synchronization.guard_ratio = expected_guard_ratio

        self.assertEqual(expected_guard_ratio, self.synchronization.guard_ratio)

    def test_guard_ratio_validation(self) -> None:
        """Guard ratio property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.synchronization.guard_ratio = -0.1

        with self.assertRaises(ValueError):
            self.synchronization.guard_ratio = 1.1

    def test_synchronize(self) -> None:
        """Synchronization should properly order pilot sections into frames"""

        pilot_sequence = Signal(np.ones(20, dtype=complex), 1.0)

        waveform = Mock()
        waveform.pilot_signal = pilot_sequence
        waveform.samples_per_frame = 20
        self.synchronization.waveform = waveform

        shifted_sequence = np.append(np.zeros((1, 10), dtype=complex), pilot_sequence.samples, axis=1)

        pilot_indices = self.synchronization.synchronize(shifted_sequence)
        self.assertSequenceEqual([10], pilot_indices)

    def test_default_synchronize(self) -> None:
        """Synchronization should properly order pilot sections into frames"""

        pilot_sequence = Signal(np.ones(20, dtype=complex), 1.0)

        waveform = Mock()
        waveform.pilot_signal = pilot_sequence
        waveform.samples_per_frame = 20
        self.synchronization.waveform = waveform

        empty_sequence = np.zeros((1, 40), dtype=complex)

        pilot_indices = self.synchronization.synchronize(empty_sequence)
        self.assertSequenceEqual([], pilot_indices)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.synchronization)
