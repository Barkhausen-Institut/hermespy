# -*- coding: utf-8 -*-

from __future__ import annotations
from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.core import Signal
from hermespy.hardware_loop import DelayCalibration, PhysicalDeviceDummy
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestDelayCalibration(TestCase):
    """Test delay calibration class."""

    def setUp(self) -> None:
        self.delay = 1.2345
        self.calibration = DelayCalibration(self.delay)

    def test_delay_setget(self) -> None:
        """Delay property getter should return setter argument"""

        expected_delay = 2.3456
        self.calibration.delay = expected_delay

        self.assertEqual(expected_delay, self.calibration.delay)

    def test_correct_transmit_delay(self) -> None:
        """Delays should be correctly corrected during transmission"""

        test_samples = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]], dtype=np.complex128)
        test_signal = Signal.Create(test_samples, 5 / abs(self.delay))

        self.calibration.delay = abs(self.delay)
        assert_array_equal(test_signal.getitem(), self.calibration.correct_transmit_delay(test_signal).getitem())

        self.calibration.delay = -abs(self.delay)
        expected_delayed_samples = np.concatenate((np.zeros((test_signal.num_streams, 5), dtype=complex), test_signal.getitem()), axis=1)
        assert_array_equal(expected_delayed_samples, self.calibration.correct_transmit_delay(test_signal).getitem())

    def test_correct_receive_delay(self) -> None:
        """Delays should be correctly corrected during reception"""

        test_samples = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]], dtype=np.complex128)
        test_signal = Signal.Create(test_samples, 1 / abs(self.delay))

        self.calibration.delay = -abs(self.delay)
        assert_array_equal(test_signal.getitem(), self.calibration.correct_receive_delay(test_signal).getitem())

        self.calibration.delay = abs(self.delay)
        expected_delayed_samples = test_samples[:, 1:]
        assert_array_equal(expected_delayed_samples, self.calibration.correct_receive_delay(test_signal).getitem())

    def test_estimate_validation(self) -> None:
        """Delay estimation routine should raise ValueErrors for invalid arguments"""

        device = PhysicalDeviceDummy(sampling_rate=1e-3)

        with self.assertRaises(ValueError):
            DelayCalibration.Estimate(device, 15e-3, 0)

        with self.assertRaises(ValueError):
            DelayCalibration.Estimate(device, 15e-3 - 1)

        with self.assertRaises(ValueError):
            DelayCalibration.Estimate(device, 15e-3, 1, -1)

        with self.assertRaises(ValueError):
            DelayCalibration.Estimate(device, 0, 1, 0)

    @patch("hermespy.hardware_loop.physical_device_dummy.PhysicalDeviceDummy.trigger_direct")
    def test_estimate(self, _trigger_direct: MagicMock) -> None:
        """Test the physical device calibration routine"""

        device = PhysicalDeviceDummy(sampling_rate=1e3)

        for expected_delay_samples in [0, 10, 12]:
            expected_delay = expected_delay_samples / device.sampling_rate

            # Configure the download routine to mirror the uploaded samples back
            # Results in a zero second calibration time of flight delay
            def trigger_side_effect(signal: Signal) -> Signal:
                # Prepend delay samples
                delayed_signal: Signal = signal.copy()
                delayed_signal.set_samples(np.append(np.zeros((delayed_signal.num_streams, expected_delay_samples), dtype=complex), delayed_signal.getitem()))

                return delayed_signal

            _trigger_direct.side_effect = trigger_side_effect

            delay_estimate = DelayCalibration.Estimate(device, 15e-3)
            self.assertAlmostEqual(expected_delay, delay_estimate.delay)

    def test_serialization(self) -> None:
        """Test delay calibration serialization"""

        test_roundtrip_serialization(self, self.calibration)
