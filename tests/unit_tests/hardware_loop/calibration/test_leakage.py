# -*- coding: utf-8 -*-

from __future__ import annotations
from unittest import TestCase

import numpy as np
from unittest.mock import patch
from numpy.testing import assert_array_almost_equal, assert_array_equal

from hermespy.core import Signal
from hermespy.hardware_loop import SelectiveLeakageCalibration, PhysicalDeviceDummy
from hermespy.simulation import SimulatedIdealAntenna, SimulatedUniformArray, SelectiveLeakage
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSelectiveLeakageCalibration(TestCase):
    """Test leakage calibration base class."""

    def setUp(self) -> None:
        self.leakage_response = np.array([[[0, 0, 1, 0, 1j]]], dtype=np.complex128)
        self.calibration = SelectiveLeakageCalibration(self.leakage_response, 1.0, 0.0)

    def test_init_validation(self) -> None:
        """Initialization should raise exceptions for invalid arguments"""

        with self.assertRaises(ValueError):
            SelectiveLeakageCalibration(np.array([[[0, 0, 1, 0, 1j]]], dtype=np.complex128), 0.0, 0.0)

        with self.assertRaises(ValueError):
            SelectiveLeakageCalibration(np.array([[[0, 0, 1, 0, 1j]]], dtype=np.complex128), -1.0, 0.0)

        with self.assertRaises(ValueError):
            SelectiveLeakageCalibration(np.array([[[0, 0, 1, 0, 1j]]], dtype=np.complex128), 0.5, -1.0)

        with self.assertRaises(ValueError):
            SelectiveLeakageCalibration(np.array([[[[0, 0, 1, 0, 1j]]]], dtype=np.complex128), 1.0, 0.0)

    def test_leakage_response_get(self) -> None:
        """Leakage response should be correctly retrieved"""

        assert_array_equal(self.leakage_response, self.calibration.leakage_response)

    def test_sampling_rate_get(self) -> None:
        """Sampling rate should be correctly retrieved"""

        self.assertEqual(1.0, self.calibration.sampling_rate)

    def test_delay_get(self) -> None:
        """Delay should be correctly retrieved"""

        self.assertEqual(0.0, self.calibration.delay)

    def test_remove_leakage_validation(self) -> None:
        """Leakage removal should raise exceptions for invalid arguments"""

        transmitted_signal = Signal.Create(np.array([[1, 2, 3, 4, 5]], dtype=np.complex128), 1.0)
        received_signal = Signal.Create(np.array([[1, 2, 3, 4, 5]], dtype=np.complex128), 1.0)

        with self.assertRaises(ValueError):
            _ = self.calibration.remove_leakage(Signal.Create(np.repeat(transmitted_signal.getitem(), 2, axis=0), transmitted_signal.sampling_rate), received_signal)

        with self.assertRaises(ValueError):
            _ = self.calibration.remove_leakage(transmitted_signal, Signal.Create(np.repeat(received_signal.getitem(), 2, axis=0), received_signal.sampling_rate))

        with self.assertRaises(ValueError):
            _ = self.calibration.remove_leakage(Signal.Create(transmitted_signal.getitem(), 2 * transmitted_signal.sampling_rate), received_signal)

        with self.assertRaises(ValueError):
            _ = self.calibration.remove_leakage(Signal.Create(transmitted_signal.getitem(), transmitted_signal.sampling_rate, 10), received_signal)

    def test_plot(self) -> None:
        """Test the visualization of the calibration"""

        with patch("matplotlib.pyplot.figure") as mock_figure:
            try:
                _ = self.calibration.plot()

            except Exception:
                self.fail("Plotting should not raise an exception")

    def test_etimate_delay(self) -> None:
        """The delay estimation should return the correct delay"""

        expected_delay = 2 / self.calibration.sampling_rate
        estimated_delay = self.calibration.estimate_delay().delay

        self.assertEqual(expected_delay, estimated_delay)

    def test_mmse_estimate_validation(self) -> None:
        """MMSE estimation should raise exceptions for invalid arguments"""

        device = PhysicalDeviceDummy(carrier_frequency=1e9, sampling_rate=1e8, seed=42, antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (2, 1, 1)), receive_transmission=False)

        with self.assertRaises(ValueError):
            _ = SelectiveLeakageCalibration.MMSEEstimate(device, num_probes=0)

        with self.assertRaises(ValueError):
            _ = SelectiveLeakageCalibration.MMSEEstimate(device, num_wavelet_samples=0)

        with self.assertRaises(ValueError):
            _ = SelectiveLeakageCalibration.MMSEEstimate(device, noise_power=np.array([-1.0, 0.0]))

        with self.assertRaises(ValueError):
            _ = SelectiveLeakageCalibration.MMSEEstimate(device, noise_power=np.array([0.0]))

    def test_mmse_estimate_without_noise_power(self) -> None:
        """Minimum mean square estimation should return the correct leakage response"""

        num_samples = 32
        sampling_rate = 1e8

        device = PhysicalDeviceDummy(carrier_frequency=1e9, sampling_rate=sampling_rate, max_receive_delay=num_samples / sampling_rate, seed=42, antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (2, 1, 1)), receive_transmission=False)
        leakage_model = SelectiveLeakage.Normal(device, num_samples=num_samples)
        device.isolation = leakage_model

        calibration = SelectiveLeakageCalibration.MMSEEstimate(device)
        assert_array_almost_equal(leakage_model.leakage_response, calibration.leakage_response[:, :, :num_samples])

    def test_mmse_estimate_with_noise_power(self) -> None:
        """Minimum mean square estimation should return the correct leakage response"""

        num_samples = 32
        sampling_rate = 1e8

        device = PhysicalDeviceDummy(carrier_frequency=1e9, sampling_rate=sampling_rate, max_receive_delay=num_samples / sampling_rate, seed=42, antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (2, 1, 1)), receive_transmission=False, noise_power=np.zeros(2))
        leakage_model = SelectiveLeakage.Normal(device, num_samples=num_samples)
        device.isolation = leakage_model

        calibration = SelectiveLeakageCalibration.MMSEEstimate(device)
        assert_array_almost_equal(leakage_model.leakage_response, calibration.leakage_response[:, :, :num_samples])

    def test_least_squares_estimate(self) -> None:
        num_samples = 32
        sampling_rate = 1e8

        device = PhysicalDeviceDummy(carrier_frequency=1e9, sampling_rate=sampling_rate, max_receive_delay=num_samples / sampling_rate, seed=42, antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (2, 1, 1)), receive_transmission=False, noise_power=np.zeros(2))
        leakage_model = SelectiveLeakage.Normal(device, num_samples=num_samples)
        device.isolation = leakage_model

        calibration = SelectiveLeakageCalibration.LeastSquaresEstimate(device)

    def test_yaml_serialization(self) -> None:
        """Test YAML serialization and deserialization"""

        test_yaml_roundtrip_serialization(self, self.calibration)
