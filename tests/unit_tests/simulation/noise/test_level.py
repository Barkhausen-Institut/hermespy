# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
from unittest import TestCase
from unittest.mock import patch, PropertyMock

import numpy as np

from hermespy.core import DenseSignal, SignalReceiver
from hermespy.simulation import NoiseLevel, N0, SimulatedDevice, SNR, AWGN
from hermespy.channel import IdealChannel
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class _TestNoiseLevel(ABC, TestCase):
    """Base class for testing noise level implementations."""

    level: NoiseLevel

    def test_level_setget(self) -> None:
        """Level property getter should return setter argument"""

        expected_level = 1.2345
        self.level.level = expected_level
        self.assertEqual(expected_level, self.level.level)

    @abstractmethod
    def test_get_power(self) -> None:
        """Power should be returned as expected"""
        ...  # pragma: no cover

    def test_title(self) -> None:
        """Title property should return a string"""

        self.assertIsInstance(self.level.title, str)

    def test_lshift(self) -> None:
        """Lshfit operator should update the level property"""

        expected_level = 1.2345
        self.level << expected_level
        self.assertEqual(expected_level, self.level.level)

    def test_yaml_roundtrip_serialization(self) -> None:
        """Noise level should be serializable"""

        test_yaml_roundtrip_serialization(self, self.level)


class TestN0(_TestNoiseLevel):
    """Test class for the N0 noise level implementation."""

    level: N0

    def setUp(self) -> None:
        """Set up the test case"""

        self.level = N0(0.0)

    def test_power_setget(self) -> None:
        """Power property getter should return setter argument"""

        expected_power = 1.2345
        self.level.power = expected_power
        self.assertEqual(expected_power, self.level.power)

    def test_power_validation(self) -> None:
        """Power should be non-negative"""

        with self.assertRaises(ValueError):
            self.level.power = -1.0

    def test_get_power(self) -> None:
        """Power should be returned as expected"""

        expected_power = 1.2345
        self.level.power = expected_power
        self.assertEqual(expected_power, self.level.get_power())


class TestSNR(_TestNoiseLevel):
    """Test class for the SNR noise level implementation."""

    level: SNR

    def setUp(self) -> None:
        self.reference = SimulatedDevice()
        self.level = SNR(1.5432, self.reference)

    def test_init(self) -> None:
        """Initialization should set the SNR"""

        expected_snr = 1.2345
        level = SNR(expected_snr, self.reference)
        self.assertEqual(expected_snr, level.snr)
        self.assertIs(self.reference, level.reference)

    def test_snr_setget(self) -> None:
        """SNR property getter should return setter argument"""

        expected_snr = 1.2345
        self.level.snr = expected_snr
        self.assertEqual(expected_snr, self.level.snr)

    def test_snr_validation(self) -> None:
        """SNR should be non-negative"""

        with self.assertRaises(ValueError):
            self.level.snr = -1.0

        with self.assertRaises(ValueError):
            self.level.snr = 0.0

    def test_get_power(self) -> None:
        """Power should be returned as expected"""

        reference_power = 1.2345
        with patch('hermespy.core.device.Device.power', new_callable=PropertyMock) as mock_power:
            mock_power.return_value = reference_power
            self.assertEqual(reference_power / self.level.snr, self.level.get_power())

    def test_power_scaling(self) -> None:
        """Power should be scaled by the expected channel scale"""

        num_samples = 10000
        sampling_rate = 1e8
        expected_energy_scale = 0.5
        channel = IdealChannel(expected_energy_scale)
        tx_device = self.reference
        tx_device.sampling_rate = sampling_rate
        rx_device = SimulatedDevice(sampling_rate=sampling_rate)
        rx_device.noise_model = AWGN(seed=42)
        
        rx_dsp = SignalReceiver(num_samples, sampling_rate)
        rx_dsp.device = rx_device
        
        noise_level = SNR(1.5, tx_device, channel)
        rx_device.noise_level = noise_level
        
        unit_power_signal = DenseSignal(np.ones(num_samples), sampling_rate, 0)
        
        propagated_signal = channel.propagate(unit_power_signal, tx_device, rx_device)
        rx_device.receive(propagated_signal)
        
        expected_received_power = (unit_power_signal.power * (1 + 1/noise_level.snr)) * expected_energy_scale
        received_power = rx_dsp.signal.power[0]
        self.assertAlmostEqual(expected_received_power, received_power, delta=.1)
        

del _TestNoiseLevel
