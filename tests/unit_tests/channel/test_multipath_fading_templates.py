# -*- coding: utf-8 -*-
"""Test Multipath Fading Channel Model Templates."""

import unittest
from itertools import product
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from scipy.constants import pi

from hermespy.channel import MultipathFadingCost259, Cost259Type, MultipathFading5GTDL, TDLType, MultipathFadingExponential, DeviceType, CorrelationType, StandardAntennaCorrelation
from hermespy.core.device import FloatingError
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestStandardAntennaCorrelation(unittest.TestCase):
    """Test standard antenna correlation models"""

    def setUp(self) -> None:
        self.device = Mock()
        self.num_antennas = [1, 2, 4]

        self.correlation = StandardAntennaCorrelation(0, CorrelationType.LOW, device=self.device)

    def test_device_type_setget(self) -> None:
        """Device type property getter should return setter argument"""

        self.correlation.device_type = DeviceType.BASE_STATION
        self.assertIs(DeviceType.BASE_STATION, self.correlation.device_type)

    def test_correlation_setget(self) -> None:
        """Correlation type property getter should return setter argument"""

        self.correlation.correlation = CorrelationType.HIGH
        self.assertIs(CorrelationType.HIGH, self.correlation.correlation)

        self.correlation.correlation = CorrelationType.MEDIUM
        self.assertIs(CorrelationType.MEDIUM, self.correlation.correlation)

    def test_covariance(self) -> None:
        """Test covariance matrix generation"""

        for device_type, correlation_type, num_antennas in product(DeviceType, CorrelationType, self.num_antennas):
            self.device.num_antennas = num_antennas
            self.correlation.device_type = device_type
            self.correlation.correlation = correlation_type

            covariance = self.correlation.covariance

            self.assertCountEqual([num_antennas, num_antennas], covariance.shape)
            self.assertTrue(np.allclose(covariance, covariance.T.conj()))  # Hermitian check

    def test_covariance_validation(self) -> None:
        """Covariance matrix generation should exceptions on invalid parameters"""

        with self.assertRaises(RuntimeError):
            self.device.num_antennas = 5
            _ = self.correlation.covariance

        with self.assertRaises(FloatingError):
            self.correlation.device = None
            _ = self.correlation.covariance


class TestCost259(unittest.TestCase):
    """Test the Cost256 template for the multipath fading channel model."""

    def setUp(self) -> None:
        self.alpha_device = Mock()
        self.beta_device = Mock()
        self.alpha_device.antennas.num_antennas = 1
        self.beta_device.antennas.num_antennas = 1
        self.alpha_device.position = np.array([100, 0, 0])
        self.beta_device.position = np.array([0, 100, 0])
        self.alpha_device.orientation = np.array([0, 0, 0])
        self.beta_device.orientation = np.array([0, 0, pi])

    def test_init(self) -> None:
        """Test the template initializations."""

        for model_type in Cost259Type:
            channel = MultipathFadingCost259(model_type=model_type, alpha_device=self.alpha_device, beta_device=self.beta_device)

            self.assertIs(self.alpha_device, channel.alpha_device)
            self.assertIs(self.beta_device, channel.beta_device)

    def test_init_validation(self) -> None:
        """Template initialization should raise ValueError on invalid model type."""

        with self.assertRaises(ValueError):
            _ = MultipathFadingCost259(100000)

        with self.assertRaises(ValueError):
            _ = MultipathFadingCost259(Cost259Type.HILLY, los_angle=0.0)

    def test_model_type(self) -> None:
        """The model type property should return"""

        for model_type in Cost259Type:
            channel = MultipathFadingCost259(model_type)
            self.assertEqual(model_type, channel.model_type)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        with patch("hermespy.channel.multipath_fading_templates.MultipathFadingCost259.alpha_device", new=PropertyMock) as alpha_device, patch("hermespy.channel.multipath_fading_templates.MultipathFadingCost259.beta_device", new=PropertyMock) as beta_device:
            alpha_device.return_value = self.alpha_device
            beta_device.return_value = self.beta_device

            test_yaml_roundtrip_serialization(self, MultipathFadingCost259(Cost259Type.HILLY), {"num_outputs", "num_inputs"})


class Test5GTDL(unittest.TestCase):
    """Test the 5GTDL template for the multipath fading channel model."""

    def setUp(self) -> None:
        self.rms_delay = 1e-6
        self.alpha_device = Mock()
        self.beta_device = Mock()
        self.alpha_device.antennas.num_antennas = 1
        self.beta_device.antennas.num_antennas = 1
        self.alpha_device.position = np.array([100, 0, 0])
        self.beta_device.position = np.array([0, 100, 0])
        self.alpha_device.orientation = np.array([0, 0, 0])
        self.beta_device.orientation = np.array([0, 0, pi])

    def test_init(self) -> None:
        """Test the template initializations."""

        for model_type in TDLType:
            channel = MultipathFading5GTDL(model_type=model_type, alpha_device=self.alpha_device, beta_device=self.beta_device)

            self.assertIs(self.alpha_device, channel.alpha_device)
            self.assertIs(self.beta_device, channel.beta_device)

    def test_init_validation(self) -> None:
        """Template initialization should raise ValueError on invalid model type."""

        with self.assertRaises(ValueError):
            _ = MultipathFading5GTDL(model_type=100000)

        with self.assertRaises(ValueError):
            _ = MultipathFading5GTDL(rms_delay=-1.0)

        with self.assertRaises(ValueError):
            _ = MultipathFading5GTDL(model_type=TDLType.D, los_doppler_frequency=0.0)

        with self.assertRaises(ValueError):
            _ = MultipathFading5GTDL(model_type=TDLType.E, los_doppler_frequency=0.0)

    def test_model_type(self) -> None:
        """The model type property should return the proper model type."""

        for model_type in TDLType:
            channel = MultipathFading5GTDL(model_type=model_type)
            self.assertEqual(model_type, channel.model_type)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        channel = MultipathFading5GTDL(model_type=TDLType.B, alpha_device=self.alpha_device, beta_device=self.beta_device)

        with patch("hermespy.channel.multipath_fading_templates.MultipathFading5GTDL.alpha_device", new=PropertyMock) as alpha_device, patch("hermespy.channel.multipath_fading_templates.MultipathFading5GTDL.beta_device", new=PropertyMock) as beta_device:
            alpha_device.return_value = self.alpha_device
            beta_device.return_value = self.beta_device

            test_yaml_roundtrip_serialization(self, channel, {"num_outputs", "num_inputs"})


class TestExponential(unittest.TestCase):
    """Test the exponential template for the multipath fading channel model."""

    def setUp(self) -> None:
        self.tap_interval = 1e-5
        self.rms_delay = 1e-8

        self.channel = MultipathFadingExponential(tap_interval=self.tap_interval, rms_delay=self.rms_delay)

    def test_init(self) -> None:
        """Initialization arguments should be properly parsed."""
        pass

    def test_init_validation(self) -> None:
        """Object initialization should raise ValueErrors on negative tap intervals and rms delays."""

        with self.assertRaises(ValueError):
            _ = MultipathFadingExponential(0.0, 1.0)

        with self.assertRaises(ValueError):
            _ = MultipathFadingExponential(1.0, 0.0)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.channel, {"num_outputs", "num_inputs"})
