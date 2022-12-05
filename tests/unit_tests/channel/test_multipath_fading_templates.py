# -*- coding: utf-8 -*-
"""Test Multipath Fading Channel Model Templates."""

import unittest
from itertools import product
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from scipy.constants import pi

from hermespy.channel import MultipathFadingCost256, Cost256Type, MultipathFading5GTDL, TDLType, MultipathFadingExponential, DeviceType, CorrelationType, StandardAntennaCorrelation
from hermespy.core.device import FloatingError
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
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
        
        self.correlation.device_type = 1
        self.assertIs(DeviceType.TERMINAL, self.correlation.device_type)
        
        self.correlation.device_type = "BASE_STATION"
        self.assertIs(DeviceType.BASE_STATION, self.correlation.device_type)
        
    def test_device_type_validation(self) -> None:
        """Device type property setter should raise ValueErrors on invalid argument types"""
        
        with self.assertRaises(ValueError):
            self.correlation.device_type = Mock()
    
    def test_correlation_setget(self) -> None:
        """Correlation type property getter should return setter argument"""
        
        self.correlation.correlation = CorrelationType.HIGH
        self.assertIs(CorrelationType.HIGH, self.correlation.correlation)
        
        self.correlation.correlation = "MEDIUM"
        self.assertIs(CorrelationType.MEDIUM, self.correlation.correlation)
        
    def test_correlation_validation(self) -> None:
        """Correlation type property setter should raise ValueErrors on invalid argument types"""
        
        with self.assertRaises(ValueError):
            self.correlation.correlation = Mock()
    
    def test_covariance(self) -> None:
        """Test covariance matrix generation"""
        
        for device_type, correlation_type, num_antennas in product(DeviceType, CorrelationType, self.num_antennas):
            
            self.device.num_antennas = num_antennas
            self.correlation.device_type = device_type
            self.correlation.correlation = correlation_type
            
            covariance = self.correlation.covariance
            
            self.assertCountEqual([num_antennas, num_antennas], covariance.shape)
            self.assertTrue(np.allclose(covariance, covariance.T.conj()))   # Hermitian check

    def test_covariance_validation(self) -> None:
        """Covariance matrix generation should exceptions on invalid parameters"""

        with self.assertRaises(RuntimeError):
            
            self.device.num_antennas = 5
            _ = self.correlation.covariance
            
        with self.assertRaises(FloatingError):
            
            self.correlation.device = None
            _ = self.correlation.covariance
        
        
class TestCost256(unittest.TestCase):
    """Test the Cost256 template for the multipath fading channel model."""

    def setUp(self) -> None:

        self.transmitter = Mock()
        self.receiver = Mock()
        self.transmitter.antennas.num_antennas = 1
        self.receiver.antennas.num_antennas = 1
        self.transmitter.position = np.array([100, 0, 0])
        self.receiver.position = np.array([0, 100, 0])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.receiver.orientation = np.array([0, 0, pi])
        self.sync_offset_low = 3
        self.sync_offset_high = 5

    def test_init(self) -> None:
        """Test the template initializations."""

        for model_type in Cost256Type:

            channel = MultipathFadingCost256(model_type=model_type,
                                             transmitter=self.transmitter,
                                             receiver=self.receiver,
                                             sync_offset_low=self.sync_offset_low,
                                             sync_offset_high=self.sync_offset_high)

            self.assertIs(self.transmitter, channel.transmitter)
            self.assertIs(self.receiver, channel.receiver)
            self.assertEqual(self.sync_offset_low, channel.sync_offset_low)
            self.assertEqual(self.sync_offset_high, channel.sync_offset_high)

    def test_init_validation(self) -> None:
        """Template initialization should raise ValueError on invalid model type."""

        with self.assertRaises(ValueError):
            _ = MultipathFadingCost256(100000)

        with self.assertRaises(ValueError):
            _ = MultipathFadingCost256(Cost256Type.HILLY, los_angle=0.0)

    def test_model_type(self) -> None:
        """The model type property should return """

        for model_type in Cost256Type:

            channel = MultipathFadingCost256(model_type)
            self.assertEqual(model_type, channel.model_type)

    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.channel.multipath_fading_templates.MultipathFadingCost256.transmitter', new=PropertyMock) as transmitter, \
             patch('hermespy.channel.multipath_fading_templates.MultipathFadingCost256.receiver', new=PropertyMock) as receiver:

            transmitter.return_value = self.transmitter
            receiver.return_value = self.receiver

            test_yaml_roundtrip_serialization(self, MultipathFadingCost256(Cost256Type.HILLY), {'num_outputs', 'num_inputs'})


class Test5GTDL(unittest.TestCase):
    """Test the 5GTDL template for the multipath fading channel model."""

    def setUp(self) -> None:

        self.rms_delay = 1e-6
        self.transmitter = Mock()
        self.receiver = Mock()
        self.transmitter.antennas.num_antennas = 1
        self.receiver.antennas.num_antennas = 1
        self.transmitter.position = np.array([100, 0, 0])
        self.receiver.position = np.array([0, 100, 0])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.receiver.orientation = np.array([0, 0, pi])
        self.sync_offset_low = 3
        self.sync_offset_high = 5

    def test_init(self) -> None:
        """Test the template initializations."""

        for model_type in TDLType:

            channel = MultipathFading5GTDL(model_type,
                                           transmitter=self.transmitter,
                                           receiver=self.receiver,
                                           sync_offset_low=self.sync_offset_low,
                                           sync_offset_high=self.sync_offset_high)

            self.assertIs(self.transmitter, channel.transmitter)
            self.assertIs(self.receiver, channel.receiver)
            self.assertEqual(self.sync_offset_low, channel.sync_offset_low)
            self.assertEqual(self.sync_offset_high, channel.sync_offset_high)

    def test_init_validation(self) -> None:
        """Template initialization should raise ValueError on invalid model type."""

        with self.assertRaises(ValueError):
            _ = MultipathFading5GTDL(100000)

        with self.assertRaises(ValueError):
            _ = MultipathFading5GTDL(rms_delay=-1.0)

        with self.assertRaises(ValueError):
            _ = MultipathFading5GTDL(TDLType.D, los_doppler_frequency=0.0)

        with self.assertRaises(ValueError):
            _ = MultipathFading5GTDL(TDLType.E, los_doppler_frequency=0.0)

    def test_model_type(self) -> None:
        """The model type property should return the proper model type."""

        for model_type in TDLType:

            channel = MultipathFading5GTDL(model_type)
            self.assertEqual(model_type, channel.model_type)

    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        channel = MultipathFading5GTDL(TDLType.B,
                                       transmitter=self.transmitter,
                                       receiver=self.receiver,
                                       sync_offset_low=self.sync_offset_low,
                                       sync_offset_high=self.sync_offset_high)
        
        with patch('hermespy.channel.multipath_fading_templates.MultipathFading5GTDL.transmitter', new=PropertyMock) as transmitter, \
             patch('hermespy.channel.multipath_fading_templates.MultipathFading5GTDL.receiver', new=PropertyMock) as receiver:

            transmitter.return_value = self.transmitter
            receiver.return_value = self.receiver

            test_yaml_roundtrip_serialization(self, channel, {'num_outputs', 'num_inputs'})


class TestExponential(unittest.TestCase):
    """Test the exponential template for the multipath fading channel model."""

    def setUp(self) -> None:

        self.tap_interval = 1e-5
        self.rms_delay = 1e-8

        self.channel = MultipathFadingExponential(tap_interval=self.tap_interval,
                                                  rms_delay=self.rms_delay)

    def test_init(self) -> None:
        """Initialization arguments should be properly parsed."""
        pass

    def test_init_validation(self) -> None:
        """Object initialization should raise ValueErrors on negative tap intervals and rms delays."""

        with self.assertRaises(ValueError):
            _ = MultipathFadingExponential(tap_interval=-1.0)

        with self.assertRaises(ValueError):
            _ = MultipathFadingExponential(rms_delay=-1.0)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.channel, {'num_outputs', 'num_inputs'})
