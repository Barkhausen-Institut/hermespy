# -*- coding: utf-8 -*-

import unittest
from unittest.mock import Mock, patch, PropertyMock

import numpy as np

from hermespy.modem.precoding import SpatialMultiplexing
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSpatialMultiplexing(unittest.TestCase):

    def setUp(self) -> None:

        # Random rng
        self.generator = np.random.default_rng(42)

        # Precoder to be tested
        self.precoder = SpatialMultiplexing()

        # Mock the precoding configuration
        self.precoding = Mock()
        self.precoding.required_outputs = lambda precoder: 4
        self.precoding.required_inputs = lambda precoder: 1
        self.precoder.precoding = self.precoding
        
    def test_encode(self) -> None:
        """Encoding should return original input unaltered"""

        symbols = Mock()
        self.assertIs(symbols, self.precoder.encode(symbols))
        
    def test_decode(self) -> None:
        """Decoding should return original input unaltered"""
        
        symbols = Mock()
        self.assertIs(symbols, self.precoder.decode(symbols))
        
    def test_num_input_streams(self) -> None:
        """The number of input streams is always one."""

        for num_outputs in [1, 5, 10]:

            self.precoding.required_outputs = lambda precoder: num_outputs
            self.assertEqual(num_outputs, self.precoder.num_input_streams)

    def test_num_output_streams(self) -> None:
        """The number of output streams should always be equal to the number of required output streams."""

        for num_outputs in [1, 5, 10]:

            self.precoding.required_outputs = lambda precoder: num_outputs
            self.assertEqual(num_outputs, self.precoder.num_output_streams)

    def test_rate(self) -> None:
        """The rate should always be the fraction between input and output streams."""

        for num_outputs in [1, 5, 10]:

            self.precoding.required_outputs = lambda precoder: num_outputs
            self.assertEqual(1, float(self.precoder.rate))
    
    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.modem.precoding.spatial_multiplexing.SpatialMultiplexing.precoding', new_callable=PropertyMock) as precoding, \
             patch('hermespy.modem.precoding.spatial_multiplexing.SpatialMultiplexing.property_blacklist', new_callable=PropertyMock) as blacklist:
        
            precoding.return_value = self.precoding
            blacklist.return_value = {'precoding'}
            
            test_yaml_roundtrip_serialization(self, self.precoder, {'precoding',})
