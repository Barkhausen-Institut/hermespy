# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.modem import StatedSymbols
from hermespy.modem.precoding import MaximumRatioCombining
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMaximumRatioCombinging(TestCase):
    """Test maximum ratio combining precoder"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        self.precoder = MaximumRatioCombining()
        
    def test_properties(self) -> None:
        """Properties should return the expected values"""
        
        self.assertEqual(-1, self.precoder.num_input_streams)
        self.assertEqual(1, self.precoder.num_output_streams)
        
    def test_encode(self) -> None:
        """Encoding should raise a NotImplementedError"""

        with self.assertRaises(NotImplementedError):
            self.precoder.encode(Mock())
            
    def test_decode_validation(self) -> None:
        """Decoding should raise a RuntimeError if the number of input streams is not 1"""
        
        raw_symbols = self.rng.standard_normal((2, 2, 100)) + 1j * self.rng.standard_normal((2, 2, 100))
        states = self.rng.standard_normal((2, 2, 2, 100)) + 1j * self.rng.standard_normal((2, 2, 2, 100))
        
        with self.assertRaises(RuntimeError):
            self.precoder.decode(StatedSymbols(raw_symbols, states))
            
    def test_decode(self) -> None:
        """Decoding should return the expected values"""
        
        raw_symbols = self.rng.standard_normal((1, 2, 100)) + 1j * self.rng.standard_normal((1, 2, 100))
        states = self.rng.standard_normal((2, 2, 100)) + 1j * self.rng.standard_normal((2, 2, 100))
        
        propagated_raw_symbols = states * raw_symbols
        propaged_stated_symbols = StatedSymbols(propagated_raw_symbols, states[:, None, :, :])
        
        decoded_symbols = self.precoder.decode(propaged_stated_symbols)
        assert_array_almost_equal(raw_symbols, decoded_symbols.raw)
        
    def test_yaml_roundtrip_serialization(self) -> None:
        """Test YAML roundtrip serialization"""
        
        test_yaml_roundtrip_serialization(self, self.precoder)
