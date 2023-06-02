# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.core import IdealAntenna, UniformArray
from hermespy.modem import StatedSymbols, DuplexModem
from hermespy.modem.precoding import SymbolPrecoding, SingleCarrier
from hermespy.simulation import SimulatedDevice
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
         
        self.device = SimulatedDevice(antennas=UniformArray(IdealAntenna, 1e-3, (2, 1, 1)))
        self.modem = DuplexModem()
        self.modem.device = self.device
        self.precoding = SymbolPrecoding(self.modem)

        self.precoder = SingleCarrier()
        self.precoding[0] = self.precoder
        
    def test_properties(self) -> None:
        """Properties should return the expected values"""
        
        self.assertEqual(1, self.precoder.num_input_streams)
        self.assertEqual(2, self.precoder.num_output_streams)
        
    def test_encode_validation(self) -> None:
        """Encoding should raise a RuntimeError if the number of output streams is not 1"""
        
        raw_symbols = self.rng.standard_normal((2, 2, 100)) + 1j * self.rng.standard_normal((2, 2, 100))
        states = self.rng.standard_normal((2, 1, 2, 100)) + 1j * self.rng.standard_normal((2, 1, 2, 100))
        
        with self.assertRaises(RuntimeError):
            self.precoder.encode(StatedSymbols(raw_symbols, states))
        
    def test_encode(self) -> None:
        """Encoding should raise a NotImplementedError"""

        raw_symbols = self.rng.standard_normal((1, 2, 100)) + 1j * self.rng.standard_normal((1, 2, 100))
        states = self.rng.standard_normal((1, 2, 2, 100)) + 1j * self.rng.standard_normal((1, 2, 2, 100))
        stated_symbols = StatedSymbols(raw_symbols, states)
        
        encoded_symbols = self.precoder.encode(stated_symbols)
        
        self.assertEqual(2, encoded_symbols.num_streams)

    def test_decode(self) -> None:
        """Decoding should return the expected values"""
        
        raw_symbols = self.rng.standard_normal((1, 2, 100)) + 1j * self.rng.standard_normal((1, 2, 100))
        states = self.rng.standard_normal((2, 2, 100)) + 1j * self.rng.standard_normal((2, 2, 100))
        
        propagated_raw_symbols = states * raw_symbols
        propaged_stated_symbols = StatedSymbols(propagated_raw_symbols, states[:, None, :, :])
        
        decoded_symbols = self.precoder.decode(propaged_stated_symbols)
        
        equalized_decoded_symbols = decoded_symbols.raw / decoded_symbols.states[:, 0, :, :]
        assert_array_almost_equal(raw_symbols, equalized_decoded_symbols)
        
    def test_yaml_roundtrip_serialization(self) -> None:
        """Test YAML roundtrip serialization"""
        
        self.precoder.precoding = None
        test_yaml_roundtrip_serialization(self, self.precoder)
