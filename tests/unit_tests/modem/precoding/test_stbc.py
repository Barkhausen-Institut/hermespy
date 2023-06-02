# -*- coding: utf-8 -*-

from itertools import product
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.core import IdealAntenna, UniformArray
from hermespy.modem import StatedSymbols, DuplexModem, SymbolPrecoding, Alamouti
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


class TestAlamouti(TestCase):
    """Test alamouti soace time block coding precoder"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
         
        self.device = SimulatedDevice(antennas=UniformArray(IdealAntenna, 1e-3, (2, 1, 1)))
        self.modem = DuplexModem()
        self.modem.device = self.device
        self.precoding = SymbolPrecoding(self.modem)

        self.precoder = Alamouti()
        self.precoding[0] = self.precoder
        
    def test_properties(self) -> None:
        """Properties should return the expected values"""
        
        self.assertEqual(1, self.precoder.num_input_streams)
        self.assertEqual(2, self.precoder.num_output_streams)
        
    def test_encode_validation(self) -> None:
        """Encoding routine should raise errors on invalid calls"""

        raw_symbols = self.rng.standard_normal((2, 2, 100)) + 1j * self.rng.standard_normal((2, 2, 100))
        states = self.rng.standard_normal((2, 1, 2, 100)) + 1j * self.rng.standard_normal((2, 1, 2, 100))
        
        with self.assertRaises(ValueError):
            self.precoder.encode(StatedSymbols(raw_symbols, states))
        
        raw_symbols = self.rng.standard_normal((1, 2, 100)) + 1j * self.rng.standard_normal((1, 2, 100))
        states = self.rng.standard_normal((1, 1, 2, 100)) + 1j * self.rng.standard_normal((1, 1, 2, 100))
        
        self.device.antennas = UniformArray(IdealAntenna, 1e-3, (1, 1, 1))
        with self.assertRaises(RuntimeError):
            self.precoder.encode(StatedSymbols(raw_symbols, states))
            
        self.device.antennas = UniformArray(IdealAntenna, 1e-3, (2, 1, 1))
        raw_symbols = self.rng.standard_normal((1, 1, 100)) + 1j * self.rng.standard_normal((1, 1, 100))
        states = self.rng.standard_normal((1, 1, 1, 100)) + 1j * self.rng.standard_normal((1, 1, 1, 100))
        
        with self.assertRaises(ValueError):
            self.precoder.encode(StatedSymbols(raw_symbols, states))
        
    def test_encode(self) -> None:
        """Test Alamouti MIMO encoding"""
        
        self.device.antennas = UniformArray(IdealAntenna, 1e-3, (2, 1, 1))

        raw_symbols = self.rng.standard_normal((1, 2, 100)) + 1j * self.rng.standard_normal((1, 2, 100))
        states = self.rng.standard_normal((1, 1, 2, 100)) + 1j * self.rng.standard_normal((1, 1, 2, 100))
        stated_symbols = StatedSymbols(raw_symbols, states)
        
        encoded_symbols = self.precoder.encode(stated_symbols)
        
        self.assertEqual(2, encoded_symbols.num_streams)
        
    def test_decode_validation(self) -> None:
        """Decoding routine should raise errors on invalid calls"""
        
        self.device.antennas = UniformArray(IdealAntenna, 1e-3, (2, 1, 1))
        raw_symbols = self.rng.standard_normal((2, 1, 100)) + 1j * self.rng.standard_normal((2, 1, 100))
        states = self.rng.standard_normal((2, 2, 1, 100)) + 1j * self.rng.standard_normal((2, 2, 1, 100))
        
        with self.assertRaises(ValueError):
            self.precoder.decode(StatedSymbols(raw_symbols, states))

    def test_decode(self) -> None:
        """Test Alamouti MIMO decoding"""

        self.device.antennas = UniformArray(IdealAntenna, 1e-3, (2, 1, 1))
        num_blocks = 8
        num_symbols = 2

        raw_symbols = self.rng.standard_normal((1, num_blocks, num_symbols)) + 1j * self.rng.standard_normal((1, num_blocks, num_symbols))
        states = self.rng.standard_normal((1, 1, num_blocks, num_symbols)) + 1j * self.rng.standard_normal((1, 1, num_blocks, num_symbols))
        stated_symbols = StatedSymbols(raw_symbols, states)
        
        encoded_symbols = self.precoder.encode(stated_symbols)
        
        ideal_channel_state = np.zeros((2, 2, num_blocks, num_symbols), dtype=np.complex_)
        ideal_channel_state[0, 0, :, :] = 1.
        ideal_channel_state[1, 1, :, :] = 1j
        
        ideal_received_symbols = encoded_symbols.raw.copy()
        ideal_decoded_symbols = self.precoder.decode(StatedSymbols(ideal_received_symbols, ideal_channel_state))
        
        assert_array_almost_equal(raw_symbols, ideal_decoded_symbols.raw[[0], ::])
        
        channel_state = self.rng.standard_normal((2, 2, int(.5 * num_blocks), num_symbols)) + 1j * self.rng.standard_normal((2, 2, int(.5 * num_blocks), num_symbols))
        channel_state = np.repeat(channel_state, 2, axis=2)  # Make the channel coherent over two symbol blocks
        
        received_encoded_symbols = np.zeros((2, num_blocks, num_symbols), dtype=np.complex_)
        for b, s in product(range(num_blocks), range(num_symbols)):
            received_encoded_symbols[:, b, s] = channel_state[:, :, b, s] @ encoded_symbols.raw[:, b, s]
        
        decoded_symbols = self.precoder.decode(StatedSymbols(received_encoded_symbols, channel_state))
        assert_array_almost_equal(raw_symbols, decoded_symbols.raw[[0], ::])

    def test_yaml_roundtrip_serialization(self) -> None:
        """Test YAML roundtrip serialization"""
        
        self.precoder.precoding = None
        test_yaml_roundtrip_serialization(self, self.precoder)
