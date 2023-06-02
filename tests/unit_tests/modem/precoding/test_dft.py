# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.modem import DuplexModem, StatedSymbols, DFT, SymbolPrecoding
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


class TestDFT(TestCase):
    """Test DFT precoder"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        self.num_symbols = 100
        
        self.device = SimulatedDevice()
        self.modem = DuplexModem()
        self.modem.device = self.device
        
        self.precoding = SymbolPrecoding(self.modem)
        self.dft = DFT()
        self.precoding[0] = self.dft
        
    def test_encode_decode(self) -> None:
        """Encoding and decoding should be the identity operation"""
        
        symbols = self.rng.standard_normal((1, 1, self.num_symbols)) + 1j * self.rng.standard_normal((1, 1, self.num_symbols))
        states = np.ones((1, 1, 1, self.num_symbols), dtype=np.complex_)
        stated_symbols = StatedSymbols(symbols, states)
        
        encoded_symbols = self.dft.encode(stated_symbols)
        decoded_symbols = self.dft.decode(encoded_symbols)
        
        assert_array_almost_equal(symbols, decoded_symbols.raw)

    def test_num_input_streams(self) -> None:
        """The number of input streams property should always match the required number of inputs"""

        self.assertEqual(1, self.dft.num_input_streams)
        
    def test_num_output_streams(self) -> None:
        """The number of output streams property should always match the required number of outputs"""

        self.assertEqual(1, self.dft.num_output_streams)

    def test_yaml_serialization(self) -> None:
        """YAML serialization should be possible"""
        
        self.dft.precoding = None
        test_yaml_roundtrip_serialization(self, self.dft)
                              