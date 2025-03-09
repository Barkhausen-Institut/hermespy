# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.modem import StatedSymbols, DFT
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestDFT(TestCase):
    """Test DFT precoder"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.num_symbols = 100
        self.dft = DFT()

    def test_encode_decode(self) -> None:
        """Encoding and decoding should be the identity operation"""

        symbols = self.rng.standard_normal((1, 1, self.num_symbols)) + 1j * self.rng.standard_normal((1, 1, self.num_symbols))
        states = np.ones((1, 1, 1, self.num_symbols), dtype=np.complex128)
        stated_symbols = StatedSymbols(symbols, states)

        encoded_symbols = self.dft.encode_symbols(stated_symbols, 1)
        decoded_symbols = self.dft.decode_symbols(encoded_symbols, 1)

        assert_array_almost_equal(symbols, decoded_symbols.raw)

    def test_properties(self) -> None:
        """Test DFT precoding's static properties"""

        self.assertEqual(1, self.dft.num_transmit_input_symbols)
        self.assertEqual(1, self.dft.num_transmit_output_symbols)
        self.assertEqual(1, self.dft.num_receive_input_symbols)
        self.assertEqual(1, self.dft.num_receive_output_symbols)

    def test_num_transmit_output_streams(self) -> None:
        """Number of output streams should be equal to the number of input streams"""

        self.assertEqual(123, self.dft._num_transmit_input_streams(123))
    
    def test_num_receive_output_streams(self) -> None:
        """Number of output streams should be equal to the number of input streams"""

        self.assertEqual(234, self.dft.num_receive_output_streams(234))
    
    def test_serialization(self) -> None:
        """Test DFT precoding serialization"""

        test_roundtrip_serialization(self, self.dft)
