# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.modem import StatedSymbols
from hermespy.modem.precoding import SingleCarrier
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSingleCarrier(TestCase):
    """Test the single carrier precoder"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.precoder = SingleCarrier()

    def test_properties(self) -> None:
        """Properties should return the expected values"""

        self.assertEqual(1, self.precoder.num_receive_input_symbols)
        self.assertEqual(1, self.precoder.num_receive_output_symbols)

    def test_decode(self) -> None:
        """Decoding should return the expected values"""

        raw_symbols = self.rng.standard_normal((1, 2, 100)) + 1j * self.rng.standard_normal((1, 2, 100))
        states = self.rng.standard_normal((2, 2, 100)) + 1j * self.rng.standard_normal((2, 2, 100))

        propagated_raw_symbols = states * raw_symbols
        propaged_stated_symbols = StatedSymbols(propagated_raw_symbols, states[:, None, :, :])

        decoded_symbols = self.precoder.decode_symbols(propaged_stated_symbols, 1)

        equalized_decoded_symbols = decoded_symbols.raw / decoded_symbols.states[:, 0, :, :]
        assert_array_almost_equal(raw_symbols, equalized_decoded_symbols)
        
    def test_num_receive_output_streams(self) -> None:
        """Number of output streams should always be 1"""

        self.assertEqual(1, self.precoder.num_receive_output_streams(13))

    def test_serialization(self) -> None:
        """Test single carrier serialization"""

        self.precoder.precoding = None
        test_roundtrip_serialization(self, self.precoder)
