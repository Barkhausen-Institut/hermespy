# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.modem import StatedSymbols
from hermespy.modem.precoding import MaximumRatioCombining
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMaximumRatioCombinging(TestCase):
    """Test maximum ratio combining precoder"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.mrc = MaximumRatioCombining()

    def test_properties(self) -> None:
        """Properties should return the expected values"""

        self.assertEqual(1, self.mrc.num_receive_input_symbols)
        self.assertEqual(1, self.mrc.num_receive_output_symbols)

    def test_decode_validation(self) -> None:
        """Decoding should raise a RuntimeError if the number of input streams is not 1"""

        raw_symbols = self.rng.standard_normal((2, 2, 100)) + 1j * self.rng.standard_normal((2, 2, 100))
        states = self.rng.standard_normal((2, 2, 2, 100)) + 1j * self.rng.standard_normal((2, 2, 2, 100))

        with self.assertRaises(RuntimeError):
            self.mrc.decode_symbols(StatedSymbols(raw_symbols, states), 1)

    def test_decode(self) -> None:
        """Decoding should return the expected values"""

        raw_symbols = self.rng.standard_normal((1, 2, 100)) + 1j * self.rng.standard_normal((1, 2, 100))
        states = self.rng.standard_normal((2, 2, 100)) + 1j * self.rng.standard_normal((2, 2, 100))

        propagated_raw_symbols = states * raw_symbols
        propaged_stated_symbols = StatedSymbols(propagated_raw_symbols, states[:, None, :, :])

        decoded_symbols = self.mrc.decode_symbols(propaged_stated_symbols, 1)
        assert_array_almost_equal(raw_symbols, decoded_symbols.raw)

    def test_num_receive_output_streams(self) -> None:
        """Number of output streams should always be 1"""

        self.assertEqual(1, self.mrc.num_receive_output_streams(12))

    def test_serialization(self) -> None:
        """Test maximum ratio combining serialization"""

        test_roundtrip_serialization(self, self.mrc)
