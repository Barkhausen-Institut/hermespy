# -*- coding: utf-8 -*-

from unittest import TestCase
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal
from ray.cloudpickle.cloudpickle_fast import dump
from ray.cloudpickle import load

from hermespy.fec import RSCCoding

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRSCCoding(TestCase):
    """Test the RSC coding"""

    def setUp(self) -> None:
        self.data_block_size = 20
        self.code_block_size = 46
        self.poly_A = 13
        self.poly_B = 15
        self.buffer = False
        self.num_attempts = 10

        self.rng = default_rng(42)
        self.coding = RSCCoding(self.data_block_size, self.code_block_size, self.buffer, self.poly_A, self.poly_B)

    def test_encode_decode(self) -> None:
        """Encoding a data block should yield a valid code"""

        for _ in range(self.num_attempts):
            data_block = self.rng.integers(0, 2, self.data_block_size, dtype=np.int32)
            flip_index = self.rng.integers(0, self.code_block_size)

            code_block = self.coding.encode(data_block)
            code_block[flip_index] = not bool(code_block[flip_index])

            decoded_block = self.coding.decode(code_block)
            assert_array_equal(data_block, decoded_block)

    def test_pickle(self) -> None:
        """Pickeling and unpickeling the C++ wrapper"""

        with NamedTemporaryFile() as file:
            dump(self.coding, file)
            file.seek(0)

            coding = load(file)
            self.assertEqual(self.data_block_size, coding.bit_block_size)
            self.assertEqual(self.code_block_size, coding.code_block_size)
