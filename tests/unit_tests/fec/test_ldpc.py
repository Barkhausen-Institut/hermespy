# -*- coding: utf-8 -*-

from os import path
from shutil import rmtree
from tempfile import mkdtemp, NamedTemporaryFile
from unittest import TestCase

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal
from ray.cloudpickle.cloudpickle_fast import dump
from ray.cloudpickle import load

from hermespy.fec import LDPCCoding

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestLDPCCoding(TestCase):
    """Test Low Differential Parity Check Coding."""

    def setUp(self) -> None:
        # Infer the aff3ct folder location of matrix files relative to this test
        self.h_directory = path.join(path.dirname(path.realpath(__file__)), "..", "..", "..", "submodules", "affect", "conf", "dec", "LDPC")
        
        self.h_candidates = [
            'DEBUG_6_3',
            'CCSDS_64_128',
            'MACKAY_504_1008',
            'WIMAX_288_576',
        ]

        self.g_directory = mkdtemp()
        self.g_path = path.join(self.g_directory, "test.alist")

        self.rng = default_rng(42)
        self.num_attempts = 20
        self.num_iterations = 100

    def tearDown(self) -> None:
        rmtree(self.g_directory)

    def test_encode_decode(self) -> None:
        """Encoding a data block should yield a valid code."""
        
        for h_candidate in self.h_candidates:
            with self.subTest(h_candidate=h_candidate):
                h_path = path.join(self.h_directory, h_candidate + ".alist")
                coding = LDPCCoding(self.num_iterations, h_path, self.g_path, True, 10)

                errors = 0
                for _ in range(self.num_attempts):
                    data_block = self.rng.integers(0, 2, coding.bit_block_size, dtype=np.int32)
                    flip_index = self.rng.integers(0, coding.bit_block_size, dtype=np.int32)

                    code_block = coding.encode(data_block)
                    code_block[flip_index] = not bool(code_block[flip_index])

                    decoded_block = coding.decode(code_block)
                    errors += np.sum(data_block != decoded_block)
                
                self.assertGreater(1, errors, msg=f"Too many errors: {errors}")

    def test_pickle(self) -> None:
        """Pickeling and unpickeling the C++ wrapper"""

        coding = LDPCCoding(self.num_iterations, path.join(self.h_directory, self.h_candidates[0] + '.alist'), "", False, 10)

        with NamedTemporaryFile() as file:
            dump(coding, file)
            file.seek(0)

            deserialized_coding = load(file)
            self.assertEqual(self.num_iterations, deserialized_coding.num_iterations)

            # Actuall run a full encoding and decoding with the unpickled object
            data_block = self.rng.integers(0, 2, deserialized_coding.bit_block_size, dtype=np.int32)
            code_block = deserialized_coding.encode(data_block)
            decoded_block = deserialized_coding.decode(code_block)
            
            assert_array_equal(data_block, decoded_block)
