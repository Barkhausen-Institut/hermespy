# -*- coding: utf-8 -*-

from os import path
from shutil import rmtree
from tempfile import mkdtemp
from unittest import TestCase

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal

from hermespy.fec import LDPCCoding

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestLDPCCoding(TestCase):
    """Test Low Differential Parity Check Coding."""
    
    def setUp(self) -> None:
        
        # Infer the aff3ct folder location of matrix files relative to this test
        h_directory = path.join(
            path.dirname(path.realpath(__file__)), 
            '..', '..', '..', 
            'submodules', 'affect', 'conf', 'dec', 'LDPC'
        )
        self.h_path = path.join(h_directory, 'DEBUG_6_3.alist')
        
        self.g_directory = mkdtemp()
        self.g_path = path.join(self.g_directory, 'test.alist')
        
        self.rng = default_rng(42)
        self.num_attempts = 10
        
        self.coding = LDPCCoding(10, self.h_path, self.g_path, False, 10)
        
    def tearDown(self) -> None:
        
        rmtree(self.g_directory)
        
    def test_encode_decode(self) -> None:
        """Encoding a data block should yield a valid code."""
        
        for i in range(self.num_attempts):
            
            data_block = self.rng.integers(0, 2, self.coding.bit_block_size, dtype=np.int32)
            flip_index = self.rng.integers(0, self.coding.bit_block_size)
            
            code_block = self.coding.encode(data_block)
            code_block[flip_index] = not bool(code_block[flip_index])
            
            decoded_block = self.coding.decode(code_block)
            assert_array_equal(data_block, decoded_block)
