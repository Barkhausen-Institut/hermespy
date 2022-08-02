# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal

from hermespy.fec import BCHCoding

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestBCHCoding(TestCase):
    """Test Bose-Chaudhuri-Hocquenghem Coding."""
    
    def setUp(self) -> None:
        
        self.data_block_size = 99
        self.code_block_size = 127
        self.power = 4
        
        self.rng = default_rng(42)
        self.num_attempts = 5
        
        self.coding = BCHCoding(self.data_block_size, self.code_block_size, self.power)
        
    def test_encode_decode(self) -> None:
        """Encoding a data block should yield a valid code."""
        
        for _ in range(self.num_attempts):
            
            data_block = self.rng.integers(0, 2, self.data_block_size, dtype=np.int32)
            flip_index = self.rng.integers(0, self.coding.code_block_size)
            
            code_block = self.coding.encode(data_block)
            code_block[flip_index] = not bool(code_block[flip_index])
            
            decoded_block = self.coding.decode(code_block)
            assert_array_equal(data_block, decoded_block)
