# -*- coding: utf-8 -*-

from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from h5py import File

from hermespy.modem import Symbols, StatedSymbols

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSymbols(TestCase):
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        self.raw_symbols = self.rng.normal(size=(3,4,5)) + 1j * self.rng.normal(size=(3,4,5))
        
        self.symbols = Symbols(self.raw_symbols)
        
    def test_hdf_serialization(self) -> None:
        """Serialization to and from HDF5 should yield the correct object reconstruction"""
        
        symbols: Symbols = None
        
        with TemporaryDirectory() as tempdir:
            
            file_location = path.join(tempdir, 'testfile.hdf5')
            
            with File(file_location, 'a') as file:
                
                group = file.create_group('testgroup')
                self.symbols.to_HDF(group)
                
            with File(file_location, 'r') as file:
                
                group = file['testgroup']
                symbols = self.symbols.from_HDF(group)
                
        np.testing.assert_array_equal(self.raw_symbols, symbols.raw)


class TestStatedSymbols(TestCase):
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        self.raw_symbols = self.rng.normal(size=(3,4,5)) + 1j * self.rng.normal(size=(3,4,5))
        self.raw_states = self.rng.normal(size=(3,2,4,5)) + 1j * self.rng.normal(size=(3,2,4,5))
        
        self.symbols = StatedSymbols(self.raw_symbols, self.raw_states)
        
    def test_hdf_serialization(self) -> None:
        """Serialization to and from HDF5 should yield the correct object reconstruction"""
        
        symbols: StatedSymbols = None
        
        with TemporaryDirectory() as tempdir:
            
            file_location = path.join(tempdir, 'testfile.hdf5')
            
            with File(file_location, 'a') as file:
                
                group = file.create_group('testgroup')
                self.symbols.to_HDF(group)
                
            with File(file_location, 'r') as file:
                
                group = file['testgroup']
                symbols = self.symbols.from_HDF(group)
                
        np.testing.assert_array_equal(self.raw_symbols, symbols.raw)
        np.testing.assert_array_equal(self.raw_states, symbols.states)
