# -*- coding: utf-8 -*-
"""Cyclic Redundancy Check bit encoding."""

from __future__ import annotations
from typing import Type

import numpy as np
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode

from .encoder import Encoder

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class CyclicRedundancyCheck(Encoder):
    """Cyclic Redundancy Check Bit Encoding.
    
    Note that redundancy checking does NOT correct errors!
    
    Attributes:
        
        __bit_block_size (int):
            Number of bits per encoded block.
            
        __check_block_size (int):
            Number of bits appended to bit blocks.
    """

    yaml_tag = u'CRC'
    __bit_block_size: int
    __check_block_size: int

    def __init__(self,
                 bit_block_size: int = 1,
                 check_block_size: int = 0) -> None:
        """Cyclic Redundancy Check initialization.
        
        Args:
            
            bit_block_size (int, optional):
                Number of bits per encoded block.
                
            check_block_size (int, optional):
                Number of bits appended to bit blocks.
        """

        Encoder.__init__(self)
        
        self.bit_block_size = bit_block_size
        self.check_block_size = check_block_size

    def encode(self, data: np.ndarray) -> np.ndarray:
        
        return data.append(self.manager.modem.random_generator.randint(2, self.__check_block_size))

    def decode(self, code: np.ndarray) -> np.ndarray:
        
        return code[:-self.__check_block_size]

    @property
    def bit_block_size(self) -> int:
        return self.__bit_block_size
    
    @bit_block_size.setter
    def bit_block_size(self, value: int) -> None:
        """Modify the bit block size.
        
        Args:
            value (int): New bit block size.
            
        Raises:
            ValueError: If `value` is smaller than one.
        """
        
        if value < 1:
            raise ValueError("CRC bit block size must be greater or equal to one")
        
        self.__bit_block_size = value
        
    @property
    def check_block_size(self) -> int:
        """Number of check bits per bit block.
        
        Returns:
            int: Number of check bits.
        """
        
        return self.__check_block_size
    
    @check_block_size.setter
    def check_block_size(self, value: int) -> None:
        """Modify the number of check bits per bit block.
        
        Args:
            value (int): New number of check bits.
            
        Raises:
            ValueError: If `value` is smaller than zero.
        """
        
        if value < 0:
            raise ValueError("Number of check bits must be greater or equal to zero")
        
        self.__check_block_size = value
        
    @property
    def code_block_size(self) -> int:
        return self.__bit_block_size + self.__check_block_size

    @classmethod
    def to_yaml(cls: Type[CyclicRedundancyCheck],
                representer: SafeRepresenter,
                node: CyclicRedundancyCheck) -> MappingNode:
        """Serialize a `CyclicRedundancyCheck` to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (CyclicRedundancyCheck):
                The `CyclicRedundancyCheck` instance to be serialized.

        Returns:
            MappingNode:
                The serialized YAML node.
        """

        state = {
            'bit_block_size': node.__bit_block_size,
            'check_block_size': node.__check_block_size,
        }

        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[CyclicRedundancyCheck],
                  constructor: SafeConstructor,
                  node: MappingNode) -> CyclicRedundancyCheck:
        """Recall a new `CyclicRedundancyCheck` from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `CyclicRedundancyCheck` serialization.

        Returns:
            CyclicRedundancyCheck:
                Newly created `CyclicRedundancyCheck` instance.

        Note that the created instance is floating by default.
        """

        state = constructor.construct_mapping(node)
        return cls(**state)
