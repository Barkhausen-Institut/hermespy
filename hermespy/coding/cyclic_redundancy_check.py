# -*- coding: utf-8 -*-
"""
========================
Cyclic Redundancy Checks
========================

Cyclic Redundancy Check (CRC) channel coding schemes introduce redundancy in order to detect the occurrence
of errors within a block of coded bits after reception.
CRC codings usually only detect errors, they do not correct them.
"""

from __future__ import annotations
from typing import Type

import numpy as np
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode

from hermespy.core.factory import Serializable
from .coding import Encoder

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class CyclicRedundancyCheck(Encoder, Serializable):
    """Cyclic Redundancy Check Mock.
    
    This channel coding step mocks CRC algorithms by appending a random checksum of
    :math:`Q` :meth:`.check_block_size` bits to data bit blocks of size :math:`K_n` :meth:`.bit_block_size`.
    The achieved coding rate is therefore

    .. math::

        R_{n} = \\frac{K_n}{K_n + Q} \\mathrm{.}
    """

    yaml_tag = u'CRC'
    __bit_block_size: int       # Number of bits per encoded block.
    __check_block_size: int     # Number of bits appended to bit blocks.

    def __init__(self,
                 bit_block_size,
                 check_block_size) -> None:
        """
        Args:
            
            bit_block_size (int):
                Number of bits per encoded block.
                
            check_block_size (int):
                Number of bits appended to bit blocks.
        """

        Encoder.__init__(self)
        
        self.bit_block_size = bit_block_size
        self.check_block_size = check_block_size

    def encode(self, data: np.ndarray) -> np.ndarray:
        
        return data.append(self.manager.modem._rng.randint(2, self.__check_block_size))

    def decode(self, code: np.ndarray) -> np.ndarray:
        
        return code[:-self.__check_block_size]

    @property
    def bit_block_size(self) -> int:
        return self.__bit_block_size
    
    @bit_block_size.setter
    def bit_block_size(self, value: int) -> None:
        
        if value < 1:
            raise ValueError("CRC bit block size must be greater or equal to one")
        
        self.__bit_block_size = value
        
    @property
    def check_block_size(self) -> int:
        """Number of appended check bits per bit block.
        
        Returns:
            int: Number of check bits :math:`Q`.


        Raises:
            ValueError: If `check_block_size` is smaller than zero.
        """
        
        return self.__check_block_size
    
    @check_block_size.setter
    def check_block_size(self, value: int) -> None:

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

        :meta private:
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

        :meta private:
        """

        state = constructor.construct_mapping(node)
        return cls(**state)
