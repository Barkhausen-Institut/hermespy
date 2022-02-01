# -*- coding: utf-8 -*-
"""
============
Turbo Coding
============

http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.135.9318&rep=rep1&type=pdf
"""
from __future__ import annotations
from typing import List, Tuple, Type

import numpy as np
from ruamel.yaml import SafeConstructor, Node, SafeRepresenter

from .encoder import Encoder
from ..core.factory import Serializable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RSC(Encoder):
    """Recursive Systematical Convolutional Encoding.

    The decoding routine applies the Viterbi algorithm, see :footcite:t:`1967:viterbi`.
    """

    __bit_block_size: int
    __memory: int
    __state_map: List[Tuple[int, int, int]]
    __memory_edges: List[List[Tuple[int]]]

    def __init__(self,
                 bit_block_size: int = 128,
                 memory: int = 4) -> None:

        # Init base class
        Encoder.__init__(self)

        # Init attributes
        self.bit_block_size = bit_block_size
        self.memory = memory

    def encode(self, bits: np.ndarray) -> np.ndarray:

        # Init result container
        encoded_bits = np.empty(2 * len(bits), dtype=np.uint8)

        # Init state
        last_state = 0

        # Encode bits
        for bit_idx, bit in enumerate(bits):

            # Compute state
            state = last_state + bit

            # Query map for state results and next state
            last_state, x, y = self.__state_map[state]

            # Store result tuple in interleaved format
            encoded_bits[2*bit_idx:2*(1+bit_idx)] = (x, y)

        return encoded_bits

    def decode(self, encoded_bits: np.ndarray) -> np.ndarray:

        # Viterbi decoding for the convolutional encoding
        num_decoded_bits = int(.5 * len(encoded_bits))
        decoded_bits = np.empty(num_decoded_bits, dtype=np.int8)

        # Build the trellis diagram
        num_memory_states = 2 ** self.__memory
        trellis_errors = -np.ones((num_memory_states, num_decoded_bits), dtype=int)
        #trellis_errors[0, 0] = 0

        for b in range(0, num_decoded_bits):

            bit_tuple = decoded_bits[2*b:2*(1+b)]

            for m in range(num_memory_states):


                for source_node, _ in self.__memory_edges[m]:

                    distance = np.sum(abs())

                #trellis_errors[m, b] =



        return decoded_bits

    @staticmethod
    def __build_state_map(memory_size: int) -> Tuple[List[Tuple[int, int, int]], List[List[int]]]:
        """Build the state transition map.

        Args:

            memory_size (int):
                Size of the memory buffer.

        Returns:

            List[Tuple[int, int, int]]:
                List of state transitions.
        """

        num_state_bits = (1 + memory_size)
        num_input_states = 2 ** num_state_bits
        num_memory_states = 2 ** memory_size

        state_map: List[Tuple[int, int, int]] = []
        memory_edges: List[List[Tuple[int, int]]] = [[] for _ in range(num_memory_states)]

        for i in range(num_input_states):

            # This might be a little confusing.
            # We convert the state integer to an array of state bits, where the least significant bit is the first
            state_bits = np.flip(np.array(list(format(i, '0' + str(num_state_bits) + 'b')), dtype=np.uint8))

            # Code output tuple
            x = state_bits[0]
            y = (np.sum(state_bits) + state_bits[-1]) % 2

            # Compute output state in bits
            output_state_bits = np.roll(state_bits, 1)
            output_state_bits[0] = 0

            # Compute output state as an integer state index
            output_state = 0
            for bit in np.flip(output_state_bits):
                output_state = (output_state << 1) | bit

            memory_edges[output_state].append((i, 0))
            memory_edges[1+output_state].append((i, 1))

            state_map.append((output_state, x, y))

        return state_map, memory_edges

    @property
    def bit_block_size(self) -> int:

        return self.__bit_block_size

    @bit_block_size.setter
    def bit_block_size(self, value: int) -> None:

        if value < 1:
            raise ValueError("Bit block size must greater or equal to one")

        self.__bit_block_size = value

    @property
    def code_block_size(self) -> int:

        return 2 * self.__bit_block_size

    @property
    def memory(self) -> int:
        """Number of bits in the encoder's memory.

        Returns:
            int: Number of bits.

        Raises:
            ValueError: If number if bits is smaller than zero.
        """

        return self.__memory

    @memory.setter
    def memory(self, value: int) -> None:

        if value < 0:
            raise ValueError("Memory length must be greater or equal to zero")

        self.__memory = value
        self.__state_map, self.__memory_edges = self.__build_state_map(value)

    @classmethod
    def to_yaml(cls: Type[Serializable], representer: SafeRepresenter, node: Serializable) -> Node:
        pass

    @classmethod
    def from_yaml(cls: Type[Serializable], constructor: SafeConstructor, node: Node) -> Serializable:
        pass
