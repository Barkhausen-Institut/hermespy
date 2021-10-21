# -*- coding: utf-8 -*-
"""Repetition Encoder."""

from __future__ import annotations
from typing import Type
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode
import numpy as np

from coding.encoder import Encoder


__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RepetitionEncoder(Encoder):
    """Exemplary implementation of a repetition channel encoder."""

    yaml_tag = 'Repetition'
    __bit_block_size: int
    __repetitions: int

    def __init__(self,
                 bit_block_size: int = 32,
                 repetitions: int = 2) -> None:
        """Object initialization.

        Args:
            bit_block_size (int, optional): The number of input bits per data block.
            repetitions (int, optional): The number of times the input bit block is repeated.

        Raises:
            ValueError: If `bit_block_size` times `repetitions` is smaller than `code_block_size`.
        """

        # Default parameters
        Encoder.__init__(self)
        self.bit_block_size = bit_block_size
        self.repetitions = repetitions

        if self.bit_block_size * repetitions > self.code_block_size:
            raise ValueError("The number of generated bits must be smaller or equal to the configured code block size")

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encodes a single block of bits.

        Args:
            bits (np.ndarray): A block of bits to be encoded by this `Encoder`.

        Returns:
            np.ndarray: The encoded `bits` block.
        """

        code = np.repeat(bits, self.repetitions)
        return code

    def decode(self, encoded_bits: np.ndarray) -> np.ndarray:
        """Decodes a single block of encoded bits.

        Args:
            encoded_bits (np.ndarray): An encoded block of bits.

        Returns:
            np.ndarray: A decoded block of bits.
        """

        code = encoded_bits.reshape((self.repetitions, self.bit_block_size), order='F')
        bits = (np.sum(code, axis=0) / self.repetitions) >= 0.5  # Majority voting

        return bits.astype(int)

    @property
    def bit_block_size(self) -> int:
        """The number of resulting bits after decoding / the number of bits required before encoding.

        Returns:
            int: The number of bits.
        """

        return self.__bit_block_size

    @bit_block_size.setter
    def bit_block_size(self, num_bits: int) -> None:
        """Configure the number of resulting bits after decoding / the number of bits required before encoding.

        Args:
            num_bits (int): The number of bits.

        Raises:
            ValueError: If `num_bits` is smaller than one.
        """

        if num_bits < 1:
            raise ValueError("Number data bits must be greater or equal to one")

        self.__bit_block_size = num_bits

    @property
    def code_block_size(self) -> int:
        """The number of resulting bits after encoding / the number of bits required before decoding.

        Returns:
            int: The number of bits.
        """

        return self.__repetitions * self.__bit_block_size

    @property
    def repetitions(self) -> int:
        """The number of bit repetitions during coding.

        Returns:
            int: The number of bits.
        """

        return self.__repetitions

    @repetitions.setter
    def repetitions(self, num: int) -> None:
        """Configure the number of bit repetitions during coding.

        Args:
            num (int): The number of repetitions.

        Raises:
            ValueError: If `num` is smaller than one.
        """

        if num < 1:
            raise ValueError("The number of data bit repetitions must be at least one")

        self.__repetitions = num

    @classmethod
    def to_yaml(cls: Type[RepetitionEncoder], representer: SafeRepresenter, node: RepetitionEncoder) -> MappingNode:
        """Serialize a `RepetitionEncoder` to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (RepetitionEncoder):
                The `RepetitionEncoder` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            "bit_block_size": node.bit_block_size,
            "repetitions": node.repetitions
        }

        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[RepetitionEncoder], constructor: SafeConstructor, node: MappingNode) -> RepetitionEncoder:
        """Recall a new `RepetitionEncoder` from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `RepetitionEncoder` serialization.

        Returns:
            RepetitionEncoder:
                Newly created `RepetitionEncoder` instance.

        Note that the created instance is floating by default.
        """

        state = constructor.construct_mapping(node)
        return cls(**state)
