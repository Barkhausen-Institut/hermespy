# -*- coding: utf-8 -*-
"""Interleaving Encoder."""

from __future__ import annotations
from typing import Type
from ruamel.yaml import SafeConstructor, SafeRepresenter, ScalarNode, MappingNode
import numpy as np

from coding import Encoder


__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class BlockInterleaver(Encoder):
    """A bit block interleaving encoder.

    Attributes:

        __block_size (int):
            The number of bits the interleaver operates on.

        __interleave_blocks (int):
            The number of sub-blocks the interleaver divides `__block_size` in.
    """

    yaml_tag = 'BlockInterleaver'
    __block_size: int
    __interleave_blocks: int

    def __init__(self,
                 block_size: int,
                 interleave_blocks: int) -> None:
        """Block interleaving encoder object initialization.

        Args:

            block_size (int):
                The input / output number of bits the interleaver requires / generates.

            interleave_blocks (int):
                The number of sections being interleaved.

        The block interleaver accepts bit blocks of length `block_size` and divides them into `interleave_blocks`
        sections. Afterwards, bits within the sections will be swapped.
        The first output section will contain the first bits of each input section,
        the second output section the second bits of each input section, and so on.

        Raises:
            ValueError: If `block_size` is not dividable into `interleave_blocks`.
        """

        # Default parameters
        Encoder.__init__(self)
        self.__block_size = block_size
        self.__interleave_blocks = interleave_blocks

        if self.block_size % self.interleave_blocks != 0:
            raise ValueError("The block size must be an integer multiple of the number of interleave blocks")

    @classmethod
    def to_yaml(cls: Type[BlockInterleaver], representer: SafeRepresenter, node: BlockInterleaver) -> MappingNode:
        """Serialize a `Interleaver` encoder to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (BlockInterleaver):
                The `Interleaver` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            "block_size": node.block_size,
            "interleave_blocks": node.interleave_blocks
        }

        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[BlockInterleaver], constructor: SafeConstructor, node: MappingNode) -> BlockInterleaver:
        """Recall a new `Interleaver` encoder from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Interleaver` serialization.

        Returns:
            BlockInterleaver:
                Newly created `Interleaver` instance.

        Note that the created instance is floating by default.
        """

        if isinstance(node, ScalarNode):
            return cls()

        state = constructor.construct_mapping(node)
        return cls(**state)

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Interleaves a single block of bits.

        Args:
            bits (np.ndarray): A block of bits to be encoded by this `Encoder`.

        Returns:
            np.ndarray: The encoded `bits` block.

        Raises:
            ValueError: If the number of `bits` does not match the `Encoder` requirements.
        """

        return bits.reshape((self.interleave_blocks, -1)).T.flatten()

    def decode(self, encoded_bits: np.ndarray) -> np.ndarray:
        """De-interleaves a single block of encoded bits.

        Args:
            encoded_bits (np.ndarray): An encoded block of bits.

        Returns:
            np.ndarray: A decoded block of bits.

        Raises:
            ValueError: If the number of `bits` does not match the `Encoder` requirements.
        """

        return encoded_bits.reshape((-1, self.interleave_blocks)).T.flatten()

    @property
    def bit_block_size(self) -> int:
        """The number of resulting bits after decoding / the number of bits required before encoding.

        Returns:
            int: The number of bits.
        """

        return self.block_size

    @property
    def code_block_size(self) -> int:
        """The number of resulting bits after encoding / the number of bits required before decoding.

        Returns:
            int: The number of bits.
        """

        return self.block_size

    @property
    def block_size(self) -> int:
        """The configured block size.

        Returns:
            int: The number of bits per block.
        """

        return self.__block_size

    @block_size.setter
    def block_size(self, num_bits: int) -> None:
        """Modify the configured block size.

        Args:
            num_bits (int): The number of bits per block.

        Raises:
            ValueError: If the number of bits is less than one.
        """

        if num_bits < 1:
            raise ValueError("The block size must be greater or equal to one")

        self.__block_size = num_bits

    @property
    def interleave_blocks(self) -> int:
        """The configured number of interleaved sections.

        Returns:
            int: The number of interleaved sections.
        """

        return self.__interleave_blocks

    @interleave_blocks.setter
    def interleave_blocks(self, num_blocks: int) -> None:
        """Modify configured number of interleaved sections.

        Args:
            num_blocks (int): The new number of interleaved sections.

        Raises:
            ValueError: If the number of interleaved sections is less than one.
        """

        if num_blocks < 1:
            raise ValueError("The number of interleaved sections must be at least one")

        self.__interleave_blocks = num_blocks

    @property
    def rate(self) -> float:
        """Code rate.

        The relation between the number of source bits to the number of code bits.
        Always one in proper interleavers.

        Returns:
            float: The code rate.
        """

        return 1.0