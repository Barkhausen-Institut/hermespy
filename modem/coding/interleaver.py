from __future__ import annotations
from typing import Type
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node
import numpy as np

from modem.coding import Encoder


class Interleaver(Encoder):
    """A bit interleaver.

    TODO: Proper documentation.
    """

    yaml_tag = 'Interleaver'
    __block_size: int
    __interleave_blocks: int

    def __init__(self,
                 block_size: int = None,
                 interleave_blocks: int = None) -> None:
        """Object initialization.

        Args:
            block_size (int, optional): The input / output number of bits the interleaver requires / generates.
            interleave_blocks (int, optional): The number of sections being interleaved.

        Raises:
            ValueError: If `block_size` is not dividable into `interleave_blocks`.
        """

        # Default parameters
        Encoder.__init__(self)
        self.__block_size = 32
        self.__interleave_blocks = 4

        if block_size is not None:
            self.block_size = block_size

        if interleave_blocks is not None:
            self.interleave_blocks = interleave_blocks

        if self.block_size % self.interleave_blocks != 0:
            raise ValueError("The block size must be an integer multiple of the number of interleave blocks")

    @classmethod
    def to_yaml(cls: Type[Interleaver], representer: SafeRepresenter, node: Interleaver) -> Node:
        """Serialize a `Interleaver` encoder to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Interleaver):
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
    def from_yaml(cls: Type[Interleaver], constructor: SafeConstructor, node: Node) -> Interleaver:
        """Recall a new `Interleaver` encoder from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Interleaver` serialization.

        Returns:
            Interleaver:
                Newly created `Interleaver` instance.

        Note that the created instance is floating by default.
        """

        state = constructor.construct_mapping(node)
        return cls(**state)

    def encode(self, bits: np.array) -> np.array:
        """Interleaves a single block of bits.

        Args:
            bits (np.array): A block of bits to be encoded by this `Encoder`.

        Returns:
            np.array: The encoded `bits` block.

        Raises:
            ValueError: If the number of `bits` does not match the `Encoder` requirements.
        """

        return bits.reshape((self.interleave_blocks, -1)).T.flatten()

    def decode(self, encoded_bits: np.array) -> np.array:
        """De-interleaves a single block of encoded bits.

        Args:
            encoded_bits (np.array): An encoded block of bits.

        Returns:
            np.array: A decoded block of bits.

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

        self.interleave_blocks = num_blocks

    @property
    def rate(self) -> float:
        """Code rate.

        The relation between the number of source bits to the number of code bits.
        Always one in proper interleavers.

        Returns:
            float: The code rate.
        """

        return 1.0
