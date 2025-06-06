# -*- coding: utf-8 -*-
"""
============
Interleaving
============

The term interleaving describes the channel process of exchanging the bit positions during coding.
This is usually being done in order to distribute the bit errors resulting from a wrong symbol decision
during waveform demodulation over the communication data frame.
Therefore, most interleaving coding operations do not introduce redundancy to the interleaved blocks,
i.e. the code rate is :math:`R = 1`.
"""

from __future__ import annotations
from typing_extensions import override

import numpy as np

from hermespy.core import Serializable, SerializationProcess, DeserializationProcess
from .coding import Encoder

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class BlockInterleaver(Encoder, Serializable):
    """An encoding operation interleaving bits in a block-wise fashion.

    During encoding, the block interleaver divides a block of :math:`K_n` :meth:`.bit_block_size` bits
    into :math:`\\tilde{M}` :meth:`.interleave_blocks` of length :math:`\\tilde{K}`.
    Let

    .. math::

       \\mathbf{x}  = \\left[ x_1, x_2, \\dots, x_{K_n} \\right]^\\intercal \\in \\left\\lbrace 0, 1 \\right\\rbrace^{K_n}

    be the vector of input bits and

    .. math::

       \\mathbf{y}  = \\left[ y_1, y_2, \\dots, y_{K_n} \\right]^\\intercal \\in \\left\\lbrace 0, 1 \\right\\rbrace^{K_n}

    be the vector of interleaved output bits, then

    .. math::

        y_k = x_{(k \\cdot \\tilde{M}) \\mod{K_n}}

    describes the block interleaving scheme.
    """

    __block_size: int  # The number of bits the interleaver operates on
    # The number of sub-blocks the interleaver divides `__block_size` in
    __interleave_blocks: int

    def __init__(self, block_size: int, interleave_blocks: int) -> None:
        """
        Args:

            block_size (int):
                The input / output number of bits the interleaver requires / generates.

            interleave_blocks (int):
                The number of sections being interleaved.

        Raises:

            ValueError:
                If `block_size` is not dividable into `interleave_blocks`.
        """

        # Default parameters
        Encoder.__init__(self)
        self.__block_size = block_size
        self.__interleave_blocks = interleave_blocks

        if self.block_size % self.interleave_blocks != 0:
            raise ValueError(
                "The block size must be an integer multiple of the number of interleave blocks"
            )

    def encode(self, bits: np.ndarray) -> np.ndarray:
        return bits.reshape((self.interleave_blocks, -1)).T.flatten()

    def decode(self, encoded_bits: np.ndarray) -> np.ndarray:
        return encoded_bits.reshape((-1, self.interleave_blocks)).T.flatten()

    @property
    def bit_block_size(self) -> int:
        return self.block_size

    @property
    def code_block_size(self) -> int:
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
        """The number of sub-blocks in which which the input block is divided.

        Returns:
            int: The number of interleaved sections :math:`\\tilde{M}`.

        Raises:
            ValueError: If the number of interleaved sections is less than one.
        """

        return self.__interleave_blocks

    @interleave_blocks.setter
    def interleave_blocks(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError("The number of interleaved sections must be at least one")

        self.__interleave_blocks = num_blocks

    @property
    def rate(self) -> float:
        return 1.0

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.block_size, "block_size")
        process.serialize_integer(self.interleave_blocks, "interleave_blocks")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> BlockInterleaver:
        block_size = process.deserialize_integer("block_size")
        interleave_blocks = process.deserialize_integer("interleave_blocks")

        return cls(block_size, interleave_blocks)
