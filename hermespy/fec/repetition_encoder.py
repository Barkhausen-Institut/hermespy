# -*- coding: utf-8 -*-
"""
=================
Repetition Coding
=================

Repetition codes are among the most basic channel coding schemes.
The achieve redundancy by repeating all bits within a block during encoding.
"""

from __future__ import annotations

import numpy as np

from hermespy.core import Serializable
from .coding import Encoder

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RepetitionEncoder(Encoder, Serializable):
    """A channel coding scheme based on block-wise repetition of bits.

    During encoding, the repetition encoder repeats a block of :math:`K_n` :meth:`.bit_block_size` bits
    :math:`\\tilde{M}` :meth:`.repetitions` times, leading to a :meth:`.code_block_size` of

    .. math::

       L_n = \\tilde{M} \\cdot K_n

    bits and a coding rate of

    .. math::

       R_n = \\frac{K_n}{L_n} = \\frac{1}{\\tilde{M}} \\mathrm{.}

    Let

    .. math::

       \\mathbf{x}  = \\left[ x_1, x_2, \\dots, x_{K_n} \\right]^\\intercal \\in \\left\\lbrace 0, 1 \\right\\rbrace^{K_n}

    be the vector of input bits and

    .. math::

       \\mathbf{y}  = \\left[ y_1, y_2, \\dots, y_{K_n} \\right]^\\intercal \\in \\left\\lbrace 0, 1 \\right\\rbrace^{L_n}

    be the vector of repeated output bits. The implemented block repetition scheme can be described by

    .. math::

        y_k = x_{k \\mod{K_n}} \\mathrm{,}

    assigning input bits to output bits by index.
    """

    yaml_tag = "Repetition"
    __bit_block_size: int
    __repetitions: int

    def __init__(self, bit_block_size: int = 32, repetitions: int = 3) -> None:
        """
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

    def encode(self, bits: np.ndarray) -> np.ndarray:
        code = np.tile(bits, self.repetitions)
        return code

    def decode(self, encoded_bits: np.ndarray) -> np.ndarray:
        if self.repetitions == 1:
            return encoded_bits

        code = encoded_bits.reshape((self.repetitions, self.bit_block_size))
        # Majority voting
        bits = (np.sum(code, axis=0) / self.repetitions) >= 0.5

        return bits.astype(int)

    @property
    def bit_block_size(self) -> int:
        return self.__bit_block_size

    @bit_block_size.setter
    def bit_block_size(self, num_bits: int) -> None:
        if num_bits < 1:
            raise ValueError("Number data bits must be greater or equal to one")

        self.__bit_block_size = num_bits

    @property
    def code_block_size(self) -> int:
        return self.__repetitions * self.__bit_block_size

    @property
    def repetitions(self) -> int:
        """Number of times the bit block is repeated during encoding.

        Returns:
            int: Number of repetitions :math:`\\tilde{M}`.

        Raises:

            ValueError:
                If `repetitions` is smaller than one.

            ValueError:
                If `repetitions` is even.
        """

        return self.__repetitions

    @repetitions.setter
    def repetitions(self, num: int) -> None:
        if num < 1:
            raise ValueError("The number of data bit repetitions must be at least one")

        if num % 2 == 0:
            raise ValueError("Repetitions must be an uneven integer")

        self.__repetitions = num
