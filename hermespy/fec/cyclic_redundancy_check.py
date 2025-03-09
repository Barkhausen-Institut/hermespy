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
from typing_extensions import override

import numpy as np

from hermespy.core import RandomNode, Serializable, SerializationProcess, DeserializationProcess
from .coding import Encoder

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class CyclicRedundancyCheck(Encoder, RandomNode, Serializable):
    """Cyclic Redundancy Check Mock.

    This channel coding step mocks CRC algorithms by appending a random checksum of
    :math:`Q` :meth:`.check_block_size` bits to data bit blocks of size :math:`K_n` :meth:`.bit_block_size`.
    The achieved coding rate is therefore

    .. math::

        R_{n} = \\frac{K_n}{K_n + Q} \\mathrm{.}
    """

    __bit_block_size: int  # Number of bits per encoded block.
    __check_block_size: int  # Number of bits appended to bit blocks.

    def __init__(self, bit_block_size: int, check_block_size: int) -> None:
        """
        Args:

            bit_block_size (int):
                Number of bits per encoded block.

            check_block_size (int):
                Number of bits appended to bit blocks.
        """

        Encoder.__init__(self)
        RandomNode.__init__(self)
        Serializable.__init__(self)

        self.bit_block_size = bit_block_size
        self.check_block_size = check_block_size

    def encode(self, data: np.ndarray) -> np.ndarray:
        return np.append(data, self._rng.integers(0, 2, self.__check_block_size))

    def decode(self, code: np.ndarray) -> np.ndarray:
        return code[: -self.__check_block_size]

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

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.bit_block_size, "bit_block_size")
        process.serialize_integer(self.check_block_size, "check_block_size")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> CyclicRedundancyCheck:
        return cls(
            bit_block_size=process.deserialize_integer("bit_block_size"),
            check_block_size=process.deserialize_integer("check_block_size"),
        )
