# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from math import ceil
from typing import Optional
from io import BufferedReader

import numpy as np

from hermespy.core.factory import Serializable
from hermespy.core.random_node import RandomNode

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class BitsSource(ABC, RandomNode):
    """Base Class for Arbitrary Streams of Communication Bits.

    Inheriting classes are required to implement the :meth:`.generate_bits` routine.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Args:
            seed (int, optional): Seed used to initialize the pseudo-random number generator.
        """

        RandomNode.__init__(self, seed=seed)

    @abstractmethod
    def generate_bits(self, num_bits: int) -> np.ndarray:
        """Generate a new sequence of bits.

        Args:

            num_bits (int):
                Number of bits to be generated.

        Returns:
            np.ndarray:
                A numpy vector of `num_bits` generated bits.
        """
        ...  # pragma: no cover


class RandomBitsSource(BitsSource, Serializable):
    """Bit stream generator for pseudo-random sequences of bits."""

    yaml_tag = "RandomBits"

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Args:
            seed (int, optional):
                Seed used to initialize the pseudo-random number generator.
        """

        # Initialize base classes
        BitsSource.__init__(self, seed=seed)

    def generate_bits(self, num_bits: int) -> np.ndarray:
        return self._rng.integers(0, 2, size=num_bits, dtype=int)


class StreamBitsSource(BitsSource, Serializable):
    """Bit-stream generator mapping representing file system streams as bit sources."""

    __stream: BufferedReader

    def __init__(self, path: str) -> None:
        """
        Args:

            path (str):
                Path to the stream bits source.
        """

        BitsSource.__init__(self)
        self.__stream = open(path, mode="rb")

    def __del__(self) -> None:
        self.__stream.close()

    def generate_bits(self, num_bits: int) -> np.ndarray:
        num_bytes = int(ceil(num_bits / 8))
        bit_overflow = num_bytes * 8 - num_bits

        if bit_overflow > 0:
            raise RuntimeError("Bit caching not yet supported")

        byte_string = self.__stream.read(num_bytes)
        array = np.unpackbits(np.frombuffer(byte_string, dtype=np.uint8))

        return array
