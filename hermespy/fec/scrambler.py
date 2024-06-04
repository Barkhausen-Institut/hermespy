# -*- coding: utf-8 -*-
"""
==========
Scrambling
==========

Scrambling as a channel coding step masks transmitted bits by a pseudo-random sequence
known at both receiver and transmitter in order to prevent long sequences of identical
bits.
Therefore, most scrambling coding operations do not introduce redundancy to the scrambled bit blocks,
i.e. the code rate is usually :math:`R = 1`.
"""

from __future__ import annotations
from collections import deque
from typing import Optional

import numpy as np

from hermespy.core import Serializable
from .coding import Encoder

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PseudoRandomGenerator:
    """A generator for pseudo-random bit sequences.

    Generators with identical initialization will output identical random sequences.
    Implements pseudo-random generator as described in the 3GPP standard for :footcite:t:`2018:ts138211`.
    """

    __queue_x1: deque
    __queue_x2: deque
    __initial_queue_x1: deque
    __initial_queue_x2: deque

    def __init__(self, init_sequence: np.ndarray, offset: int = 1600) -> None:
        """
        Args:

            init_sequence(np.ndarray):
                A sequence of 31 bits initializing the generator.

            offset(int):
                Gold sequence parameter controlling the sequence offset.
        """

        # The required sequence buffer length is inferred from the length of the init sequence
        m = init_sequence.shape[0]

        # Make sure the buffer / init sequence is at least 4 bits lon
        if m < 4:
            raise ValueError("The init sequence must contain at least 4 bits")

        # Init the first fifo queue as [1 0 0 ... 0]
        self.__queue_x1 = deque(np.zeros(m, dtype=np.int8), m)
        self.__queue_x1.appendleft(1)

        # Init the second fifo queue by the provided init sequence
        self.__queue_x2 = deque(init_sequence, m)

        # Fast-forward the queues to compensate for the offset
        for _ in range(offset - m):
            self.__forward_x1()
            self.__forward_x2()

        # Store the initial queues in order to reset the rng to n = 0
        self.__initial_queue_x1 = self.__queue_x1.copy()
        self.__initial_queue_x2 = self.__queue_x2.copy()

    def generate(self) -> int:
        """Generate the next bit within the pseudo-random sequence.

        Returns:
            int:
                The generated bit.
        """

        return (self.__forward_x1() + self.__forward_x2()) % 2

    def generate_sequence(self, length: int) -> np.ndarray:
        """Generate a new sequence of random numbers.

        Args:
            length(int):
                Length of the sequence to be generated.

        Returns: A numpy array of dimension length containing a sequence of pseudo-random bits.
        """

        sequence = np.empty(length, dtype=int)
        for n in range(length):
            sequence[n] = self.generate()

        return sequence

    def reset(self) -> None:
        """Resets the gernator to its default state.

        This implies reverting the queues back to their original state (at rng position n = 0).
        """

        self.__queue_x1 = self.__initial_queue_x1.copy()
        self.__queue_x2 = self.__initial_queue_x2.copy()

    def __forward_x1(self) -> int:
        x1 = (self.__queue_x1[-3] + self.__queue_x1[0]) % 2

        self.__queue_x1.append(x1)
        return x1

    def __forward_x2(self) -> int:
        x2 = (
            self.__queue_x2[-3] + self.__queue_x2[-2] + self.__queue_x2[-1] + self.__queue_x1[0]
        ) % 2

        self.__queue_x2.append(x2)
        return x2


class Scrambler3GPP(Encoder, Serializable):
    """Scrambler channel coding in the physical up- and down-link standard of the 3GPP.

    See section 7.3.1.1 of the respective technical standard :footcite:t:`2018:ts138211` for details.
    """

    yaml_tag: str = "SCRAMBLER_3GPP"

    __transmit_rng: PseudoRandomGenerator
    __receive_rng: PseudoRandomGenerator
    __default_seed = np.array([0, 1, 0, 1, 1, 0, 1], int)

    def __init__(self, seed: Optional[np.ndarray] = None) -> None:
        """
        Args:

            seed (np.ndarray, optional):
                Seed used to initialize the scrambling sequence generation.
                Must contain a sequence of bits.
        """

        # Init base class (Encoder)
        Encoder.__init__(self)

        # Initialize the pseudo random number generator
        seed = self.__default_seed.copy() if seed is None else seed

        self.__transmit_rng = PseudoRandomGenerator(seed)
        self.__receive_rng = PseudoRandomGenerator(seed)

    def encode(self, data: np.ndarray) -> np.ndarray:
        codeword = self.__transmit_rng.generate_sequence(data.shape[0])
        code = (data + codeword) % 2

        return code

    def decode(self, code: np.ndarray) -> np.ndarray:
        codeword = self.__receive_rng.generate_sequence(code.shape[0])
        data = (code + codeword) % 2

        return data

    @property
    def bit_block_size(self) -> int:
        return 1

    @property
    def code_block_size(self) -> int:
        return 1


class Scrambler80211a(Encoder, Serializable):
    """Scrambler channel coding in the the `802.11a` standard.

    Refer to section 17.3.5.4 of :footcite:t:`80211a:1999` for further details.
    """

    yaml_tag: str = "SCRAMBLER_80211A"

    __seed: np.ndarray
    __queue: deque
    __default_seed: np.ndarray = np.array([0, 1, 0, 1, 1, 0, 1], dtype=int)

    def __init__(self, seed: Optional[np.ndarray] = None) -> None:
        """
        Args:

            seed (np.ndarray, optional):
                Seed used to initialize the scrambling sequence generation.
                Must contain a sequence of 7 bits.
        """

        # Init base class (Encoder)
        Encoder.__init__(self)

        # The default seed is all zeros
        self.__seed = self.__default_seed.copy() if seed is None else seed
        self.__queue = deque(self.__seed, 7)

    @property
    def seed(self) -> np.ndarray:
        """Random sequence generator seed.

        Resets the internal register queue used to generate the scrambling sequence.

        Returns:

            np.ndarray:
                Numpy vector containing the generator seed.
                Must be an array of dimension 7 containing only soft bits.

        Raises:
            ValueError: If `seed` does not contain exactly 7 bits.
        """

        return self.__seed

    @seed.setter
    def seed(self, value: np.ndarray) -> None:
        if value.shape[0] != 7:
            raise ValueError("The seed must contain exactly 7 bit")

        for bit in value:
            if bit != 0 and bit != 1:
                raise ValueError("Only bits (i.e. 0 or 1) represent valid seed fields")

        self.__seed = value
        self.__queue = deque(self.__seed, 7)

    def encode(self, data: np.ndarray) -> np.ndarray:
        code = np.array([self.__scramble_bit(bit) for bit in data], dtype=int)
        return code

    def decode(self, code: np.ndarray) -> np.ndarray:
        data = np.array([self.__scramble_bit(bit) for bit in code], dtype=int)
        return data

    def __forward_code_bit(self) -> int:
        """Generate the next bit in the scrambling sequence.

        The sequence depends on the configured seed.

        Returns:
            int:
                The next bit within the scrambling sequence
        """

        code_bit = (self.__queue[3] + self.__queue[6]) % 2
        self.__queue.appendleft(code_bit)

        return code_bit

    def __scramble_bit(self, bit: int) -> int:
        """Scramble a given bit.

        Args:
            bit(int):
                The bit to be scrambled.

        Returns:
            int: The scrambled bit."""

        return (self.__forward_code_bit() + bit) % 2

    @property
    def bit_block_size(self) -> int:
        return 1

    @property
    def code_block_size(self) -> int:
        return 1
