from typing import List, Dict
from collections import namedtuple

import numpy as np
from numpy import random as rnd

ErrorStats = namedtuple(
    'ErrorStats',
    'number_of_bits number_of_bit_errors number_of_blocks number_of_block_errors')


class BitsSource:
    """Implements a random bit source, with calculation of error statistics."""

    def __init__(self, rng: rnd.RandomState) -> None:

        self.bits_in_drop: List[np.array] = []

        self._random: rnd.RandomState = rng

    def init_drop(self) -> None:
        self.bits_in_drop.clear()

    def get_bits(self, number_of_bits: int,
                 number_of_blocks: int = 1) -> List[np.array]:
        """Returns a list of bits with each list item being a block.

        These bits are appended to the internal bits vector of the current drop.
        """
        bits_in_frame: List[np.array] = []
        for block in range(number_of_blocks):
            bits_in_frame.append(self._random.randint(2, size=number_of_bits))

        self.bits_in_drop.extend(bits_in_frame)

        return bits_in_frame

    def get_number_of_generated_bits(self) -> Dict[str, int]:
        """returns the number of bits that have been generated in the current drop."""
        number_of_bits = 0
        for block in self.bits_in_drop:
            number_of_bits += block.size

        output = {
            'number_of_bits': number_of_bits,
            'number_of_blocks': len(
                self.bits_in_drop)}
        return output

    def get_number_of_errors(
            self, received_bits: List[np.ndarray]) -> ErrorStats:
        """returns the number of errors in 'received_bits', when compared with the generated bits inside the object.

        Args:
            received_bits(List[np.ndarray]):
                Each list element corresponds to one frame. A frame is np.ndarray
                of size `blocks x bits`.
        """

        number_of_block_errors = 0
        number_of_bit_errors = 0

        number_of_blocks = 0
        number_of_bits = 0

        if len(received_bits) == 1:
            received_blocks = received_bits
        elif received_bits[0].ndim == 1:
            received_blocks = [frame for frame in received_bits]
        else:
            received_blocks = [frame_block.squeeze()
                               for frame in received_bits for frame_block in frame]

        for generated_block, received_block in zip(
                self.bits_in_drop, received_blocks):
            number_of_bit_errors += np.sum(generated_block != received_block)
            number_of_bits += generated_block.size

            number_of_block_errors += not np.array_equal(
                generated_block, received_block)
            number_of_blocks += 1

        output = ErrorStats(
            number_of_bits,
            number_of_bit_errors,
            number_of_blocks,
            number_of_block_errors)

        return output
