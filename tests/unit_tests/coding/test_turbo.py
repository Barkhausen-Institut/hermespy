from unittest import TestCase

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal

from hermespy.coding.turbo import RSC
from hermespy.hardware_loop.physical_device import PhysicalDevice


class TestRSC(TestCase):
    """Test Recursive Systematical Convolutional Encoding."""

    def setUp(self) -> None:

        self.rng = default_rng(42)

        self.memory = 4
        self.bit_block_size = 128

        self.rsc = RSC(self.bit_block_size, self.memory)

    def test_encode_decode(self) -> None:
        """Encoding and subsequent decoding should yield the input bit sequence."""

        bit_sequence = self.rng.integers(0, 2, self.bit_block_size)

        encoded_sequence = self.rsc.encode(bit_sequence)
        decoded_sequence = self.rsc.decode(encoded_sequence)

        assert_array_equal(bit_sequence, decoded_sequence)

class Test(PhysicalDevice):
    def trigger(self) -> None:
        pass

    @property
    def carrier_frequency(self) -> float:
        pass


@property
def sampling_rate(self) -> float:
    pass