import unittest
import numpy as np
from unittest.mock import Mock

from coding.encoder import Encoder


class StubEncoder(Encoder):
    """Encoder mock for testing only."""

    __block_size: int

    def __init__(self, manager: Mock, block_size: int) -> None:

        Encoder.__init__(self, manager)
        self.__block_size = block_size

    def encode(self, bits: np.array) -> np.array:
        return bits.repeat(2)

    def decode(self, encoded_bits: np.array) -> np.array:
        return encoded_bits[::2]

    @property
    def bit_block_size(self) -> int:
        return self.__block_size

    @property
    def code_block_size(self) -> int:
        return 2 * self.__block_size


class TestEncoder(unittest.TestCase):
    """Test the abstract Encoder base class."""

    def setUp(self) -> None:

        self.bits_in_frame = 100
        self.manager = Mock()
        self.encoder = StubEncoder(self.manager, self.bits_in_frame)

    def test_init(self) -> None:
        """Test that the init properly stores all parameters."""

        self.assertIs(self.encoder.manager, self.manager, "Manager init failed")

    def test_manager(self) -> None:
        """Encoder manager getter must return setter value."""

        manager = Mock()
        self.encoder.manager = manager
        self.assertIs(manager, self.encoder.manager, "Manager get / set failed")

    def test_rate(self) -> None:
        """Rate property check."""

        expected_rate = 0.5
        self.assertAlmostEquals(expected_rate, self.encoder.rate, "Rate produced unexpected value")
