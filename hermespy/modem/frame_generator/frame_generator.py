from abc import ABC, abstractmethod

import numpy as np

from hermespy.core import Serializable
from ..bits_source import BitsSource


class FrameGenerator(ABC, Serializable):
    """Base class for frame generators."""

    @abstractmethod
    def pack_frame(self, source: BitsSource, num_bits: int) -> np.ndarray:
        """Generate a frame of num_bits bits from the given bitsource.

        Args:
            source (BitsSource): payload source.
            num_bits (int): number of bits in the whole resulting frame.

        Return:
            frame (np.ndarray): array of ints with each element beeing an individual bit.
        """
        ...

    @abstractmethod
    def unpack_frame(self, frame: np.ndarray) -> np.ndarray:
        """Extract the original payload from the frame generated with pack_frame.

        Args:
            frame (np.ndarray): array of bits of a frame, generated with pack_frame.

        Return:
            payload (np.ndarray): array of payload bits."""
        ...


class FrameGeneratorStub(FrameGenerator):
    """A dummy placeholder frame generator, packing and unpacking payload without any overhead."""

    yaml_tag = "GeneratorStub"

    def pack_frame(self, source: BitsSource, num_bits: int) -> np.ndarray:
        return source.generate_bits(num_bits)

    def unpack_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame
