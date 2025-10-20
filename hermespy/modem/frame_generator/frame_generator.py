# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from typing_extensions import override

import numpy as np

from hermespy.core import DeserializationProcess, Serializable, SerializationProcess
from ..bits_source import BitsSource

__author__ = "Egor Achkasov"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Egor Achkasov", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FrameGenerator(ABC, Serializable):
    """Base class for frame generators."""

    @abstractmethod
    def pack_frame(
        self, source: BitsSource, num_bits: int
    ) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
        """Generate a frame of num_bits bits from the given bitsource.

        Args:
            source: Payload source.
            num_bits: Number of bits in the whole resulting frame.

        Returns: Array of ints with each element beeing an individual bit.
        """
        ...  # pragma: no cover

    @abstractmethod
    def unpack_frame(
        self, frame: np.ndarray[tuple[int], np.dtype[np.uint8]]
    ) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
        """Extract the original payload from the frame generated with pack_frame.

        Args:
            frame: Array of bits of a frame, generated with pack_frame.

        Returns: Array of payload bits."""
        ...  # pragma: no cover


class FrameGeneratorStub(FrameGenerator):
    """A dummy placeholder frame generator, packing and unpacking payload without any overhead."""

    def pack_frame(
        self, source: BitsSource, num_bits: int
    ) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
        return source.generate_bits(num_bits)

    def unpack_frame(
        self, frame: np.ndarray[tuple[int], np.dtype[np.uint8]]
    ) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
        return frame

    @override
    def serialize(self, process: SerializationProcess) -> None:
        return

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> FrameGeneratorStub:
        return cls()
