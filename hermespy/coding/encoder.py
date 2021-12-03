# -*- coding: utf-8 -*-
"""Encoder."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
import numpy as np

if TYPE_CHECKING:
    from . import EncoderManager


__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Encoder(ABC):
    """This class serves as an abstract class for all encoders.

    All deriving classes must overwrite the `encode(data_bits)` and
    `decode(encoded_bits)` function.
    """

    yaml_tag = u'Encoder'
    __manager: Optional[EncoderManager]

    def __init__(self, manager: EncoderManager = None) -> None:
        """Object initialization.

        Args:
            manager (EncoderManager, optional): The encoding configuration this encoder belongs to.
        """

        # Default settings
        self.__manager = None

        if manager is not None:
            self.manager = manager

    @property
    def manager(self) -> EncoderManager:
        """Access the configuration this encoding step is attached to.

        Returns:
            EncoderManager:
                Handle to the configuration object.

        Raises:
            RuntimeError: If the encoder is floating.
        """

        if self.__manager is None:
            raise RuntimeError("Trying to access the manager of a floating encoding")

        return self.__manager

    @manager.setter
    def manager(self, manager: EncoderManager) -> None:
        """Modify the configuration this encoding step is attached to.

        Args:
            manager (EncoderManager): Handle to the encoding manager.
        """

        if self.__manager is not manager:
            self.__manager = manager

    @abstractmethod
    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encodes a single block of bits.

        Args:
            bits (np.ndarray): A block of bits to be encoded by this `Encoder`.

        Returns:
            np.ndarray: The encoded `bits` block.

        Raises:
            ValueError: If the number of `bits` does not match the `Encoder` requirements.
        """
        ...

    @abstractmethod
    def decode(self, encoded_bits: np.ndarray) -> np.ndarray:
        """Decodes a single block of encoded bits.

        Args:
            encoded_bits (np.ndarray): An encoded block of bits.

        Returns:
            np.ndarray: A decoded block of bits.

        Raises:
            ValueError: If the number of `bits` does not match the `Encoder` requirements.
        """
        ...

    @property
    @abstractmethod
    def bit_block_size(self) -> int:
        """The number of resulting bits after decoding / the number of bits required before encoding.

        Returns:
            int: The number of bits.
        """
        ...

    @property
    @abstractmethod
    def code_block_size(self) -> int:
        """The number of resulting bits after encoding / the number of bits required before decoding.

        Returns:
            int: The number of bits.
        """
        ...

    @property
    def rate(self) -> float:
        """Code rate.

        The relation between the number of source bits to the number of code bits.

        Returns:
            float: The code rate.
        """

        return self.bit_block_size / self.code_block_size
