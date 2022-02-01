# -*- coding: utf-8 -*-
"""
===================
Precoding Step Base
===================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Type, Tuple
from fractions import Fraction

import numpy as np
from ruamel.yaml import SafeConstructor, SafeRepresenter, ScalarNode

from . import SymbolPrecoding
from hermespy.channel import ChannelStateInformation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SymbolPrecoder(ABC):
    """Abstract base class for signal processing algorithms operating on complex data symbols streams.

    A symbol precoder represents one coding step of a full symbol precoding configuration.
    It features the `encoding` and `decoding` routines, meant to encode and decode multidimensional symbol streams
    during transmission and reception, respectively.
    """

    yaml_tag: str
    __precoding: Optional[SymbolPrecoding]

    def __init__(self) -> None:
        """Symbol Precoder initialization."""

        self.__precoding = None

    @property
    def precoding(self) -> SymbolPrecoding:
        """Access the precoding configuration this precoder is attached to.

        Returns:
            SymbolPrecoding: Handle to the precoding.

        Raises:
            RuntimeError: If this precoder is currently floating.
        """

        if self.__precoding is None:
            raise RuntimeError("Trying to access the precoding of a floating precoder")

        return self.__precoding

    @precoding.setter
    def precoding(self, precoding: SymbolPrecoding) -> None:
        """Modify the precoding configuration this precoder is attached to.
        
        Args:
            precoding (SymbolPrecoding): Handle to the precoding configuration.
        """

        self.__precoding = precoding

    @abstractmethod
    def encode(self, symbol_stream: np.ndarray) -> np.ndarray:
        """Encode a data stream before transmission.

        This operation may modify the number of streams as well as the number of data symbols per stream.

        Args:
            
            symbol_stream (np.ndarray):
                An MxN matrix of data symbol streams feeding into the precoding step to be encoded.
                The first matrix dimension M represents the number of streams,
                the second dimension N the number of discrete data symbols.

        Returns:
            
            np.ndarray:
                A matrix of M'xN' encoded data symbol streams.
                The first matrix dimension M' represents the number of streams after encoding,
                the second dimension N' the number of discrete data symbols after encoding.

        Raises:
            NotImplementedError: If an encoding operation is not supported
        """
        ...

    @abstractmethod
    def decode(self,
               symbol_stream: np.ndarray,
               channel_state: ChannelStateInformation,
               stream_noises: np.ndarray) -> Tuple[np.ndarray, ChannelStateInformation, np.ndarray]:
        """Decode a data stream before reception.

        This operation may modify the number of streams as well as the number of data symbols per stream.

        Args:

            symbol_stream (np.ndarray):
                An MxN matrix of data symbol streams feeding into the precoding step to be decoded.
                The first matrix dimension M represents the number of streams,
                the second dimension N the number of discrete data symbols.

            channel_state (ChannelStateInformation):
                The channel state estimates for each input symbol within `input_stream`.

            stream_noises (np.ndarray):
                The noise variances for each data symbol within `symbol_stream`.
                Identical dimensionality to `input_stream`.

        Returns:

            np.ndarray:
                A matrix of M'xN' decoded data symbol streams.
                The first matrix dimension M' represents the number of streams after decoding,
                the second dimension N' the number of discrete data symbols after decoding.

            ChannelStateInformation:
                Updated channel state information after decoding.

            np.ndarray:
                A matrix of M'xN' data symbol noise estimations after this decoding step.

        Raises:
            NotImplementedError: If a decoding operation is not supported
        """
        ...

    @property
    @abstractmethod
    def num_input_streams(self) -> int:
        """Required number of input symbol streams for encoding / number of resulting output streams after decoding.

        Returns:
            int:
                The number of symbol streams.
        """
        ...

    @property
    @abstractmethod
    def num_output_streams(self) -> int:
        """Required number of input symbol streams for decoding / number of resulting output streams after encoding.

        Returns:
            int:
                The number of symbol streams.
        """
        ...

    @property
    def required_num_output_streams(self) -> int:
        """Number of output streams required by the precoding configuration.

        Returns:
            int: Required number of output streams.

        Raises:
            RuntimeError: If precoder is not attached to a precoding configuration.
        """

        if self.__precoding is None:
            raise RuntimeError("Error trying to access requirements of a floating precoder")

        return self.precoding.required_outputs(self)

    @property
    def required_num_input_streams(self) -> int:
        """Number of input streams required by the precoding configuration.

        Returns:
            int: Required number of input streams.

        Raises:
            RuntimeError: If precoder is not attached to a precoding configuration.
        """

        if self.__precoding is None:
            raise RuntimeError("Error trying to access requirements of a floating precoder")

        return self.precoding.required_inputs(self)

    @property
    def rate(self) -> Fraction:
        """Rate between input symbol slots and output symbol slots.

        For example, a rate of one indicates that no symbols are getting added or removed during precoding.

        Return:
            Fraction: The precoding rate.
        """

        return Fraction(1, 1)

    @classmethod
    def to_yaml(cls: Type[SymbolPrecoder],
                representer: SafeRepresenter,
                node: SymbolPrecoder) -> ScalarNode:
        """Serialize an `SymbolPrecoder` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (SymbolPrecoder):
                The `SymbolPrecoder` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        return representer.represent_scalar(cls.yaml_tag, None)

    @classmethod
    def from_yaml(cls: Type[SymbolPrecoder],
                  constructor: SafeConstructor,
                  node: ScalarNode) -> SymbolPrecoder:
        """Recall a new `SymbolPrecoder` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `SymbolPrecoder` serialization.

        Returns:
            SymbolPrecoder:
                Newly created `SymbolPrecoder` instance.
        """

        return cls()
