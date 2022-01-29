# -*- coding: utf-8 -*-
"""Spatial Multiplexing encoding step of communication data symbols."""

from __future__ import annotations
from typing import Type, Tuple
from fractions import Fraction

import numpy as np
from ruamel.yaml import SafeConstructor, SafeRepresenter, ScalarNode

from hermespy.core.factory import Serializable
from . import SymbolPrecoder
from hermespy.channel import ChannelStateInformation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SpatialMultiplexing(SymbolPrecoder, Serializable):
    """Spatial Multiplexing data symbol precoding step.

    Takes a on-dimensional input stream and distributes the symbols to multiple output streams.
    Does not support decoding by default!
    """

    yaml_tag: str = u'SM'

    def __init__(self) -> None:
        """Spatial Multiplexing object initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, symbol_stream: np.ndarray) -> np.ndarray:

        if symbol_stream.shape[0] > 1:
            raise ValueError("Spatial multiplexing supports only one-dimensional input symbol streams")

        number_of_streams = self.num_output_streams

        # Distribute data symbols to the stream
        encoded_symbol_stream = np.reshape(symbol_stream, (number_of_streams, -1))   # 'F'

        return encoded_symbol_stream

    def decode(self,
               symbol_stream: np.ndarray,
               channel_state: ChannelStateInformation,
               stream_noises: np.ndarray) -> Tuple[np.ndarray, ChannelStateInformation, np.ndarray]:

        # Collect data symbols from the stream
        symbol_stream = np.reshape(symbol_stream, (1, -1))  # 'F'
        stream_noises = np.reshape(stream_noises, (1, -1))  # 'F'

        state = channel_state.state
        channel_state.state = state.reshape((1, state.shape[1], -1, state.shape[3]))  # 'F'

        return symbol_stream, channel_state, stream_noises

    @property
    def num_input_streams(self) -> int:

        # Accepts only one-dimensional input streams
        return 1

    @property
    def num_output_streams(self) -> int:

        # Always outputs the required number of streams
        return self.required_num_output_streams

    @property
    def rate(self) -> Fraction:

        # Spatial multiplexing distributes the incoming stream symbols
        # equally over the outgoing streams.
        return Fraction(self.num_input_streams, self.num_output_streams)

    @classmethod
    def to_yaml(cls: Type[SpatialMultiplexing],
                representer: SafeRepresenter,
                node: SpatialMultiplexing) -> ScalarNode:
        """Serialize an `SpatialMultiplexing` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (SpatialMultiplexing):
                The `SpatialMultiplexing` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        return representer.represent_scalar(cls.yaml_tag, None)

    @classmethod
    def from_yaml(cls: Type[SpatialMultiplexing],
                  constructor: SafeConstructor,
                  node: ScalarNode) -> SpatialMultiplexing:
        """Recall a new `SpatialMultiplexing` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `SpatialMultiplexing` serialization.

        Returns:
            SpatialMultiplexing:
                Newly created `SpatialMultiplexing` instance.
        """

        return cls()
