# -*- coding: utf-8 -*-
"""Zero-Forcing channel equalization."""

from __future__ import annotations
from typing import Type, Tuple

import numpy as np
from ruamel.yaml import SafeRepresenter, SafeConstructor, ScalarNode
from numpy import tensordot
from sparse import tensordot
from scipy.linalg.decomp_svd import svd

from hermespy.channel import ChannelStateInformation
from hermespy.core.factory import Serializable
from .symbol_precoder import SymbolPrecoder


__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ZFTimeEqualizer(SymbolPrecoder, Serializable):
    """Zero-Forcing channel equalization in time domain."""

    yaml_tag: str = u'ZF-Time'

    def __init__(self) -> None:
        """Zero-Forcing channel equalization object initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, symbol_stream: np.ndarray) -> np.ndarray:

        # Stub
        return symbol_stream

    def decode(self,
               symbol_stream: np.ndarray,
               channel_state: ChannelStateInformation,
               stream_noises: np.ndarray) -> Tuple[np.ndarray, ChannelStateInformation, np.ndarray]:

        equalized_symbols = np.empty((channel_state.num_receive_streams, channel_state.num_samples), dtype=complex)
        equalized_noises = np.empty((channel_state.num_receive_streams, channel_state.num_samples), dtype=float)
        equalized_channel_state = ChannelStateInformation(channel_state.state_format)

        # Equalize in space in a first step
        for idx, (symbols, stream_state, noise) in enumerate(zip(symbol_stream,
                                                                 channel_state.received_streams(),
                                                                 stream_noises)):

            # Combine the responses of all superimposed transmit antennas for equalization
            linear_state = stream_state.linear
            transform = np.sum(linear_state[0, ::], axis=0, keepdims=False)

            # Compute the pseudo-inverse from the singular-value-decomposition of the linear channel transform
            # noinspection PyTupleAssignmentBalance
            u, s, vh = svd(transform.todense(), full_matrices=False, check_finite=False)
            u /= s

            equalizer = (u @ vh).conj().T

            equalized_symbols[idx, :] = equalizer @ symbols
            equalized_csi_slice = tensordot(equalizer, linear_state, axes=(1, 2)).transpose((1, 2, 0, 3))
            equalized_channel_state.append_linear(equalized_csi_slice, 0)
            equalized_noises[idx, :] = noise[:stream_state.num_samples] * s

        return equalized_symbols, channel_state, equalized_noises

    @property
    def num_input_streams(self) -> int:
        return self.required_num_output_streams

    @property
    def num_output_streams(self) -> int:
        return self.required_num_input_streams

    @classmethod
    def to_yaml(cls: Type[ZFTimeEqualizer],
                representer: SafeRepresenter,
                node: ZFTimeEqualizer) -> ScalarNode:
        """Serialize an `ZFTimeEqualizer` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (ZFTimeEqualizer):
                The `ZFTimeEqualizer` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        return representer.represent_scalar(cls.yaml_tag, None)

    @classmethod
    def from_yaml(cls: Type[ZFTimeEqualizer],
                  constructor: SafeConstructor,
                  node: ScalarNode) -> ZFTimeEqualizer:
        """Recall a new `ZFTimeEqualizer` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `ZFTimeEqualizer` serialization.

        Returns:
            ZFTimeEqualizer:
                Newly created `ZFTimeEqualizer` instance.
        """

        return cls()


class ZFSpaceEqualizer(SymbolPrecoder, Serializable):
    """Zero-Forcing channel equalization in space domain."""

    yaml_tag: str = u'ZF-Space'

    def __init__(self) -> None:
        """Zero-Forcing channel equalization object initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, symbol_stream: np.ndarray) -> np.ndarray:

        # Stub
        return symbol_stream

    def decode(self,
               symbol_stream: np.ndarray,
               channel_state: ChannelStateInformation,
               stream_noises: np.ndarray) -> Tuple[np.ndarray, ChannelStateInformation, np.ndarray]:

        for time_idx, (symbols, csi, noise) in enumerate(zip(symbol_stream.T,
                                                             channel_state.samples(),
                                                             stream_noises.T)):

            # Combine the responses of all superimposed transmit antennas for equalization
            transform = np.sum(csi.linear[:, :, 0, :], axis=2, keepdims=False)

            # Compute the pseudo-inverse from the singular-value-decomposition of the linear channel transform
            # noinspection PyTupleAssignmentBalance
            u, s, vh = svd(transform.todense(), full_matrices=False, check_finite=False)
            u /= s

            equalizer = (u @ vh).conj().T

            symbol_stream[:, time_idx] = equalizer @ symbols
            channel_state.state[:, :, time_idx, :] = np.tensordot(equalizer, csi.linear[:, :, 0, :], axes=(1, 0))
            stream_noises[:, time_idx] = noise * s

        return symbol_stream, channel_state, stream_noises

    @property
    def num_input_streams(self) -> int:
        return self.required_num_output_streams

    @property
    def num_output_streams(self) -> int:
        return self.required_num_input_streams

    @classmethod
    def to_yaml(cls: Type[ZFSpaceEqualizer],
                representer: SafeRepresenter,
                node: ZFSpaceEqualizer) -> ScalarNode:
        """Serialize an `ZFSpaceEqualizer` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (ZFSpaceEqualizer):
                The `ZFSpaceEqualizer` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        return representer.represent_scalar(cls.yaml_tag, None)

    @classmethod
    def from_yaml(cls: Type[ZFSpaceEqualizer],
                  constructor: SafeConstructor,
                  node: ScalarNode) -> ZFSpaceEqualizer:
        """Recall a new `ZFSpaceEqualizer` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `ZFSpaceEqualizer` serialization.

        Returns:
            ZFSpaceEqualizer:
                Newly created `ZFSpaceEqualizer` instance.
        """

        return cls()
