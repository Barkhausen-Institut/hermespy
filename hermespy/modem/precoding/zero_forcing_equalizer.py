# -*- coding: utf-8 -*-
"""Zero-Forcing channel equalization."""

from __future__ import annotations
from typing import Type, Tuple

from ruamel.yaml import SafeRepresenter, SafeConstructor, ScalarNode
import numpy as np
from numpy import tensordot

from .symbol_precoder import SymbolPrecoder
from hermespy.channel import Channel

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ZFTimeEqualizer(SymbolPrecoder):
    """Zero-Forcing channel equalization in time domain."""

    yaml_tag: str = u'ZF-Time'

    def __init__(self) -> None:
        """Zero-Forcing channel equalization object initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, symbol_stream: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Equalization can only applied to receiver stream coding configurations")

    def decode(self,
               symbol_stream: np.ndarray,
               stream_responses: np.ndarray,
               stream_noises: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        equalized_symbols = np.empty(symbol_stream.shape, dtype=complex)
        equalized_responses = np.empty(stream_responses.shape, dtype=complex)
        equalized_noises = np.empty(stream_noises.shape, dtype=complex)

        num_symbols = symbol_stream.shape[1]

        # Equalize in space in a first step
        for idx, (symbols, response, noise) in enumerate(zip(symbol_stream,
                                                             stream_responses,
                                                             stream_noises)):

            # Combine the responses of all superimposed transmit antennas for equalization
            response_sum = np.sum(response, axis=1, keepdims=False)

            delay_matrix = Channel.DelayMatrix(response_sum)[:num_symbols, :num_symbols]

            # ToDo: Optimization opportunity, lower triangular multiplied by upper triangular matrix
            # We should be able to skip creating the whole convolution matrix
            inverse = np.linalg.inv(delay_matrix.T.conj() @ delay_matrix)
            equalizer = inverse @ delay_matrix.T.conj()

            equalized_symbols[idx, :] = equalizer @ symbols
            equalized_responses[idx, :, :, :] = tensordot(equalizer, response, axes=(1, 0))
            equalized_noises[idx, :] = noise * np.diag(inverse).real

        return equalized_symbols, equalized_responses, equalized_noises

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


class ZFSpaceEqualizer(SymbolPrecoder):
    """Zero-Forcing channel equalization in space domain."""

    yaml_tag: str = u'ZF-Space'

    def __init__(self) -> None:
        """Zero-Forcing channel equalization object initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, symbol_stream: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Equalization can only applied to receiver stream coding configurations")

    def decode(self,
               symbol_stream: np.ndarray,
               stream_responses: np.ndarray,
               stream_noises: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        equalized_symbols = np.empty(symbol_stream.shape, dtype=complex)
        equalized_responses = np.empty(stream_responses.shape, dtype=complex)
        equalized_noises = np.empty(stream_noises.shape, dtype=complex)

        # Equalize in space in a first step
        for idx, (symbol, response, noise) in enumerate(zip(symbol_stream.T,
                                                            np.rollaxis(stream_responses, 1),
                                                            np.rollaxis(stream_noises, 1))):

            response_sum = np.sum(response, axis=2, keepdims=False)

            inverse = np.linalg.inv(response_sum.T.conj() @ response_sum)
            equalizer = inverse @ response_sum.T.conj()

            equalized_symbols[:, idx] = equalizer @ symbol
            equalized_responses[:, idx, :, :] = equalizer @ response
            equalized_noises[:, idx] = noise * np.diag(inverse).real

        return equalized_symbols, equalized_responses, equalized_noises

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
