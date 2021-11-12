# -*- coding: utf-8 -*-
"""Minimum-Mean-Square channel equalization."""

from __future__ import annotations
from typing import Type, Tuple
from itertools import product, repeat

from ruamel.yaml import SafeRepresenter, SafeConstructor, ScalarNode
import numpy as np
from numpy import tensordot
from numpy.linalg import tensorinv

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


class MMSETimeEqualizer(SymbolPrecoder):
    """Minimum-Mean-Square channel equalization in time domain."""

    yaml_tag: str = u'MMSE-Time'

    def __init__(self) -> None:
        """Minimum-Mean-Square channel equalization object initialization."""

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
            noise_covariance = np.diag(noise)

            # ToDo: Optimization opportunity, lower triangular multiplied by upper triangular matrix
            # We should be able to skip creating the whole convolution matrix
            inverse = np.linalg.inv(delay_matrix.T.conj() @ delay_matrix + noise_covariance)
            equalizer = inverse @ delay_matrix.T.conj()

            equalized_symbols[idx, :] = equalizer @ symbols
            equalized_responses[idx, :, :, :] = tensordot(equalizer, response, axes=(1, 0))
            equalized_noises[idx, :] = 1 / (1 / (noise * np.diag(inverse).real) - 1)

        return equalized_symbols, equalized_responses, equalized_noises

    @property
    def num_input_streams(self) -> int:
        return self.required_num_output_streams

    @property
    def num_output_streams(self) -> int:
        return self.required_num_input_streams

    @classmethod
    def to_yaml(cls: Type[MMSETimeEqualizer],
                representer: SafeRepresenter,
                node: MMSETimeEqualizer) -> ScalarNode:
        """Serialize an `MMSETimeEqualizer` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (MMSETimeEqualizer):
                The `MMSETimeEqualizer` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        return representer.represent_scalar(cls.yaml_tag, None)

    @classmethod
    def from_yaml(cls: Type[MMSETimeEqualizer],
                  constructor: SafeConstructor,
                  node: ScalarNode) -> MMSETimeEqualizer:
        """Recall a new `MMSETimeEqualizer` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `MMSETimeEqualizer` serialization.

        Returns:
            MMSETimeEqualizer:
                Newly created `MMSETimeEqualizer` instance.
        """

        return cls()


class MMSESpaceEqualizer(SymbolPrecoder):
    """Minimum-Mean-Square channel equalization in space domain."""

    yaml_tag: str = u'MMSE-Space'

    def __init__(self) -> None:
        """Minimum-Mean-Square channel equalization object initialization."""

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
            noise_covariance = np.diag(noise)
            response_sum = np.sum(response, axis=2, keepdims=False)

            inverse = np.linalg.inv(response_sum.T.conj() @ response_sum + noise_covariance)
            equalizer = inverse @ response_sum.T.conj()

            equalized_symbols[:, idx] = equalizer @ symbol
            equalized_responses[:, idx, :, :] = equalizer @ response
            equalized_noises[:, idx] = 1 / (1 / (noise * np.diag(inverse).real) - 1)

        return equalized_symbols, equalized_responses, equalized_noises

    @property
    def num_input_streams(self) -> int:
        return self.required_num_output_streams

    @property
    def num_output_streams(self) -> int:
        return self.required_num_input_streams

    @classmethod
    def to_yaml(cls: Type[MMSESpaceEqualizer],
                representer: SafeRepresenter,
                node: MMSESpaceEqualizer) -> ScalarNode:
        """Serialize an `MMSESpaceEqualizer` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (MMSESpaceEqualizer):
                The `MMSESpaceEqualizer` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        return representer.represent_scalar(cls.yaml_tag, None)

    @classmethod
    def from_yaml(cls: Type[MMSESpaceEqualizer],
                  constructor: SafeConstructor,
                  node: ScalarNode) -> MMSESpaceEqualizer:
        """Recall a new `MMSESpaceEqualizer` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `MMSESpaceEqualizer` serialization.

        Returns:
            MMSESpaceEqualizer:
                Newly created `MMSESpaceEqualizer` instance.
        """

        return cls()
