# -*- coding: utf-8 -*-
"""Minimum-Mean-Square channel equalization."""

from __future__ import annotations
from typing import Type, Tuple
from itertools import product, repeat

from ruamel.yaml import SafeRepresenter, SafeConstructor, ScalarNode
import numpy as np

from .symbol_precoder import SymbolPrecoder

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MMSEqualizer(SymbolPrecoder):
    """Minimum-Mean-Square channel equalization."""

    yaml_tag: str = u'MMSE'

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

        for idx, (symbol, response, noise) in enumerate(zip(symbol_stream.T,
                                                            np.rollaxis(stream_responses, 1),
                                                            np.rollaxis(stream_noises, 1))):

            noise_covariance = np.diag(noise)
            equalizer = np.linalg.inv(response.T.conj() @ response + noise_covariance) @ response.T.conj()

            equalized_symbols[:, idx] = equalizer @ symbol
            equalized_responses[:, idx, :] = equalizer @ response
            equalized_noises[:, idx] = np.diag(equalized_responses[:, idx, :]).real ** -1 - 1

        return equalized_symbols, equalized_responses, equalized_noises

    @property
    def num_input_streams(self) -> int:
        return self.required_num_output_streams

    @property
    def num_output_streams(self) -> int:
        return self.required_num_output_streams

    @classmethod
    def to_yaml(cls: Type[MMSEqualizer],
                representer: SafeRepresenter,
                node: MMSEqualizer) -> ScalarNode:
        """Serialize an `MMSEqualizer` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (MMSEqualizer):
                The `MMSEqualizer` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        return representer.represent_scalar(cls.yaml_tag, None)

    @classmethod
    def from_yaml(cls: Type[MMSEqualizer],
                  constructor: SafeConstructor,
                  node: ScalarNode) -> MMSEqualizer:
        """Recall a new `MMSEqualizer` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `MMSEqualizer` serialization.

        Returns:
            MMSEqualizer:
                Newly created `MMSEqualizer` instance.
        """

        return cls()
