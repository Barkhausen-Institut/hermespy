# -*- coding: utf-8 -*-
"""Minimum-Mean-Square channel equalization."""

from __future__ import annotations
from typing import Type, Tuple
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

        snr = (stream_responses * np.conj(stream_responses)) ** 2 / stream_noises
        equalizer = 1 / stream_responses * (snr / (snr + 1.))

        resulting_symbols = symbol_stream * equalizer
        resulting_responses = (snr / (snr + 1.))
        resulting_noises = stream_noises * np.abs(equalizer)**2

        return resulting_symbols, resulting_responses, resulting_noises

    @property
    def num_input_streams(self) -> int:
        return self.required_num_input_streams

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
