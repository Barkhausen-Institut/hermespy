from __future__ import annotations
from typing import Type, List
import numpy as np
from ruamel.yaml import RoundTripRepresenter, RoundTripConstructor, Node
from ruamel.yaml.comments import CommentedOrderedMap

from modem.coding.encoder import Encoder


class EncoderManager:

    yaml_tag = 'Encoding'
    _encoders = List[Encoder]

    """Serves as a wrapper class for multiple encoders."""
    def __init__(self) -> None:
        self._encoders: List[Encoder] = []

    @classmethod
    def to_yaml(cls: Type[EncoderManager], representer: RoundTripRepresenter, node: EncoderManager) -> Node:
        """Serialize an EnoderManager to YAML.

        Args:
            representer (RoundTripRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (EncoderManager):
                The EncoderManager instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        return representer.represent_sequence(cls.yaml_tag, node.encoders)

    @classmethod
    def from_yaml(cls: Type[EncoderManager], constructor: RoundTripConstructor, node: Node) -> EncoderManager:
        """Recall a new `EncoderManager` instance from YAML.

        Args:
            constructor (RoundTripConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `EncoderManager` serialization.

        Returns:
            EncoderManager:
                Newly created `EncoderManager` instance.
        """

        manager = cls()
        manager._encoders = constructor.construct_sequence(node, deep=True)

        return manager

    def add_encoder(self, encoder: Encoder) -> None:
        self._encoders.append(encoder)
        self._encoders = sorted(
            self._encoders,
            key=lambda encoder: encoder.data_bits_k)

    @property
    def encoders(self) -> List[Encoder]:
        return self._encoders

    @property
    def code_rate(self) -> float:
        R = 1
        for encoder in self._encoders:
            R *= encoder.data_bits_k / encoder.encoded_bits_n

        return R

    def encode(self, data_bits: List[np.array]) -> List[np.array]:
        encoded_bits: List[np.array] = data_bits
        for encoder in self._encoders:
            encoded_bits = encoder.encode(encoded_bits)

        return encoded_bits

    def decode(self, encoded_bits: List[np.array]) -> List[np.array]:
        decoded_bits: List[np.array] = encoded_bits
        for encoder in reversed(self._encoders):
            decoded_bits = encoder.decode(decoded_bits)

        return decoded_bits

    @property
    def num_input_bits(self) -> int:
        """The number of bits required by the encoder input.

        Returns:
            int:
                The number of input bits required by the encoder input.
        """

        if len(self.encoders) < 1:
            return 1

        return self.encoders[0].source_bits
