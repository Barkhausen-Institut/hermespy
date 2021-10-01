from __future__ import annotations
from typing import Type, List, Optional, TYPE_CHECKING
import numpy as np
from ruamel.yaml import SafeRepresenter, SafeConstructor, Node
from math import ceil

if TYPE_CHECKING:
    from modem import Modem
    from . import Encoder


class EncoderManager:
    """Serves as a wrapper class for multiple encoders."""

    yaml_tag = 'Encoding'
    __modem: Optional[Modem]
    _encoders: List[Encoder]

    def __init__(self,
                 modem: Modem = None) -> None:
        """Object initialization.

        Args:
            modem (Modem, optional): The modem this `EncoderManager` belongs to.
        """

        # Default parameters
        self.__modem = None
        self._encoders: List[Encoder] = []

        if modem is not None:
            self.modem = modem

    @classmethod
    def to_yaml(cls: Type[EncoderManager], representer: SafeRepresenter, node: EncoderManager) -> Node:
        """Serialize an EncoderManager to YAML.

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

        if len(node.encoders) < 1:
            return representer.represent_none(None)

        return representer.represent_sequence(cls.yaml_tag, node.encoders)

    @classmethod
    def from_yaml(cls: Type[EncoderManager], constructor: SafeConstructor, node: Node) -> EncoderManager:
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

    @property
    def modem(self) -> Modem:
        """Access the modem this encoding configuration is attached to.

        Returns:
            Modem:
                Handle to the modem object.

        Raises:
            RuntimeError: If the encoding configuration is floating.
        """

        if self.__modem is None:
            raise RuntimeError("Trying to access the modem of a floating encoding configuration")

        return self.__modem

    @modem.setter
    def modem(self, modem: Modem) -> None:
        """Modify the modem this encoding configuration is attached to.

        Args:
            modem (Modem):
                Handle to the modem object.
        """

        if self.__modem is not modem:
            self.__modem = modem

    def add_encoder(self, encoder: Encoder) -> None:
        """Add a new encoder to this configuration.

        Args:
            encoder (Encoder): The new encoder to be added.
        """

        # Register this encoding configuration to the encoder
        encoder.manager = self

        # Add new encoder to the queue of configured encoders
        self._encoders.append(encoder)
        self._encoders = self.__execution_order()

    @property
    def encoders(self) -> List[Encoder]:
        """"""

        return self._encoders

    def encode(self, data_bits: np.array) -> np.array:
        """Encode a stream of source bits.

        By default, the input `data_bits` will be padded with zeros
        to match the next integer multiple of the expected `bit_block_size`.

        Args:
            data_bits (np.array): The data bits to be encoded.

        Returns:
            np.array: The encoded source bits.
        """

        bits: np.ndarray = data_bits.copy()

        for encoder in self._encoders:

            num_input_blocks = int(ceil(bits.shape[0] / encoder.bit_block_size))
            # num_padding_bits = input_bits.shape[0] % encoder.bit_block_size

            bits = encoder.encode(bits)

        return bits

    def decode(self, encoded_bits: List[np.array]) -> List[np.array]:
        decoded_bits: List[np.array] = encoded_bits
        for encoder in reversed(self._encoders):
            decoded_bits = encoder.decode(decoded_bits)

        return decoded_bits

    @property
    def bit_block_size(self) -> int:
        """"The number of resulting bits after decoding / the number of bits required before encoding.

        Returns:
            int: The number of bits.
        """

        if len(self._encoders) < 1:
            return 1

        block_size = self._encoders[0].bit_block_size

        for encoder_index in range(1, (len(self._encoders))):

            repetitions = int(self.encoders[encoder_index].bit_block_size /
                              self.encoders[encoder_index-1].code_block_size)
            block_size *= repetitions

        return block_size

    @property
    def code_block_size(self) -> int:
        """The number of resulting bits after encoding / the number of bits required before decoding.

        Returns:
            int: The number of bits.
        """

        if len(self._encoders) < 1:
            return 1

        return self.encoders[-1].code_block_size

    def __execution_order(self) -> List[Encoder]:
        """Sort the encoders into an order of execution.

        Returns:
            List[Encoder]: A list of encoders in order of transmit execution (reversed receive execution).
        """

        return sorted(self._encoders, key=lambda encoder: encoder.bit_block_size)

    @property
    def rate(self) -> float:
        """Code rate.

        The relation between the number of source bits to the number of code bits.

        Returns:
            float: The code rate.
        """

        code_rate = 1.0
        for encoder in self._encoders:
            code_rate *= encoder.rate

        return code_rate
