from __future__ import annotations
from typing import Optional, List, Type, TYPE_CHECKING
from ruamel.yaml import SafeRepresenter, SafeConstructor, Node
import numpy as np

if TYPE_CHECKING:
    from modem import Modem
    from . import Precoder


class Precoding:
    """Channel precoding configuration for wireless transmission of modulated data symbols.
    """

    yaml_tag = 'Precoding'
    __modem: Optional[Modem]
    __precoders: List[Precoder]

    def __init__(self,
                 modem: Modem = None) -> None:
        """Object initialization.

        Args:
            modem (Modem, Optional):
                The modem this `Precoding` configuration belongs to.
        """

        self.__modem = None
        self.__precoders = []

        if modem is not None:
            self.modem = modem

    @classmethod
    def to_yaml(cls: Type[Precoding], representer: SafeRepresenter, node: Precoding) -> Node:
        """Serialize a `Precoding` configuration to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Precoding):
                The `Precoding` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
                None if the object state is default.
        """

        if len(node.__precoders) < 1:
            return representer.represent_none(None)

        return representer.represent_sequence(cls.yaml_tag, node.__precoders)

    @classmethod
    def from_yaml(cls: Type[Precoding], constructor: SafeConstructor, node: Node) -> Precoding:
        """Recall a new `Precoding` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Precoding` serialization.

        Returns:
            Precoding:
                Newly created `Precoding` instance.
        """

        state = constructor.construct_sequence(node, deep=True)

        precoding = Precoding()
        yield precoding

        precoding.__precoders = state

        for precoder in precoding.__precoders:
            precoder.precoding = precoding

    @property
    def modem(self) -> Modem:
        """Access the modem this precoding configuration is attached to.

        Returns:
            Modem:
                Handle to the modem object.

        Raises:
            RuntimeError: If the precoding configuration is floating.
        """

        if self.__modem is None:
            raise RuntimeError("Trying to access the modem of a floating precoding configuration")

        return self.__modem

    @modem.setter
    def modem(self, modem: Modem) -> None:
        """Modify the modem this precoding configuration is attached to.

        Args:
            modem (Modem):
                Handle to the modem object.
        """

        if self.__modem is not modem:
            self.__modem = modem

    def encode(self, output_stream: np.array) -> np.matrix:
        """Encode a data symbol stream before transmission.

        Args:
            output_stream (np.array):
                Stream of modulated data symbols feeding into the `Precoder`.

        Returns:
            np.matrix:
                The encoded data streams.
        """

        # Prepare the stream
        if len(self.__precoders) < 1:
            num_streams = self.__modem.num_antennas
        else:
            num_streams = self.__precoders[0].num_inputs

        stream = output_stream.reshape((num_streams, -1))

        for precoder in self.__precoders:
            stream = precoder.encode(stream)

        return stream

    def decode(self, input_stream: np.matrix) -> np.array:
        """Decode a data symbol stream after reception.

        Args:
            input_stream (np.matrix):
                The data streams feeding into the `Precoder` to be decoded.
                The first matrix dimension is the number of streams,
                the second dimension the number of discrete samples within each respective stream.

        Returns:
            np.array:
                The decoded data symbols
        """

        input_stream = input_stream.copy()

        for precoder in reversed(self.__precoders):
            input_stream = precoder.encode(input_stream)

        return input_stream

    def get_outputs(self, precoder: Precoder) -> int:
        """Query the number output streams of a given precoder within a transmitter.

        Args:
            precoder (Precoder): Handle to the precoder in question.

        Returns:
            int: Number of streams

        Raises:
            ValueError: If the precoder is not registered with this configuration.
        """

        if precoder not in self.__precoders:
            raise ValueError("Precoder not found in precoding configuration")

        precoder_index = self.__precoders.index(precoder)

        if precoder_index >= len(self.__precoders) - 1:
            return self.__modem.num_antennas

        return self.__precoders[precoder_index + 1].num_inputs

    def get_inputs(self, precoder: Precoder) -> int:
        """Query the number input streams of a given precoder within a receiver.

        Args:
            precoder (Precoder): Handle to the precoder in question.

        Returns:
            int: Number of streams

        Raises:
            ValueError: If the precoder is not registered with this configuration.
        """

        if precoder not in self.__precoders:
            raise ValueError("Precoder not found in precoding configuration")

        precoder_index = self.__precoders.index(precoder)

        if precoder_index <= 0:
            return self.__modem.num_antennas

        return self.__precoders[precoder_index - 1].num_outputs

    def __getitem__(self, index: int) -> Precoder:
        """Access a precoder at a given index.

        Args:
            index (int):
                Precoder index.

        Raises:
            ValueError: If the given index is out of bounds.
        """

        return self.__precoders[index]

    def __setitem__(self, index: int, precoder: Precoder) -> None:
        """Register a precoder within the configuration chain.

        This function automatically register the `Precoding` instance to the `Precoder`.

        Args:
            index (int): Position at which to register the precoder.
                Set to -1 to insert at the beginning of the chain.
            precoder (Precoder): The precoder object to be added.
        """

        if index < 0:
            self.__precoders.insert(0, precoder)

        elif index == len(self.__precoders):
            self.__precoders.append(precoder)

        else:
            self.__precoders[index] = precoder

        precoder.precoding = self
