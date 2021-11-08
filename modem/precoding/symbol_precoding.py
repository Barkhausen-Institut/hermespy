# -*- coding: utf-8 -*-
"""Precoding configuration of communication symbols."""

from __future__ import annotations
from typing import Optional, List, Type, TYPE_CHECKING
from ruamel.yaml import SafeRepresenter, SafeConstructor, Node
from fractions import Fraction
import numpy as np

if TYPE_CHECKING:
    from modem import Modem
    from . import Precoder

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SymbolPrecoding:
    """Channel SymbolPrecoding configuration for wireless transmission of modulated data symbols.

    Symbol precoding may occur as an intermediate step between bit-mapping and base-band symbol modulations.
    In order to account for the possibility of multiple antenna data-streams,
    waveform generators may access the `SymbolPrecoding` configuration to encode one-dimensional symbol
    streams into multi-dimensional symbol streams during transmission and subsequently decode during reception.
    """

    yaml_tag = u'Precoding'
    __modem: Optional[Modem]
    __symbol_precoders: List[Precoder]

    def __init__(self,
                 modem: Modem = None) -> None:
        """Symbol Precoding object initialization.

        Args:
            modem (Modem, Optional):
                The modem this `SymbolPrecoding` configuration is attached to.
        """

        self.modem = modem
        self.__symbol_precoders = []

    @classmethod
    def to_yaml(cls: Type[SymbolPrecoding], representer: SafeRepresenter, node: SymbolPrecoding) -> Node:
        """Serialize a `SymbolPrecoding` configuration to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (SymbolPrecoding):
                The `SymbolPrecoding` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
                None if the object state is default.
        """

        if len(node.__symbol_precoders) < 1:
            return representer.represent_none(None)

        return representer.represent_sequence(cls.yaml_tag, node.__symbol_precoders)

    @classmethod
    def from_yaml(cls: Type[SymbolPrecoding], constructor: SafeConstructor, node: Node) -> SymbolPrecoding:
        """Recall a new `SymbolPrecoding` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `SymbolPrecoding` serialization.

        Returns:
            SymbolPrecoding:
                Newly created `SymbolPrecoding` instance.
        """

        state = constructor.construct_sequence(node, deep=True)
        symbol_precoding = cls()

        symbol_precoding.__symbol_precoders = state

        for precoder in symbol_precoding.__symbol_precoders:
            precoder.precoding = symbol_precoding

        return symbol_precoding

    @property
    def modem(self) -> Modem:
        """Access the modem this SymbolPrecoding configuration is attached to.

        Returns:
            Modem:
                Handle to the modem object.

        Raises:
            RuntimeError: If the SymbolPrecoding configuration is floating.
        """

        if self.__modem is None:
            raise RuntimeError("Trying to access the modem of a floating SymbolPrecoding configuration")

        return self.__modem

    @modem.setter
    def modem(self, modem: Modem) -> None:
        """Modify the modem this SymbolPrecoding configuration is attached to.

        Args:
            modem (Modem):
                Handle to the modem object.
        """

        # if modem is not None and self.__modem is not modem:
        #    raise RuntimeError("Re-attaching a precoder is not permitted")

        self.__modem = modem

    def encode(self, output_stream: np.ndarray) -> np.ndarray:
        """Encode a data symbol stream before transmission.

        Args:
            output_stream (np.ndarray):
                Stream of modulated data symbols feeding into the `Precoder`.

        Returns:
            np.ndarray:
                The encoded data streams.
        """

        # Prepare the stream
        stream = output_stream
        if stream.ndim == 1:
            stream = stream[np.newaxis, :]

        for precoder in self.__symbol_precoders:
            stream = precoder.encode(stream)

        return stream

    def decode(self,
               input_stream: np.ndarray,
               stream_responses: np.ndarray,
               stream_noises: np.ndarray) -> np.array:
        """Decode a data symbol stream after reception.

        Args:

            input_stream (np.ndarray):
                The data streams feeding into the `Precoder` to be decoded.
                The first matrix dimension is the number of streams,
                the second dimension the number of discrete samples within each respective stream.

            stream_responses (np.ndarray):
                The channel impulse response for each data symbol within `input_stream`.
                Identical dimensionality to `input_stream`.

            stream_noises (np.ndarray):
                The noise variances for each data symbol within `input_stream`.
                Identical dimensionality to `input_stream`.

        Returns:
            np.array:
                The decoded data symbols

        Raises:
            ValueError: If dimensions of `stream_responses`, `stream_noises` and `input_streams` do not match.
        """

        if np.any(input_stream.shape != stream_responses.shape) or np.any(input_stream.shape != stream_noises.shape):
            raise ValueError("Dimensions of input_stream and stream_responses must be identical")

        symbols_iteration = input_stream.copy()
        streams_iteration = stream_responses.copy()
        noises_iteration = stream_noises.copy()

        # Recursion through all precoders, each one may update the stream as well as the responses
        for precoder in reversed(self.__symbol_precoders):
            symbols_iteration, stream_responses, noises_iteration = precoder.decode(symbols_iteration,
                                                                                    streams_iteration,
                                                                                    noises_iteration)

        return input_stream.flatten()

    def required_outputs(self, precoder: Precoder) -> int:
        """Query the number output streams of a given precoder within a transmitter.

        Args:
            precoder (Precoder): Handle to the precoder in question.

        Returns:
            int: Number of streams

        Raises:
            ValueError: If the precoder is not registered with this configuration.
        """

        if precoder not in self.__symbol_precoders:
            raise ValueError("Precoder not found in SymbolPrecoding configuration")

        precoder_index = self.__symbol_precoders.index(precoder)

        if precoder_index >= len(self.__symbol_precoders) - 1:
            return self.__modem.num_antennas

        return self.__symbol_precoders[precoder_index + 1].num_input_streams

    def required_inputs(self, precoder: Precoder) -> int:
        """Query the number input streams of a given precoder within a receiver.

        Args:
            precoder (Precoder): Handle to the precoder in question.

        Returns:
            int: Number of streams

        Raises:
            ValueError: If the precoder is not registered with this configuration.
        """

        if precoder not in self.__symbol_precoders:
            raise ValueError("Precoder not found in SymbolPrecoding configuration")

        precoder_index = self.__symbol_precoders.index(precoder)

        if precoder_index <= 0:
            return 1

        return self.__symbol_precoders[precoder_index - 1].num_output_streams

    @property
    def rate(self) -> Fraction:
        """Rate between input symbol slots and output symbol slots.

        For example, a rate of one indicates that no symbols are getting added or removed during precoding.

        Return:
            Fraction: The precoding rate.
        """

        r = Fraction(1, 1)
        for symbol_precoder in self.__symbol_precoders:
            r *= symbol_precoder.rate

        return r

    def __getitem__(self, index: int) -> Precoder:
        """Access a precoder at a given index.

        Args:
            index (int):
                Precoder index.

        Raises:
            ValueError: If the given index is out of bounds.
        """

        return self.__symbol_precoders[index]

    def __setitem__(self, index: int, precoder: Precoder) -> None:
        """Register a precoder within the configuration chain.

        This function automatically register the `SymbolPrecoding` instance to the `Precoder`.

        Args:
            index (int): Position at which to register the precoder.
                Set to -1 to insert at the beginning of the chain.
            precoder (Precoder): The precoder object to be added.
        """

        if index < 0:
            self.__symbol_precoders.insert(0, precoder)

        elif index == len(self.__symbol_precoders):
            self.__symbol_precoders.append(precoder)

        else:
            self.__symbol_precoders[index] = precoder

        precoder.SymbolPrecoding = self
