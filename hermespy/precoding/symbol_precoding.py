# -*- coding: utf-8 -*-
"""Precoding configuration of communication symbols."""

from __future__ import annotations
from typing import Optional, List, Type, TYPE_CHECKING, Union
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt
from ruamel.yaml import SafeRepresenter, SafeConstructor, Node

from hermespy.channel import ChannelStateInformation

if TYPE_CHECKING:
    from hermespy.modem import Modem
    from .symbol_precoder import SymbolPrecoder

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SymbolPrecoding:
    """Channel SymbolPrecoding configuration for wireless transmission of modulated data symbols.

    Symbol precoding may occur as an intermediate step between bit-mapping and base-band symbol modulations.
    In order to account for the possibility of multiple antenna data-streams,
    waveform generators may access the `SymbolPrecoding` configuration to encode one-dimensional symbol
    streams into multi-dimensional symbol streams during transmission and subsequently decode during reception.

    Attributes:

        __modem (Optional[Modem]):
            Communication modem (transmitter or receiver) this precoding configuration is attached to.

        __symbol_precoders (List[SymbolPrecoder]):
            List of individual precoding steps.
            The full precoding results from a sequential execution of each precoding step.

        debug (bool):
            Debug flag.
            If enabled, the precoding will visualize the individual precoding steps after decoding.
    """

    yaml_tag = u'Precoding'
    __modem: Optional[Modem]
    __symbol_precoders: List[SymbolPrecoder]
    debug: bool

    def __init__(self,
                 modem: Modem = None) -> None:
        """Symbol Precoding object initialization.

        Args:
            modem (Modem, Optional):
                The modem this `SymbolPrecoding` configuration is attached to.
        """

        self.modem = modem
        self.__symbol_precoders = []
        self.debug = False

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
               channel_states: ChannelStateInformation,
               stream_noises: Union[float, np.ndarray]) -> np.array:
        """Decode a data symbol stream after reception.

        Args:

            input_stream (np.ndarray):
                The data streams feeding into the `Precoder` to be decoded.
                The first matrix dimension is the number of streams,
                the second dimension the number of discrete samples within each respective stream.

            channel_states (ChannelStateInformation):
                The channel state estimates for each input symbol within `input_stream`.

            stream_noises (Union[float, np.ndarray]):
                The noise variances for each data symbol within `input_stream`.
                Identical dimensionality to `input_stream`.

        Returns:
            np.array:
                The decoded data symbols

        Raises:
            ValueError: If dimensions of `stream_responses`, `stream_noises` and `input_streams` do not match.
        """

        if input_stream.shape[0] != channel_states.num_receive_streams:
            raise ValueError("Input streams and channel states must have identical number of streams")

        # if input_stream.shape[1] != channel_states.num_symbols:
        #     raise ValueError("Input streams and channel states must have identical number of symbols")

        # If only a nuclear noise variance is provided, expand it to an array
        if isinstance(stream_noises, float) or isinstance(stream_noises, int):
            stream_noises = np.array([[stream_noises]], dtype=float).repeat(input_stream.shape[0], axis=0)\
                .repeat(input_stream.shape[1], axis=1)

        symbols_iteration = input_stream.copy()
        channel_state_iteration = channel_states
        noises_iteration = stream_noises.copy()

        if self.debug:
            fig, ax = plt.subplots(3, 1+len(self.__symbol_precoders), squeeze=False)

            ax[0, 0].set_title("Input")
            ax[0, 0].set_ylabel("Signal")
            ax[0, 0].plot(abs(symbols_iteration.flatten()))
            ax[1, 0].set_ylabel("CSI")
            ax[1, 0].plot(abs(channel_state_iteration.state.sum(axis=1).sum(axis=2).flatten()))
            ax[2, 0].set_ylabel("Noise")
            ax[2, 0].plot(abs(noises_iteration.flatten()))
            i = 0

        # Recursion through all precoders, each one may update the stream as well as the responses
        for precoder in reversed(self.__symbol_precoders):
            symbols_iteration, channel_state_iteration, noises_iteration = precoder.decode(symbols_iteration,
                                                                                           channel_state_iteration,
                                                                                           noises_iteration)

            if self.debug:
                i += 1
                ax[0, i].set_title(precoder.__class__.__name__)
                ax[0, i].plot(abs(symbols_iteration.flatten()))
                ax[1, i].plot(abs(channel_state_iteration.state.sum(axis=1).sum(axis=2).flatten()))
                ax[2, i].plot(abs(noises_iteration.flatten()))

        if self.debug:
            plt.show()

        # Make sure the output stream is one-dimensional
        # A multi-dimensional output stream indicates a invalid precoding configuration
        if symbols_iteration.shape[0] > 1:
            raise RuntimeError("More than one stream resulting from precoding decoding, the configuration is invalid")

        return symbols_iteration.flatten()

    def required_outputs(self, precoder: SymbolPrecoder) -> int:
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

    def required_inputs(self, precoder: SymbolPrecoder) -> int:
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

    def __getitem__(self, index: int) -> SymbolPrecoder:
        """Access a precoder at a given index.

        Args:
            index (int):
                Precoder index.

        Raises:
            ValueError: If the given index is out of bounds.
        """

        return self.__symbol_precoders[index]

    def __setitem__(self, index: int, precoder: SymbolPrecoder) -> None:
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

        precoder.precoding = self

    def __len__(self):
        """Length of the precoding is the number of precoding steps."""

        return len(self.__symbol_precoders)
