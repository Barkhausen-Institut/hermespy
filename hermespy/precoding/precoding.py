# -*- coding: utf-8 -*-
"""
=======================
Precoding Configuration
=======================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, overload, List, Type, TYPE_CHECKING, TypeVar, Generic
from fractions import Fraction

from ruamel.yaml import SafeRepresenter, SafeConstructor, Node

from hermespy.core.factory import Serializable

if TYPE_CHECKING:
    from hermespy.modem.modem import BaseModem  # pragma: no cover

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Precoder(ABC):
    """Abstract base class for signal processing algorithms operating on complex data symbols streams.

    A symbol precoder represents one coding step of a full symbol precoding configuration.
    It features the `encoding` and `decoding` routines, meant to encode and decode multidimensional symbol streams
    during transmission and reception, respectively.
    """

    yaml_tag: str
    __precoding: Optional[Precoding]

    def __init__(self) -> None:
        self.__precoding = None

    @property
    def precoding(self) -> Precoding | None:
        """Access the precoding configuration this precoder is attached to.

        Returns:
            Handle to the precoding.
            `None` if the precoder is considered floating.

        Raises:
            RuntimeError: If this precoder is currently floating.
        """

        return self.__precoding

    @precoding.setter
    def precoding(self, precoding: Precoding) -> None:
        """Modify the precoding configuration this precoder is attached to.

        Args:
            precoding (Precoding): Handle to the precoding configuration.
        """

        self.__precoding = precoding

    @property
    @abstractmethod
    def num_input_streams(self) -> int:
        """Required number of input symbol streams for encoding / number of resulting output streams after decoding.

        Returns:
            int:
                The number of symbol streams.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def num_output_streams(self) -> int:
        """Required number of input symbol streams for decoding / number of resulting output streams after encoding.

        Returns:
            int:
                The number of symbol streams.
        """
        ...  # pragma: no cover

    @property
    def required_num_output_streams(self) -> int:
        """Number of output streams required by the precoding configuration.

        Returns:
            int: Required number of output streams.

        Raises:
            RuntimeError: If precoder is not attached to a precoding configuration.
        """

        if self.precoding is None:
            raise RuntimeError("Error trying to access requirements of a floating precoder")

        return self.precoding.required_outputs(self)

    @property
    def required_num_input_streams(self) -> int:
        """Number of input streams required by the precoding configuration.

        Returns:
            int: Required number of input streams.

        Raises:
            RuntimeError: If precoder is not attached to a precoding configuration.
        """

        if self.precoding is None:
            raise RuntimeError("Error trying to access requirements of a floating precoder")

        return self.precoding.required_inputs(self)

    @property
    def rate(self) -> Fraction:
        """Rate between input symbol slots and output symbol slots.

        For example, a rate of one indicates that no symbols are getting added or removed during precoding.

        Return:
            Fraction: The precoding rate.
        """

        return Fraction(1, 1)


PrecoderType = TypeVar("PrecoderType", bound=Precoder)
"""Type of precoder."""


class Precoding(Sequence, Serializable, Generic[PrecoderType]):
    """Channel Precoding configuration for wireless transmission of modulated data symbols.

    Symbol precoding may occur as an intermediate step between bit-mapping and base-band symbol modulations.
    In order to account for the possibility of multiple antenna data-streams,
    waveform generators may access the `Precoding` configuration to encode one-dimensional symbol
    streams into multi-dimensional symbol streams during transmission and subsequently decode during reception.

    Attributes:

        __modem (Optional[Modem]):
            Communication modem (transmitter or receiver) this precoding configuration is attached to.

        __precoders (List[Precoder]):
            List of individual precoding steps.
            The full precoding results from a sequential execution of each precoding step.
    """

    __modem: Optional[BaseModem]
    __precoders: List[PrecoderType]

    def __init__(self, modem: BaseModem = None) -> None:
        """Symbol Precoding object initialization.

        Args:
            modem (Modem, Optional):
                The modem this `Precoding` configuration is attached to.
        """

        self.modem = modem
        self.__precoders = []

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

        state: List[Precoder] = constructor.construct_sequence(node, deep=True)
        precoding = cls()

        precoding.__precoders = state

        for precoder in state:
            precoder.precoding = precoding

        return precoding

    @property
    def modem(self) -> Optional[BaseModem]:
        """Access the modem this Precoding configuration is attached to.

        Returns:  Handle to the modem object.
        """

        return self.__modem

    @modem.setter
    def modem(self, modem: BaseModem) -> None:
        """Modify the modem this Precoding configuration is attached to.

        Args:
            modem (Modem):
                Handle to the modem object.
        """

        # if modem is not None and self.__modem is not modem:
        #    raise RuntimeError("Re-attaching a precoder is not permitted")

        self.__modem = modem

    def required_outputs(self, precoder: PrecoderType) -> int:
        """Query the number output streams of a given precoder within a transmitter.

        Args:
            precoder (Precoder): Handle to the precoder in question.

        Returns:
            int: Number of streams

        Raises:
            ValueError: If the precoder is not registered with this configuration.
        """

        if precoder not in self.__precoders:
            raise ValueError("Precoder not found in Precoding configuration")

        precoder_index = self.__precoders.index(precoder)

        if precoder_index >= len(self.__precoders) - 1:
            if self.modem.transmitting_device is not None:
                return self.modem.transmitting_device.antennas.num_transmit_antennas

            else:
                return self.modem.receiving_device.antennas.num_receive_antennas

        return self.__precoders[precoder_index + 1].num_input_streams

    def required_inputs(self, precoder: PrecoderType) -> int:
        """Query the number input streams of a given precoder within a receiver.

        Args:
            precoder (Precoder): Handle to the precoder in question.

        Returns:
            int: Number of streams

        Raises:
            ValueError: If the precoder is not registered with this configuration.
        """

        if precoder not in self.__precoders:
            raise ValueError("Precoder not found in Precoding configuration")

        precoder_index = self.__precoders.index(precoder)

        if precoder_index <= 0:
            return 1

        return self.__precoders[precoder_index - 1].num_output_streams

    @property
    def rate(self) -> Fraction:
        """Rate between input symbol slots and output symbol slots.

        For example, a rate of one indicates that no symbols are getting added or removed during precoding.

        Return:
            Fraction: The precoding rate.
        """

        r = Fraction(1, 1)
        for symbol_precoder in self.__precoders:
            r *= symbol_precoder.rate

        return r

    @property
    def num_input_streams(self) -> int:
        """Number of input streams required to perform the precoding.

        Returns: The number of inputs.
        """

        if len(self.__precoders) < 1:
            return 1

        return self.__precoders[0].num_input_streams

    @property
    def num_output_streams(self) -> int:
        """Number of output streams resulting from the precoding.

        Returns: Number of outputs
        """

        if len(self.__precoders) < 1:
            return 1

        return self.__precoders[-1].num_output_streams

    @overload
    def __getitem__(self, index: int) -> PrecoderType:
        ...  # pragma: no cover

    @overload
    def __getitem__(self, index: slice) -> List[PrecoderType]:
        ...  # pragma: no cover

    def __getitem__(self, index: int | slice) -> PrecoderType | List[PrecoderType]:
        """Access a precoder at a given index.

        Args:
            index (int | slice):
                Precoder index.

        Raises:
            ValueError: If the given index is out of bounds.
        """

        return self.__precoders[index]

    def __setitem__(self, index: int, precoder: PrecoderType) -> None:
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

    def __len__(self):
        """Length of the precoding is the number of precoding steps."""

        return len(self.__precoders)

    def pop_precoder(self, index: int) -> Precoder:
        """Remove a precoder from the processing chain.

        Args:

            index (int): Index of the precoder to be removed.

        Returns: Handle to the removed precoder.
        """

        return self.__precoders.pop(index)
