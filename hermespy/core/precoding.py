# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import overload, TypeVar, Generic
from typing_extensions import override

from .factory import DeserializationProcess, Serializable, SerializationProcess
from .signal_model import Signal
from .state import ReceiveState, TransmitState

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


PrecoderType = TypeVar("PrecoderType", bound="Precoder")
"""Type of precoder."""

TransmitEncoderType = TypeVar("TransmitEncoderType", bound="TransmitEncoder")
"""Type of transmit encoder."""

ReceiveDecoderType = TypeVar("ReceiveDecoderType", bound="ReceiveDecoder")
"""Type of receive decoder."""

PrecodingType = TypeVar("PrecodingType", bound="Precoding")
"""Type of precoding."""

TransmitPrecodingType = TypeVar("TransmitPrecodingType", bound="TransmitPrecoding")
"""Type of transmit precoding."""

ReceivePrecodingType = TypeVar("ReceivePrecodingType", bound="ReceivePrecoding")
"""Type of receive precoding."""


class Precoder(Generic[PrecodingType], Serializable):
    """Base class for signal processing algorithms operating on parallel complex data streams."""

    __precoding: PrecodingType | None

    def __init__(self) -> None:
        # Initialize class attributes
        self.__precoding = None

    @property
    def precoding(self) -> PrecodingType | None:
        """Access the precoding configuration this precoder is attached to.

        Returns:
            Handle to the precoding.
            `None` if the precoder is considered floating.

        Raises:
            RuntimeError: If this precoder is currently floating.
        """

        return self.__precoding

    @precoding.setter
    def precoding(self, precoding: PrecodingType) -> None:
        """Modify the precoding configuration this precoder is attached to.

        Args:
            precoding (Precoding): Handle to the precoding configuration.
        """

        self.__precoding = precoding


class TransmitEncoder(ABC, Precoder[TransmitPrecodingType], Generic[TransmitPrecodingType]):
    """Base class of precoding steps within transmit precoding configurations."""

    def __init__(self) -> None:
        # Initialize base class
        Precoder.__init__(self)

    @abstractmethod
    def num_transmit_input_streams(self, num_output_streams: int) -> int:
        """Get required number of input streams during encoding.

        Args:

            num_output_streams (int): Number of desired output streams.

        Returns: The number of input streams. Negative numbers indicate infeasible configurations.
        """
        ...  # pragma: no cover


class ReceiveDecoder(ABC, Precoder[ReceivePrecodingType], Generic[ReceivePrecodingType]):
    """Base class of precoding steps within receive precoding configurations."""

    def __init__(self) -> None:
        # Initialize base class, required for static type checking
        Precoder.__init__(self)

    @abstractmethod
    def num_receive_output_streams(self, num_input_streams: int) -> int:
        """Get required number of output streams during decoding.

        Args:

            num_input_streams (int): Number of input streams.

        Returns: The number of output streams. Negative numbers indicate infeasible configurations.
        """
        ...  # pragma: no cover


class Precoding(Sequence[PrecoderType], Generic[PrecoderType], Serializable):
    """Base class of precoding configurations."""

    __precoders: list[PrecoderType]  # Sequence of precoding steps

    def __init__(self) -> None:
        # Initialize base class
        Serializable.__init__(self)

        # Initialize class attributes
        self.__precoders = []

    @overload
    def __getitem__(self, index: int) -> PrecoderType: ...  # pragma: no cover

    @overload
    def __getitem__(self, index: slice) -> list[PrecoderType]: ...  # pragma: no cover

    def __getitem__(self, index: int | slice) -> PrecoderType | list[PrecoderType]:
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

    def pop_precoder(self, index: int) -> PrecoderType:
        """Remove a precoder from the processing chain.

        Args:

            index (int): Index of the precoder to be removed.

        Returns: Handle to the removed precoder.
        """

        return self.__precoders.pop(index)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object_sequence(self.__precoders, "precoders")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> Precoding[PrecoderType]:
        precoding = cls()
        precoding.__precoders = process.deserialize_object_sequence("precoders", Precoder)  # type: ignore
        return precoding


class TransmitPrecoding(Precoding[TransmitEncoderType], Generic[TransmitEncoderType]):
    """Base class for transmit encoding configurations."""

    def _collect_encoder_num_io_streams(self, num_output_streams: int) -> list[int]:
        """Collect the number of input and output streams for each encoder in the chain.

        Args:

            num_output_streams (int): Number of desired output streams.

        Returns:
            List containing the number of input and output streams for each encoder in the chain.
            The first entry is the number of input streams for the first encoder.
            The last entry is the number of output streams for the last encoder.
            The numbers are the connecting number of output and input streams for each encoder.
        """

        # Assert that the number of output streams is positive
        if num_output_streams < 1:
            raise ValueError("Number of output streams must be greater than zero")

        # Query the number of input and output streams for each precoder
        num_encoder_io_streams: list[int] = [num_output_streams]
        for encoder in reversed(self):
            # The input of the current precoder is dependent on its required output
            # In turn, the output of the current precoder is the input of the next precoder
            num_input_streams = encoder.num_transmit_input_streams(num_encoder_io_streams[0])

            # Negative numbers of input streams indicate that the configuration is not feasible
            if num_input_streams < 0:
                raise ValueError("Invalid number of input streams for encoder.")

            # Store the number of input streams for the next precoder
            num_encoder_io_streams.insert(0, num_input_streams)

        return num_encoder_io_streams

    def num_transmit_input_streams(self, num_output_streams: int) -> int:
        """Get number of input streams required for encoding.

        Args:

            num_output_streams (int): Number of desired output streams.

        Returns: The number of input streams
        """

        if len(self) < 1:
            return num_output_streams
        else:
            return self._collect_encoder_num_io_streams(num_output_streams)[0]


class ReceivePrecoding(Precoding[ReceiveDecoderType], Generic[ReceiveDecoderType]):
    """Base class for receive decoding configurations."""

    def _collect_decoder_num_io_streams(self, num_input_streams: int) -> list[int]:
        # Assert that the number of output streams is positive
        if num_input_streams < 1:
            raise ValueError("Number of input streams must be greater than zero")

        # Query the number of input and output streams for each precoder
        num_decoder_io_streams: list[int] = [num_input_streams]
        for decoder in self:
            num_output_streams = decoder.num_receive_output_streams(num_decoder_io_streams[-1])

            # Negative numbers of input streams indicate that the configuration is not feasible
            if num_output_streams < 0:
                raise ValueError("Invalid number of input streams for decoder.")

            # Store the number of input streams for the next precoder
            num_decoder_io_streams.append(num_output_streams)

        return num_decoder_io_streams

    def num_receive_output_streams(self, num_input_streams: int) -> int:
        """Get number of output streams after decoding.

        Args:

            num_input_streams (int): Number of input streams.

        Returns: The number of output streams.
        """

        if len(self) < 1:
            return num_input_streams
        else:
            return self._collect_decoder_num_io_streams(num_input_streams)[-1]


class TransmitStreamEncoder(TransmitEncoder["TransmitSignalCoding"]):
    """Base class for multi-stream MIMO coding steps during signal transmission."""

    def __init__(self) -> None:
        # Initialize base class
        TransmitEncoder.__init__(self)

    @abstractmethod
    def encode_streams(
        self, streams: Signal, num_output_streams: int, device: TransmitState
    ) -> Signal:
        """Encode a signal MIMO stream during transmission.

        Args:

            streams (Signal): The signal stream to be encoded.
            num_output_streams (int): Number of desired output streams.
            device (TransmitState): Physical state of the device.

        Returns: The encoded signal stream.
        """
        ...  # pragma: no cover


class ReceiveStreamDecoder(ReceiveDecoder["ReceiveSignalCoding"]):
    """Base class for multi-stream MIMO coding steps during signal reception."""

    def __init__(self) -> None:
        # Initialize base class, required for static type checking
        ReceiveDecoder.__init__(self)

    @abstractmethod
    def decode_streams(
        self, streams: Signal, num_output_streams: int, device: ReceiveState
    ) -> Signal:
        """Encode a signal MIMO stream during signal recepeption.

        Args:

            streams (Signal): The signal stream to be decoded.
            num_output_streams (int): Number of desired output streams.
            device (ReceiveState): Physical state of the device.

        Returns: The decoded signal stream.
        """
        ...  # pragma: no cover


class TransmitSignalCoding(TransmitPrecoding[TransmitStreamEncoder]):
    """Stream MIMO coding configuration during signal transmission."""

    def encode_streams(self, signal: Signal, device: TransmitState) -> Signal:
        """Encode a signal MIMO stream during transmission.

        This operation may modify the number of streams.

        Args:

            streams (Signal): The signal stream to be encoded.
            device (TransmitState): Physical state of the device.

        Returns: The encoded signal stream.

        Raises:

            ValueError: If the number of input streams does not match the configuration.
        """

        # Collect the number of required output streams at each encoding step
        encoder_num_io_streams = self._collect_encoder_num_io_streams(
            device.num_digital_transmit_ports
        )

        # Assert that the number of input streams is correct
        if signal.num_streams != encoder_num_io_streams[0]:
            raise ValueError(
                f"The number of streams to be encoded does not match the configuration ({signal.num_streams} != {encoder_num_io_streams[0]})"
            )

        # Iteratively apply each encoding step
        encoded_signal = signal.copy()
        for precoder, num_output_streams in zip(self, encoder_num_io_streams[1:]):
            encoded_signal = precoder.encode_streams(encoded_signal, num_output_streams, device)

        return encoded_signal


class ReceiveSignalCoding(ReceivePrecoding[ReceiveStreamDecoder], Serializable):
    """Stream MIMO coding configuration during signal transmission."""

    def decode_streams(self, signal: Signal, device: ReceiveState) -> Signal:
        """Decode a signal MIMO stream during reception.

        This operation may modify the number of streams.

        Args:

            streams (Signal): The signal stream to be decoded.
            device (ReceiveState): Physical state of the device.

        Returns: The decode signal stream.

        Raises:

            ValueError: If the number of input streams does not match the configuration.
        """

        # Collect the number of required output streams at each decoding step
        decoder_num_io_streams = self._collect_decoder_num_io_streams(
            device.num_digital_receive_ports
        )

        # Iteratively apply each encoding step
        decoded_signal = signal.copy()
        for decoder, num_output_streams in zip(self, decoder_num_io_streams[1:]):
            decoded_signal = decoder.decode_streams(decoded_signal, num_output_streams, device)

        return decoded_signal
