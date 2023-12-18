# -*- coding: utf-8 -*-
"""
================
Stream Precoding
================

Stream precodings implement MIMO algorithms on a base-band signal sample level
during both signal transmission and reception.
"""

from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty

from hermespy.core import Serializable, Signal
from .precoding import Precoder, Precoding

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TransmitStreamEncoder(Precoder, ABC):
    """Stream MIMO coding during signal transmission."""

    @abstractmethod
    def encode_streams(self, streams: Signal) -> Signal:
        """Encode a signal MIMO stream during transmission.

        This operation may modify the number of streams.

        Args:

            streams (Signal): The signal stream to be encoded.

        Returns: The encoded signal stream.
        """
        ...  # pragma: no cover

    @abstractproperty
    def num_transmit_input_streams(self) -> int:
        """Number of input streams required by this coding.

        Returns: Number of input streams.
        """
        ...  # pragma: no cover

    @abstractproperty
    def num_transmit_output_streams(self) -> int:
        """Number of output streams generated by this coding.

        Returns: Number of output streams.
        """
        ...  # pragma: no cover

    @property
    def num_input_streams(self) -> int:
        return self.num_transmit_input_streams

    @property
    def num_output_streams(self) -> int:
        return self.num_transmit_output_streams


class ReceiveStreamDecoder(Precoder, ABC):
    """Stream MIMO coding during signal reception."""

    @abstractmethod
    def decode_streams(self, streams: Signal) -> Signal:
        """Encode a signal MIMO stream during signal recepeption.

        This operation may modify the number of streams.

        Args:

            streams (Signal): The signal stream to be decoded.

        Returns: The decoded signal stream.
        """
        ...  # pragma: no cover

    @abstractproperty
    def num_receive_input_streams(self) -> int:
        """Number of input streams required by this coding.

        Returns: Number of input streams.
        """
        ...  # pragma: no cover

    @abstractproperty
    def num_receive_output_streams(self) -> int:
        """Number of output streams generated by this coding.

        Returns: Number of output streams.
        """
        ...  # pragma: no cover

    @property
    def num_input_streams(self) -> int:
        return self.num_receive_output_streams

    @property
    def num_output_streams(self) -> int:
        return self.num_input_streams


class TransmitStreamCoding(Precoding[TransmitStreamEncoder], Serializable):
    """Stream MIMO coding configuration during signal transmission."""

    yaml_tag = "TransmitCoding"
    """YAML serialization tag."""

    def encode(self, signal: Signal) -> Signal:
        """Encode a signal MIMO stream during transmission.

        This operation may modify the number of streams.

        Args:

            streams (Signal): The signal stream to be encoded.

        Returns: The encoded signal stream.
        """

        # Iteratively apply each encoding step
        encoded_signal = signal.copy()
        for precoder in self:
            encoded_signal = precoder.encode_streams(encoded_signal)

        return encoded_signal


class ReceiveStreamCoding(Precoding[ReceiveStreamDecoder], Serializable):
    """Stream MIMO coding configuration during signal transmission."""

    yaml_tag = "ReceiveCoding"
    """YAML serialization tag."""

    def decode(self, signal: Signal) -> Signal:
        """Decode a signal MIMO stream during reception.

        This operation may modify the number of streams.

        Args:

            streams (Signal): The signal stream to be decoded.

        Returns: The decode signal stream.
        """

        # Iteratively apply each encoding step
        decoded_signal = signal.copy()
        for precoder in self:
            decoded_signal = precoder.decode_streams(decoded_signal)

        return decoded_signal
