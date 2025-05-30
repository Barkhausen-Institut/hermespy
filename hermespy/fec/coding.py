# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from math import ceil
from typing_extensions import override

import numpy as np

from hermespy.core import Serializable, RandomNode, SerializationProcess, DeserializationProcess

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Encoder(ABC, Serializable):
    """Base class of a single coding step within a channel coding pipeline.

    Instances of this class represent the :math:`n`-th coding step within an :class:`.EncoderManager` configuration,
    encoding blocks of :math:`K_n` bits into blocks of :math:`L_n` bits, respectively, therefore achieving a rate of

    .. math::

       R_n = \\frac{K_n}{L_n} \\mathrm{.}

    All inheriting classes represent implementations of coding steps and are required to implement the methods

        * :meth:`.encode` for encoding blocks during transmission
        * :meth:`.decode` for decoding blocks during reception
        * :meth:`.bit_block_size` reporting the input bit block length
        * :meth:`.code_block_size` reporting the output bit block length

    """

    # Coding pipeline configuration this encoder is registered to
    __manager: EncoderManager | None

    def __init__(self, manager: EncoderManager | None = None) -> None:
        """
        Args:

            manager: The coding pipeline configuration this encoder is registered in.
        """

        # Default settings
        self.__manager = None
        self.enabled = True

        if manager is not None:
            self.manager = manager

    @property
    def manager(self) -> EncoderManager | None:
        """Coding pipeline configuration this encoder is registered in.

        Returns: Handle to the coding pipeline.
        """

        return self.__manager

    @manager.setter
    def manager(self, value: EncoderManager | None) -> None:
        if self.__manager is not value:
            self.__manager = value

    @property
    def enabled(self) -> bool:
        """Is the encoding currently enabled within its chain?"""

        return self.__enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.__enabled = value

    @abstractmethod
    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encodes a single block of bits.

        Bit encoding routine during data transmission,
        encoding a block of :math:`K_n` input bits into a block of :math:`L_n` code bits.

        Args:
            bits:
                A numpy vector of :math:`K_n` bits, representing a single bit block to be encoded.

        Returns: A numpy vector of :math:`L_n` bits, representing a single code block.

        Raises:
            ValueError: If the length of ``bits`` does not equal :meth:`bit_block_size`.
        """
        ...  # pragma: no cover

    @abstractmethod
    def decode(self, encoded_bits: np.ndarray) -> np.ndarray:
        """Decodes a single block of bits.

        Bit decoding routine during data reception,
        decoding a block of :math:`L_n` code bits into a block of :math:`K_n` data bits.

        Args:

            encoded_bits:
                A numpy vector of :math:`L_n` code bits, representing a single code block to be decoded.

        Returns: A numpy vector of :math:`K_n` bits, representing a single data block.

        Raises:

            ValueError: If the length of ``encoded_bits`` does not equal :meth:`code_block_size`.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def bit_block_size(self) -> int:
        """Data bit block size of a single coding operation.

        In other words, the number of input bits within a single code block during transmit encoding,
        or the number of output bits during receive decoding.
        Referred to as :math:`K_n` within the respective equations.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def code_block_size(self) -> int:
        """Code bit block size of a single coding operation.

        In other words, the number of input bits within a single code block during receive decoding,
        or the number of output bits during transmit encoding.
        Referred to as :math:`L_n` within the respective equations.
        """
        ...  # pragma: no cover

    @property
    def rate(self) -> float:
        """Code rate achieved by this coding step.

        Defined as the relation

        .. math::

           R_n = \\frac{K_n}{L_n}

        between the :meth:`.bit_block_size` :math:`K_n` and :meth:`code_block_size` :math:`L_n`.
        """

        return self.bit_block_size / self.code_block_size


class EncoderManager(RandomNode, Serializable):
    """Configuration managing a channel coding pipeline."""

    allow_padding: bool
    """Tolerate padding of data bit blocks during encoding."""

    allow_truncating: bool
    """Tolerate truncating of data code blocks during decoding."""

    # List of encoding steps defining the internal pipeline configuration
    _encoders: list[Encoder]

    def __init__(
        self,
        allow_padding: bool = True,
        allow_truncating: bool = True,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            allow_padding:
                Tolerate padding of data bit blocks during encoding.
                Enabled by default.

            allow_truncating:
                Tolerate truncating of data code blocks during decoding.
                Enabled by default.

            seed:
                Seed for the random number generator.
                By default, a random seed is used.
        """

        # Initialize base class
        RandomNode.__init__(self, seed=seed)

        # Default parameters
        self._encoders: list[Encoder] = []
        self.allow_padding = allow_padding
        self.allow_truncating = allow_truncating

    def add_encoder(self, encoder: Encoder) -> None:
        """Register a new encoder instance to this pipeline configuration.

        Args:
            encoder: The new encoder to be added.
        """

        # Register this encoding configuration to the encoder
        if hasattr(encoder, "manager"):
            encoder.manager = self

        # Add new encoder to the queue of configured encoders
        self._encoders.append(encoder)
        self._encoders = self.__execution_order()

    @property
    def encoders(self) -> list[Encoder]:
        """list of encoders registered within  this pipeline.

        list of :math:`N` :class:`Encoder` instances where the :math:`n`-th entry represents
        the :math:`n`-th coding operation during transmit encoding, or, inversely,
        the :math:`1 + N - n`-th coding operation during receive decoding.
        """

        return self._encoders

    def encode(self, data_bits: np.ndarray, num_code_bits: int | None = None) -> np.ndarray:
        """Encode a stream of data bits to a stream of code bits.

        By default, the input `data_bits` will be padded with zeros
        to match the next integer multiple of the expected :meth:`.Encoder.bit_block_size`.

        The resulting code will be padded with zeros to match the requested `num_code_bits`.

        Args:
            data_bits: Numpy vector of data bits to be encoded.
            num_code_bits: The expected resulting number of code bits.

        Returns: Numpy vector of encoded bits.

        Raises:
            ValueError: If `num_code_bits` is smaller than the resulting code bits after encoding.
        """

        code_state = data_bits.copy()

        # Loop through the encoders and encode the data, using the output of the last encoder as input to the next
        for encoder in self._encoders:
            # Skip if the respective encoder is disabled
            if not encoder.enabled:
                continue

            data_block_size = encoder.bit_block_size
            code_block_size = encoder.code_block_size

            # Compute the number of blocks within the code for this coding step
            num_blocks = ceil(len(code_state) / data_block_size)

            # Pad if allowed
            data_state = code_state
            code_state = np.empty(num_blocks * code_block_size, dtype=bool)

            required_num_data_bits = num_blocks * data_block_size
            if len(data_state) < required_num_data_bits:
                if not self.allow_padding:
                    raise RuntimeError("Encoding would require padding, but padding is not allowed")

                num_padding_bits = required_num_data_bits - len(data_state)
                data_state = np.append(
                    data_state, self._rng.integers(0, 2, num_padding_bits, dtype=bool)
                )

            # Encode all blocks sequentially
            for block_idx in range(num_blocks):
                encoded_block = encoder.encode(
                    data_state[block_idx * data_block_size : (1 + block_idx) * data_block_size]
                )
                code_state[block_idx * code_block_size : (1 + block_idx) * code_block_size] = (
                    encoded_block
                )

        if num_code_bits and len(code_state) > num_code_bits:
            raise RuntimeError(
                "Too many input bits provided for encoding, truncating would destroy information"
            )

        if num_code_bits and len(code_state) < num_code_bits:
            if not self.allow_padding:
                raise RuntimeError("Encoding would require padding, but padding is not allowed")

            num_padding_bits = num_code_bits - len(code_state)

            if num_padding_bits >= self.code_block_size:
                raise ValueError("Insufficient number of input blocks provided for encoding")

            code_state = np.append(
                code_state, self._rng.integers(0, 2, num_padding_bits, dtype=bool)
            )

        # Return resulting overall code
        return code_state

    def decode(self, encoded_bits: np.ndarray, num_data_bits: int | None = None) -> np.ndarray:
        """Decode a stream of code bits to a stream of plain data bits.

        By default, decoding `encoded_bits` may ignore bits in order
        to match the next integer multiple of the expected `code_block_size`.

        The resulting data might be cut to match the requested `num_data_bits`.

        Args:
            encoded_bits: Numpy vector of code bits to be decoded to data bits.
            num_data_bits: The expected number of resulting data bits.

        Returns: Numpy vector of the resulting data bit stream after decoding.

        Raises:
            RuntimeError: If `num_data_bits` is bigger than the resulting data bits after decoding.
            RuntimeError: If truncating is required but disabled by `allow_truncating`.
        """

        bit_block_size = self.bit_block_size
        code_block_size = self.code_block_size
        # Float to int conversion floors by default
        num_blocks = int(encoded_bits.shape[0] / self.code_block_size)

        if num_data_bits is not None:
            num_data_bits_full = num_blocks * bit_block_size

            if num_data_bits > num_data_bits_full:
                raise RuntimeError(
                    "The requested number of data bits is larger than number "
                    "of bits recovered by decoding"
                )

            if not self.allow_truncating and num_data_bits != num_data_bits_full:
                raise RuntimeError("Data truncating is required but not allowed")

        else:
            num_data_bits = num_blocks * bit_block_size

        data_state = encoded_bits.copy()

        # Loop through the encoders decode the code using the output of the last encoder as input to the next
        for encoder in reversed(self._encoders):
            # Skip if the respective encoder is disabled
            if not encoder.enabled:
                continue

            code_block_size = encoder.code_block_size
            data_block_size = encoder.bit_block_size

            # Compute the number of blocks within the code for this coding step
            num_blocks = int(len(data_state) / code_block_size)

            # Truncate if allowed, otherwise throw an exception
            code_state = data_state[: num_blocks * code_block_size]
            data_state = np.empty(num_blocks * data_block_size, dtype=bool)

            # Decode all blocks sequentially
            for block_idx in range(num_blocks):
                data_state[block_idx * data_block_size : (1 + block_idx) * data_block_size] = (
                    encoder.decode(
                        code_state[block_idx * code_block_size : (1 + block_idx) * code_block_size]
                    )
                )

        # Return resulting data
        return data_state[:num_data_bits]

    @property
    def bit_block_size(self) -> int:
        """Data bit block size of a single coding operation.

        In other words, the number of input bits within a single code block during transmit encoding,
        or the number of output bits during receive decoding.
        Referred to as :math:`K` within the respective equations."""

        if len(self._encoders) < 1:
            return 1

        encoder_index = 0
        block_size = 1
        num_bits = 1

        for encoder_index, encoder in enumerate(self._encoders):
            if encoder.enabled:
                block_size = encoder.bit_block_size
                num_bits = encoder.code_block_size
                break

        for encoder in self._encoders[encoder_index + 1 :]:
            if not encoder.enabled:
                continue

            repetitions = int(encoder.bit_block_size / num_bits)
            block_size *= repetitions
            num_bits *= int(repetitions / encoder.rate)

        return block_size

    @property
    def code_block_size(self) -> int:
        """Code bit block size of a single coding operation.

        In other words, the number of input bits within a single code block during receive decoding,
        or the number of output bits during transmit encoding.
        Referred to as :math:`L` within the respective equations.
        """

        for encoder in reversed(self.encoders):
            if encoder.enabled:
                return encoder.code_block_size

        return 1

    def __execution_order(self) -> list[Encoder]:
        """Sort the encoders into an order of execution.

        Returns:
            list[Encoder]: A list of encoders in order of transmit execution (reversed receive execution).
        """

        return sorted(self._encoders, key=lambda encoder: encoder.bit_block_size)

    @property
    def rate(self) -> float:
        """Code rate achieved by this coding pipeline configuration.

        Defined as the relation

        .. math::

           R = \\frac{K}{L}

        between the :meth:`.bit_block_size` :math:`K` and :meth:`.code_block_size` :math:`L`.
        """

        code_rate = 1.0
        for encoder in self._encoders:
            if encoder.enabled:
                code_rate *= encoder.rate

        return code_rate

    def required_num_data_bits(self, num_code_bits: int) -> int:
        """Compute the number of input bits required to produce a certain number of output bits.

        Args:
            num_code_bits: The expected number of output bits.

        Returns: The required number of input bits.
        """

        num_blocks = int(num_code_bits / self.code_block_size)
        return self.bit_block_size * num_blocks

    def __getitem__(self, item: int) -> Encoder:
        """Select an encoder from the current configuration chain.

        Args:

            item: Index of the encoder within the chain.

        Returns: The selected encoder.
        """

        return self._encoders[item]

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object_sequence(self._encoders, "encoders")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> EncoderManager:
        manager = cls()
        encoders = process.deserialize_object_sequence("encoders", Encoder)
        for encoder in encoders:
            manager.add_encoder(encoder)
        return manager
