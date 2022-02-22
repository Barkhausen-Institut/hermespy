# -*- coding: utf-8 -*-
"""
===============
Coding Pipeline
===============

This module introduces the concept of bit :class:`.Encoder` steps,
which form single chain link within a channel coding processing chain.

Considering an arbitrary coding scheme consisting of multiple steps,
the process of encoding bit streams during transmission and decoding them during
subsequent reception is modeled by a chain of :class:`.Encoder` instances:

.. mermaid::

   %%{init: {'theme': 'dark'}}%%
   flowchart LR

      input([Input Bits]) --> n_i[...]
      n_i --> n_a[Encoder N-1] --> n_b[Encoder N] --> n_c[Encoder N+1]  --> n_o[...]
      n_o --> output([Coded Bits])

During transmission encoding the processing chain is sequentially executed from left to right,
during reception decoding in reverse order.

Within bit streams, :class:`.Encoder` instances sequentially encode block sections of :math:`K_n` bits into
code sections of :math:`L_n` bits.
Therefore, the rate of the :math:`n`-th :class:`.Encoder`

.. math::

   R_n = \\frac{K_n}{L_n}

is defined as the relation between input and output block length.
The pipeline configuration as well as the encoding step execution is managed by the :class:`.EncoderManager`.
Provided with a frame of :math:`K` input bits, the manager will generate a coded frame of :math:`L` bits by
sequentially executing all :math:`N` configured encoders.
Considering a frame of :math:`K_{\\mathrm{Frame}, n}` input bits to the :math:`n`-th encoder within the pipeline,
the manager will split the frame into

.. math::

   M_n(K_{\\mathrm{Frame}, n}) = \\left\\lceil \\frac{K_{\\mathrm{Frame}, n}}{K_n} \\right\\rceil

blocks to be encoded independently.
The last block will be padded with zeros should it not contain sufficient bits.
While this may not be exactly standard-compliant behaviour, it is a necessary simplification to enable
arbitrary combinations of encoders.
Therefore, the coding rate of the whole pipeline

.. math::

   R = \\frac{K}{L} = \\frac{K}{M_N \\cdot R_N}

can only be defined recursively considering the number of input blocks :math:`M_N` and rate :math:`R_N` of the last
encoder with in the pipeline, respectively.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from math import ceil

from typing import TYPE_CHECKING, Type, List, Optional

import numpy as np
from ruamel.yaml import SafeRepresenter, SafeConstructor, Node

from hermespy.core.factory import Serializable

if TYPE_CHECKING:
    from hermespy.modem import Modem


__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
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

    yaml_tag: Optional[str] = u'Encoder'
    __manager: Optional[EncoderManager]     # Coding pipeline configuration this encoder is registered to

    def __init__(self, manager: EncoderManager = None) -> None:
        """
        Args:

            manager (EncoderManager, optional):
                The coding pipeline configuration this encoder is registered in.
        """

        # Default settings
        self.__manager = None

        if manager is not None:
            self.manager = manager

    @property
    def manager(self) -> EncoderManager:
        """Coding pipeline configuration this encoder is registered in.

        Returns:
            EncoderManager:
                Handle to the coding pipeline.

        Raises:
            RuntimeError: If the encoder is considered floating.
        """

        if self.__manager is None:
            raise RuntimeError("Trying to access the manager of a floating encoding")

        return self.__manager

    @manager.setter
    def manager(self, manager: EncoderManager) -> None:

        if self.__manager is not manager:
            self.__manager = manager

    @abstractmethod
    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encodes a single block of bits.

        Bit encoding routine during data transmission,
        encoding a block of :math:`K_n` input bits into a block of :math:`L_n` code bits.

        Args:

            bits (np.ndarray):
                A numpy vector of :math:`K_n` bits, representing a single bit block to be encoded.

        Returns:

            np.ndarray:
                A numpy vector of :math:`L_n` bits, representing a single code block.

        Raises:

            ValueError:
                If the length of ``bits`` does not equal :meth:`bit_block_size`.
        """
        ...

    @abstractmethod
    def decode(self, encoded_bits: np.ndarray) -> np.ndarray:
        """Decodes a single block of bits.

        Bit decoding routine during data reception,
        decoding a block of :math:`L_n` code bits into a block of :math:`K_n` data bits.

        Args:

            encoded_bits (np.ndarray):
                A numpy vector of :math:`L_n` code bits, representing a single code block to be decoded.

        Returns:

            np.ndarray:
                A numpy vector of :math:`K_n` bits, representing a single data block.

        Raises:

            ValueError:
                If the length of ``encoded_bits`` does not equal :meth:`code_block_size`.
        """
        ...

    @property
    @abstractmethod
    def bit_block_size(self) -> int:
        """Data bit block size of a single coding operation.

        In other words, the number of input bits within a single code block during transmit encoding,
        or the number of output bits during receive decoding.
        Referred to as :math:`K_n` within the respective equations.

        Returns:

            int:
                Number of bits :math:`K_n`.
        """
        ...

    @property
    @abstractmethod
    def code_block_size(self) -> int:
        """Code bit block size of a single coding operation.

        In other words, the number of input bits within a single code block during receive decoding,
        or the number of output bits during transmit encoding.
        Referred to as :math:`L_n` within the respective equations.

        Returns:

            int:
                Number of bits :math:`L_n`.
        """
        ...

    @property
    def rate(self) -> float:
        """Code rate achieved by this coding step.

        Defined as the relation

        .. math::

           R_n = \\frac{K_n}{L_n}

        between the :meth:`.bit_block_size` :math:`K_n` and :meth:`code_block_size` :math:`L_n`.

        Returns:

            float:
                The code rate :math:`R_n`
        """

        return self.bit_block_size / self.code_block_size


class EncoderManager(Serializable):
    """Configuration managing a channel coding pipeline."""

    yaml_tag: str = u'Encoding'

    allow_padding: bool
    """Tolerate padding of data bit blocks during encoding."""

    allow_truncating: bool
    """Tolerate truncating of data code blocks during decoding."""

    __modem: Optional[Modem]        # Communication modem instance this coding pipeline configuration is attached to
    _encoders: List[Encoder]        # List of encoding steps defining the internal pipeline configuration

    def __init__(self,
                 modem: Modem = None,
                 allow_padding: bool = True,
                 allow_truncating: bool = True) -> None:
        """
        Args:

            modem (Modem, optional):
                Communication modem instance this coding pipeline configuration is attached to.
                By default, the coding pipeline is considered to be floating.

            allow_padding(bool, optional):
                Tolerate padding of data bit blocks during encoding.
                Enabled by default.

            allow_truncating(bool, optional):
                Tolerate truncating of data code blocks during decoding.
                Enabled by default.
        """

        # Default parameters
        self.__modem = None
        self._encoders: List[Encoder] = []
        self.allow_padding = allow_padding
        self.allow_truncating = allow_truncating

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

        :meta private:
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

        :meta private:
        """

        manager = cls()
        manager._encoders = constructor.construct_sequence(node, deep=True)

        return manager

    @property
    def modem(self) -> Modem:
        """Communication modem instance this coding pipeline configuration is attached to.

        Returns:

            Modem:
                Handle to the modem instance.

        Raises:

            RuntimeError:
                If the encoding configuration is floating, i.e. not attached to a modem.
        """

        if self.__modem is None:
            raise RuntimeError("Trying to access the modem of a floating encoding configuration")

        return self.__modem

    @modem.setter
    def modem(self, modem: Modem) -> None:

        if self.__modem is not modem:
            self.__modem = modem

    def add_encoder(self, encoder: Encoder) -> None:
        """Register a new encoder instance to this pipeline configuration.

        Args:
            encoder (Encoder):
                The new encoder to be added.
        """

        # Register this encoding configuration to the encoder
        encoder.manager = self

        # Add new encoder to the queue of configured encoders
        self._encoders.append(encoder)
        self._encoders = self.__execution_order()

    @property
    def encoders(self) -> List[Encoder]:
        """List of encoders registered within  this pipeline.

        Returns:
            List[Encoder]:
                List of :math:`N` :class:`Encoder` instances where the :math:`n`-th entry represents
                the :math:`n`-th coding operation during transmit encoding, or, inversely,
                the :math:`1 + N - n`-th coding operation during receive decoding.
        """

        return self._encoders

    def encode(self,
               data_bits: np.ndarray,
               num_code_bits: Optional[int] = None) -> np.ndarray:
        """Encode a stream of data bits to a stream of code bits.

        By default, the input `data_bits` will be padded with zeros
        to match the next integer multiple of the expected :meth:`.Encoder.bit_block_size`.

        The resulting code will be padded with zeros to match the requested `num_code_bits`.

        Args:

            data_bits (np.ndarray):
                Numpy vector of data bits to be encoded.

            num_code_bits (int, optional):
                The expected resulting number of code bits.

        Returns:

            np.ndarray:
                Numpy vector of encoded bits.

        Raises:

            ValueError:
                If `num_code_bits` is smaller than the resulting code bits after encoding.
        """

        bit_block_size = self.bit_block_size
        code_block_size = self.code_block_size
        num_blocks = int(ceil(data_bits.shape[0] / self.bit_block_size))

        num_data_bits = num_blocks * bit_block_size
        num_data_padding_bits = num_data_bits - data_bits.shape[0]

        if num_code_bits is None:

            num_code_bits = num_blocks * code_block_size
            num_code_padding_bits = 0

        else:

            num_minimal_code_bits = num_blocks * code_block_size  # Number of bits required to completely encode data
            num_code_padding_bits = num_code_bits - num_minimal_code_bits

            if num_code_bits < num_minimal_code_bits:
                raise ValueError("The requested number of code bits would discard parts for vital recoverability")

        padded_bits = np.concatenate((data_bits, np.zeros(num_data_padding_bits, dtype=int)))
        code = np.empty(num_code_bits, dtype=int)

        # Iterate over data blocks to be individually encoded
        for b in range(num_blocks):

            # Initialize the encoding configuration input
            code_state = padded_bits[b*bit_block_size:(b+1)*bit_block_size]

            # Loop through the encoders and encode the data, using the output of the last encoder as input to the next
            for encoder in self._encoders:
                code_state = self.__encoding_step(encoder, code_state)

            # Save result in the respective code block
            code[b*code_block_size:(b+1)*code_block_size] = code_state

        # Pad overall result with zeros if some bits are missing at the end
        if num_code_padding_bits > 0:
            code[-num_code_padding_bits:] = np.zeros(num_code_padding_bits, dtype=int)

        # Return resulting overall code
        return code

    def __encoding_step(self, encoder: Encoder, data_bits: np.ndarray) -> np.ndarray:
        """Internal function running a single encoding.

        Incoming `data_bits` may be padded with zeros to match the encoder
        `bit_block_size` specifications.

        Args:
            encoder (Encoder): The encoder processing the incoming data.
            data_bits (np.ndarray): Data feeding into the encoder.

        Returns:
            np.ndarray: Bits of encoded data processed by the encoder.
        """

        bit_block_size = encoder.bit_block_size
        code_block_size = encoder.code_block_size

        num_blocks = int(ceil(data_bits.shape[0] / bit_block_size))
        num_input_bits = bit_block_size * num_blocks
        num_padding_bits = num_input_bits - data_bits.shape[0]
        num_output_bits = code_block_size * num_blocks

        if num_padding_bits < 0:
            raise RuntimeError("Encoder chain configuration invalid, "
                               "since an output produces more bits than required at the next input")

        if not self.allow_padding and num_padding_bits > 0:
            raise RuntimeError("Padding required but not allowed")

        data_bits = np.append(data_bits, np.zeros(num_padding_bits, dtype=int))
        code_bits = np.empty(num_output_bits, dtype=int)

        for b in range(num_blocks):

            input_bits = data_bits[b*bit_block_size:(b+1)*bit_block_size]
            code_bits[b*code_block_size:(b+1)*code_block_size] = encoder.encode(input_bits)

        return code_bits

    def decode(self,
               encoded_bits: np.ndarray,
               num_data_bits: Optional[int] = None) -> np.ndarray:
        """Decode a stream of code bits to a stream of plain data bits.

        By default, decoding `encoded_bits` may ignore bits in order
        to match the next integer multiple of the expected `code_block_size`.

        The resulting data might be cut to match the requested `num_data_bits`.

        Args:
            encoded_bits (np.ndarray):
                Numpy vector of code bits to be decoded to data bits.

            num_data_bits (int, optional):
                The expected number of resulting data bits.

        Returns:

            np.ndarray:
                Numpy vector of the resulting data bit stream after decoding.

        Raises:

            RuntimeError:
                If `num_data_bits` is bigger than the resulting data bits after decoding.

            RuntimeError:
                If truncating is required but disabled by :meth:`.allow_truncating`.
        """

        bit_block_size = self.bit_block_size
        code_block_size = self.code_block_size
        num_blocks = int(encoded_bits.shape[0] / self.code_block_size)  # Float to int conversion floors by default

        if num_data_bits is not None:

            num_data_bits_full = num_blocks * bit_block_size

            if num_data_bits > num_data_bits_full:
                raise RuntimeError("The requested number of data bits is larger than number "
                                   "of bits recovered by decoding")

            if not self.allow_truncating and num_data_bits != num_data_bits_full:
                raise RuntimeError("Data truncating is required but not allowed")

        else:
            num_data_bits = num_blocks * bit_block_size

        data_bits = np.empty(num_blocks * bit_block_size, dtype=int)

        # Iterate over code blocks to be individually decoded
        for b in range(num_blocks):

            # Initialize the decoding "pipeline" input
            data_state = encoded_bits[b*code_block_size:(b+1)*code_block_size]

            # Loop through the encoders decode the code using the output of the last encoder as input to the next
            for encoder in reversed(self._encoders):
                data_state = self.__decoding_step(encoder, data_state)

            # Save resulting decoded data block
            data_bits[b*bit_block_size:(b+1)*bit_block_size] = data_state

        # Return resulting data, truncate if required
        return data_bits[:num_data_bits]

    def __decoding_step(self, encoder: Encoder, code_bits: np.ndarray) -> np.ndarray:
        """Internal function running a single decoding.

        Args:
            encoder (Encoder): The encoder processing the incoming code.
            code_bits (np.ndarray): Code block feeding into the encoder.

        Returns:
            np.ndarray: Bits of decoded codes processed by the encoder.

        Raises:
            RuntimeError: If truncating is required but not allowed.
        """

        bit_block_size = encoder.bit_block_size
        code_block_size = encoder.code_block_size

        num_blocks = int(code_bits.shape[0] / code_block_size)
        num_data_bits = bit_block_size * num_blocks

        if not self.allow_truncating:

            num_code_bits = code_block_size * num_blocks

            if num_code_bits != code_bits.shape[0]:
                raise ValueError("Code truncating required but not allowed")

        data_bits = np.empty(num_data_bits, dtype=int)

        for b in range(num_blocks):

            code_block = code_bits[b*code_block_size:(b+1)*code_block_size]
            data_bits[b*bit_block_size:(b+1)*bit_block_size] = encoder.decode(code_block)

        return data_bits

    @property
    def bit_block_size(self) -> int:
        """Data bit block size of a single coding operation.

        In other words, the number of input bits within a single code block during transmit encoding,
        or the number of output bits during receive decoding.
        Referred to as :math:`K` within the respective equations.

        Returns:

            int:
                Number of bits :math:`K`.
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
        """Code bit block size of a single coding operation.

        In other words, the number of input bits within a single code block during receive decoding,
        or the number of output bits during transmit encoding.
        Referred to as :math:`L` within the respective equations.

        Returns:

            int:
                Number of bits :math:`L`.
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
        """Code rate achieved by this coding pipeline configuration.

        Defined as the relation

        .. math::

           R = \\frac{K}{L}

        between the :meth:`.bit_block_size` :math:`K` and :meth:`.code_block_size` :math:`L`.

        Returns:

            float:
                The code rate :math:`R`.
        """

        code_rate = 1.0
        for encoder in self._encoders:
            code_rate *= encoder.rate

        return code_rate

    def required_num_data_bits(self, num_code_bits: int) -> int:
        """Compute the number of input bits required to produce a certain number of output bits.

        Args:
            num_code_bits (int): The expected number of output bits.

        Returns:
            int: The required number of input bits.
        """

        num_blocks = int(num_code_bits / self.code_block_size)
        return self.bit_block_size * num_blocks

    def __getitem__(self, item: int) -> Encoder:
        """Select an encoder from the current configuration chain.

        Args:

            item (int):
                Index of the encoder within the chain.

        Returns:

            Encoder:
                The selected encoder.
        """

        return self._encoders[item]
