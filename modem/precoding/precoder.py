from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
import numpy as np


from . import Precoding


class Precoder(ABC):
    """Abstract base class for signal processing algorithms operating on complex data symbols.

    A `Precoder` may compress or expand the number of data streams, however, the number of data streams before encoding
    and after decoding should generally be identical.

    `Precoders` who just encode or decode data are allowed, in this case unsupported operations are required to raise a
    `NotImplementedError` exception.
    """

    __precoding = Optional[Precoding]

    def __init__(self,
                 precoding: Precoding = None) -> None:
        """Object initialization.
        """

        self.__precoding = None

        if precoding is not None:
            self.precoding = precoding

    @property
    def precoding(self) -> Precoding:
        """Access the precoding configuration this precoder is attached to.

        Returns:
            Precoding: Handle to the precoding.

        Raises:
            RuntimeError: If this precoder is currently floating.
        """

        if self.__precoding is None:
            raise RuntimeError("Trying to access the precoding of a floating precoder")

        return self.__precoding

    @precoding.setter
    def precoding(self, precoding: Precoding) -> None:
        """Modify the precoding configuration this precoder is attached to.
        
        Args:
            precoding (Precoding): Handle to the precoding configuration.
        """

        self.__precoding = precoding

    @abstractmethod
    def encode(self, output_stream: np.matrix) -> np.matrix:
        """Encode a data stream before transmission.

        This operation may modify the number of streams.

        Args:
            output_stream (np.matrix):
                The data streams feeding into the `Precoder` to be encoded.
                The first matrix dimension is the number of streams,
                the second dimension the number of discrete samples within each respective stream.

        Returns:
            np.matrix:
                The encoded data streams.
                The first matrix dimension is the number of streams,
                the second dimension the number of discrete samples.

        Raises:
            NotImplementedError: If the `Precoder` does not support an encoding operation.
        """
        ...

    @abstractmethod
    def decode(self, input_stream: np.matrix) -> np.matrix:
        """Decode a data stream after reception.

        This operation may modify the number of streams.

        Args:
            input_stream (np.matrix):
                The data streams feeding into the `Precoder` to be decoded.
                The first matrix dimension is the number of streams,
                the second dimension the number of discrete samples within each respective stream.

        Returns:
            np.matrix:
                The decoded data streams.
                The first matrix dimension is the number of streams,
                the second dimension the number of discrete samples.

        Raises:
            NotImplementedError: If the `Precoder` does not support a decoding operation.
        """
        ...

    @property
    @abstractmethod
    def num_inputs(self) -> int:
        """The required number of input symbol streams during encoding.

        Returns:
            int:
                The number of symbol streams.
        """
        ...

    @property
    @abstractmethod
    def num_outputs(self) -> int:
        """The generated number of output symbol streams after decoding.

        Returns:
            int:
                The number of symbol streams.
        """
        ...
