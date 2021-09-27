from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class Precoder(ABC):
    """Abstract base class for signal processing algorithms operating on complex data streams.

    A `Precoder` may compress or expand the number of data streams, however, the number of data streams before encoding
    and after decoding should generally be identical.

    `Precoders` who just encode or decode data are allowed, in this case unsupported operations are required to raise a
    `NotImplementedError` exception.
    """

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
        """Encode a data stream before transmission.

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
    def num_streams(self) -> int:
        """The resulting number of data streams after precoding.

        Returns:
            int:
                The number of data streams.
        """
        ...
