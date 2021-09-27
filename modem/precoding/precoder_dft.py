from __future__ import annotations
import numpy as np

from . import Precoder


class DFT(Precoder):
    """A precoder applying the Discrete Fourier Transform to each data stream.
    """

    __fft_norm: str

    def __init__(self,
                 fft_norm: str = None) -> None:
        """Object initialization.

        Args:
            fft_norm (str, optional):
                The norm applied to the discrete fourier transform.
                See also numpy.fft.fft for details
        """

        self.__fft_norm = 'ortho'

        if fft_norm is not None:
            self.__fft_norm = fft_norm

        Precoder.__init__(self)

    def encode(self, output_stream: np.matrix) -> np.matrix:
        """Apply a DFT to data streams before transmission.

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
        """

        return np.fft.fft(output_stream, norm=self.__fft_norm)

    def decode(self, input_stream: np.matrix) -> np.matrix:
        """Apply an inverse DFT to data streams after reception

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
        """

        return np.fft.ifft(input_stream, norm=self.__fft_norm)

    @property
    def num_inputs(self) -> int:
        """The required number of input symbol streams during encoding.

        DFT precoding does not alter the number of symbol streams,
        therefore the number of inputs is equal to the number of outputs.

        Returns:
            int:
                The number of symbol streams.
        """
        ...

    @property
    def num_outputs(self) -> int:
        """The generated number of output symbol streams after decoding.

        DFT precoding does not alter the number of symbol streams,
        therefore the number of inputs is equal to the number of outputs.

        Returns:
            int:
                The number of symbol streams.
        """
        ...
