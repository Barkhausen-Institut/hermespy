from __future__ import annotations
from typing import Type
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node
import numpy as np

from . import Precoder


class DFT(Precoder):
    """A precoder applying the Discrete Fourier Transform to each data stream.
    """

    yaml_tag = 'DFT'
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

    @classmethod
    def to_yaml(cls: Type[DFT], representer: SafeRepresenter, node: DFT) -> Node:
        """Serialize a `DFT` precoder to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (DFT):
                The `DFT` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        return representer.represent_scalar(cls.yaml_tag, "")

    @classmethod
    def from_yaml(cls: Type[DFT], constructor: SafeConstructor, node: Node) -> DFT:
        """Recall a new `DFT` precoder to YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `DFT` serialization.

        Returns:
            DFT:
                Newly created `DFT` instance.
        """

        return cls()

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
