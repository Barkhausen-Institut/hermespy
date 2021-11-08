from __future__ import annotations
from typing import Type, Tuple
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node
import numpy as np

from . import SymbolPrecoder

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DFT(SymbolPrecoder):
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

        SymbolPrecoder.__init__(self)

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
        """Recall a new `DFT` precoder from YAML.

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

    def encode(self, symbol_stream: np.ndarray) -> np.ndarray:

        # There will be an FFT conversion over the antenna streams
        return np.fft.fft(symbol_stream, axis=0, norm=self.__fft_norm)

    def decode(self,
               symbol_stream: np.ndarray,
               stream_responses: np.ndarray,
               stream_noises: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # There will be an inverse FFT conversion over the antenna streams
        decoded_stream = np.fft.ifft(symbol_stream, axis=0, norm=self.__fft_norm)
        decoded_responses = np.fft.ifft(stream_responses, axis=0, norm=self.__fft_norm)

        return decoded_stream, decoded_responses, stream_noises

    @property
    def num_input_streams(self) -> int:

        # DFT precoding does not alter the number of symbol streams
        return self.precoding.required_inputs(self)

    @property
    def num_output_streams(self) -> int:

        # DFT precoding does not alter the number of symbol streams
        return self.precoding.required_inputs(self)
