from typing import List
from parameters_parser.parameters_encoder import ParametersEncoder
from modem.coding.encoder import Encoder

import numpy as np


class CrcEncoder(Encoder):
    """Implements CRC Encoder only for throughput calculations."""

    def __init__(self, params: ParametersEncoder,
                 bits_in_frame: int,
                 rng_source: np.random.RandomState) -> None:
        """
        Args:
            params (ParametersEncoder): Parameters necessary for Encoder.
            bits_in_frame (int): Number of bits that fit into one frame.
            rng_source (RandomState): Random number generator of bit source
            crc_bits (int): default 0, number of crc bits to add
        """
        super().__init__(params, bits_in_frame)
        self.rng = rng_source

    def encode(self, data_bits: List[np.array]) -> List[np.array]:
        encoded_bits: List[np.array] = []
        if self.encoded_bits_n == self.data_bits_k:
            encoded_bits = data_bits
        else:
            for block in data_bits:
                while block.size > 0:
                    encoded_block = np.append(
                        block[:self.data_bits_k], 
                        self.rng.randint(2, size=self.encoded_bits_n - self.data_bits_k))
                    encoded_bits.append(encoded_block)

                    block = block[self.data_bits_k:]

            encoded_bits.append(np.zeros(
                    self.bits_in_frame 
                    - self.code_blocks * self.params.encoded_bits_n))
        return encoded_bits

    def decode(self, encoded_bits: List[np.array]) -> List[np.array]:
        decoded_bits: List[np.array] = []
        if self.encoded_bits_n == self.data_bits_k:
            decoded_bits = encoded_bits
        else:
            # if there are some bits appended, discard them
            if encoded_bits[-1].size < self.params.encoded_bits_n:
                del encoded_bits[-1]

            decoded_bits = [block[:self.params.data_bits_k] for block in encoded_bits]
        return decoded_bits

    @property
    def encoded_bits_n(self) -> int:
        """int: Number of encoded bits that the encoding of k data bits result in."""
        return self.params.encoded_bits_n

    @property
    def data_bits_k(self) -> int:
        """int: Number of bits that are to be encoded into n bits."""
        return self.params.data_bits_k

    @property
    def code_blocks(self) -> int:
        """int: Number of code blocks which are to encoded."""
        return int(np.floor(self.bits_in_frame / self.encoded_bits_n))

    @property
    def source_bits(self) -> int:
        """int: Number of bits to be generated by the source given n/k."""
        return int(self.code_blocks * self.data_bits_k)
