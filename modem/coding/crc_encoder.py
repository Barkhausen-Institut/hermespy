from typing import List
from parameters_parser.parameters_crc_encoder import ParametersCrcEncoder
from modem.coding.encoder import Encoder

import numpy as np


class CrcEncoder(Encoder):
    """Implements CRC Encoder only for throughput calculations."""

    def __init__(self, params: ParametersCrcEncoder,
                 bits_in_frame: int,
                 rng: np.random.RandomState) -> None:
        super().__init__(params, bits_in_frame, rng)

    def encode(self, data_bits: List[np.array]) -> List[np.array]:
        encoded_bits: List[np.array] = []
        if self.encoded_bits_n == self.data_bits_k:
            encoded_bits = data_bits
        else:
            for block in data_bits:
                num_subblocks = int(np.ceil(block.size / self.data_bits_k))

                for n in range(num_subblocks):
                    encoded_block = np.append(
                        block[n*self.data_bits_k:(n+1)*self.data_bits_k],
                        self.rng.randint(2, size=self.encoded_bits_n-self.data_bits_k)
                    )
                    encoded_bits.append(encoded_block)
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