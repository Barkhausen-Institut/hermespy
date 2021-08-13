from typing import List
import numpy as np

from modem.coding.encoder import Encoder


class EncoderManager:
    def __init__(self) -> None:
        self._encoders: List[Encoder] = []

    def add_encoder(self, encoder: Encoder) -> bool:
        encoders = self._encoders
        encoders.append(encoder)
        encoders = sorted(encoders, key=lambda e: e.data_bits_k)

        if len(encoders) > 1:
            for enc, enc_next in zip(encoders[:-1], encoders[1:]):
                if enc.encoded_bits_n != enc_next.data_bits_k:
                    return False

        self._encoders = encoders
        return True

    @property
    def encoders(self) -> List[Encoder]:
        return self._encoders

    def encode(self, data_bits: List[np.array]) -> List[np.array]:
        encoded_bits: List[np.array] = data_bits
        for encoder in self._encoders:
            encoded_bits = encoder.encode(encoded_bits)

        return encoded_bits

    def decode(self, encoded_bits: List[np.array]) -> List[np.array]:
        decoded_bits: List[np.array] = encoded_bits
        for encoder in self._encoders:
            decoded_bits = encoder.decode(encoded_bits)

        return decoded_bits