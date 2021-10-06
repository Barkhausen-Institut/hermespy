from typing import List
import numpy as np

from modem.coding.encoder import Encoder


class EncoderManager:
    """Serves as a wrapper class for multiple encoders."""
    def __init__(self) -> None:
        self._encoders: List[Encoder] = []

    def add_encoder(self, encoder: Encoder) -> None:
        self._encoders.append(encoder)
        self._encoders = sorted(
            self._encoders,
            key=lambda encoder: encoder.data_bits_k)

    @property
    def encoders(self) -> List[Encoder]:
        return self._encoders

    @property
    def code_rate(self) -> float:
        R = 1
        for encoder in self._encoders:
            R *= encoder.data_bits_k / encoder.encoded_bits_n

        return R

    @property
    def source_bits(self) -> int:
        """Returns lowest number of source bits of all of the encoders."""
        min_source_bits_encoder = min(self._encoders, key=lambda encoder: encoder.source_bits)
        return min_source_bits_encoder.source_bits

    def encode(self, data_bits: List[np.array]) -> List[np.array]:
        encoded_bits: List[np.array] = data_bits
        for encoder in self._encoders:
            encoded_bits = encoder.encode(encoded_bits)

        return encoded_bits

    def decode(self, encoded_bits: List[np.array]) -> List[np.array]:
        decoded_bits: List[np.array] = encoded_bits
        for encoder in reversed(self._encoders):
            decoded_bits = encoder.decode(decoded_bits)

        return decoded_bits