# -*- coding: utf-8 -*-
"""LDPC Encoding with Cpp bindings."""

from __future__ import annotations
from typing import Any
import numpy as np

from modem.coding.ldpc import LDPC
from modem.coding.ldpc_binding.bin import ldpc_binding

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer, Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class LDPCBinding(LDPC):
    """Cpp binding of the LDPC encoder."""

    def __init__(self, **args: Any) -> None:
        """Object initialization.

        Shadows the base constructor.

        Args:
            **args (Any): All arguments get passed to the base constructor.
        """
        LDPC.__init__(self, *args)

    def encode(self, bits: np.array) -> np.array:

        encoded_words = ldpc_binding.encode(
            data_bits, self.G, self.Z, self.num_info_bits, self.encoded_bits_n,
            self.data_bits_k, self.code_blocks, self.bits_in_frame
        )

    def decode(self, encoded_bits: np.array) -> np.array:
        decoded_blocks = ldpc_binding.decode(
            encoded_bits, self.encoded_bits_n, self.code_blocks, self.number_parity_bits,
            self.num_total_bits, self.Z, self.params.no_iterations, self.H, self.num_info_bits
        )
        return decoded_blocks


