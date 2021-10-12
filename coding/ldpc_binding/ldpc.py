# -*- coding: utf-8 -*-
"""LDPC Encoding with Cpp bindings."""

from __future__ import annotations
from typing import Any
import numpy as np

from coding.ldpc import LDPC
import coding.ldpc_binding.bin.ldpc_binding as ldpc_binding  # type: ignore

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

    def __init__(self, *args: Any) -> None:
        """Object initialization.

        Shadows the base constructor.

        Args:
            **args (Any): All arguments get passed to the base constructor.
        """
        LDPC.__init__(self, *args)

    def encode(self, bits: np.ndarray) -> np.ndarray:

        return ldpc_binding.encode(
            [bits], self._G, 0, self.bit_block_size, self.code_block_size,
            self.bit_block_size, 1, self.bit_block_size
        )[0]

    def decode(self, encoded_bits: np.ndarray) -> np.ndarray:

        codes = encoded_bits.copy()
        codes[codes < .5] = -1.

        return ldpc_binding.decode(
            [codes], self.code_block_size, 1, self.num_parity_bits,
            self.code_block_size, int(.5 * (self._H.shape[1] - self._G.shape[1])), self.iterations,
            self._H, self.bit_block_size)[0]
