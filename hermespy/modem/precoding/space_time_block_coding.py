# -*- coding: utf-8 -*-
"""
=======================
Space-Time Block Coding
=======================
"""

from __future__ import annotations

import numpy as np

from hermespy.core import Serializable
from ..symbols import StatedSymbols
from .symbol_precoding import SymbolPrecoder

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Alamouti(SymbolPrecoder, Serializable):
    """Alamouti precoder distributing symbols in space and time.

    Support for 2 transmit antennas only.
    Refer to :cite:t:`1998:alamouti` for further information.
    """

    yaml_tag = "ALAMOUTI"

    def encode(self, symbols: StatedSymbols) -> StatedSymbols:
        """Encode data into multiple antennas with space-time/frequency block codes.

        Args:
            symbols (StatedSymbols): Input signal featuring :math:`K` blocks.

        Returns: Encoded data with size :math:`2 \\times K` symbols

        Raises:

            ValueError: If more than a single symbol stream is provided.
            RuntimeError: If the number of transmit antennas is not two.
            ValueError: If the number of data symbols is not even.
        """

        if symbols.num_streams != 1:
            raise ValueError("Space-Time block codings require a single symbol input stream")

        num_tx_streams = self.required_num_output_streams
        input_data = symbols.raw[0, :, :]

        # 2x2 MIMO Alamouti code
        if num_tx_streams != 2:
            raise RuntimeError(f"Alamouti encoding requires two transmit antennas ({num_tx_streams} requested)")

        if symbols.num_blocks % 2 != 0:
            raise ValueError("Alamouti encoding must contain an even amount of data symbols blocks")

        output = np.empty((2, symbols.num_blocks, symbols.num_symbols), dtype=np.complex_)
        output[0, :, :] = input_data
        output[1, 0::2, :] = -input_data[1::2, :].conj()
        output[1, 1::2, :] = input_data[0::2, :].conj()

        state = np.repeat(symbols.states, num_tx_streams, axis=0)
        return StatedSymbols(output, state)

    def decode(self, symbols: StatedSymbols) -> StatedSymbols:
        """Decode data for STBC with 2 antenna streams

        Received signal with equal noise power is assumed, the decoded signal has same noise level as input.
        If more than 2 receive antennas are employed, then MRC is applied on the STBC decoding output of all antennas.

        Args:
            symbols (StatedSymbols): Input signal with :math:`N \\times K` symbol blocks.

        Returns: Decoded data with size :math:`N \\times K`
        """

        if symbols.num_blocks % 2 != 0:
            raise ValueError("Alamouti decoding must contain an even amount of data symbols blocks")

        channel_state = symbols.states[:, :2, 0::2, :]
        weight_norms = np.sum(np.abs(channel_state) ** 2, axis=1, keepdims=False)

        decoded_symbols = np.empty((symbols.num_streams, symbols.num_blocks, symbols.num_symbols), dtype=complex)
        decoded_symbols[:, 0::2, :] = (channel_state[:, 0, ::].conj() * symbols.raw[:, 0::2, :] + channel_state[:, 1, ::] * symbols.raw[:, 1::2, :].conj()) / weight_norms
        decoded_symbols[:, 1::2, :] = (channel_state[:, 0, ::].conj() * symbols.raw[:, 1::2, :] - channel_state[:, 1, ::] * symbols.raw[:, 0::2, :].conj()) / weight_norms

        return StatedSymbols(decoded_symbols, np.ones((symbols.num_streams, 1, symbols.num_blocks, symbols.num_symbols), dtype=complex))

    @property
    def num_input_streams(self) -> int:
        # Alamouti coding requires a single symbol input stream
        return 1

    @property
    def num_output_streams(self) -> int:
        # Alamouti coding will always produce 2 symbol output streams
        return 2
