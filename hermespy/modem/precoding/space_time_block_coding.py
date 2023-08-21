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


class Ganesan(SymbolPrecoder, Serializable):
    """Girish Ganesan and Petre Stoica general precoder distributing symbols in space and time.

    Supports 4 transmit antennas. Features a :math:`\\frac{3}{4}` symbol rate.
    Refer to :cite:t:`2001:ganesan` for further information.
    """

    yaml_tag = "GANESAN"

    def encode(self, symbols: StatedSymbols) -> StatedSymbols:
        """Encode data into multiple antennas with space-time/frequency block codes.
        Note that Ganesan schema's symbol rate is :math:`\\frac{3}{4}` so the encoding process increases the number of blocks by :math:`\\frac{4}{3}`.

        Args:
            symbols (StatedSymbols): Input signal featuring :math:`K` blocks.

        Returns:
            Encoded data with size :math:`\\frac{4}{3} \\times K` symbol blocks. Thus num_blocks is changed to num_blocks / 3 * 4.
            Returned channel states are initialized with ones (np.ones is used).

        Raises:
            ValueError: If more than a single symbol stream is provided.
            RuntimeError: If the number of transmit antennas is not four.
            ValueError: If the number of data symbols blocks is not divisable by three.
        """

        if symbols.num_streams != 1:
            raise ValueError("Space-Time block codings require a single symbol input stream")

        num_tx_streams = self.required_num_output_streams
        input_data = symbols.raw[0, :, :]

        if num_tx_streams != 4:
            raise RuntimeError(f"Ganesan encoding requires 4 transmit antennas ({num_tx_streams} requested)")

        if symbols.num_blocks % 3 != 0:
            raise ValueError("Number of blocks must be divisable by 3.")

        # Change symbol block amount because of the 3/4 symbol rate.
        output = np.empty((4, symbols.num_blocks // 3 * 4, symbols.num_symbols), dtype=np.complex_)
        zero = np.zeros((symbols.num_blocks // 3, symbols.num_symbols), dtype=np.complex_)
        # Encode data explicitly element-wise.
        # Notice that matrix Z (Eq. 41) is m by N in the paper,
        # where m = num Tx, and N = num symbol periods,
        # so each column is a symbol period and each row is TX
        # Note that input_data[i::3, :] relates to symbol s_{i-1} in the paper.
        # Tx 1
        output[0, 0::4, :] = input_data[0::3, :]
        output[0, 1::4, :] = zero
        output[0, 2::4, :] = input_data[1::3, :]
        output[0, 3::4, :] = -input_data[2::3, :]
        # Tx 2
        output[1, 0::4, :] = zero
        output[1, 1::4, :] = input_data[0::3, :]
        output[1, 2::4, :] = input_data[2::3, :].conj()
        output[1, 3::4, :] = input_data[1::3, :].conj()
        # Tx 3
        output[2, 0::4, :] = -input_data[1::3, :].conj()
        output[2, 1::4, :] = -input_data[2::3, :]
        output[2, 2::4, :] = input_data[0::3, :].conj()
        output[2, 3::4, :] = zero
        # Tx 4
        output[3, 0::4, :] = input_data[2::3, :].conj()
        output[3, 1::4, :] = -input_data[1::3, :]
        output[3, 2::4, :] = zero
        output[3, 3::4, :] = input_data[0::3, :].conj()

        # Cast the result to StatedSymbols
        st = symbols.states
        st = np.ones((4, st.shape[1], st.shape[2] // 3 * 4, st.shape[3]))
        return StatedSymbols(output, st)

    def decode(self, symbols: StatedSymbols) -> StatedSymbols:
        """Decode data for STBC with 4 antenna streams
        Note that Ganesan schema's symbol rate is :math:`\\frac{3}{4}` so the decoding process decreases the number of blocks by :math:`\\frac{3}{4}`.

        Args:
            symbols (StatedSymbols): Input signal with :math:`4 \\times N` symbol blocks.

        Returns:
            Decoded data with size :math:`3 \\times N`
            Returned channel states are initialized with ones (np.ones is used).
        """

        # check the number of blocks (we expect them to be divisable by 4)
        if symbols.num_blocks % 4 != 0:
            raise ValueError("Ganesan decoding must be given an amount of data symbols blocks that is divisable by 4")
        # check number of Tx (must be 4)
        if symbols.num_transmit_streams != 4:
            raise ValueError(f"Ganesan decoding must be given 4 transmit antennas ({symbols.num_transmit_streams} were given)")

        # Init the decoded symbols ndarray. Notice that num_blocks is reduced because of the 3/4 symbol rate.
        num_rx = symbols.num_streams
        decoded_symbols = np.empty((num_rx, symbols.num_blocks // 4 * 3, symbols.num_symbols), dtype=np.complex_)

        # split the r vector onto real and imag vectors and concatenate them
        b = symbols.raw
        b = np.concatenate((b.real, b.imag), axis=1)

        # Let each Rx antenna receive 4 signals over 4 time moments (=1 symbol period).
        # Let R be a vector of 4 received signals by a Rx antenna
        # Let A be a matrix of channel states of size of 4x4 (4 Tx to 4 symbols in a symbol period):
        # R = A@s, where
        # R = {r1, r2, r3, r4}
        # s = {s1, s2, s3, 0}
        # Split each variable into real and imag parts, expanding the system
        # R' = A' @ s', where
        # R' = {r1.real, r2.real, r3.real, r4.real, r1.imag, r2.imag, r3.imag, r4.imag}, where
        # s' = {s1.real, s2.real, s3.real, s1.imag, s2.imag, s3.imag}
        # Then matrix A can be constructed with the following sings and index matrices
        signs_matrix_real = np.array([[1, -1,  1, -1, -1,  1],
                                      [1, -1, -1, -1,  1,  1],
                                      [1,  1,  1,  1, -1,  1],
                                      [1,  1, -1,  1,  1,  1]])
        signs_matrix_imag = signs_matrix_real.copy()
        signs_matrix_imag[:, 3:] *= -1
        signs_matrix = np.concatenate((signs_matrix_real, signs_matrix_imag), axis=0)
        index_matrix = np.array([[0, 2, 3],
                                 [1, 3, 2],
                                 [2, 0, 1],
                                 [3, 1, 0]])

        # Init result(decoded_symbols), matrix A(an) and estimator with lhs(b) of the linear system
        decoded_symbols = np.empty((num_rx, symbols.num_blocks * 3 // 4, symbols.num_symbols), dtype=np.complex_)
        an = np.empty((num_rx, 6, 8, symbols.num_symbols), dtype=np.float_)
        estimator = np.empty((symbols.num_symbols, symbols.num_streams, 6, 8), dtype=np.float_)
        b = np.empty((num_rx, 8, symbols.num_symbols), dtype=np.float_)

        # Init einsum paths to optimize einsum in the future
        an_path = np.einsum_path('ikjl,jk->lijk', an, signs_matrix, optimize='optimal')[0]
        estimation_path = np.einsum_path('ijkl,jli->jki', estimator, b, optimize='optimal')[0]

        # For each symbol period (which is 4 blocks) decode 3 encoded symbol blocks
        for n in range(symbols.num_blocks // 4):
            # Assemble matrix A'
            for n_ in range(4):
                an[:, 3:, n_+4, :] = an[:, :3, n_, :] = symbols.states.real[:, index_matrix[n_], n*4+n_, :]
                an[:, :3, n_+4, :] = an[:, 3:, n_, :] = symbols.states.imag[:, index_matrix[n_], n*4+n_, :]

            # Calculate estimator such that estimator @ R' = s'
            # this einsum applies the signs matrix to A' and transposes it
            estimator = np.linalg.pinv(np.einsum('ikjl,jk->lijk', an, signs_matrix, optimize=an_path))

            # Init R' for this symbol period
            received_symbols_blocks = symbols.raw[:, n*4:n*4+4, :]
            b = np.concatenate((received_symbols_blocks.real, received_symbols_blocks.imag), axis=1)

            # Solve the system and assemble extended results from 6 floats back to 3 complex
            estimated_split_symbols = np.einsum('ijkl,jli->jki', estimator, b, optimize=estimation_path)
            decoded_symbols[:, n*3:n*3+3, :] = estimated_split_symbols[:, :3, :] + 1j * estimated_split_symbols[:, 3:, :]

        # Construct ideal channel states to cast result to StatedSymbols
        ideal_states = np.ones((num_rx, 1, decoded_symbols.shape[1], decoded_symbols.shape[2]))
        return StatedSymbols(decoded_symbols, ideal_states)

    @property
    def num_input_streams(self) -> int:
        return 1

    @property
    def num_output_streams(self) -> int:
        return 4
