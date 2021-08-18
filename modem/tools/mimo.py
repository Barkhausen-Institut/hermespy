from typing import Tuple

import numpy as np


class Mimo:
    """Implements a generic MIMO (multiple-input multiple-output) system

    This class implements the encoder and decoder for a narrowband MIMO system.
    It supports the following MIMO schemes:
    - SIMO with receiver diversity, either selection combining (SC) or maximum-ratio combining (MRC)
    - Transmit diversity using Alamouti's space-time/frequency block codes scheme (STBC/SFBC) for 2 or 4 antennas
    - Spatial multiplexing with zero forcing receiver (SM-ZF)
    - Spatial multiplexing with zero MMSE receiver (SM-MMSE)

    Attributes:
        method (str): current MIMO method, can be one of the following:
                      'NONE', 'SC', 'MRC', 'STBC', 'SFBC', 'SM-ZF', 'SM-MMSE'
        number_tx_antennas (int): number of transmit antennas
        number_of_streams (int): number of spatial streams (only relevant for 'SM'). Currently no precoding is supported
                                 and number_tx_antennas must be equal to number_of_streams.
    """

    def __init__(self,
                 mimo_method: str,
                 number_of_streams: int = 1,
                 number_tx_antennas: int = 1) -> None:
        self.method = mimo_method
        self.number_of_streams = number_of_streams
        self.number_tx_antennas = number_tx_antennas

        if self.method in {'NONE', 'SC', 'MRC'} and self.number_tx_antennas > 1:
            raise ValueError(f'Number of transmit antennas must ve equal to 1 with MIMO scheme {self.method}')

        if self.number_of_streams > self.number_tx_antennas:
            raise ValueError(f"Number of transmit antennas ({self.number_tx_antennas})"
                             f" cannot be less than number of streams (self.number_of_streams)")

        if self.method.find('SM') == 0 and self.number_of_streams != self.number_tx_antennas:
            raise ValueError("In 'SM' Number of streams must be equal to number of transmit antennas")

    def encode(self, input_data: np.array) -> np.ndarray:
        """Encode data into multiple antennas.

        Args:
            input_data(np.array): Input signal with K symbols.

        Returns:
            output (np.array): Encoded data with size N_tx x (K/M),
                               with N_tx the number of transmit antennas and M the number of spatial streams
        """

        if self.method in {"NONE", "SC", "MRC"}:
            output = input_data
        elif self.method in {"STBC", "SFBC"}:
            output = self._encode_stbc(input_data)
        elif self.method in {"SM-ZF", "SM-MMSE"}:
            output = self._encode_sm(input_data)
        else:
            raise ValueError(f"MIMO encoding method '{self.method}' not supported")
        return output

    def decode(self, input_data: np.ndarray,
               channel_estimation: np.ndarray,
               noise_var: np.ndarray = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode data with multiple antennas.

        Args:
            input_data (np.ndarray):
                Input signal with N_rx x K symbols, with N_rx the number of receive antennas.
            channel_estimation (np.ndarray):
                channel estimation between every pair of antennas, of size N_rx x N_tx x K
                values, with N_tx the number of transmit antennas.
            noise_var (np.ndarray):
                Noise variance at each output symbol, with either N_rx x K values, or a scalar.

        Returns:
            (np.ndarray, np.ndarray, np.ndarray):
                output (np.ndarray): Decoded data with M x K symbols,
                    with M the number of spatial streams.
                channel_estimation (np.ndarray): post-processing channel estimation,
                    with the same size as the output.
                noise_var (np.ndarray): post-processing noise variance,
                    with the same size as the output.
        """
        if isinstance(noise_var, float):
            noise_var = np.ones(input_data.shape) * noise_var

        if self.method == "NONE":
            output = input_data
            channel_estimation = np.squeeze(channel_estimation, axis=1)
        elif self.method == "MRC":
            # Maximum ratio Combining
            output, channel_estimation, noise_var = self._decode_mrc(input_data, channel_estimation, noise_var)
        elif self.method == "SC":
            # SC: Selection Combining
            output, channel_estimation, noise_var = self._decode_sc(input_data, channel_estimation, noise_var)
        elif self.method in {"STBC", "SFBC"}:
            if self.number_tx_antennas == 2:
                output, channel_estimation, noise_var = self._decode_stbc_2_tx_antennas(input_data, channel_estimation,
                                                                                        noise_var)
            elif self.number_tx_antennas == 4:
                output, channel_estimation = self._decode_stbc_4_tx_antennas(input_data, channel_estimation)
        elif self.method in {"SM-ZF", "SM-MMSE"}:
            if input_data.shape[0] < self.number_tx_antennas:
                raise ValueError("Number of rx antennas must be larger than number of tx ant.")
            output, channel_estimation, noise_var = self._decode_sm(
                input_data, channel_estimation, noise_var)
        else:
            raise ValueError(f"MIMO decoding method '{self.method}' not supported")
        return output, channel_estimation, noise_var

    def _encode_stbc(self, input_data: np.ndarray) -> np.ndarray:
        """Encode data into multiple antennas with space-time/frequency block codes

        Currently STBCs with 2 or 4 transmit antennas are supported,
        following 3GPP TS 36.211, Sec, 6.3.3.3)

        Args:
            input_data(np.array): Input signal with K symbols.

        Returns:
            output (np.array): Encoded data with size N_tx x K symbols
        """
        number_of_symbols = input_data.size

        if self.number_tx_antennas == 2:

            output = np.zeros((2, number_of_symbols), dtype=complex)

            # alamouti precoding
            even_idx = np.arange(0, number_of_symbols, 2)
            odd_idx = even_idx + 1

            output[0, :] = input_data
            output[1, even_idx] = -np.conj(input_data[odd_idx])
            output[1, odd_idx] = np.conj(input_data[even_idx])

            output = output / np.sqrt(2)

        elif self.number_tx_antennas == 4:
            output = np.zeros((4, number_of_symbols), dtype=complex)

            idx0 = np.arange(0, number_of_symbols, 4)
            idx1 = idx0 + 1
            idx2 = idx0 + 2
            idx3 = idx0 + 3

            output[0, idx0] = input_data[idx0]
            output[0, idx1] = input_data[idx1]
            output[1, idx2] = input_data[idx2]
            output[1, idx3] = input_data[idx3]
            output[2, idx0] = -np.conj(input_data[idx1])
            output[2, idx1] = np.conj(input_data[idx0])
            output[3, idx2] = -np.conj(input_data[idx3])
            output[3, idx3] = np.conj(input_data[idx2])

            output = output / np.sqrt(2)
        else:
            raise ValueError(f"number of transmit antennas ({self.number_tx_antennas}) "
                             "not supported in space-time/frequency code")

        return output

    def _encode_sm(self, input_data: np.ndarray) -> np.ndarray:
        """Encode data into multiple antennas with spatial multiplexing

        Currently no precoding is supported, and each spatial stream is mapped to a transmit antenna.

        Args:
            input_data(np.array): Input signal with N symbols.

        Returns:
            np.ndarray:
                Encoded data with size N_tx x (N/M), with N_tx the number of transmit antennas
                and M the number of spatial streams.
        """

        number_of_symbols = np.int(input_data.size / self.number_of_streams)
        output = np.reshape(input_data, (self.number_of_streams, number_of_symbols), 'F')
        return output

    def _decode_sc(self, input_data: np.ndarray,
                   channel_estimation: np.ndarray,
                   noise_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode data using SC receive diversity with N_rx received antennas.

        Received signal with equal noise power is assumed, the decoded signal has same noise
        level as input. It is assumed that all data have equal noise levels.

        Args:
            input_data(np.ndarray): Input signal with N_rx x K symbols.
            channel_estimation(np.ndarray): channel estimation with N_rx x 1 x K values
            noise_var(np.ndarray): noise variance with N_rx x K values

        Returns:
            (np.ndarray, np.ndarray, np.ndarray):
                output (np.ndarray): Decoded data with size 1 x K
                channel_estimation (np.ndarray): post-processed channel estimation with
                    size 1 x K.
                noise_var (np.ndarray): post-processing noise variance with size 1 x K.
        """

        channel_estimation = np.squeeze(channel_estimation, axis=1)
        antenna_index = np.argmax(np.abs(channel_estimation) ** 2 / noise_var, axis=0)
        output = np.take_along_axis(input_data, antenna_index[np.newaxis, :], axis=0)
        channel_estimation = np.take_along_axis(channel_estimation, antenna_index[np.newaxis, :], axis=0)
        noise_var = np.take_along_axis(noise_var, antenna_index[np.newaxis, :], axis=0)

        return output, channel_estimation, noise_var

    def _decode_mrc(self, input_data: np.ndarray,
                    channel_estimation: np.ndarray,
                    noise_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode data using MRC receive diversity with N_rx received antennas.

        Received signal with equal noise power is assumed, the decoded signal has same noise
        level as input. It is assumed that all data have equal noise levels.

        Args:
            input_data(np.ndarray): Input signal with N_rx x K symbols.
            channel_estimation(np.ndarray): channel estimation with N_rx x 1 x K values
            noise_var(np.ndarray): noise variance with N_rx x K values

        Returns:
            (np.ndarray, np.ndarray, np.ndarray):
                output: Decoded data with size 1 x K.
                channel_estimation: post-processing channel estimation with size 1 x K.
                noise_var: post-processing noise variance with size 1 x K.
        """

        channel_estimation = np.squeeze(channel_estimation, axis=1)
        output = np.sum(input_data * channel_estimation.conj(), axis=0)[np.newaxis]
        noise_var = np.sum(noise_var * (np.abs(channel_estimation) ** 2), axis=0)[np.newaxis]
        channel_estimation = np.sum(np.abs(channel_estimation) ** 2, axis=0)[np.newaxis]

        return output, channel_estimation, noise_var

    def _decode_stbc_2_tx_antennas(self, input_data: np.ndarray,
                                   channel_estimation: np.ndarray, noise_var) -> Tuple[np.ndarray, np.ndarray,
                                                                                       np.ndarray]:
        """Decode data for STBC with 2 transmit antennas

        Received signal with equal noise power is assumed, the decoded signal has same noise level as input.
        If more than 2 receive antennas are employed, then MRC is applied on the STBC decoding output of all antennas.

        Args:
            input_data(np.ndarray): Input signal with N_rx x N symbols.
            channel_estimation(np.ndarray): Channel estimation with N_rx x N_tx x N symbols.

        Returns:
            (np.ndarray, np.ndarray):
                output: Decoded data with size 1 x N.
                channel_estimation_out: updated channel estimation with size 1 x N.
        """

        number_of_rx_antennas = input_data.shape[0]
        number_of_symbols = input_data.shape[1]
        output = np.zeros((number_of_rx_antennas, number_of_symbols), dtype=complex)
        channel_estimation_out = np.zeros((number_of_rx_antennas, number_of_symbols), dtype=complex)

        channel = channel_estimation / np.sqrt(2)

        even_idx = np.arange(0, number_of_symbols, 2)
        odd_idx = even_idx + 1

        y_even = (np.conj(channel[:, 0, even_idx]) * input_data[:, even_idx] +
                  channel[:, 1, odd_idx] * np.conj(input_data[:, odd_idx]))
        y_odd = (-channel[:, 1, even_idx] * np.conj(input_data[:, even_idx]) +
                 np.conj(channel[:, 0, odd_idx]) * input_data[:, odd_idx])

        norm = np.sqrt(np.abs(channel[:, 0, even_idx]) ** 2 + np.abs(channel[:, 1, even_idx]) ** 2)

        output[:, even_idx] = y_even / norm
        output[:, odd_idx] = y_odd / norm

        # the channel coefficient is the norm
        channel_estimation_out[:, even_idx] = norm
        channel_estimation_out[:, odd_idx] = norm

        if number_of_rx_antennas > 1:
            output, channel_estimation_out, noise_var = self._decode_mrc(output,
                                                                         channel_estimation_out[:, np.newaxis, :],
                                                                         noise_var)

        return output, channel_estimation_out, noise_var

    def _decode_stbc_4_tx_antennas(self, input_data: np.ndarray,
                                   channel_estimation: np.ndarray) -> Tuple[np.ndarray,
                                                                            np.ndarray]:
        """Decode data for STBC with 4 transmit antennas.

        Received signal with equal noise power is assumed, the decoded signal has same noise
        level as input.

        Args:
            input_data(np.ndarray): Input signal with N symbols.
            channel_estimation(np.ndarray): Input signal with N_rx x N_tx x N symbols.

        Returns:
            (np.ndarray, np.ndarray):
                output (np.ndarray): Decoded data with size 1 x N
                channel_estimation_out (np.ndarray): Updated channel estimation.
        """

        N = input_data.size

        channel_estimation = np.squeeze(channel_estimation)

        tx0_idx = np.arange(0, N, 4)
        tx1_idx = tx0_idx + 1
        tx2_idx = tx0_idx + 2
        tx3_idx = tx0_idx + 3

        # antenna 0 and 2
        input_stbc_tx_antennas_0_2 = np.zeros(
            (1, len(tx0_idx) + len(tx1_idx)), dtype=complex)
        input_stbc_tx_antennas_0_2[0, ::2] = input_data[0, tx0_idx]
        input_stbc_tx_antennas_0_2[0, 1::2] = input_data[0, tx1_idx]

        ce_0_2 = np.zeros((2, (len(tx0_idx) + len(tx1_idx))), dtype=complex)
        ce_0_2[0, ::2] = channel_estimation[0, tx0_idx]
        ce_0_2[0, 1::2] = channel_estimation[0, tx1_idx]
        ce_0_2[1, ::2] = channel_estimation[2, tx0_idx]
        ce_0_2[1, 1::2] = channel_estimation[2, tx1_idx]

        out_tx_0_2, ce_0_2 = self._decode_stbc_2_tx_antennas(
            input_stbc_tx_antennas_0_2, ce_0_2)

        # reshape, so that we can concatenate t and t+1 accordingly
        out_tx_0_2 = np.reshape(out_tx_0_2, (-1, 2))
        ce_0_2 = np.reshape(ce_0_2, (-1, 2))

        # antenna 1 and 3
        input_stbc_tx_antennas_1_3 = np.zeros(
            (1, len(tx2_idx) + len(tx3_idx)), dtype=complex)
        input_stbc_tx_antennas_1_3[0, ::2] = input_data[0, tx2_idx]
        input_stbc_tx_antennas_1_3[0, 1::2] = input_data[0, tx3_idx]

        ce_1_3 = np.zeros((2, (len(tx2_idx) + len(tx3_idx))), dtype=complex)
        ce_1_3[0, ::2] = channel_estimation[1, tx2_idx]
        ce_1_3[0, 1::2] = channel_estimation[1, tx3_idx]
        ce_1_3[1, ::2] = channel_estimation[3, tx2_idx]
        ce_1_3[1, 1::2] = channel_estimation[3, tx3_idx]

        out_tx_1_3, ce_1_3 = self._decode_stbc_2_tx_antennas(
            input_stbc_tx_antennas_1_3, ce_1_3)

        # reshape so that we can concatenate t and t+1 accordingly
        out_tx_1_3 = np.reshape(out_tx_1_3, (-1, 2))
        ce_1_3 = np.reshape(ce_1_3, (-1, 2))

        output = np.hstack((out_tx_0_2, out_tx_1_3))
        output = np.reshape(output, (1, -1))

        channel_estimation_out = np.hstack((ce_0_2, ce_1_3))
        channel_estimation_out = np.reshape(channel_estimation_out, (1, -1))
        return output, channel_estimation_out

    def _decode_sm(self, input_data: np.ndarray, channel_estimation: np.ndarray,
                   noise_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode data for SM using Zero Forcing or MMSE.

        For MMSE it is assumed that the symbol energy Es = 1.

        Args:
            input_data(np.ndarray): Input signal with N_rx x K symbols.
            channel_estimation(np.ndarray): Input signal with N_rx x N_tx x K symbols.
            noise_var(np.ndarray): Noise variance at each antenna with N_rx x K values

        Returns:
            (np.ndarray, np.ndarray, np.ndarray)
                output: Decoded data with size (N_tx * K)
                estimated_channe: est. channel after decoding with size (N_tx * K).
                Since SM decoder is also an equalizer, it is a vector of ones.
                noise_var: Variance of noise of estimated data with size (N_tx * K)
        """

        number_of_symbols = input_data.shape[1]
        number_rx_antennas = input_data.shape[0]
        ch = np.moveaxis(channel_estimation, -1, 0)

        # convert noise variance into noise covariance diagonal matrices
        idx = np.arange(number_rx_antennas)
        idx = idx + number_rx_antennas * idx
        noise_covariance = np.zeros((number_of_symbols, number_rx_antennas * number_rx_antennas))
        noise_covariance[:, idx] = noise_var.T
        noise_covariance = np.reshape(noise_covariance, (number_of_symbols, number_rx_antennas,
                                                         number_rx_antennas))

        if self.method == 'SM-ZF':
            linear_decoder = np.linalg.pinv(ch)
        elif self.method == 'SM-MMSE':
            ch_hermitian = np.transpose(ch, axes=[0, 2, 1]).conj()

            linear_decoder = ch_hermitian @ np.linalg.inv(ch @ ch_hermitian + noise_covariance)
            norm = np.diagonal(linear_decoder @ ch, axis1=1, axis2=2).T

        else:
            raise ValueError(f'unsupported MIMO receiver {self.method}')

        output = np.matmul(linear_decoder, input_data.T[:, :, np.newaxis])

        if self.method == 'SM-MMSE':
            output = 1 / norm.T[:, :, np.newaxis] * output

        output = output.squeeze(axis=2).T

        if self.method == 'SM-MMSE':
            noise_var_out = 1./np.real(norm) - 1
        else:
            noise_covariance = (linear_decoder @ np.transpose(linear_decoder.conj(), axes=[0, 2, 1])
                                @ noise_covariance)
            noise_var_out = np.real(np.diagonal(noise_covariance, axis1=1, axis2=2)).T

        return output, np.ones((self.number_of_streams, number_of_symbols)), noise_var_out
