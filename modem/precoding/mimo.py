# -*- coding: utf-8 -*-
"""HermesPy Multiple-Input-Multiple-Output spatial channel precodings."""

from __future__ import annotations
from typing import Tuple, Type
from ruamel.yaml import SafeConstructor, SafeRepresenter, ScalarNode
import numpy as np

from .symbol_precoder import SymbolPrecoder

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "André Noll Barreto"
__email__ = "andre.nollbarreto@barkhauseninstitut.org"
__status__ = "Prototype"


class Mimo(SymbolPrecoder):
    """Implements a generic MIMO (multiple-input multiple-output) system

    This class implements the encoder and decoder for a narrowband MIMO system.
    It supports the following MIMO schemes:
    - SIMO with receiver diversity, either selection combining (SC) or maximum-ratio combining (MRC)
    - Transmit diversity using Alamouti's space-time/frequency block codes scheme (STBC/SFBC) for 2 or 4 antennas
    - Spatial multiplexing with zero forcing receiver (SM-ZF)
    - Spatial multiplexing with zero MMSE receiver (SM-MMSE)
    """

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

        if self.num_output_streams == 2:

            output = np.zeros((2, number_of_symbols), dtype=complex)

            # alamouti precoding
            even_idx = np.arange(0, number_of_symbols, 2)
            odd_idx = even_idx + 1

            output[0, :] = input_data
            output[1, even_idx] = -np.conj(input_data[odd_idx])
            output[1, odd_idx] = np.conj(input_data[even_idx])

            output = output / np.sqrt(2)

        elif self.num_output_streams == 4:
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
            raise ValueError(f"number of output streams ({self.num_output_streams}) "
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
                stream_responses (np.ndarray): post-processed channel estimation with
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
                stream_responses: post-processing channel estimation with size 1 x K.
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
                                   channel_estimation: np.ndarray,
                                   noise_var) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        number_of_rx_antennas = input_data.shape[0]
        number_of_symbols = input_data.shape[1]

        if isinstance(noise_var, float):
            noise_var = noise_var * np.ones(input_data.shape)

        # stream_responses = np.squeeze(stream_responses)

        tx0_idx = np.arange(0, number_of_symbols, 4)
        tx1_idx = tx0_idx + 1
        tx2_idx = tx0_idx + 2
        tx3_idx = tx0_idx + 3

        # antenna 0 and 2
        input_stbc_tx_antennas_0_2 = np.zeros((number_of_rx_antennas, len(tx0_idx) + len(tx1_idx)), dtype=complex)
        input_stbc_tx_antennas_0_2[:, ::2] = input_data[:, tx0_idx]
        input_stbc_tx_antennas_0_2[:, 1::2] = input_data[:, tx1_idx]

        noise_var_0_2 = np.zeros(input_stbc_tx_antennas_0_2.shape)
        noise_var_0_2[:, ::2] = noise_var[:, tx0_idx]
        noise_var_0_2[:, 1::2] = noise_var[:, tx1_idx]

        ce_0_2 = np.zeros((number_of_rx_antennas, 2, (len(tx0_idx) + len(tx1_idx))), dtype=complex)
        ce_0_2[:, 0, ::2] = channel_estimation[:, 0, tx0_idx]
        ce_0_2[:, 0, 1::2] = channel_estimation[:, 0, tx1_idx]
        ce_0_2[:, 1, ::2] = channel_estimation[:, 2, tx0_idx]
        ce_0_2[:, 1, 1::2] = channel_estimation[:, 2, tx1_idx]

        out_tx_0_2, ce_0_2, noise_var_0_2 = self._decode_stbc_2_tx_antennas(input_stbc_tx_antennas_0_2, ce_0_2,
                                                                            noise_var_0_2)

        # reshape, so that we can concatenate t and t+1 accordingly
        out_tx_0_2 = np.reshape(out_tx_0_2, (-1, 2))
        ce_0_2 = np.reshape(ce_0_2, (-1, 2))
        noise_var_0_2 = np.reshape(noise_var_0_2, (-1, 2))

        # antenna 1 and 3
        input_stbc_tx_antennas_1_3 = np.zeros((number_of_rx_antennas, len(tx2_idx) + len(tx3_idx)), dtype=complex)
        input_stbc_tx_antennas_1_3[:, ::2] = input_data[:, tx2_idx]
        input_stbc_tx_antennas_1_3[:, 1::2] = input_data[:, tx3_idx]

        noise_var_1_3 = np.zeros(input_stbc_tx_antennas_0_2.shape)
        noise_var_1_3[:, ::2] = noise_var[:, tx2_idx]
        noise_var_1_3[:, 1::2] = noise_var[:, tx3_idx]

        ce_1_3 = np.zeros((number_of_rx_antennas, 2, (len(tx2_idx) + len(tx3_idx))), dtype=complex)
        ce_1_3[:, 0, ::2] = channel_estimation[:, 1, tx2_idx]
        ce_1_3[:, 0, 1::2] = channel_estimation[:, 1, tx3_idx]
        ce_1_3[:, 1, ::2] = channel_estimation[:, 3, tx2_idx]
        ce_1_3[:, 1, 1::2] = channel_estimation[:, 3, tx3_idx]

        out_tx_1_3, ce_1_3, noise_var_1_3 = self._decode_stbc_2_tx_antennas(input_stbc_tx_antennas_1_3,
                                                                            ce_1_3, noise_var_1_3)

        # reshape so that we can concatenate t and t+1 accordingly
        out_tx_1_3 = np.reshape(out_tx_1_3, (-1, 2))
        ce_1_3 = np.reshape(ce_1_3, (-1, 2))
        noise_var_1_3 = np.reshape(noise_var_1_3, (-1, 2))

        output = np.hstack((out_tx_0_2, out_tx_1_3))
        output = np.reshape(output, (1, -1))

        channel_estimation_out = np.hstack((ce_0_2, ce_1_3))
        channel_estimation_out = np.reshape(channel_estimation_out, (1, -1))

        noise_var = np.hstack((noise_var_0_2, noise_var_1_3))
        noise_var = np.reshape(noise_var, (1, -1))

        return output, channel_estimation_out, noise_var

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


class MaximumRatioCombining(Mimo):

    yaml_tag: str = u'MRC'

    def __init__(self) -> None:
        """Maximum Ration Combining Precoding initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, input_data: np.array) -> np.ndarray:
        return input_data

    def decode(self, input_stream: np.ndarray) -> np.ndarray:
        # Decode data using MRC receive diversity with N_rx received antennas.
        #
        # Received signal with equal noise power is assumed, the decoded signal has same noise
        # level as input. It is assumed that all data have equal noise levels.

        channel_estimation = self.precoding.modem.reference_channel.estimate()
        noise_var = 0.0

        channel_estimation = np.squeeze(channel_estimation, axis=1)
        output_stream = np.sum(input_stream * channel_estimation.conj(), axis=0)[np.newaxis]
        # noise_var = np.sum(noise_var * (np.abs(stream_responses) ** 2), axis=0)[np.newaxis]
        # stream_responses = np.sum(np.abs(stream_responses) ** 2, axis=0)[np.newaxis]

        return output_stream

    @classmethod
    def to_yaml(cls: Type[MaximumRatioCombining], representer: SafeRepresenter, node: MaximumRatioCombining) -> ScalarNode:
        """Serialize a MaximumRatioCombining object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (MaximumRatioCombining):
                The MaximumRatioCombining instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        return representer.represent_scalar(cls.yaml_tag, None)

    @classmethod
    def from_yaml(cls: Type[MaximumRatioCombining], constructor: SafeConstructor, node: ScalarNode) -> MaximumRatioCombining:
        """Recall a new `MaximumRatioCombining` instance from YAML.

        Args:
            constructor (RoundTripConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `MaximumRatioCombining` serialization.

        Returns:
            MaximumRatioCombining:
                Newly created `MaximumRatioCombining` instance.
            """

        return cls()


class SpaceTimeBlockCoding(Mimo):

    yaml_tag: str = u'STBC'

    def __init__(self) -> None:
        """Space Time Block Coding initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, input_data: np.array) -> np.ndarray:
        return self._encode_stbc(input_data)

    def decode(self, input_stream: np.ndarray) -> np.ndarray:

        num_input_streams = input_stream.shape[0]
        channel_estimation = self.precoding.channel_estimate
        noise_var = 0.0

        if num_input_streams == 2:
            output, _, _ = self._decode_stbc_2_tx_antennas(input_stream, channel_estimation, noise_var)

        elif num_input_streams == 4:
            output, _, _ = self._decode_stbc_4_tx_antennas(input_stream, channel_estimation, noise_var)

        else:
            raise RuntimeError("Space-Time Block decoding is currently only available for 2 and 4 input streams")


class SpaceFrequencyBlockCoding(Mimo):

    yaml_tag: str = u'SFBC'

    def __init__(self) -> None:
        """Space Frequency Block Coding initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, input_data: np.array) -> np.ndarray:
        return self._encode_stbc(input_data)

    def decode(self, input_stream: np.ndarray) -> np.ndarray:

        num_input_streams = input_stream.shape[0]
        channel_estimation = self.precoding.channel_estimate
        noise_var = 0.0

        if num_input_streams == 2:
            output_stream, _, _ = self._decode_stbc_2_tx_antennas(input_stream, channel_estimation, noise_var)

        elif num_input_streams == 4:
            output_stream, _, _ = self._decode_stbc_4_tx_antennas(input_stream, channel_estimation, noise_var)

        else:
            raise RuntimeError("Space-Frequency Block decoding is currently only available for 2 and 4 input streams")

        return output_stream


class SpatialMultiplexingZeroForcing(Mimo):

    yaml_tag: str = u'SM-ZF'

    def __init__(self) -> None:
        """Spatial Multiplexing Zero Forcing initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, input_data: np.array) -> np.ndarray:

        number_of_symbols = np.int(input_data.size / self.number_of_streams)
        output = np.reshape(input_data, (self.number_of_streams, number_of_symbols), 'F')
        return output

    def decode(self, input_stream: np.ndarray) -> np.ndarray:

        channel_estimation = self.precoding.channel_estimate
        # noise_var = 0.0

        # number_of_symbols = input_stream.shape[1]
        # number_rx_antennas = input_stream.shape[0]
        ch = np.moveaxis(channel_estimation, -1, 0)

        # convert noise variance into noise covariance diagonal matrices
        # idx = np.arange(number_rx_antennas)
        # idx = idx + number_rx_antennas * idx
        # noise_covariance = np.zeros((number_of_symbols, number_rx_antennas * number_rx_antennas))
        # noise_covariance[:, idx] = noise_var
        # noise_covariance = np.reshape(noise_covariance, (number_of_symbols, number_rx_antennas,
        #                                                  number_rx_antennas))

        linear_decoder = np.linalg.pinv(ch)

        output = np.matmul(linear_decoder, input_stream.T[:, :, np.newaxis])
        output = output.squeeze(axis=2).T

        # noise_covariance = (linear_decoder @ np.transpose(linear_decoder.conj(), axes=[0, 2, 1])
        #                     @ noise_covariance)
        # noise_var_out = np.real(np.diagonal(noise_covariance, axis1=1, axis2=2)).T
        return output


class SpatialMultiplexingMinimumMeanSquareError(Mimo):

    yaml_tag: str = u'SM-MMSE'

    def __init__(self) -> None:
        """Spatial Multiplexing MMSE initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, input_data: np.array) -> np.ndarray:

        number_of_symbols = np.int(input_data.size / self.number_of_streams)
        output = np.reshape(input_data, (self.number_of_streams, number_of_symbols), 'F')
        return output

    def decode(self, input_stream: np.ndarray) -> np.ndarray:

        channel_estimation = self.precoding.channel_estimate
        noise_var = 0.0

        number_of_symbols = input_stream.shape[1]
        number_rx_antennas = input_stream.shape[0]
        ch = np.moveaxis(channel_estimation, -1, 0)

        # convert noise variance into noise covariance diagonal matrices
        idx = np.arange(number_rx_antennas)
        idx = idx + number_rx_antennas * idx
        noise_covariance = np.zeros((number_of_symbols, number_rx_antennas * number_rx_antennas))
        noise_covariance[:, idx] = noise_var
        noise_covariance = np.reshape(noise_covariance, (number_of_symbols, number_rx_antennas,
                                                         number_rx_antennas))

        ch_hermitian = np.transpose(ch, axes=[0, 2, 1]).conj()

        linear_decoder = ch_hermitian @ np.linalg.inv(ch @ ch_hermitian + noise_covariance)
        norm = np.diagonal(linear_decoder @ ch, axis1=1, axis2=2).T

        output = np.matmul(linear_decoder, input_stream.T[:, :, np.newaxis])
        output = 1 / norm.T[:, :, np.newaxis] * output
        output = output.squeeze(axis=2).T

        # noise_var_out = 1./np.real(norm) - 1
        return output
