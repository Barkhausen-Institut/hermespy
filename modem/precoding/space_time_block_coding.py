from __future__ import annotations
from typing import Type, Tuple
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node
import numpy as np

from . import SymbolPrecoder


class SpaceTimeBlockCoding(SymbolPrecoder):
    """A precoder distributing symbols in space and time.
    
    Cool.
    """

    yaml_tag: str = u'STBC'

    def __init__(self) -> None:
        """Space-Time Block Coding object initialization."""

        SymbolPrecoder.__init__(self)

    def encode(self, symbol_stream: np.ndarray) -> np.ndarray:
        """Encode data into multiple antennas with space-time/frequency block codes

        Currently STBCs with 2 or 4 transmit antennas are supported,
        following 3GPP TS 36.211, Sec, 6.3.3.3)

        Args:
            symbol_stream(np.array): Input signal with K symbols.

        Returns:
            output (np.array): Encoded data with size N_tx x K symbols
        """

        number_of_symbols = symbol_stream.shape[1]
        num_tx_streams = self.required_num_output_streams
        input_data = symbol_stream[0, :]

        if num_tx_streams == 2:

            output = np.zeros((2, number_of_symbols), dtype=complex)

            # alamouti precoding
            even_idx = np.arange(0, number_of_symbols, 2)
            odd_idx = even_idx + 1

            output[0, :] = input_data
            output[1, even_idx] = -np.conj(input_data[odd_idx])
            output[1, odd_idx] = np.conj(input_data[even_idx])

            output = output / np.sqrt(2)

        elif num_tx_streams == 4:
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
            raise ValueError(f"Number of transmit streams ({num_tx_streams}) "
                             "not supported in space-time/frequency code")

        return output

    def decode(self, symbol_stream: np.ndarray, stream_responses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        num_rx_streams = symbol_stream.shape[1]

        if num_rx_streams == 2:
            return self.__decode_stbc_2_rx_antennas(symbol_stream, stream_responses, 1.0)

        if num_rx_streams == 4:
            return self.__decode_stbc_4_rx_antennas(symbol_stream, stream_responses, 1.0)

        raise ValueError(f"Number of receive streams ({num_rx_streams}) "
                         "not supported in space-time/frequency code")

    def __decode_stbc_2_rx_antennas(self,
                                   input_data: np.ndarray,
                                   channel_estimation: np.ndarray, noise_var) -> Tuple[np.ndarray, np.ndarray]:
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

        return output, channel_estimation_out

    def __decode_stbc_4_rx_antennas(self,
                                    input_data: np.ndarray,
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

        # channel_estimation = np.squeeze(channel_estimation)

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

        out_tx_0_2, ce_0_2, noise_var_0_2 = self.__decode_stbc_2_rx_antennas(input_stbc_tx_antennas_0_2, ce_0_2,
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

        out_tx_1_3, ce_1_3, noise_var_1_3 = self.__decode_stbc_2_rx_antennas(input_stbc_tx_antennas_1_3,
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

    @property
    def num_input_streams(self) -> int:

        # DFT precoding does not alter the number of symbol streams
        return self.precoding.required_inputs(self)

    @property
    def num_output_streams(self) -> int:

        # DFT precoding does not alter the number of symbol streams
        return self.precoding.required_inputs(self)

    @classmethod
    def to_yaml(cls: Type[SpaceTimeBlockCoding], representer: SafeRepresenter, node: SpaceTimeBlockCoding) -> Node:
        """Serialize a `SpaceTimeBlockCoding` precoder to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (SpaceTimeBlockCoding):
                The `SpaceTimeBlockCoding` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        return representer.represent_scalar(cls.yaml_tag, "")

    @classmethod
    def from_yaml(cls: Type[SpaceTimeBlockCoding], constructor: SafeConstructor, node: Node) -> SpaceTimeBlockCoding:
        """Recall a new `SpaceTimeBlockCoding` precoder from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `SpaceTimeBlockCoding` serialization.

        Returns:
            SpaceTimeBlockCoding:
                Newly created `SpaceTimeBlockCoding` instance.
        """

        return cls()
