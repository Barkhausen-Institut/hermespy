from typing import List
from scipy.io import loadmat
import os
import warnings

import numpy as np


from modem.coding.encoder import Encoder
from parameters_parser.parameters_ldpc_encoder import ParametersLdpcEncoder
try:
    from modem.coding import ldpc_binding
except ImportError:
    pass


class LdpcEncoder(Encoder):
    """Implementation of an LDPC Encoder.

    LDPC decoder using a serial C (check node) schedule and  message-passing as introduced in
    [E. Sharon, S. Litsyn and J. Goldberger, "An efficient message-passing schedule for LDPC
    decoding," 2004 23rd IEEE Convention of Electrical and Electronics Engineers in Israel,
    2004, pp. 223-226].
    """

    def __init__(self, params: ParametersLdpcEncoder, bits_in_frame: int) -> None:

        self.__block_size = 1
        self.__

        self.params = params
        self.bits_in_frame = bits_in_frame
        self._read_precalculated_codes()

        if self.params.use_binding and 'ldpc_binding' not in globals():
            self.params.use_binding = False
            warnings.warn("LDPC C++ binding could not ne imported, falling back to slower Python LDPC implementation")

        if self.code_blocks < 1:
            raise ValueError("Code block must not be longer than bits in frame")

    @property
    def source_bits(self) -> int:
        return self.code_blocks * self.data_bits_k

    @property
    def encoded_bits_n(self) -> int:
        return self.num_total_bits

    @property
    def data_bits_k(self) -> int:
        return self.num_info_bits

    def encode(self, data_bits: List[np.array]) -> List[np.array]:
        if self.params.use_binding:
            return self.__encode_binding(data_bits)
        else:
            return self.__encode_python(data_bits)

    def __encode_python(self, data_bits: List[np.array]) -> List[np.array]:
        no_bits = 0
        encoded_words = []
        for block in data_bits:
            if not (len(block) % self.data_bits_k == 0):
                raise ValueError("Block length must be an integer multiple of k")
            for code_block_idx in range(self.code_blocks):
                code_word = (block[:self.num_info_bits] @ self.G) % 2
                # Puncturing the 2*Z first systematic bits to ensure the correct code rate
                code_word = code_word[2 * self.Z:]
                encoded_words.append(code_word)

                block = block[self.num_info_bits:]
                no_bits += self.encoded_bits_n

        if (self.bits_in_frame - no_bits) > 0:
            encoded_words.append(np.random.randint(2, size=self.bits_in_frame - no_bits))

        return encoded_words

    def __encode_binding(self, data_bits: List[np.array]) -> List[np.array]:
        encoded_words = ldpc_binding.encode(
            data_bits, self.G, self.Z, self.num_info_bits, self.encoded_bits_n,
            self.data_bits_k, self.code_blocks, self.bits_in_frame
        )
        return encoded_words

    def decode(self, encoded_bits: List[np.array]) -> List[np.array]:
        if self.params.use_binding:
            return self.__decode_binding(encoded_bits)
        else:
            return self.__decode_python(encoded_bits)

    def __decode_python(self, encoded_bits: List[np.array]) -> List[np.array]:
        eps = 2.22045e-16
        decoded_blocks: List[np.array] = []
        for block in encoded_bits:
            dec_block: np.array = np.array([])
            for code_block in range(self.code_blocks):
                curr_code_block = -block[:self.encoded_bits_n]

                Rcv = np.zeros((self.number_parity_bits, self.num_total_bits + 2 * self.Z))
                punc_bits = np.zeros(2 * self.Z)
                Qv = np.concatenate((punc_bits, curr_code_block))
                # Loop over the number of iteration in the SPA algorithm
                for spa_ind in range(self.params.no_iterations):

                    # Loop over the check nodes
                    for check_ind in range(self.number_parity_bits):

                        # Finds the neighbouring variable nodes connected to the current check node
                        nb_var_nodes = np.nonzero(self.H[check_ind, :])

                        # Temporary updated of encoded_bits
                        temp_llr = Qv[nb_var_nodes] - Rcv[check_ind, nb_var_nodes]

                        # Magnitude of S
                        S_mag = np.sum(-np.log(eps + np.tanh(np.abs(temp_llr) / 2)))

                        # Sign of S - counting the number of negative elements in temp_llr
                        if np.sum(temp_llr < 0) % 2 == 0:
                            S_sign = +1
                        else:
                            S_sign = -1
                        # Loop over the variable nodes
                        for var_ind in range(len(nb_var_nodes[0])):
                            var_pos = nb_var_nodes[0][var_ind]
                            Q_temp = Qv[var_pos] - Rcv[check_ind, var_pos]
                            Q_temp_mag = -np.log(eps + np.tanh(np.abs(Q_temp) / 2))
                            Q_temp_sign = np.sign(Q_temp + eps)

                            # Update message passing matrix
                            Rcv[check_ind, var_pos] = S_sign * Q_temp_sign * (
                                -np.log(eps + np.tanh(np.abs(S_mag - Q_temp_mag) / 2)))

                            # Update Qv
                            Qv[var_pos] = Q_temp + Rcv[check_ind, var_pos]

                dec_code_block = np.array(Qv[:self.num_info_bits] < 0, dtype=int)
                dec_block = np.append(dec_block, dec_code_block)

                block = block[self.encoded_bits_n:]
            decoded_blocks.append(dec_block)

        return decoded_blocks

    def __decode_binding(self, encoded_bits: List[np.array]) -> List[np.array]:
        decoded_blocks = ldpc_binding.decode(
            encoded_bits, self.encoded_bits_n, self.code_blocks, self.number_parity_bits,
            self.num_total_bits, self.Z, self.params.no_iterations, self.H, self.num_info_bits
        )
        return decoded_blocks

    def _read_precalculated_codes(self):
        # Supports code rates Rc = ['1/3', '1/2', '2/3', '3/4', '5/6']
        # and codeword block size n = [256, 512, 1024, 2048, 4096, 8192]
        precalculated_codes_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'precalculated_codes'
        )
        mat_filename = "BS{0}_CR{1}_{2}.mat".format(
                            self.params.block_size,
                            self.params.code_rate_fraction.numerator,
                            self.params.code_rate_fraction.denominator
                        )
        path_mat_files: List[str] = [os.path.join(precalculated_codes_dir, mat_filename)]

        if self.params.custom_ldpc_codes != "":
            path_mat_files.append(
                os.path.join(self.params.custom_ldpc_codes, mat_filename))

        ldpc_file_found = False
        for path in path_mat_files:
            if os.path.exists(path):
                LDPC = loadmat(path, squeeze_me=True)
                ldpc_file_found = True

        if not ldpc_file_found:
            raise ValueError('Error: The specified block size or code rate are not supported.')

        self.H = LDPC['LDPC']['H'].item()
        self.G = LDPC['LDPC']['G'].item()

        self.number_parity_bits = LDPC['LDPC']['numParBits'].item()
        self.num_total_bits = LDPC['LDPC']['numTotBits'].item()
        self.Z = LDPC['LDPC']['Z'].item()
        self.num_info_bits = LDPC['LDPC']['numInfBits'].item()
