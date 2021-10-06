# -*- coding: utf-8 -*-
"""LDPC Encoding."""

from __future__ import annotations
from typing import Tuple, Set, Optional
from scipy.io import loadmat
from fractions import Fraction
import os
import numpy as np

from modem.coding.encoder import Encoder

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer, Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class LDPC(Encoder):
    """Implementation of an LDPC Encoder.

    LDPC decoder using a serial C (check node) schedule and  message-passing as introduced in
    [E. Sharon, S. Litsyn and J. Goldberger, "An efficient message-passing schedule for LDPC
    decoding," 2004 23rd IEEE Convention of Electrical and Electronics Engineers in Israel,
    2004, pp. 223-226].

    Attributes:
        CODE_RATES (Set[Fraction]): The supported code rates.
        BLOCK_SIZES (Set[int]): The supported input block sizes.
    """

    CODE_RATES: Set[Fraction] = [Fraction(1, 3),
                                 Fraction(1, 2),
                                 Fraction(2, 3),
                                 Fraction(3, 4),
                                 Fraction(4, 5),
                                 Fraction(5, 6)]
    BLOCK_SIZES: Set[int] = [256, 512, 1024, 2048, 4096, 8192]

    yaml_tag = u'LDPC'
    __rate: Fraction
    _G: np.ndarray
    _H: np.ndarray
    __iterations: int
    __custom_codes: Set[str]

    def __init__(self,
                 block_size: int = 256,
                 rate: Fraction = Fraction(2, 3),
                 iterations: int = 20,
                 custom_codes: Set[str] = None) -> None:
        """Object initialization.

        Args:
            block_size (int, optional): LDPC coding matrix block size.
            rate: (Fraction, optional): Coding rate.
            iterations (int, optional): Number of decoding iterations.
            custom_codes (Set[str], optional): Discovery path for custom LDPC codings.
        """

        Encoder.__init__(self)

        # Will be initialized and managed by set_rate
        if custom_codes is None:
            custom_codes = set()
        self._G = np.empty(0)
        self._H = np.empty(0)
        self.__rate = Fraction(1, 2)    # Directly overwritten by set_rate, so value does not matter

        self.iterations = iterations

        if custom_codes is not None:
            self.__custom_codes = custom_codes
        else:
            self.__custom_codes = set()

        self.set_rate(block_size, rate)

    @property
    def iterations(self) -> int:
        """Access the configured number of coding iterations.

        Returns:
            int: The number of coding iterations.
        """

        return self.__iterations

    @iterations.setter
    def iterations(self, num: int) -> None:
        """Modify the configured number of coding iterations.

        Args:
            num (int): The new number of coding iterations.

        Raises:
            ValueError: If the number of iterations is less than one.
        """

        if num < 1:
            raise ValueError("Number of iterations must be greater or equal to zero")

        self.__iterations = num

    @property
    def custom_codes(self) -> Set[str]:
        """Access and modify custom code lookup paths."""

        return self.__custom_codes

    def encode(self, bits: np.array) -> np.array:
        return (bits @ self._G) % 2

    def decode(self, encoded_bits: np.array) -> np.array:

        # Transform bits from {0, 1} format to {-1, 1}
        codes = -encoded_bits.copy()
        codes[codes > -.5] = 1.
        eps = 2.22045e-16

        Rcv = np.zeros(self._H.shape)
        punc_bits = np.zeros(self._H.shape[1] - self._G.shape[1])
        Qv = np.concatenate((punc_bits, codes))

        # Loop over the number of iteration in the SPA algorithm
        for spa_ind in range(self.iterations):

            # Loop over the check nodes
            for check_ind in range(self.num_parity_bits):

                # Finds the neighbouring variable nodes connected to the current check node
                nb_var_nodes = np.nonzero(self._H[check_ind, :])

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

        # Return bit format from {-1, 1} format to {0, 1}
        return np.array(Qv[:self.bit_block_size] < 0, dtype=int)

    @property
    def bit_block_size(self) -> int:
        return self._G.shape[0]

    @property
    def code_block_size(self) -> int:
        return self._G.shape[1]

    @property
    def num_parity_bits(self) -> int:
        """The number of parity bis introduced by the LDPC coding.

        Returns:
            int: the number of parity bits.
        """

        # The number of parity bits is identical to the first dimension of the parity check matrix H
        return self._H.shape[0]

    @property
    def rate(self) -> float:
        return float(self.__rate)

    def set_rate(self, block_size: int, rate: Fraction) -> None:
        """Configure the coding rate.

        Args:
            block_size (int): LDPC matrix coding block size.
            rate (Fraction): Code rate, i.e. the relation between number of data and code bits.

        Raises:
            ValueError: If the requested `bit_block_size` is not supported.
            ValueError: If the requested `rate` is not supported
        """

        try:

            # Update internal coding matrices
            self._G, self._H = self.__read_precalculated_codes(block_size, rate)
            self.__rate = rate

        except RuntimeError as error:

            if block_size not in self.BLOCK_SIZES:
                raise ValueError("Code block size of {} codewords is currently not supported by the LDPC encoder"
                                 .format(block_size))

            if rate not in self.CODE_RATES:
                raise ValueError("Rate of {}/{} currently not supported by the LDPC encoder".format(rate.numerator,
                                                                                                    rate.denominator))
            # The error was unexpected, re-raise it
            raise error

    def __read_precalculated_codes(self, block_size: int, rate: Fraction) -> Tuple[np.array, np.array]:
        """Read precalculated LDPC coding matrices from a Matlab save file.

        The function expects save files to be named after the scheme `BS*_*_*.mat`,
        the first wildcard being the bit block size, the following ones the rate numerator
        and denominator respectively, i.e. BS256_1_2.mat for block size 256 and rate 1/2.

        Args:
            block_size (int): LDPC matrix coding block size.
            rate (Fraction): Code rate, i.e. the relation between number of data and code bits.

        Raises:
            RuntimeError: If a valid save file could not be detected in all lookup paths.

        Returns:
            Tuple[np.array, np.array]: LDPC coding and decoding matrices.
        """

        lookup_paths = {
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'precalculated_codes')
        }.union(self.__custom_codes)

        mat_filename = "BS{}_CR{}_{}.mat".format(
            block_size,
            rate.numerator,
            rate.denominator
        )

        # Search for the mat file in all possible locations
        mat: Optional[dict] = None
        for lookup_path in lookup_paths:

            lookup_file = os.path.join(lookup_path, mat_filename)

            if os.path.exists(lookup_file):
                mat = loadmat(lookup_file, squeeze_me=True)

        if mat is None:
            raise RuntimeError('Matlab file for selected code parameters not found')

        Z = mat['LDPC']['Z'].item()
        H = mat['LDPC']['H'].item()

        G = mat['LDPC']['G'].item()
        G = G[:, 2*Z:]

        return G, H
