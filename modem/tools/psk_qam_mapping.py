"""Mapping between bits and PSK/QAM/PAM constellation


This module provides a class for a PSK/QAM mapper/demapper.

The following features are supported:
- arbitrary 2D (complex) constellation mapping can be given
- default Gray-coded constellations for BPSK, QPSK, 8-PSK, 4-, 8-, 16- PAM, 16-, 64-
    and 256-QAM are provided
- all default constellations follow 3GPP standards TS 36.211
    (except 8-PSK, which is not defined in 3GPP)
- hard and soft (LLR) output are available

This implementation has currently the following limitations:
- LLR available only for default BPSK, QPSK, 4-, 8-, 16- PAM, 16-, 64- and 256-QAM
- only linear approximation of LLR is considered, similar to the one described in:
    Tosato, Bisaglia, "Simplified Soft-Output Demapper for Binary Interleaved COFDM with
    Application to HIPERLAN/2", Proceedings of IEEE International Commun. Conf. (ICC) 2002
"""
from typing import Union
import numpy as np


class PskQamMapping(object):
    """Implements the mapping of bits into complex numbers, following a PSK/QAM modulation.

    Attributes:
        modulation_order(int): size of modulation constellation.
        mapping_available(list of int, class level): list of available default constellations.
        bits_per_symbol(int): number of bits in modulation symbol.
        soft_output(bool):
            if True, then soft output (LLR) will be provided. #
            If False, then estimated bits are returned.
    """

    mapping_available = [2, 4, 8, 16, 64, 64, 256]
    mapping_available_pam = [2, 4, 8, 16]

    _psk8_map = np.exp(1j *
                       np.array([0, 1 /
                                 4, 3 /
                                 4, 1 /
                                 2, -
                                 1 /
                                 4, -
                                 1 /
                                 2, 1, -
                                 3 /
                                 4]) *
                       np.pi)  # gray-coded 8-PSK map

    def __init__(
            self,
            modulation_order: int,
            mapping: np.ndarray = None,
            soft_output: bool = False,
            is_complex: bool = True):
        """
        Args:
            modulation_order (int):
                Number of points in the constellation. Must be a power of two.
            mapping (numpy.ndarray, optional):
                Vector with length `modulation_order` defining the mapping between bits
                and modulation symbols. At each symbol, bits are input MSB first.
                For example, with a 32-point constellation, the bit sequence 01101
                corresponds to the decimal 13,  and hence will be mapped to the
                13-th element in 'mapping' vector.
                It is optional for certain modulation orders, which are given in
                PskQamMapping.mapping_available, and for which a default mapping is provided.
            soft_output(bool):
                if True, then soft output (LLR) will be provided.
                If False, then estimated bits (0 or 1) are returned.
            is_complex(bool):
                if True, then complex modulation is considered (PSK/QAM),
                if False, then real-valued modulation is considered (PAM)
        """
        if modulation_order <= 0 or (
                modulation_order & (modulation_order - 1)) != 0:
            raise ValueError('modulation_order must be a power of two')

        if is_complex and modulation_order not in PskQamMapping.mapping_available and mapping is None:
            raise ValueError(
                'constellation must be provided for this modulation order')

        if not is_complex and modulation_order not in PskQamMapping.mapping_available_pam and mapping is None:
            raise ValueError(
                'constellation must be provided for this modulation order')

        if mapping is not None and mapping.size != modulation_order:
            raise ValueError(
                'mapping must have the same number of elements as the modulation order')

        self.is_complex = is_complex

        self.modulation_order = modulation_order
        self.bits_per_symbol = int(np.log2(self.modulation_order))

        self.mapping = mapping
        if self.mapping is None and modulation_order == 8 and is_complex:
            self.mapping = PskQamMapping._psk8_map
        elif self.mapping is not None:
            # normalize mapping
            energy = np.mean(np.abs(self.mapping)**2)
            self.mapping = mapping / np.sqrt(energy)

        self.soft_output = soft_output

    def get_symbols(self, bits: np.ndarray) -> np.ndarray:
        """Calculates the complex numbers corresponding to the information in 'bits'.

        Note:
            The constellation is normalized, such that the mean symbol energy is unitary.

        Args:
            bits (np.ndarray):
                Vector with N elements,
                corresponding to the bits to be modulated.

        Returns:
            symbols(numpy.ndarray):
                Vector of N/log2(modulation_order) elements with modulated symbols.
        """
        number_symbols = int(bits.size / self.bits_per_symbol)
        # bits in rows, symbols in columns
        bits = np.reshape(
            bits, (self.bits_per_symbol, number_symbols), order='F')

        if self.mapping is not None:
            # e.g. [8, 4, 2, 1]
            power_of_2 = 2 ** np.arange(self.bits_per_symbol - 1, -1, -1)
            idx = np.matmul(power_of_2, bits)  # multiply to get symbol value
            symbols = self.mapping[idx]

        else:
            # use 3GPP mapping for BPSK, QPSK, 16-,64- and 256-QAM (or PAM
            # equivalent)
            if self.modulation_order == 2:
                # BPSK
                symbols = self.generate_pam_symbol_3gpp(2, bits) + 1j * 0
            elif self.modulation_order == 4 and self.is_complex:
                # QPSK
                real_part = self.generate_pam_symbol_3gpp(2, bits[0, :])
                imag_part = self.generate_pam_symbol_3gpp(2, bits[1, :])
                symbols = (real_part + 1j * imag_part) / np.sqrt(2)
            elif self.modulation_order == 4 and not self.is_complex:
                # 4-PAM
                symbols = self.generate_pam_symbol_3gpp(
                    4, bits) / np.sqrt(5) + 1j * 0
            elif self.modulation_order == 8 and not self.is_complex:
                # 8-PAM
                symbols = self.generate_pam_symbol_3gpp(
                    8, bits) / np.sqrt(21) + 1j * 0
            elif self.modulation_order == 16 and self.is_complex:
                # 16-QAM
                real_part = self.generate_pam_symbol_3gpp(4, bits[[0, 2], :])
                imag_part = self.generate_pam_symbol_3gpp(4, bits[[1, 3], :])
                symbols = (real_part + 1j * imag_part) / np.sqrt(10)
            elif self.modulation_order == 16 and not self.is_complex:
                # 16-PAM
                symbols = self.generate_pam_symbol_3gpp(
                    16, bits) / np.sqrt(85) + 1j * 0
            elif self.modulation_order == 64 and self.is_complex:
                real_part = self.generate_pam_symbol_3gpp(
                    8, bits[[0, 2, 4], :])
                imag_part = self.generate_pam_symbol_3gpp(
                    8, bits[[1, 3, 5], :])
                symbols = (real_part + 1j * imag_part) / np.sqrt(42)
            elif self.modulation_order == 256 and self.is_complex:
                real_part = self.generate_pam_symbol_3gpp(
                    16, bits[[0, 2, 4, 6], :])
                imag_part = self.generate_pam_symbol_3gpp(
                    16, bits[[1, 3, 5, 7], :])
                symbols = (real_part + 1j * imag_part) / np.sqrt(170)
            else:
                if self.is_complex:
                    modulation_type = 'QAM'
                else:
                    modulation_type = 'PAM'
                raise ValueError(
                    f"Modulation ({self.modulation_order}-{modulation_type}) not supported")

        return np.ravel(symbols)

    def detect_bits(self,
                    rx_symbols: np.ndarray,
                    noise_variance: Union[np.ndarray,
                                          float] = 1) -> np.ndarray:
        """Returns either bits or LLR for the provided symbols.

        Args:
            rx_symbols(np.ndarray):
                Vector of N received symbols, for which the bits/LLR will be estimated
            noise_variance (float or np.ndarray, optional):
                vector with the noise variance in each received symbol. If a
                scalar is given, then the same variance is assumed for all symbols.
                This is only relevant if 'self.soft_output' is true.

        Returns:
            bits(np.ndarray):
                Vector of N * self.bits_per_symbol elements containing either the estimated
                bits or the LLR values of each bit, depending on the value of 'self.soft_output'
        """
        number_of_bits = rx_symbols.size * self.bits_per_symbol
        llr = np.zeros(number_of_bits)

        # set starting index of encoded symbol (MSB)
        bits_idx = np.arange(
            0,
            number_of_bits,
            self.bits_per_symbol,
            dtype=int)

        if self.mapping is not None:
            if not self.soft_output:
                # get closest point in constellation diagram
                dist = np.abs(rx_symbols - self.mapping.reshape(-1, 1))
                min_index = np.argmin(dist, axis=0)

                for bit_offset in range(
                        self.bits_per_symbol):  # iterate from the MSB to the LSB
                    # calculate encoded value
                    power_of_2 = int(
                        2**(self.bits_per_symbol - bit_offset - 1))
                    llr[bits_idx +
                        bit_offset] = np.bitwise_and(min_index, power_of_2) > 0
                    llr = llr * 2 - 1
            else:
                raise ValueError(
                    "soft output not yet supported for this modulation scheme")

        # use 3GPP mapping for BPSK, QPSK, 16-,64- and 256-QAM
        elif self.modulation_order == 2:
            # BPSK
            llr = self.get_llr_3gpp(
                2, np.real(rx_symbols), False) / noise_variance

        elif self.modulation_order == 4 and self.is_complex:
            # QPSK
            llr[0::2] = self.get_llr_3gpp(
                2, np.real(rx_symbols), True) / noise_variance
            llr[1::2] = self.get_llr_3gpp(
                2, np.imag(rx_symbols), True) / noise_variance

        elif self.modulation_order == 4 and not self.is_complex:
            # 4-PAM
            llr = self.get_llr_3gpp(
                4, np.real(rx_symbols), False) / noise_variance

        elif self.modulation_order == 8 and not self.is_complex:
            # 8-PAM
            llr = self.get_llr_3gpp(
                8, np.real(rx_symbols), False) / noise_variance

        elif self.modulation_order == 16 and not self.is_complex:
            # 16-PAM
            llr = self.get_llr_3gpp(
                16, np.real(rx_symbols), False) / noise_variance

        elif self.modulation_order == 16 and self.is_complex:
            # 16-QAM
            llr[0::2] = self.get_llr_3gpp(
                4, np.real(rx_symbols), True) / noise_variance
            llr[1::2] = self.get_llr_3gpp(
                4, np.imag(rx_symbols), True) / noise_variance

        elif self.modulation_order == 64 and self.is_complex:
            # 64-QAM
            llr[0::2] = self.get_llr_3gpp(
                8, np.real(rx_symbols), True) / noise_variance
            llr[1::2] = self.get_llr_3gpp(
                8, np.imag(rx_symbols), True) / noise_variance

        elif self.modulation_order == 256 and self.is_complex:
            # 256-QAM
            llr[0::2] = self.get_llr_3gpp(
                16, np.real(rx_symbols), True) / noise_variance
            llr[1::2] = self.get_llr_3gpp(
                16, np.imag(rx_symbols), True) / noise_variance

        else:
            raise ValueError("Unsupported modulation scheme")

        if not self.soft_output:
            bits = llr > 0
            return bits
        else:
            return llr

    @staticmethod
    def generate_pam_symbol_3gpp(
            modulation_order, bits: np.ndarray) -> np.ndarray:
        """Returns 1D amplitudes following 3GPP modulation mapping.

        3GPP has defined in TS 36.211 mapping tables from bits into complex symbols.
        Since the mapping from bits into amplitudes is the same for both I and Q components,
        and this function maps blocks of N bits into one of M=2^N possible (real-valued)
        amplitudes.

        Args:
            modulation_order(int): modulation order M. M=2, 4, 8, 16 are supported
            bits(np.ndarray): N x K array, with K the number of symbols

        Returns:
            symbols(np.ndarray):
                Vector of K real-valued symbols. Note that the symbols are not normalized,
                and range from -(M-1) to (M+1) with step 2, e..g., for M=8,
                values can be -7, -5, -3, -1, 1, 3, 5, 7.
        """

        if modulation_order == 2:
            symbols = 1.0 - 2 * bits
        elif modulation_order == 4:
            symbols = (1 - 2 * bits[0, :]) * (1 + 2 * bits[1, :])
        elif modulation_order == 8:
            symbols = (2 * bits[0, :] - 1) * \
                ((1 - 2 * bits[1, :]) * (1 + 2 * bits[2, :]) - 4)
        elif modulation_order == 16:
            symbols = (((((1 - 2 * bits[2,
                                        :]) * (1 + 2 * bits[3,
                                                            :]) - 4) * (-1 + 2 * bits[1,
                                                                                      :]))
                        - 8)
                       * (-1 + 2 * bits[0,
                                        :]))
        else:
            raise ValueError(
                f"unsupported modulation order ({modulation_order})")

        return symbols

    @staticmethod
    def get_llr_3gpp(modulation_order, rx_symbols: np.ndarray,
                     is_complex: bool) -> np.ndarray:
        """Returns LLR for each bit based on a received symbol, following 1D 3GPP modulation mapping.

        3GPP has defined in TS 36.211 mapping tables from bits into complex symbols.
        Since the mapping from bits into amplitudes is the same for both I and Q
        components, and this function maps received real-valued amplitudes into
        blocks of N = log2(M) log-likelihood ratios (LLR) for all bits, with M the modulation
        order.

        Only linear approximation of LLR is considered, similar to the one described in:
        Tosato, Bisaglia, "Simplified Soft-Output Demapper for Binary Interleaved COFDM with
        Application to HIPERLAN/2", Proceedings of IEEE International Commun. Conf. (ICC) 2002

        LLR calculation is available for real-valued modulations of order 2, 4, 8 or 16.

        LLR is returned considering unit-power Gaussian noise at all symbols.

        Args:
            modulation_order(int): modulation order M. M=2, 4, 8, 16 are supported
            rx_symbols(np.ndarray): array with K received symbols
            is_complex(bool):
                if True, then complex modulation is considered (PSK/QAM),
                If False, then real-valued modulation is considered (PAM)

        Returns:
            llr(np.ndarray): Vector of N x K elements with the LLR values.
        """

        if is_complex:
            rx_symbols = rx_symbols * np.sqrt(2)

        if modulation_order == 2:
            llr = -2 * rx_symbols

        elif modulation_order == 4:
            llr = np.zeros([2, rx_symbols.size])

            rx_symbols = rx_symbols * np.sqrt(5)

            llr[0, :] = 2 * ((rx_symbols <= -2) * (-4 * (1 + rx_symbols))
                             + np.logical_and(rx_symbols > -2, rx_symbols <= 2)
                             * (-2 * rx_symbols)
                             + (rx_symbols > 2) * (4 * (1 - rx_symbols)))

            llr[1, :] = 2 * ((rx_symbols <= 0) * (-2 * (2 + rx_symbols)
                                                  ) + (rx_symbols > 0)
                             * (-2 * (2 - rx_symbols)))

            llr = llr / 5

        elif modulation_order == 8:
            llr = np.zeros([3, rx_symbols.size])

            rx_symbols = rx_symbols * np.sqrt(21)

            llr[0, :] = 4 * ((rx_symbols <= -6) * (-4 * (3 + rx_symbols)) +
                             np.bitwise_and(rx_symbols > -6, rx_symbols <= -4)
                             * (-3 * (2 + rx_symbols)) +
                             np.bitwise_and(rx_symbols > -4, rx_symbols <= -2)
                             * (-2 * (1 + rx_symbols)) +
                             np.bitwise_and(rx_symbols > -2, rx_symbols <= 2)
                             * (-rx_symbols) +
                             np.bitwise_and(rx_symbols > 2, rx_symbols <= 4)
                             * (2 * (1 - rx_symbols)) +
                             np.bitwise_and(rx_symbols > 4, rx_symbols <= 6)
                             * (3 * (2 - rx_symbols)) +
                             (rx_symbols > 6) * (4 * (3 - rx_symbols)))

            llr[1, :] = 4 * ((rx_symbols <= -6) * (-2 * (5 + rx_symbols)) +
                             np.bitwise_and(rx_symbols > -6, rx_symbols <= -2)
                             * (-(4 + rx_symbols)) +
                             np.bitwise_and(rx_symbols > -2, rx_symbols <= 0)
                             * (-2 * (3 + rx_symbols)) +
                             np.bitwise_and(rx_symbols > 0, rx_symbols <= 2)
                             * (-2 * (3 - rx_symbols)) +
                             np.bitwise_and(rx_symbols > 2, rx_symbols <= 6)
                             * (-(4 - rx_symbols)) +
                             (rx_symbols > 6) * (-2 * (5 - rx_symbols)))

            llr[2, :] = 4 * ((rx_symbols <= -4) * (-(6 + rx_symbols)) +
                             np.bitwise_and(rx_symbols > -4, rx_symbols <= 0)
                             * (2 + rx_symbols) +
                             np.bitwise_and(rx_symbols > 0, rx_symbols <= 4)
                             * (2 - rx_symbols) +
                             (rx_symbols > 4) * (-(6 - rx_symbols)))
            llr = llr / 21

        elif modulation_order == 16:
            llr = np.zeros([4, rx_symbols.size])

            rx_symbols = rx_symbols * np.sqrt(85)

            llr[0, :] = 8 * ((rx_symbols <= -14) * (-4 * (7 + rx_symbols)) -
                             np.bitwise_and(rx_symbols > -14, rx_symbols <= -12)
                             * 3.5 * (6 + rx_symbols) -
                             np.bitwise_and(rx_symbols > -12, rx_symbols <= -10)
                             * 3 * (5 + rx_symbols) -
                             np.bitwise_and(rx_symbols > -10, rx_symbols <= -8)
                             * 2.5 * (4 + rx_symbols) -
                             np.bitwise_and(rx_symbols > -8, rx_symbols <= -6)
                             * 2 * (3 + rx_symbols) -
                             np.bitwise_and(rx_symbols > -6, rx_symbols <= -4)
                             * 1.5 * (2 + rx_symbols) -
                             np.bitwise_and(rx_symbols > -4, rx_symbols <= -2)
                             * (1 + rx_symbols) -
                             np.bitwise_and(rx_symbols > -2, rx_symbols <= 2)
                             * 0.5 * rx_symbols +
                             np.bitwise_and(rx_symbols > 2, rx_symbols <= 4)
                             * (1 - rx_symbols) +
                             np.bitwise_and(rx_symbols > 4, rx_symbols <= 6)
                             * 1.5 * (2 - rx_symbols) +
                             np.bitwise_and(rx_symbols > 6, rx_symbols <= 8)
                             * 2 * (3 - rx_symbols) +
                             np.bitwise_and(rx_symbols > 8, rx_symbols <= 10)
                             * 2.5 * (4 - rx_symbols) +
                             np.bitwise_and(rx_symbols > 10, rx_symbols <= 12)
                             * 3 * (5 - rx_symbols) +
                             np.bitwise_and(rx_symbols > 12, rx_symbols <= 14)
                             * 3.5 * (6 - rx_symbols) +
                             (rx_symbols > 14) * 4 * (7 - rx_symbols))

            llr[1, :] = 8 * ((rx_symbols <= -14) * (-2 * (11 + rx_symbols)) -
                             np.bitwise_and(rx_symbols > -14, rx_symbols <= -12)
                             * 1.5 * (10 + rx_symbols) -
                             np.bitwise_and(rx_symbols > -12, rx_symbols <= -10)
                             * (9 + rx_symbols) -
                             np.bitwise_and(rx_symbols > -10, rx_symbols <= -6)
                             * 0.5 * (8 + rx_symbols) -
                             np.bitwise_and(rx_symbols > -6, rx_symbols <= -4)
                             * (7 + rx_symbols) -
                             np.bitwise_and(rx_symbols > -4, rx_symbols <= -2)
                             * 1.5 * (6 + rx_symbols) -
                             np.bitwise_and(rx_symbols > -2, rx_symbols <= 0)
                             * 2 * (5 + rx_symbols) -
                             np.bitwise_and(rx_symbols > 0, rx_symbols <= 2)
                             * 2 * (5 - rx_symbols) -
                             np.bitwise_and(rx_symbols > 2, rx_symbols <= 4)
                             * 1.5 * (6 - rx_symbols) -
                             np.bitwise_and(rx_symbols > 4, rx_symbols <= 6)
                             * (7 - rx_symbols) -
                             np.bitwise_and(rx_symbols > 6, rx_symbols <= 10)
                             * 0.5 * (8 - rx_symbols) -
                             np.bitwise_and(rx_symbols > 10, rx_symbols <= 12)
                             * (9 - rx_symbols) -
                             np.bitwise_and(rx_symbols > 12, rx_symbols <= 14)
                             * 1.5 * (10 - rx_symbols) -
                             (rx_symbols > 14) * 2 * (11 - rx_symbols))

            llr[2, :] = 8 * ((rx_symbols <= -14) * (-13 - rx_symbols) -
                             np.bitwise_and(rx_symbols > -14, rx_symbols <= -10)
                             * 0.5 * (12 + rx_symbols) -
                             np.bitwise_and(rx_symbols > -10, rx_symbols <= -8)
                             * (11 + rx_symbols) +
                             np.bitwise_and(rx_symbols > -8, rx_symbols <= -6)
                             * (5 + rx_symbols) +
                             np.bitwise_and(rx_symbols > -6, rx_symbols <= -2)
                             * 0.5 * (4 + rx_symbols) +
                             np.bitwise_and(rx_symbols > -2, rx_symbols <= 0)
                             * (3 + rx_symbols) +
                             np.bitwise_and(rx_symbols > 0, rx_symbols <= 2)
                             * (3 - rx_symbols) +
                             np.bitwise_and(rx_symbols > 2, rx_symbols <= 6)
                             * 0.5 * (4 - rx_symbols) +
                             np.bitwise_and(rx_symbols > 6, rx_symbols <= 8)
                             * (5 - rx_symbols) -
                             np.bitwise_and(rx_symbols > 8, rx_symbols <= 10)
                             * (11 - rx_symbols) -
                             np.bitwise_and(rx_symbols > 10, rx_symbols <= 14)
                             * 0.5 * (12 - rx_symbols) -
                             (rx_symbols > 14) * (13 - rx_symbols))

            llr[3, :] = 8 * ((rx_symbols <= -12) * (-0.5 * (14 + rx_symbols)) +
                             np.bitwise_and(rx_symbols > -12, rx_symbols <= -8)
                             * 0.5 * (10 + rx_symbols) -
                             np.bitwise_and(rx_symbols > -8, rx_symbols <= -4)
                             * 0.5 * (6 + rx_symbols) +
                             np.bitwise_and(rx_symbols > -4, rx_symbols <= 0)
                             * 0.5 * (2 + rx_symbols) +
                             np.bitwise_and(rx_symbols > 0, rx_symbols <= 4)
                             * 0.5 * (2 - rx_symbols) -
                             np.bitwise_and(rx_symbols > 4, rx_symbols <= 8)
                             * 0.5 * (6 - rx_symbols) +
                             np.bitwise_and(rx_symbols > 8, rx_symbols <= 12)
                             * 0.5 * (10 - rx_symbols) -
                             (rx_symbols > 12) * 0.5 * (14 - rx_symbols))

            llr = llr / 85

        else:
            raise ValueError(
                f"unsupported modulation order ({modulation_order})")

        if is_complex:
            llr = llr / 2

        return llr.ravel('F')

    def get_mapping(self) -> np.array:
        """Returns current mapping table

        Returns:
            mapping(np.ndarray):
                array with M (modulation_order) elements containing all possible modulation symbols.
                See specifications in "PskQamMapping.__init__"
        """
        if self.mapping is not None:
            mapping = self.mapping
        else:
            bits_all = np.zeros(self.modulation_order * self.bits_per_symbol)
            for symbol_idx in range(self.modulation_order):
                idx = symbol_idx * self.bits_per_symbol
                bits = np.asarray([1 if symbol_idx & (1 << (
                    self.bits_per_symbol - 1 - n)) else 0 for n in range(self.bits_per_symbol)])

                bits_all[idx: idx + self.bits_per_symbol] = bits

            mapping = self.get_symbols(bits_all)

        return mapping
