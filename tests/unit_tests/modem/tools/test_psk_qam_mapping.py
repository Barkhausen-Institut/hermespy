import unittest

import numpy as np

from hermespy.modem.tools.psk_qam_mapping import PskQamMapping
from matplotlib import pyplot as plt


class TestPskQamMapping(unittest.TestCase):

    def test_verification_of_modulation_order(self) -> None:
        modulation_order_incorrect = [-1, 0, 3]

        for modulation_order in modulation_order_incorrect:
            self.assertRaises(ValueError,
                              lambda: PskQamMapping(modulation_order))

    def test_verification_of_mapping_size(self) -> None:
        mapping = np.array([1, 1, 1])
        modulation_order = 4

        self.assertRaises(
            ValueError, lambda: PskQamMapping(
                modulation_order, mapping))

    def test_creation_of_graycoded_psk8_map(self) -> None:
        psk8_map = np.exp(
            1j * np.array([0, 1 / 4, 3 / 4, 1 / 2, -1 / 4, -1 / 2, 1, -3 / 4]) * np.pi)
        psk_qam_mapping = PskQamMapping(8)

        np.testing.assert_array_almost_equal(psk8_map, psk_qam_mapping.mapping)

    def test_symbols_if_mapping_provided(self) -> None:
        # create mapping and normalize it by energy
        mapping = np.array([10, 20, 30, 40])
        mapping_energy = np.mean(np.abs(mapping) ** 2)
        mapping = mapping / np.sqrt(mapping_energy)

        # create bits and set modulation order
        bits = np.array([1, 1,
                         0, 0,
                         0, 1,
                         1, 0])

        modulation_order = 4

        # create mapping with predefined modulation order and mapping
        psk_qam_mapping = PskQamMapping(modulation_order, mapping)

        # perform actual test
        symbols = psk_qam_mapping.get_symbols(bits)
        symbols_expected = np.array(
            [mapping[3], mapping[0], mapping[1], mapping[2]])

        np.testing.assert_array_almost_equal(symbols_expected, symbols)

    def test_symbols_bpsk(self) -> None:
        bpsk_symbols = np.array([1, -1, -1, 1])
        bits = np.array([0, 1, 1, 0])
        psk_qam_mapping = PskQamMapping(2)
        symbols = psk_qam_mapping.get_symbols(bits)

        np.testing.assert_array_almost_equal(bpsk_symbols, symbols)

    def test_symbols_qpsk(self) -> None:
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        qpsk_symbols = np.array(
            [1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)
        psk_qam_mapping = PskQamMapping(4)
        symbols = psk_qam_mapping.get_symbols(bits)

        np.testing.assert_array_almost_equal(qpsk_symbols, symbols)

    def test_symbols_4pam(self) -> None:
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
        pam4_symbols = np.array([1, 3, -1, -3]) / np.sqrt(5)
        psk_qam_mapping = PskQamMapping(4, is_complex=False)
        symbols = psk_qam_mapping.get_symbols(bits)

        np.testing.assert_array_almost_equal(pam4_symbols, symbols)

    def test_symbols_8pam(self) -> None:
        bits = np.array([0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
                         1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1])
        pam8_symbols = np.array([3, 1, 5, 7, -3, -1, -5, -7]) / np.sqrt(21)
        psk_qam_mapping = PskQamMapping(8, is_complex=False)
        symbols = psk_qam_mapping.get_symbols(bits)

        np.testing.assert_array_almost_equal(pam8_symbols, symbols)

    def test_symbols_16qam(self) -> None:
        bits = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
                         1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
        qam16_symbols = np.array([1 + 1j, 1 + 3j, 3 + 1j, 3 + 3j, 1 - 1j, 1 - 3j, 3 - 1j, 3 - 3j,
                                  -1 + 1j, -1 + 3j, -3 + 1j, -3 + 3j, -1 - 1j, -1 - 3j, -3 - 1j, -3 - 3j]) / np.sqrt(10)
        psk_qam_mapping = PskQamMapping(16, is_complex=True)
        symbols = psk_qam_mapping.get_symbols(bits)

        np.testing.assert_array_almost_equal(qam16_symbols, symbols)

    def test_symbols_16pam(self) -> None:
        bits = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
                         1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
        pam16_symbols = np.array(
            [5, 7, 3, 1, 11, 9, 13, 15, -5, -7, -3, -1, -11, -9, -13, -15]) / np.sqrt(85)
        psk_qam_mapping = PskQamMapping(16, is_complex=False)
        symbols = psk_qam_mapping.get_symbols(bits)

        np.testing.assert_array_almost_equal(pam16_symbols, symbols)

    def test_symbols_64qam(self) -> None:
        bits = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1,
                         1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1,
                         1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
                         0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])
        qam64_symbols = np.array([3 + 3j, 3 - 1j, -1 - 3j, -1 + 1j, -3 + 5j, -3 - 7j, 1 - 5j, -1 + 7j,
                                  -5 + 3j, -5 - 1j, 7 - 3j, -7 + 1j, 5 + 5j, 5 - 7j, -7 - 5j, 7 - 7j]) / np.sqrt(42)
        psk_qam_mapping = PskQamMapping(64, is_complex=True)
        symbols = psk_qam_mapping.get_symbols(bits)

        np.testing.assert_array_almost_equal(qam64_symbols, symbols)

    def test_symbols_256qam(self) -> None:
        bits = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1,
                         0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
                         1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
                         1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1])
        qam256_symbols = np.array([-5 + 5j, 5 - 9j, 9 + 11j, -9 - 7j, 11 + 3j, -11 + 15j, -7 - 13j, 9 - 1j,
                                   -13 - 5j, 13 - 9j, 1 + 11j, -15 + 7j, -3 + 3j, -3 - 15j, 15 + 13j, 1 - 15j]) / np.sqrt(170)
        psk_qam_mapping = PskQamMapping(256)
        symbols = psk_qam_mapping.get_symbols(bits)

        np.testing.assert_array_almost_equal(qam256_symbols, symbols)

    def test_demodulation_bpsk(self) -> None:
        self._demodulation_test(100, 2, False, 1 / 10, 1)

    def test_demodulation_qpsk(self) -> None:
        self._demodulation_test(100, 4, True, 1 / 20, 1 / np.sqrt(2))

    def test_demodulation_4pam(self) -> None:
        self._demodulation_test(100, 4, False, 1 / 50, 1 / np.sqrt(5))

    def test_demodulation_8pam(self) -> None:
        self._demodulation_test(100, 8, False, 1 / 210, 1 / np.sqrt(21))

    def test_demodulation_16qam(self) -> None:
        self._demodulation_test(100, 16, True, 1 / 100, 1 / np.sqrt(10))

    def test_demodulation_16pam(self) -> None:
        self._demodulation_test(100, 16, False, 1 / 850, 1 / np.sqrt(85))

    def test_demodulation_64qam(self) -> None:
        self._demodulation_test(100, 64, True, 1 / 420, 1 / np.sqrt(42))

    def test_demodulation_256qam(self) -> None:
        self._demodulation_test(100, 256, True, 1 / 1700, 1 / np.sqrt(170))

    def _demodulation_test(self, number_of_symbols: int, modulation_order: np.ndarray,
                           is_complex: bool, noise_variance: float, max_noise: float) -> None:

        bits = np.random.randint(
            2, size=number_of_symbols * int(np.log2(modulation_order)))

        psk_qam_mapping = PskQamMapping(
            modulation_order, soft_output=False, is_complex=is_complex)
        symbols = psk_qam_mapping.get_symbols(bits)

        # add noise such that no errors occur
        max_measured_noise = np.inf

        noise = None
        while max_measured_noise > max_noise:
            noise = np.zeros(symbols.shape, dtype=complex)

            noise += np.random.standard_normal(symbols.shape) * \
                np.sqrt(noise_variance)
            if is_complex:
                noise += 1j * \
                    np.random.standard_normal(
                        symbols.shape) * np.sqrt(noise_variance)
            max_measured_noise = np.max(np.abs(noise))
        symbols += noise

        rx_bits = psk_qam_mapping.detect_bits(symbols)

        np.testing.assert_array_equal(bits == 1, rx_bits)


def _plot_constellation(modulation_order: np.ndarray,
                        is_complex: bool, soft_output: bool) -> None:
    mapper = PskQamMapping(modulation_order, soft_output=False, is_complex=is_complex)
    mapping = mapper.get_mapping()

    plt.plot(np.real(mapping), np.imag(mapping), '*')

    for idx in range(modulation_order):
        bin_str = bin(idx)[2:].zfill(mapper.bits_per_symbol)
        plt.annotate(bin_str, (np.real(mapping[idx]), np.imag(mapping[idx])))


if __name__ == '__main__':

    # plot 16-QAM constellation
    plt.figure()
    _plot_constellation(16, True, False)
    plt.title("16-QAM")

    # plot 8-PSK constellation
    plt.figure()
    _plot_constellation(8, True, False)
    plt.title("8-PSK")

    # plot 8-PAM
    plt.figure()
    _plot_constellation(8, False, False)
    plt.title("8-PAM")

    # plot 64-QAM
    plt.figure()
    _plot_constellation(64, True, False)
    plt.title("64-QAM")

    plt.show()

    unittest.main()
