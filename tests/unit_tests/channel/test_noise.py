import unittest

import numpy as np

from channel.noise import Noise


class TestNoise(unittest.TestCase):
    def setUp(self) -> None:
        self.pseudo_rng = np.random.RandomState(42)
        self.noise = Noise("EB/N0(DB)", self.pseudo_rng)

    def test_invalid_snr_type(self) -> None:
        signal = np.zeros(5)
        snr = 40
        signal_energy = 1

        self.noise.snr_type = "InvalidSNRType"

        self.assertRaises(
            ValueError, lambda: self.noise.add_noise(
                signal, snr, signal_energy)
        )

    def test_snr_calculation_ebno(self) -> None:
        signal = np.zeros(5)
        snr = 40
        signal_energy = 1
        noisy_signal_expected = np.array(
            [
                0.0035123 - 0.0016556j,
                -0.00097768 + 0.01116672j,
                0.00457985 + 0.00542658j,
                0.01076945 - 0.00331969j,
                -0.00165571 + 0.00383648j,
            ]
        )

        noisy_signal = self.noise.add_noise(signal, snr, signal_energy)[0]
        np.testing.assert_array_almost_equal(
            noisy_signal_expected, noisy_signal)

    def test_custom_noise(self) -> None:
        custom_noise = Noise("custom", self.pseudo_rng)

        signal = np.zeros(5)
        noise_power = 10
        signal_energy = 1

        noisy_signal = custom_noise.add_noise(signal, noise_power, signal_energy)[0]
        noisy_signal_expected = np.array(
            [
                0.11106866 - 0.05235462j,
                -0.03091684 + 0.35312272j,
                0.14482756 + 0.17160362j,
                0.34055983 - 0.10497766j,
                -0.05235829 + 0.12132011j
            ]
        ) * noise_power

        np.testing.assert_array_almost_equal(
            noisy_signal_expected, noisy_signal)


if __name__ == '__main__':
    unittest.main()
