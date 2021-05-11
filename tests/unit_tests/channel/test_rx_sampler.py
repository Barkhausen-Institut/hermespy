import unittest
import unittest.mock
from unittest.mock import patch
import os

import numpy as np
from numpy import random
import scipy
from scipy import io

from channel.rx_sampler import RxSampler


class TestRxSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.MIN_SNR = 40

    def test_constructor(self) -> None:
        """Tests if the sampler is properly constructed."""
        rx_sampling_rate = 1000
        rx_center_freq = 2.5 * 10 ** 9

        rx_sampler = RxSampler(rx_sampling_rate, rx_center_freq)

        # perform actual test
        self.assertAlmostEqual(rx_sampler.rx_sampling_rate, rx_sampling_rate)
        self.assertAlmostEqual(rx_sampler.rx_center_freq, rx_center_freq)

    def test_set_tx_sampling_rate(self) -> None:
        """Tests if the sampling rate is properly set."""
        # construct rx sampler
        rx_sampling_rate = 1000
        rx_center_freq = 2.5 * 10 ** 9

        rx_sampler = RxSampler(rx_sampling_rate, rx_center_freq)

        # perform actual test
        tx_sampling_rate = np.array([1000])
        tx_center_freq = np.array([10])

        # verify if __computeFactors was called at least once
        with patch.object(rx_sampler, "computeFactors") as mock_method:
            rx_sampler.set_tx_sampling_rate(tx_sampling_rate, tx_center_freq)

            self.assertTrue(
                np.allclose(
                    rx_sampler.tx_sampling_rate,
                    tx_sampling_rate))
            self.assertTrue(
                np.allclose(
                    rx_sampler.tx_center_freq,
                    tx_center_freq))

            mock_method.assert_called()

    def test_compute_factors(self) -> None:
        """Tests on proper resampling factor calculation.

        Ten different sampling rates for Tx are tested, each having a different
        center frequency. """

        # construct rx sampler
        rx_sampling_rate = 1000
        rx_center_freq = 2.5e9

        rx_sampler = RxSampler(rx_sampling_rate, rx_center_freq)

        # set expected results of test by reading the matlab file
        # due to the reading manner by scipy, some reshaping/flattening
        # needs to be performed.
        test_results_mat = io.loadmat(
            os.path.join(
                "tests", "unit_tests", "channel", "res",
                "computeFactors_variable_results.mat")
        )
        test_results_struct = test_results_mat["rxSamplerMembers"]
        test_results = test_results_struct[0, 0]

        oversample_rate = np.asscalar(test_results["oversampleRate"])
        interpolate_factors_tx = (
            test_results["interpolateFactorTx"].ravel().astype(float)
        )
        interpolate_factor_rx = np.asscalar(
            test_results["interpolateFactorRx"])
        decimate_factor_tx = np.asscalar(test_results["decimateFactorTx"])
        decimate_factor_rx = np.asscalar(test_results["decimateFactorRx"])

        # set properties of tx
        tx_sampling_rate = np.array(
            [2e4, 1e4, 3.5e4, 5e4, 5e3, 2.5e4, 4.5e4, 1.5e4, 4e4, 3e4]
        )
        tx_center_freq = np.array(
            [6, 9, 10, 8, 2, 1, 7, 4, 5, 3]) * 5000 + 2.5e9
        rx_sampler.set_tx_sampling_rate(tx_sampling_rate, tx_center_freq)

        # perform tests
        self.assertAlmostEqual(oversample_rate, rx_sampler.oversample_rate)
        np.testing.assert_allclose(
            interpolate_factors_tx, rx_sampler.interpolate_factors_tx
        )
        self.assertAlmostEqual(
            interpolate_factor_rx,
            rx_sampler.interpolate_factor_rx)
        self.assertAlmostEqual(
            decimate_factor_tx,
            rx_sampler.decimate_factor_tx)
        self.assertAlmostEqual(
            decimate_factor_rx,
            rx_sampler.decimate_factor_rx)

    def test_resample_same_sampling_rate(self) -> None:
        """Tests resampling given same tx sampling rates."""
        # construct rx sampler
        rx_sampling_rate = 23.04e6
        rx_center_freq = 2.4e9

        rx_sampler = RxSampler(rx_sampling_rate, rx_center_freq)

        # set sampling frequency of transmitters
        tx_sampling_rate = np.full(3, 23.04e6)
        tx_center_freq = np.full(3, 2.4e9)

        rx_sampler.set_tx_sampling_rate(tx_sampling_rate, tx_center_freq)

        # calculate amount of samples
        receive_time = 1e-3
        no_samples = np.array([0.25, 0.5, 1]) * tx_sampling_rate * receive_time
        no_rx_antennas = 2
        no_rx_symbols = rx_sampling_rate * receive_time

        # set input signals
        signals_in = np.empty(3, dtype="object")
        for signal_iter in range(len(no_samples)):
            signals_in[signal_iter] = (
                random.rand(no_rx_antennas, int(no_samples[signal_iter]))
                - 0.5
                + np.random.rand(no_rx_antennas,
                                 int(no_samples[signal_iter])) * 1j
                + 0.5j
            )

        # perform actual test
        signal_out = rx_sampler.resample(signals_in)
        size_out = signal_out.shape

        self.assertEqual(no_rx_symbols, size_out[1])
        self.assertEqual(no_rx_antennas, size_out[0])

    def test_resample_different_sampling_rate(self) -> None:
        """Tests resampling given differing tx sampling rates."""

        # set expected results and input variables of test
        # by reading the matlab file
        test_variables = io.loadmat(
            os.path.join(
                ".", "tests", "unit_tests", "channel", "res", "resample_different_samples_rates_variables.mat"
            )
        )

        rx_sampling_rate = np.asscalar(test_variables["rxSamplingRate"])
        rx_center_freq = np.asscalar(test_variables["rxCenterFreq"])
        tx_sampling_rate = test_variables["txSamplingRate"].ravel()
        tx_center_freq = test_variables["txCenterFreq"].ravel()
        no_rx_antennas = np.asscalar(test_variables["numberOfRxAntennas"])
        no_rx_symbols = np.asscalar(test_variables["numberOfRxSymbols"])
        signal_in = test_variables["signalIn"].ravel()
        signal_resampled_matlab = test_variables["signalOut"]

        # create rx_sampler
        rx_sampler = RxSampler(rx_sampling_rate, rx_center_freq)
        rx_sampler.set_tx_sampling_rate(tx_sampling_rate, tx_center_freq)

        # need to reshape signals, since the matrix shapes of the signals in hermes
        # implementation matlab and python differ
        for signal_idx, signal in enumerate(signal_in):
            signal_in[signal_idx] = signal.T
        signal_resampled_matlab = signal_resampled_matlab.T

        # perform actual test
        signal_out = rx_sampler.resample(signal_in)
        size_out = signal_out.shape

        self.assertEqual(size_out[0], no_rx_antennas)
        self.assertTrue(
            no_rx_symbols - 1 <= size_out[1] <= no_rx_symbols + 1,
            "Number of Rx symbols should be between {0} and {1}. Actual value is {2}.".format(
                no_rx_symbols - 1, no_rx_symbols + 1, signal_out.shape[1]
            ),
        )
        self.assertLess(
            self.MIN_SNR,
            self.calculate_snr_complex_signals(
                signal_resampled_matlab, signal_out),
        )

    def test_built_in_resample_method(self) -> None:
        """Tests if matlab and scipy built-in resample methods yield same results."""
        variables_mat = io.loadmat(
            os.path.join(
                "tests",
                "unit_tests",
                "channel",
                "res",
                "resample_builtIn_test.mat")
        )

        # read variables from matfile
        signal_resampled = variables_mat["signalResampled"].ravel()
        signal_in = variables_mat["signal"].ravel()
        interpolate_factor = np.asscalar(variables_mat["interpolateFactor"])
        decimate_factor = np.asscalar(variables_mat["decimateFactor"])

        signal_resampled_scipy = scipy.signal.resample_poly(
            x=signal_in, up=interpolate_factor, down=decimate_factor
        )

        self.assertEqual(signal_resampled.shape, signal_resampled_scipy.shape)
        self.assertLess(
            self.MIN_SNR,
            self.calculate_snr_complex_signals(
                signal_resampled, signal_resampled_scipy
            ),
        )

    def calculate_snr_complex_signals(
        self, signal_matlab: np.array, signal_scipy: np.array
    ) -> float:
        """This functions calculates "SNR" between matlab and scipy signal.

        The difference between the matlab and the scipy-signal is considered
        as noise. Therefore, the method compares the matlab-signal with the
        scipy signal.

        Args:
            signal_matlab (np.array): Matlab signal.
            signal_scipy (np.array): Scipy signal.

        Returns:
            (float) signal to noise ratio.

        """
        power_scipy_signal = signal_scipy * np.conj(signal_scipy)
        power_noise = (signal_scipy - signal_matlab) * (
            np.conj(signal_scipy - signal_matlab)
        )

        snr = np.mean(10 * np.log10(power_scipy_signal / power_noise))

        return snr
