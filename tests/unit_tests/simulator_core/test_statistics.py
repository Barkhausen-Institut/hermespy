import unittest
import os
import unittest.mock
from unittest.mock import Mock, patch
from typing import List

import numpy as np
from scipy import signal

from simulator_core.statistics import Statistics
from parameters_parser.parameters import Parameters
from source.bits_source import BitsSource


class StatisticsTest(unittest.TestCase):
    def setUp(self) -> None:
        # read parameters and create scenario
        parameters_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "res_test_statistics"
        )
        self.parameters = Parameters(parameters_path)
        self.parameters.read_params()
        self.stats = Statistics(
            self.parameters.general,
            self.parameters.scenario,
            [],
            [])

        self.no_rx_modems = self.parameters.scenario.number_of_rx_modems
        self.no_snr = self.parameters.general.snr_vector.size
        self.no_tx_modems = self.parameters.scenario.number_of_tx_modems

        seed = 42
        self.rng = np.random.RandomState(seed)
        self.sources = [
            BitsSource(self.rng) for modem_count in range(self.no_tx_modems)
        ]
        self.BER_expected = [
            [np.empty(0) for idx in range(self.no_snr)]
            for i in range(self.no_rx_modems)
        ]
        self.BER_expected_mean = [
            np.full(self.no_snr, -1) for i in range(self.no_rx_modems)
        ]
        self.BER_expected_min = [
            np.full(self.no_snr, -1) for i in range(self.no_rx_modems)
        ]
        self.BER_expected_max = [
            np.full(self.no_snr, -1) for i in range(self.no_rx_modems)
        ]

        self.no_bits_in_frame = 10

    def test_update_error_rate_one_drop_one_frame(self) -> None:
        """Tests for one drop, whether BER is calculated properly.

        One drop is evaluated. A signal is sent containing ten bits. It is expected
        that the receiver is able to receive 2 bits properly and 8 bits are falsely detected
        resulting in a BER of 0.8. We add no noise."""
        # set signal that is received at rx side, no noise added
        output_rx_modems = self.generate_10bits_signals([8])

        # run the method to test
        self.stats.update_error_rate(self.sources, output_rx_modems)

        # set expected output
        self.BER_expected = [
            [np.array([0.8]) for snr_idx in range(self.no_snr)]
            for rx_modem_idx in range(self.no_rx_modems)
        ]

        # compare expected with actual result
        for rx_modem_idx in range(self.no_rx_modems):
            np.testing.assert_array_almost_equal(
                np.array(self.BER_expected[rx_modem_idx]),
                np.array(self.stats.ber[rx_modem_idx]),
            )

    def test_update_error_rate_one_drop_two_frames_multiple_blocks(
            self) -> None:
        """Tests fer calculation for one drop and one frame"""

        output_rx_modems = self.generate_10bits_signals([0, 5], num_blocks=2)
        self.stats.update_error_rate(self.sources, output_rx_modems)

        # set expected output
        self.fer_expected = [
            [np.array([0.5]) for snr_idx in range(self.no_snr)]
            for rx_modem_idx in range(self.no_rx_modems)
        ] * self.no_rx_modems

        for rx_modem_idx in range(self.no_rx_modems):
            np.testing.assert_array_almost_equal(
                np.array(self.fer_expected[rx_modem_idx]),
                np.array(self.stats.fer[rx_modem_idx]),
            )

    def test_update_error_rate_multiple_drops_one_frame(self) -> None:
        """Tests for multiple drops, whether BER is calculated properly.

        Does the same thing as one drop, except that it generates multiple drops."""
        no_drops = 2  # must be lower than params.general.min_num_drops

        # iterate over drops and generate actual ber values
        for drop in range(no_drops):

            for source in self.sources:
                source.init_drop()

            output_rx_modems = self.generate_10bits_signals([8])

            # calculate actual ber values
            self.stats.update_error_rate(self.sources, output_rx_modems)

        # set expected ber values
        self.BER_expected = [
            [np.array([0.8, 0.8]) for snr_idx in range(self.no_snr)]
            for rx_modem_idx in range(self.no_rx_modems)
        ]

        # calculate expected ber mean/min/max for tests
        for rx_modem_idx in range(self.no_rx_modems):
            self.BER_expected_mean[rx_modem_idx] = np.ones(self.no_snr) * 0.8
            self.BER_expected_min[rx_modem_idx] = np.ones(self.no_snr) * 0.8
            self.BER_expected_max[rx_modem_idx] = np.ones(self.no_snr) * 0.8

        # run actual test
        for rx_modem_idx in range(self.no_rx_modems):
            np.testing.assert_array_almost_equal(
                np.array(self.BER_expected[rx_modem_idx]),
                np.array(self.stats.ber[rx_modem_idx]),
            )
            np.testing.assert_array_almost_equal(
                np.array(self.BER_expected_mean[rx_modem_idx]),
                np.array(self.stats.bit_error_sum[rx_modem_idx]),
            )
            np.testing.assert_array_almost_equal(
                np.array(self.BER_expected_min[rx_modem_idx]),
                np.array(self.stats.bit_error_min[rx_modem_idx]),
            )
            np.testing.assert_array_almost_equal(
                np.array(self.BER_expected_max[rx_modem_idx]),
                np.array(self.stats.bit_error_max[rx_modem_idx]),
            )

    def test_update_error_rate_stopping_criteria_one_frame(self) -> None:
        """Tests if the stopping criteria is correctly evaluated.

        In total, there are five drops run. In the first three drops, all signals
        are negated yielding a BER of 1. In the fourth drop, for the first four
        snr values of each receiver, we only negate 80 percent of the bits, i.e.
        a BER of 0.8. For the remaining 26 snr values, there is no negation, i.e.
        BER = 0. Due to the sudden BER change, we expect the first four snr values in
        the fourth drop to be within the confidence margin, i.e. the stopping criteria
        is met in this case. For the other values, it is outside of the confidence margin,
        i.e. the sopping criteria is not met.
        """
        drop_ber_variation = 3
        drops_total = 5

        for drop in range(drops_total):
            for source in self.sources:
                source.init_drop()

            # iterate over rx_modems and get signals
            output_rx_modems = self.generate_10bits_signals([10])

            if drop >= drop_ber_variation:
                output_rx_modems_eighty_percent: List[np.array] = (
                    [np.array([])]*self.no_rx_modems
                )

                # auxiliary calculation: calculate eighty percent negation of signals
                # for ALL snrs
                for rx_modem_idx in range(self.no_rx_modems):
                    frame = np.dstack(
                        (
                            np.logical_not(
                                output_rx_modems[rx_modem_idx][0][:, :, :2]),
                            output_rx_modems[rx_modem_idx][0][:, :, 2:],
                        )
                    )
                    output_rx_modems_eighty_percent[rx_modem_idx] = [frame]

                # ensure that only the first four signals are negated by 80%,
                # and the other 26 signals are perfect
                for rx_modem_idx in range(self.no_rx_modems):
                    # concatenate the signals
                    frame = np.vstack(
                        (
                            output_rx_modems_eighty_percent[rx_modem_idx][0][:4, :, :],
                            np.logical_not(
                                output_rx_modems[rx_modem_idx][0][4:, :, :]),
                        )
                    )
                    output_rx_modems[rx_modem_idx] = [frame]

            # ensure that only signals are transmitted that are outside
            # of the confidence margin
            for rx_modem_idx in range(self.no_rx_modems):
                frame = output_rx_modems[rx_modem_idx][0]
                valid_signals_in_frame = frame[self.stats.run_flag[rx_modem_idx], :, :]

                # it must be a list item, as each frame corresponds to one
                # frame
                output_rx_modems[rx_modem_idx] = [valid_signals_in_frame]

            self.stats.update_error_rate(self.sources, output_rx_modems)

        # calculate expected ber
        for rx_modem_idx in range(self.no_rx_modems):
            for idx in range(4):
                self.BER_expected[rx_modem_idx][idx] = np.append(
                    np.ones(drop_ber_variation), 0.8
                )

            for idx in np.arange(4, self.no_snr):
                self.BER_expected[rx_modem_idx][idx] = np.append(
                    np.ones(drop_ber_variation), [0, 0]
                )
        # set expected amount of signals here
        no_signals_expected = list(
            np.append(np.full(4, 4), np.full(26, 5))
            for rx_modem_idx in range(self.no_rx_modems)
        )
        # calculate actual amount of signals
        no_signals_actual = [
            np.zeros(30) for rx_modem_idx in range(
                self.no_rx_modems)]

        for rx_modem_idx in range(self.no_rx_modems):
            for snr_idx, ber in enumerate(self.stats.ber[rx_modem_idx]):
                no_signals_actual[rx_modem_idx][snr_idx] = len(ber)

        # run the test
        for rx_modem_idx in range(self.no_rx_modems):
            np.testing.assert_array_almost_equal(
                no_signals_expected[rx_modem_idx], no_signals_actual[rx_modem_idx]
            )

    @patch.object(signal, 'welch')
    def test_update_tx_spectrum_welch_method_update_periodogram(
            self, mock_welch: Mock) -> None:
        # initialize fft-specific variables
        fft_size = 512
        initial_periodogram = np.full(fft_size, 3)
        welch_periodogram = np.full(fft_size, -1)

        # generate signal
        rnd = np.random.RandomState(42)
        tx_signal = rnd.rand(fft_size)[np.newaxis, :]

        # set call specific required vavriables
        self.stats._tx_sampling_rate = [1e6]
        self.stats.param_general.spectrum_fft_size = fft_size
        self.stats._periodogram_tx = [np.copy(initial_periodogram)]
        self.stats._frequency_range_tx = [np.zeros(fft_size)]

        mock_welch.return_value = (welch_periodogram, -1)
        self.stats.update_tx_spectrum([tx_signal])

        expected_periodogram = (
            initial_periodogram +
            welch_periodogram).reshape(
            1,
            fft_size)
        np.testing.assert_array_equal(
            expected_periodogram,
            self.stats._periodogram_tx)

    @patch.object(np.fft, 'fftshift')
    @patch.object(signal, 'stft')
    def test_stft_calculation_tx(
            self, mock_stft: Mock, mock_fftshift: Mock) -> None:
        # toggle calculation method
        self.stats.param_general.calc_spectrum_tx = False
        self.stats.param_general.calc_stft_tx = True
        self.stats._stft_freq_tx = [np.array([])]
        self.stats.param_general.spectrum_fft_size = 512

        self.stats._tx_sampling_rate = [1e6]

        # generate signal
        rnd = np.random.RandomState(42)
        tx_signal = rnd.rand(512)[np.newaxis, :]

        # mock functions
        f_stft = 1
        t_stft = 0
        Zxx_stft = 1
        mock_stft.return_value = (f_stft, t_stft, Zxx_stft)

        mock_ffttshift_retval = 30
        mock_fftshift.return_value = mock_ffttshift_retval

        # call method to test
        self.stats.update_tx_spectrum([tx_signal])

        # check method calls
        mock_fftshift.assert_any_call(f_stft)
        mock_fftshift.assert_any_call(Zxx_stft, axes=0)

        # check actual behavior
        self.assertListEqual(self.stats._stft_freq_tx, [mock_ffttshift_retval])
        self.assertListEqual(self.stats._stft_time_tx, [t_stft])
        self.assertListEqual(
            self.stats._stft_power_tx,
            [mock_ffttshift_retval])

    def generate_10bits_signals(
            self, erroneous_bits: List[int], num_blocks: int = 1) -> List[List[np.array]]:
        """Generates signals at receiver side.

        For each signal that a transmitter sends, the respective signal at
        receivers side is calculated with added noise according to the length
        of the snr_vector. The noise is set to zero. The sent signal is 10 bits
        long, among which `erroneous_bits` are negated and `10-erroneous_bits`
        are unchanged. This results in a BER value of erroneous_bits/10
        at the receiver side, for each receiver.

        Args:
            erroneous_bits List[int]:
                Number of bits that are erroneous, i.e. are to be negated.
                Each list item corresponds to one frame.

        Returns:
            List[List[np.array]]:
                Each List item corresponds to one receiver. Each receiver list
                contains frames, which are in turn list items. Each Frame is
                composed of np.array with the actual signal.
        """

        output_rx_modems = list()
        for rx_modem_idx in range(self.no_rx_modems):
            frames = []
            for bit_errors in erroneous_bits:
                # set the detected bits here
                # we only evaluate one drop
                detected_bits = self.sources[rx_modem_idx].get_bits(
                    self.no_bits_in_frame,
                    num_blocks)

                detected_bits_noisy = np.zeros((
                    self.no_snr, num_blocks, self.no_bits_in_frame))

                for block_iter, block in enumerate(detected_bits):
                    block_bits_partly_negated = np.append(
                        np.logical_not(block[:bit_errors]).astype(int),
                        block[bit_errors:],
                    )
                    block_with_different_snr = np.tile(
                        # replicate signal
                        block_bits_partly_negated, (self.no_snr, 1)
                    )
                    detected_bits_noisy[:, block_iter, :] = (
                        block_with_different_snr
                    )
                frames.append(detected_bits_noisy)
            output_rx_modems.append(frames)
        return output_rx_modems

    def test_retrieving_correct_snr_list(self) -> None:
        snr_vector = np.arange(10)

        run_flag_rx1 = np.array(
            [True, True, False, True, True, False, True, True, False, True])
        run_flag_rx2 = np.array(
            [True, False, True, False, True, False, True, False, True, False])
        run_flag = [run_flag_rx1, run_flag_rx2]

        self.stats.param_general.snr_vector = snr_vector
        self.stats.run_flag = run_flag

        for rx_modem_idx in range(len(run_flag)):
            np.testing.assert_array_equal(
                snr_vector[run_flag[rx_modem_idx]],
                self.stats.get_snr_list(rx_modem_idx)
            )

    def test_multiple_frames_multiple_drops_ber(self) -> None:
        no_drops = 2  # must be lower than params.general.min_num_drops

        # iterate over drops and generate actual ber values
        for drop in range(no_drops):

            for source in self.sources:
                source.init_drop()

            output_rx_modems = self.generate_10bits_signals([10, 10, 1])

            # calculate actual ber values
            self.stats.update_error_rate(self.sources, output_rx_modems)

        # set expected ber values
        self.BER_expected = [
            [np.array([0.7, 0.7]) for snr_idx in range(self.no_snr)]
            for rx_modem_idx in range(self.no_rx_modems)
        ]

        self.fer_expected = [
            [np.array([1, 1]) for snr_idx in range(self.no_snr)]
            for rx_modem_idx in range(self.no_rx_modems)
        ]

        # run actual test
        for rx_modem_idx in range(self.no_rx_modems):
            np.testing.assert_array_almost_equal(
                np.array(self.BER_expected[rx_modem_idx]),
                np.array(self.stats.ber[rx_modem_idx]),
            )
            np.testing.assert_array_almost_equal(
                np.array(self.fer_expected[rx_modem_idx]),
                np.array(self.stats.fer[rx_modem_idx])
            )

    def test_update_error_rate_stopping_criteria_multiple_frames(self) -> None:
        """Tests if the stopping criteria is correctly evaluated.

        The same as 'test_update_error_rate_stopping_criteria_one_frame' but
        with multiple frames.
        """
        drop_ber_variation = 3
        drops_total = 5
        no_frames = 3

        for drop in range(drops_total):
            for source in self.sources:
                source.init_drop()

            # iterate over rx_modems and get signals
            output_rx_modems = self.generate_10bits_signals([10, 10, 10])

            if drop >= drop_ber_variation:
                output_rx_modems_eighty_percent: List[np.array] = [
                    list() for rx_modem_idx in range(
                        self.no_rx_modems)]

                # auxiliary calculation: calculate eighty percent negation of signals
                # for ALL snrs
                for rx_modem_idx in range(self.no_rx_modems):
                    for frame_idx in range(no_frames):
                        frame = np.dstack(
                            (
                                np.logical_not(
                                    output_rx_modems[rx_modem_idx][frame_idx][:, :, :2]),
                                output_rx_modems[rx_modem_idx][frame_idx][:, :, 2:],
                            )
                        )
                        output_rx_modems_eighty_percent[rx_modem_idx].append(
                            frame)
                # ensure that only the first four signals are negated by 80%,
                # and the other 26 signals are perfect
                for rx_modem_idx in range(self.no_rx_modems):

                    # concatenate the signals
                    for frame_idx in range(no_frames):
                        frame = np.vstack(
                            (
                                output_rx_modems_eighty_percent[rx_modem_idx][frame_idx][:4, :, :],
                                np.logical_not(
                                    output_rx_modems[rx_modem_idx][frame_idx][4:, :, :]),
                            )
                        )
                        output_rx_modems[rx_modem_idx][frame_idx] = frame

            # ensure that only signals are transmitted that are outside
            # of the confidence margin
            for rx_modem_idx in range(self.no_rx_modems):
                for frame_idx in range(no_frames):
                    frame = output_rx_modems[rx_modem_idx][frame_idx]
                    valid_signals_in_frame = frame[self.stats.run_flag[rx_modem_idx], :]

                    output_rx_modems[rx_modem_idx][frame_idx] = valid_signals_in_frame
            self.stats.update_error_rate(self.sources, output_rx_modems)

        # set expected amount of signals here
        no_signals_expected = list(
            np.append(np.full(4, 4), np.full(26, 5))
            for rx_modem_idx in range(self.no_rx_modems)
        )
        # calculate actual amount of signals
        no_signals_actual = [
            np.zeros(30) for rx_modem_idx in range(
                self.no_rx_modems)]

        for rx_modem_idx in range(self.no_rx_modems):
            for snr_idx, ber in enumerate(self.stats.ber[rx_modem_idx]):
                no_signals_actual[rx_modem_idx][snr_idx] = len(ber)

        # run the test
        for rx_modem_idx in range(self.no_rx_modems):
            np.testing.assert_array_almost_equal(
                no_signals_expected[rx_modem_idx], no_signals_actual[rx_modem_idx]
            )


if __name__ == "__main__":
    os.chdir("../../..")
    unittest.main()
