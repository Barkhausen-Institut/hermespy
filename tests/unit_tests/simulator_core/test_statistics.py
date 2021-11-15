import unittest
from unittest.mock import Mock
from typing import List

import numpy as np

from hermespy.simulator_core.statistics import Statistics
from hermespy.simulator_core.drop import Drop

class FakeStatistics(Statistics):
    def update_stopping_criteria_drop(self, drop: Drop, snr_index: int) -> None:
        self._drop_updates[snr_index, :, :] = True

class TestUpdateStoppingCriteria(unittest.TestCase):
    def setUp(self) -> None:
        self.min_num_drops = 3
        self.confidence_margin = 0.99
        self.snr_loop = np.arange(5)
        self.scenario_mock = Mock()
        self.scenario_mock.num_transmitters = 1
        self.scenario_mock.num_receivers = 1
        self.stats = Statistics(scenario=self.scenario_mock,
                                snr_loop=self.snr_loop,
                                calc_theory=False,
                                calc_transmit_spectrum=False,
                                calc_receive_spectrum=False,
                                calc_transmit_stft=False,
                                calc_receive_stft=False,
                                confidence_margin=self.confidence_margin,
                                min_num_drops=self.min_num_drops)
        self.transmitted_bits = [np.ones(10)]
        self.received_bits = [np.zeros(10)]

    def create_drop(self, transmitted_bits: List[np.array], received_bits: List[np.array]) -> Drop:
        return Drop(transmitted_bits=transmitted_bits,
                    transmitted_signals=[None],
                    transmit_block_sizes=[len(bits) for bits in transmitted_bits],
                    received_signals=[None],
                    received_bits=received_bits,
                    receive_block_sizes=[len(bits) for bits in received_bits])

    def test_update_mean(self) -> None:
        samples = np.arange(5)

        expected_means = [0, 0.5, 1.0, 1.5, 2.0]
        mean = 0.0
        means: List[float] = []
        for n in range(1, 5+1):
            mean = self.stats.update_mean(
                         old_mean=mean,
                         no_old_samples=n-1,
                         new_sample=samples[n-1])
            means.append(mean)

        self.assertListEqual(expected_means, means)

    def test_no_iterations_update_if_stopping_criteria_update_for_all_drops(self) -> None:
        transmitted_bits = [np.ones(10)]
        received_bits = transmitted_bits

        drop = self.create_drop(transmitted_bits, received_bits)

        snr_loop = np.arange(5)
        scenario_mock = Mock()
        scenario_mock.num_transmitters = 1
        scenario_mock.num_receivers = 1
        stats = FakeStatistics(scenario=self.scenario_mock,
                                snr_loop=snr_loop,
                                calc_theory=False,
                                calc_transmit_spectrum=False,
                                calc_receive_spectrum=False,
                                calc_transmit_stft=False,
                                calc_receive_stft=False,
                                confidence_margin=0.99,
                                min_num_drops=3)

        for snr_idx, snr_val in enumerate(snr_loop):
            self.assertEquals(stats.no_simulation_iterations, 0)
            stats.update_stopping_criteria(drop, snr_idx)
        self.assertEquals(stats.no_simulation_iterations, 1)

    def test_min_num_drops_reached(self):
        for snr_idx, snr in enumerate(self.snr_loop):
            for drop in range(self.min_num_drops + 1):
                self.stats.update_stopping_criteria(snr_idx)
                self.assertTrue(self.next_drop_can_be_run(self.stats.flag_matrix, snr_idx))

    def test_estimation_of_confidence_intervals_of_mean(self) -> None:
        data = [0, 0, 0.5, 0.5, 0.6, 0.5]
        alpha = 0.95

        expected_lower_bound = 0.0626
        expected_upper_bound = 0.637

        lower_bound, upper_bound = self.stats.estimate_confidence_intervals_mean(
            data, alpha)

        self.assertAlmostEqual(expected_lower_bound, lower_bound, places=3)
        self.assertAlmostEqual(expected_upper_bound, upper_bound, places=3)

    def test_confidence_intervals_bounds_equals_first_sample_if_only_one_sample(self) -> None:
        data = [0]
        alpha = 0.05

        lower_bound, upper_bound = self.stats.estimate_confidence_intervals_mean(
            data, alpha)

        self.assertAlmostEqual(0, lower_bound, places=3)
        self.assertAlmostEqual(0, upper_bound, places=3)

    def test_getting_confidence_margin_for_ber_mean(self) -> None:
        NO_TX = 3
        NO_RX = 2
        NO_SNRS = 10
        self.stats.bit_error_max = np.ones((NO_TX, NO_RX, NO_SNRS))
        self.stats.bit_error_min = np.zeros(self.stats.bit_error_max.shape)
        self.stats.bit_error_mean = np.ones(self.stats.bit_error_max.shape)

        self.stats.bit_error_max[0, 0, 0] = 1
        self.stats.bit_error_min[0, 0, 0] = 0.9
        self.stats.bit_error_mean[0, 0, 0] = 0.95

        self.assertAlmostEqual(self.stats.get_confidence_margin_ber_mean(0, 0, 0), (1-0.9)/0.95)

    def test_margin_calculation_returns_inf_if_mean_is_zero(self) -> None:
        self.stats.bit_error_max[0, 0, 0] = 1
        self.stats.bit_error_min[0, 0, 0] = 0
        self.stats.bit_error_mean[0, 0, 0] = 0
        self.assertEqual(np.inf, self.stats.get_confidence_margin_ber_mean(0, 0, 0))

    def next_drop_can_be_run(self, flag_matrix: np.ndarray, snr_index: int) -> bool:
        return np.all(flag_matrix[:, :, snr_index] == True)