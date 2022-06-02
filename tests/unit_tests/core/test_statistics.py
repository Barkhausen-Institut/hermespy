# -*- coding: utf-8 -*-
"""Test HermesPy simulation statistics."""

import unittest
from unittest.mock import Mock
from typing import List

import numpy as np

from hermespy.core.statistics import Statistics, ConfidenceMetric
from hermespy.core.drop import Drop

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "3.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestUpdateStoppingCriteria(unittest.TestCase):
    """Test proper handling of stopping criteria."""

    def setUp(self) -> None:

        self.min_num_drops = 3
        self.max_num_drops = 4
        self.confidence_margin = 0.91
        self.confidence_level = 0.9
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
                                confidence_level=self.confidence_level,
                                confidence_metric=ConfidenceMetric.BER,
                                min_num_drops=self.min_num_drops,
                                max_num_drops=self.max_num_drops)
        self.transmitted_bits = [np.ones(10)]
        self.received_bits = [np.zeros(10)]

    @staticmethod
    def create_drop(transmitted_bits: List[np.ndarray], received_bits: List[np.ndarray]) -> Drop:

        return Drop(transmitted_bits=transmitted_bits,
                    transmitted_symbols=transmitted_bits,
                    transmitted_signals=[None],
                    transmit_block_sizes=[len(bits) for bits in transmitted_bits],
                    received_signals=[None],
                    received_symbols=received_bits,
                    received_bits=received_bits,
                    receive_block_sizes=[len(bits) for bits in received_bits])

    def test_update_mean(self) -> None:
        samples = np.arange(5)

        expected_means = [0, 0.5, 1.0, 1.5, 2.0]
        mean = 0.0
        means: List[float] = []
        for n in range(1, 5+1):

            mean = self.stats._Statistics__update_mean(old_mean=mean,
                                                       no_old_samples=n-1,
                                                       new_sample=samples[n-1])
            means.append(mean)

        self.assertListEqual(expected_means, means)

    def test_disabled_stopping_criteria(self) -> None:

        self.stats = Statistics(scenario=self.scenario_mock,
                                snr_loop=self.snr_loop,
                                calc_theory=False,
                                calc_transmit_spectrum=False,
                                calc_receive_spectrum=False,
                                calc_transmit_stft=False,
                                calc_receive_stft=False,
                                confidence_margin=self.confidence_margin,
                                confidence_level=self.confidence_level,
                                confidence_metric=ConfidenceMetric.DISABLED,
                                min_num_drops=self.min_num_drops,
                                max_num_drops=self.max_num_drops)

        NUM_DROPS = self.min_num_drops + 1
        transmitted_bits = [[np.ones(10)] for _ in range(NUM_DROPS)]
        received_bits = [[np.zeros(10)] for _ in range(self.min_num_drops-1)]
        received_bits.append([np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])])
        received_bits.append([np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])])

        drops = [self.create_drop(transmitted_bits[i], received_bits[i]) for i in range(NUM_DROPS)]

        for drop_idx in range(1, NUM_DROPS+1):
            for snr_idx, _ in enumerate(self.snr_loop):

                self.stats.add_drop(drops[drop_idx-1], snr_idx)
                self.assertTrue(self.next_drop_can_be_run(self.stats.run_flag_matrix, snr_idx))

    def test_update_stopping_criteria(self) -> None:

        NUM_DROPS = self.min_num_drops + 1
        transmitted_bits = [[np.ones(10)] for _ in range(NUM_DROPS)]
        received_bits = [[np.zeros(10)] for _ in range(self.min_num_drops-1)]
        received_bits.append([np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])])
        received_bits.append([np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])])

        drops = [self.create_drop(transmitted_bits[i], received_bits[i]) for i in range(NUM_DROPS)]

        for drop_idx in range(1, NUM_DROPS+1):
            for snr_idx, _ in enumerate(self.snr_loop):

                self.stats.add_drop(drops[drop_idx-1], snr_idx)

                if drop_idx == NUM_DROPS:

                    self.assertFalse(self.next_drop_can_be_run(self.stats.run_flag_matrix, snr_idx))

                else:
                    self.assertTrue(self.next_drop_can_be_run(self.stats.run_flag_matrix, snr_idx))

    def test_estimation_of_confidence_intervals_of_mean(self) -> None:

        data = np.array([0, 0, 0.5, 0.5, 0.6, 0.5])
        alpha = 0.95

        expected_lower_bound = 0.0626
        expected_upper_bound = 0.637

        lower_bound, upper_bound = self.stats.estimate_confidence_intervals_mean(data, alpha)

        self.assertAlmostEqual(expected_lower_bound, lower_bound, places=3)
        self.assertAlmostEqual(expected_upper_bound, upper_bound, places=3)

    def test_confidence_intervals_bounds_equals_first_sample_if_only_one_sample(self) -> None:

        data = np.array([0])
        alpha = 0.05

        lower_bound, upper_bound = self.stats.estimate_confidence_intervals_mean(data, alpha)

        self.assertAlmostEqual(0, lower_bound, places=3)
        self.assertAlmostEqual(0, upper_bound, places=3)

    def test_getting_confidence_margin_for_ber_mean(self) -> None:
        self.assertAlmostEqual(self.stats.get_confidence_margin(1, 0.9, 0.95), (1-0.9)/0.95)

    def test_margin_calculation_does_not_crash_if_mean_zero(self) -> None:
        _ = self.stats.get_confidence_margin(0, 0, 0)

    @staticmethod
    def next_drop_can_be_run(run_flag_matrix: np.ndarray, snr_index: int) -> bool:
        return np.all(run_flag_matrix[:, :, snr_index])
