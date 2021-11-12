import unittest
from unittest.mock import Mock
from typing import List

import numpy as np

from hermespy.simulator_core.statistics import Statistics
from hermespy.simulator_core.drop import Drop


class TestUpdateStoppingCriteria(unittest.TestCase):
    def setUp(self) -> None:
        self.min_num_drops = 3
        self.confidence_margin = 0.05
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
                    transmit_block_sizes=[None for _ in transmitted_bits],
                    received_signals=[None],
                    received_bits=received_bits,
                    receive_block_sizes=[None for _ in received_bits])

    def test_iterative_mean(self) -> None:
        samples = np.arange(5)

        expected_means = [0, 0.5, 1.0, 1.5, 2.0]
        mean = 0.0
        means: List[float] = []
        for n in range(1, 5+1):
            mean = self.stats.iterative_mean(
                         old_mean=mean,
                         old_samples=n-1,
                         new_sample=samples[n-1])
            means.append(mean)

        self.assertListEqual(expected_means, means)

    def test_min_num_drops_reached(self):
        transmitted_bits = [np.ones(10)]
        received_bits = transmitted_bits

        drops = [self.create_drop(transmitted_bits, received_bits)
                 for i in range(self.min_num_drops + 1)]

        for snr_idx, snr in enumerate(self.snr_loop):
            for drop in drops:
                self.stats.update_stopping_criteria(drop, snr_idx)
                self.assertTrue(self.next_drop_can_be_run(self.stats.flag_matrix, snr_idx))

    def next_drop_can_be_run(self, flag_matrix: np.ndarray, snr_index: int) -> bool:
        return np.all(flag_matrix[:, :, snr_index] == True)