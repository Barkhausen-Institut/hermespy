import unittest
from unittest.mock import Mock
from typing import List

import numpy as np

from hermespy.simulator_core.statistics import Statistics


class StatisticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario_mock = Mock()
        self.scenario_mock.num_transmitters = 1
        self.scenario_mock.num_receivers = 1
        self.stats = Statistics(scenario=self.scenario_mock,
                                snr_loop=[],
                                calc_theory=False)

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