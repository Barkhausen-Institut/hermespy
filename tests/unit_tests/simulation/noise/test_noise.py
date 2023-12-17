# -*- coding: utf-8 -*-
"""Testing of the noise model base class."""

import unittest
from unittest.mock import Mock

import numpy as np
import numpy.random as rnd
from numpy.testing import assert_array_almost_equal

from hermespy.core.signal_model import Signal
from hermespy.simulation.noise import AWGN
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestAWGN(unittest.TestCase):
    """Test the base class for noise models."""

    def setUp(self) -> None:
        self.random_node = Mock()
        self.random_node._rng = rnd.default_rng(42)

        self.noise = AWGN()
        self.noise.random_mother = self.random_node

    def test_power_validation(self) -> None:
        """Power property setter should raise ValueError on negative argument"""

        with self.assertRaises(ValueError):
            self.noise.power = -1.0

    def test_power_setget(self) -> None:
        """Noise power property getter should return setter argument"""

        expected_noise = 1.234
        self.noise.power = expected_noise

        self.assertEqual(expected_noise, self.noise.power)

    def test_add_noise_power(self) -> None:
        """Added noise should have correct power"""

        signal = np.zeros(1000000, dtype=complex)
        powers = np.array([0, 1, 100, 1000])

        for expected_noise_power in powers:
            self.noise.power = expected_noise_power
            noisy_signal = self.noise.add(Signal(signal, sampling_rate=1.0))
            noisy_signal_power = np.var(noisy_signal.samples)

            self.assertTrue(abs(noisy_signal_power - expected_noise_power) <= (0.001 * expected_noise_power))

    def test_add_noise_from_realization(self) -> None:
        """Adding noise from realizations should result in reproducable noises"""

        signal = Signal(np.zeros(1000000, dtype=complex), sampling_rate=1.0)
        powers = np.array([0, 1, 100, 1000])

        for expected_noise_power in powers:
            self.noise.power = expected_noise_power
            realization = self.noise.realize()

            noisy_signal_alpha = self.noise.add(signal, realization)
            noisy_signal_beta = self.noise.add(signal, realization)

            assert_array_almost_equal(noisy_signal_alpha.samples, noisy_signal_beta.samples)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.noise, {"is_random_root"})
