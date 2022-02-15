# -*- coding: utf-8 -*-
"""Testing of the noise model base class."""

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import numpy.random as rnd

from hermespy.core.signal_model import Signal
from hermespy.simulation.noise import Noise, AWGN

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestNoise(TestCase):
    """Test the base class for noise models."""

    def setUp(self) -> None:

        self.random_node = Mock()
        self.random_node._rng = rnd.default_rng(42)

        self.power = 1.0
        self.noise = Noise(power=self.power)

    def test_init(self) -> None:
        """Initialization arguments should be properly stored as class attributes."""

        self.assertEqual(self.power, self.noise.power)

    def test_power_setget(self) -> None:
        """Noise power property getter should return setter argument."""

        expected_power = 1.234
        self.noise.power = expected_power

        self.assertEqual(expected_power, self.noise.power)

    def test_power_validation(self) -> None:
        """Noise power property setter should raise ValueError on negative arguments."""

        with self.assertRaises(ValueError):
            self.noise.power = -1.

        try:
            self.noise.power = 0.

        except ValueError:
            self.fail()


class TestAWGN(TestCase):
    """Test Additive White Gaussian Noise Model."""

    def setUp(self) -> None:

        self.random_node = Mock()
        self.random_node._rng = rnd.default_rng(42)

        self.noise = AWGN()
        self.noise.random_mother = self.random_node

    def test_add_noise_power(self) -> None:
        """Added noise should have correct power"""

        signal = Signal(np.zeros(1000000, dtype=complex), sampling_rate=1.)
        powers = np.array([0, 1, 100, 1000])

        for expected_noise_power in powers:

            noisy_signal = signal.copy()
            self.noise.add(noisy_signal, expected_noise_power)

            noisy_signal.samples = noisy_signal.samples - np.mean(noisy_signal.samples)
            noise_power = sum(noisy_signal.samples.real.flatten() ** 2 +
                              noisy_signal.samples.imag.flatten() ** 2) / noisy_signal.num_samples

            self.assertTrue(abs(noise_power - expected_noise_power) <= (0.001 * expected_noise_power))

    def test_to_yaml(self) -> None:
        """Test object serialization."""
        pass

    def test_from_yaml(self) -> None:
        """Test object recall from yaml."""
        pass
