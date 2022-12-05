# -*- coding: utf-8 -*-
"""Testing of the noise model base class."""

import unittest
from unittest.mock import Mock

import numpy as np
import numpy.random as rnd

from hermespy.core.signal_model import Signal
from hermespy.simulation.noise import AWGN
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
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

    def test_add_noise_power(self) -> None:
        """Added noise should have correct power"""

        signal = np.zeros(1000000, dtype=complex)
        powers = np.array([0, 1, 100, 1000])

        for expected_noise_power in powers:

            noisy_signal = Signal(signal, sampling_rate=1.)
            self.noise.add(noisy_signal, expected_noise_power)
            noise_power = np.var(noisy_signal.samples)

            self.assertTrue(abs(noise_power - expected_noise_power) <= (0.001 * expected_noise_power))

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.noise, {'is_random_root'})
