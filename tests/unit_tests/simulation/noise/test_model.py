# -*- coding: utf-8 -*-

import unittest
from unittest.mock import Mock

import numpy as np
import numpy.random as rnd
from numpy.testing import assert_array_almost_equal

from hermespy.core.signal_model import Signal
from hermespy.simulation.noise import AWGN, AWGNRealization, NoiseModel, NoiseRealization
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class NoiseRealizationMock(NoiseRealization):
    """Mock class for testing the abstract base class."""

    def add_to(self, signal: Signal) -> Signal:
        return signal


class _TestNoiseRealization(unittest.TestCase):
    """Base class for testing noise realization implementations."""

    realization: NoiseRealization

    def test_add_to(self) -> None:
        """Noise of proper power should be added to signal"""

        realization = self.realization
        signal = Signal.Create(np.zeros((1, 1000000), dtype=np.complex128), sampling_rate=1.0)
        noisy_signal = realization.add_to(signal)

        self.assertEqual(len(signal.getitem()), len(noisy_signal.getitem()))

        expected_power = np.ones((1), dtype=np.float64) * realization.power
        assert_array_almost_equal(expected_power, noisy_signal.power, decimal=2)


class _TestNoiseModel(unittest.TestCase):
    """Base class for testing noise model implementations."""

    model: NoiseModel

    def test_realize(self) -> None:
        """Noise model should be properly realized"""

        realization = self.model.realize(0.0)
        self.assertIsInstance(realization, NoiseRealization)

    def test_add_noise(self) -> None:
        """Noise of should be properly realized and added to signal"""

        powers = np.array([0, 1, 100, 1000])
        for expected_noise_power in powers:
            signal = Signal.Create(np.zeros(1000000, dtype=complex), sampling_rate=1.0)
            noisy_signal = self.model.add_noise(signal, expected_noise_power)

            assert_array_almost_equal(np.ones((1), dtype=np.float64) * expected_noise_power, noisy_signal.power, decimal=0)
            self.assertEqual(len(signal.getitem()), len(noisy_signal.getitem()))

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.model, {"is_random_root"})


class TestNoiseRealizationMock(_TestNoiseRealization):
    """Test the mock realization of noise."""

    def setUp(self) -> None:
        self.realization = NoiseRealizationMock(Mock(), 0.0)

    def test_init(self) -> None:
        """Initialization parameters should be stored correctly"""

        expected_power = 1.234
        expected_noise_model = Mock(spec=NoiseModel)

        realization = NoiseRealizationMock(expected_noise_model, expected_power)

        self.assertEqual(expected_power, realization.power)

    def test_init_validation(self) -> None:
        """Initialization should raise ValueError on negative power"""

        with self.assertRaises(ValueError):
            NoiseRealizationMock(Mock(), -1.0)


class TestAWGNRealization(_TestNoiseRealization):
    """Test the realization of AWGN noise."""

    def setUp(self) -> None:
        self.model = AWGN()
        self.realization = AWGNRealization(self.model, 1.234)


class TestAWGN(_TestNoiseModel):
    """Test the base class for noise models."""

    def setUp(self) -> None:
        self.random_node = Mock()
        self.random_node._rng = rnd.default_rng(42)

        self.model = AWGN()
        self.model.random_mother = self.random_node


del _TestNoiseRealization
del _TestNoiseModel
