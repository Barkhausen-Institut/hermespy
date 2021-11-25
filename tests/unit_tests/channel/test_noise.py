# -*- coding: utf-8 -*-
"""Testing of the noise model base class."""

import unittest
from unittest.mock import Mock

import numpy as np
import numpy.random as rnd

from hermespy.noise import Noise


class TestNoise(unittest.TestCase):
    """Test the base class for noise models."""

    def setUp(self) -> None:

        self.receiver = Mock()
        self.receiver.random_generator = Mock()
        self.random_generator = rnd.default_rng(42)
        self.noise = Noise(self.random_generator)
        self.noise.receiver = self.receiver

    def test_init(self) -> None:
        """Init parameters should be properly stored."""

        self.assertIs(self.random_generator, self.noise.random_generator)

    def test_receiver_setget(self) -> None:
        """Receiver property getter should return setter value."""

        self.assertIs(self.receiver, self.noise.receiver)

    def test_receiver_get_validation(self) -> None:
        """Receiver property getter should raise RuntimeError if the receiver is not specified."""

        self.noise = Noise()
        with self.assertRaises(RuntimeError):
            _ = self.noise.receiver

    def test_receiver_set_validation(self) -> None:
        """Receiver property setter should raise RuntimeError if attachment would be overwritten."""

        with self.assertRaises(RuntimeError):
            self.noise.receiver = Mock()

    def test_random_generator_setget(self) -> None:
        """Random generator property getter should return setter argument."""

        random_generator = Mock()
        self.noise.random_generator = random_generator

        self.assertIs(random_generator, self.noise.random_generator)

    def test_random_generator_get(self) -> None:
        """Random generator property getter should fetch from scenario if not specified."""

        self.noise.random_generator = None
        self.assertIs(self.receiver.random_generator, self.noise.random_generator)

    def test_random_generator_get_validation(self) -> None:
        """Random generator property getter should raise RuntimeError if model is floating"""

        self.noise = Noise()
        with self.assertRaises(RuntimeError):
            _ = self.noise.random_generator

    def test_add_noise_power(self) -> None:
        """Added noise should have correct power"""

        signal = np.zeros(1000000, dtype=complex)
        powers = np.array([0, 1, 100, 1000])

        for expected_noise_power in powers:

            noisy_signal = self.noise.add_noise(signal, expected_noise_power)
            noisy_signal = noisy_signal - np.mean(noisy_signal)
            noise_power = sum(noisy_signal.real ** 2 + noisy_signal.imag ** 2) / (len(signal))

            self.assertTrue(abs(noise_power - expected_noise_power) <= (0.001 * expected_noise_power))

    def test_to_yaml(self) -> None:
        """Test object serialization."""
        pass

    def test_from_yaml(self) -> None:
        """Test object recall from yaml."""
        pass
