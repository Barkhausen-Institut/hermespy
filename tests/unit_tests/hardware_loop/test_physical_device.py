# -*- coding: utf-8 -*-
"""Test Physical Device functionalities."""

from unittest import TestCase
from unittest.mock import patch

import numpy as np
from numpy.random import default_rng

from hermespy.core.signal_model import Signal
from hermespy.hardware_loop.physical_device import PhysicalDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalDeviceMock(PhysicalDevice):
    """Mock for a physical device."""

    __sampling_rate: float

    def __init__(self,
                 sampling_rate: float) -> None:

        PhysicalDevice.__init__(self)
        self.__sampling_rate = sampling_rate

    @property
    def sampling_rate(self) -> float:
        return self.__sampling_rate

    @property
    def carrier_frequency(self) -> float:
        return 0.

    def trigger(self) -> None:
        pass

    @property
    def velocity(self) -> np.ndarray:
        return np.zeros(3, dtype=float)


class TestPhysicalDevice(TestCase):
    """Test the base class for all physical devices."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.sampling_rate = 1e6

        self.device = PhysicalDeviceMock(sampling_rate=self.sampling_rate)

    @patch.object(PhysicalDeviceMock, 'trigger')
    def test_estimate_noise_power(self, mock_trigger) -> None:
        """Noise power estimation should return the correct power estimate."""

        num_samples = 10000
        expected_noise_power = .1
        samples = 2 ** -.5 * (self.rng.normal(size=num_samples, scale=expected_noise_power ** .5) + 1j *
                              self.rng.normal(size=num_samples, scale=expected_noise_power ** .5))
        signal = Signal(samples, sampling_rate=self.sampling_rate)
        mock_trigger.side_effect = lambda: [receiver.cache_reception(signal) for receiver in self.device.receivers]

        noise_power = self.device.estimate_noise_power(num_samples)
        self.assertAlmostEqual(expected_noise_power, noise_power[0], places=2)
