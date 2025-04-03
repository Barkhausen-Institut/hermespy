# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import patch, PropertyMock

import numpy as np
from h5py import File
from numpy.testing import assert_array_almost_equal
from scipy.constants import speed_of_light

from hermespy.channel.delay.spatial import SpatialDelayChannel, SpatialDelayChannelRealization
from hermespy.channel.delay.delay import DelayChannelBase
from hermespy.channel.delay.random import RandomDelayChannel
from hermespy.core import DenseSignal, Transformation
from hermespy.simulation import StaticTrajectory, SimulatedDevice
from hermespy.tools import amplitude_path_loss
from unit_tests.core.test_factory import test_roundtrip_serialization
from unit_tests.utils import assert_signals_equal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class _TestDelayChannelBase(TestCase):
    """Test base class of all delay channels"""

    def _init_channel(self, *args, **kwargs) -> DelayChannelBase:
        ...

    def setUp(self) -> None:
        self.sampling_rate = 1e6
        self.carrier_frequency = 1.234e9
        self.alpha_device = SimulatedDevice(sampling_rate=self.sampling_rate, carrier_frequency=self.carrier_frequency, pose=Transformation.From_Translation(np.array([0, 0, 0])))
        self.beta_device = SimulatedDevice(sampling_rate=self.sampling_rate, carrier_frequency=self.carrier_frequency, pose=Transformation.From_Translation(np.array([0, 0, 10])))

        self.channel = self._init_channel()

    def test_realize(self) -> None:
        """Test channel realization"""

        realization = self.channel.realize()
        self.assertEqual(self.channel.gain, realization.gain)
        
    def test_propagate_validation(self) -> None:
        """Propagating in base-band is prohibited if propagation loss is modeled"""

        self.alpha_device.carrier_frequency = 0.0
        self.beta_device.carrier_frequency = 0.0
        self.channel.model_propagation_loss = True
        
        with self.assertRaises(RuntimeError):
            self.channel.propagate(DenseSignal(np.zeros((self.alpha_device.num_antennas, 10)), self.alpha_device.sampling_rate), self.alpha_device, self.beta_device)

    def test_propagate_state(self) -> None:
        """Propagation and channel state should match"""
        
        sample = self.channel.realize().sample(self.alpha_device, self.beta_device)
        signal = DenseSignal(np.ones((self.alpha_device.num_antennas, 10)), self.sampling_rate, self.carrier_frequency)

        channel_propagation = sample.propagate(signal)
        state_propagation = sample.state(10, 1556701).propagate(signal)

        assert_signals_equal(self, channel_propagation, state_propagation)

    def test_model_serialization(self) -> None:
        """Test delay channel model serialization"""
        
        test_roundtrip_serialization(self, self.channel)

    def test_realization_serialization(self) -> None:
        """Test delay channel serialization realization"""
        
        realization = self.channel.realize()
        test_roundtrip_serialization(self, realization)


class TestSpatialDelayChannel(_TestDelayChannelBase):
    """Test the spatial delay channel"""

    def _init_channel(self, *args, **kwargs) -> SpatialDelayChannel:
        return SpatialDelayChannel(*args, **kwargs)

    def setUp(self) -> None:
        super().setUp()

        self.expected_delay = 1.4567
        self.alpha_device.trajectory = StaticTrajectory(Transformation.From_Translation(np.zeros(3)))
        self.beta_device.trajectory = StaticTrajectory(Transformation.From_Translation(np.ones(3) / np.sqrt(3) * self.expected_delay * speed_of_light))

    def test_propagate_identical_positions(self) -> None:
        """Propagation whith identical positions should result in zero delay"""
        
        self.alpha_device.trajectory = self.beta_device.trajectory
        
        signal = DenseSignal(np.zeros((self.alpha_device.num_antennas, 1)), self.alpha_device.sampling_rate, self.alpha_device.carrier_frequency)
        propagation = self.channel.propagate(signal, self.alpha_device, self.beta_device)
        
        self.assertEqual(1, propagation.num_samples)

    def test_power_loss(self) -> None:
        """Propagated signals should loose power according to the free space propagation loss"""

        # Assert free space propagation power loss
        power_signal = DenseSignal(np.zeros((self.alpha_device.num_antennas, 10)), self.alpha_device.sampling_rate, self.alpha_device.carrier_frequency)
        power_signal[0, :] = np.ones(10)

        initial_energy = np.sum(power_signal.energy)
        expected_received_energy = initial_energy * amplitude_path_loss(self.alpha_device.carrier_frequency, self.expected_delay * speed_of_light) ** 2

        propagation = self.channel.propagate(power_signal, self.alpha_device, self.beta_device)
        received_energy = np.mean(propagation.energy)

        self.assertAlmostEqual(expected_received_energy, received_energy)

        # Assert no power loss (flag disabled)
        self.channel.model_propagation_loss = False
        propagation = self.channel.propagate(power_signal, self.alpha_device, self.beta_device)

        self.assertAlmostEqual(initial_energy, np.mean(propagation.energy))
        
    def test_reciprocal_sample(self) -> None:
        """Test reciprocal channel samples"""
        
        realization = self.channel.realize()
        sample = realization.sample(self.alpha_device, self.beta_device)
        reciprocal_sample = realization.reciprocal_sample(sample, self.beta_device, self.alpha_device)

        self.assertAlmostEqual(sample.delay, reciprocal_sample.delay)

class TestRandomDelayChannel(_TestDelayChannelBase):
    def _init_channel(self, *args, **kwargs) -> SpatialDelayChannel:
        return RandomDelayChannel(0.0, *args, seed=42, **kwargs)
    
    def test_delay_setget(self) -> None:
        """Delay property getter should return setter argument"""

        expected_delay = 1.0
        self.channel.delay = expected_delay
        self.assertEqual(expected_delay, self.channel.delay)

        expected_delay = (1.0, 2.0)
        self.channel.delay = expected_delay
        self.assertEqual(expected_delay, self.channel.delay)

    def test_delay_validation(self) -> None:
        """Invalid delay arguments should raise ValueErrors"""

        with self.assertRaises(ValueError):
            self.channel.delay = -1.0

        with self.assertRaises(ValueError):
            self.channel.delay = (-1.0, 1.0)

        with self.assertRaises(ValueError):
            self.channel.delay = (2.0, 1.0)

        with self.assertRaises(ValueError):
            self.channel.delay = (1.0, 2.0, 3.0)

        with self.assertRaises(ValueError):
            self.channel.delay = "wrong param"
            
    def test_decorrelation_distance_setget(self) -> None:
        """Decorrelation distance property getter should return setter argument"""

        expected_distance = 1.0
        self.channel.decorrelation_distance = expected_distance
        self.assertEqual(expected_distance, self.channel.decorrelation_distance)
        
    def test_decorrelation_distance_validation(self) -> None:
        """Invalid decorrelation distance should raise a ValueError"""

        with self.assertRaises(ValueError):
            self.channel.decorrelation_distance = -1.0

    def test_constant_delay(self) -> None:
        """Setting the delay as a scalar should realize a constant delay"""

        expected_delay = 1.0
        self.channel.delay = expected_delay

        for _ in range(3):
            sample = self.channel.realize().sample(self.alpha_device, self.beta_device)
            self.assertEqual(expected_delay, sample.delay)

    def test_random_delay_realization(self) -> None:
        """Setting the delay as a scalar should realize a draw from the uniform distribution"""

        min_delay = 10
        max_delay = 11

        self.channel.delay = min_delay, max_delay

        for _ in range(5):
            realization = self.channel.realize()
            self.assertSequenceEqual((min_delay, max_delay), realization.delay)
            sample = realization.sample(self.alpha_device, self.beta_device)
            self.assertTrue(min_delay <= sample.delay <= max_delay)


del _TestDelayChannelBase
