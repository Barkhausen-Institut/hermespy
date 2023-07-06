# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import patch, PropertyMock 

import numpy as np
from numpy.testing import assert_array_equal
from scipy.constants import speed_of_light

from hermespy.channel import DelayChannelBase, SpatialDelayChannel, RandomDelayChannel
from hermespy.core import UniformArray, IdealAntenna, Signal
from hermespy.simulation import SimulatedDevice
from hermespy.tools import amplitude_path_loss
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockDelayChannel(DelayChannelBase):
    """Delay channel for testing only"""

    def _realize_delay(self) -> float:
        return 1.234

    def _realize_response(self) -> np.ndarray:

        return np.eye(1, 2, dtype=complex)


class TestDelayChannelBase(TestCase):

    def setUp(self) -> None:

        self.transmitter = SimulatedDevice(carrier_frequency=1.234)
        self.receiver = SimulatedDevice(carrier_frequency=1.234)
        
        self.channel = MockDelayChannel(transmitter=self.transmitter, receiver=self.receiver)

    def test_realize(self) -> None:
        """Test impulse response generation"""

        num_samples = 10
        sampling_rate = 1.234
        realization = self.channel.realize(num_samples, sampling_rate)
        expected_signal_scale = amplitude_path_loss(self.transmitter.carrier_frequency, self.channel._realize_delay() * speed_of_light)

        self.assertSequenceEqual((1, 2, 10, 2), realization.state.shape)
        self.assertAlmostEqual(10. * expected_signal_scale, np.sum(realization.state))
        assert_array_equal(expected_signal_scale * np.eye(1, 2), realization.state[:, :, 0, -1])

    def test_realize_validation(self) -> None:
        """Realization routine should raise RuntimeError for base band transmissions"""

        self.transmitter.carrier_frequency = 0.

        with self.assertRaises(RuntimeError):
            _ = self.channel.realize(1, 1.234)


class TestSpatialDelayChannel(TestCase):

    def setUp(self) -> None:

        self.carrier_frequency = 60e9

        self.transmitter = SimulatedDevice(carrier_frequency=self.carrier_frequency, antennas=UniformArray(IdealAntenna(), .5 * speed_of_light / self.carrier_frequency, (2, 1, 1)))
        self.receiver = SimulatedDevice(carrier_frequency=self.carrier_frequency, antennas=UniformArray(IdealAntenna(), .5 * speed_of_light / self.carrier_frequency, (3, 1, 1)))

        self.expected_delay = 1.4567
        self.transmitter.position = np.zeros(3)
        self.receiver.position = np.ones(3) / np.sqrt(3) * self.expected_delay * speed_of_light

        self.channel = SpatialDelayChannel(transmitter=self.transmitter, receiver=self.receiver)

    def test_delay_realization_validation(self) -> None:
        """Delay realization should raise RuntimeErrors on invalid internal states"""
        
        self.channel.transmitter = None
        with self.assertRaises(RuntimeError):
            self.channel._realize_delay()

    def test_delay_realization(self) -> None:
        """Delay realization should yield the correct time delay"""

        delay = self.channel._realize_delay()
        self.assertAlmostEqual(self.expected_delay, delay)

    def test_response_realization(self) -> None:
        """Channel realization should yield the correct sensor array response"""

        response = self.channel._realize_response()
        
        # Assert general response shape
        self.assertSequenceEqual((self.receiver.num_antennas, self.transmitter.num_antennas), response.shape)

    def test_power_loss(self) -> None:
        """Propagated signals should loose power according to the free space propagation loss"""
        
        # Assert free space propagation power loss
        power_signal = Signal(np.zeros((self.transmitter.num_antennas, 10)), self.transmitter.sampling_rate, self.transmitter.carrier_frequency)
        power_signal.samples[0, :] = np.ones(10)
        
        initial_energy = np.sum(power_signal.energy)
        expected_received_energy = initial_energy * amplitude_path_loss(self.transmitter.carrier_frequency, self.expected_delay * speed_of_light) ** 2

        propagated_signals, _, _ = self.channel.propagate(power_signal)
        received_energy = np.mean(propagated_signals[0].energy)
        
        self.assertAlmostEqual(expected_received_energy, received_energy)
        
        # Assert no power loss (flag disabled)
        self.channel.model_propagation_loss = False
        propagated_signals, _, _ = self.channel.propagate(power_signal)
        
        self.assertAlmostEqual(initial_energy, np.mean(propagated_signals[0].energy))     

    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.channel.Channel.transmitter', new_callable=PropertyMock) as transmitter_mock, \
             patch('hermespy.channel.Channel.receiver', new_callable=PropertyMock) as receiver_mock, \
             patch('hermespy.channel.Channel.random_mother', new_callable=PropertyMock) as random_mock:
            
            transmitter_mock.return_value = None
            receiver_mock.return_value = None
            random_mock.return_value = None
            
            test_yaml_roundtrip_serialization(self, self.channel)


class TestRandomDelayChannel(TestCase):

    def setUp(self) -> None:

        self.transmitter = SimulatedDevice(antennas=UniformArray(IdealAntenna(), .1, (2, 1, 1)))
        self.receiver = SimulatedDevice(antennas=UniformArray(IdealAntenna(), .1, (3, 1, 1)))

        self.channel = RandomDelayChannel(0., seed=42, transmitter=self.transmitter, receiver=self.receiver)

    def test_delay_validation(self) -> None:
        """Invalid delay arguments should raise ValueErrors"""

        with self.assertRaises(ValueError):
            self.channel.delay = -1.

        with self.assertRaises(ValueError):
            self.channel.delay = (-1., 1.)

        with self.assertRaises(ValueError):
            self.channel.delay = (2., 1.)

        with self.assertRaises(ValueError):
            self.channel.delay = (1., 2., 3.)
            
        with self.assertRaises(ValueError):
            self.channel.delay = 'wrong param'

    def test_constant_delay_realization(self) -> None:
        """Setting the delay as a scalar should realize a constant delay"""

        expected_delay = 1.
        self.channel.delay = expected_delay

        for _ in range(3):
            self.assertEqual(expected_delay, self.channel._realize_delay())

    def test_random_delay_realization(self) -> None:
        """Setting the delay as a scalar should realize a draw from the uniform distribution"""

        min_delay = 10
        max_delay = 11

        self.channel.delay = min_delay, max_delay

        for _ in range(5):

            realization = self.channel._realize_delay()
            self.assertTrue(min_delay <= realization <= max_delay)
            
    def test_delay_realization_validation(self) -> None:
        """Delay realization should raise a runtime error on invalid delay configurations"""
            
        with patch('hermespy.channel.delay.RandomDelayChannel.delay', new_callable=PropertyMock) as delay_patch:
            
            delay_patch.return_value = 'wrong value'
            
            with self.assertRaises(RuntimeError):
                _ = self.channel._realize_delay()
            
    def test_response_realization(self) -> None:
        """Response should return a diagonal matrix"""
        
        response = self.channel._realize_response()
        self.assertSequenceEqual((3, 2), response.shape)
        self.assertAlmostEqual(np.sqrt(2), np.linalg.norm(response))

    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.channel.Channel.transmitter', new_callable=PropertyMock) as transmitter_mock, \
             patch('hermespy.channel.Channel.receiver', new_callable=PropertyMock) as receiver_mock, \
             patch('hermespy.channel.Channel.random_mother', new_callable=PropertyMock) as random_mock:
            
            transmitter_mock.return_value = None
            receiver_mock.return_value = None
            random_mock.return_value = None
            
            test_yaml_roundtrip_serialization(self, self.channel)
