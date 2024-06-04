# -*- coding: utf-8 -*-

from __future__ import annotations
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from h5py import Group

from hermespy.channel.channel import Channel, ChannelRealization, ChannelSample, ChannelSampleHook, InterpolationMode, LinkState
from hermespy.core import ChannelStateInformation, Device, DeviceOutput, Signal
from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestLinkState(TestCase):
    """Test the link state dataclass"""
    
    def setUp(self) -> None:
        
        self.alpha_state = Mock()
        self.beta_state = Mock()
        self.carrier_frequency = 1.234
        self.bandwith = 5.678
        self.timestamp = 10.234
        
        self.state = LinkState(self.alpha_state, self.beta_state, self.carrier_frequency, self.bandwith, self.timestamp)

    def test_properties(self) -> None:
        """Properties should return the correct values"""
        
        self.assertIs(self.alpha_state, self.state.transmitter)
        self.assertIs(self.beta_state, self.state.receiver)
        self.assertEqual(self.carrier_frequency, self.state.carrier_frequency)
        self.assertEqual(self.bandwith, self.state.bandwidth)
        self.assertEqual(self.timestamp, self.state.time)

class ChannelSampleMock(ChannelSample):
    
    def _propagate(self, signal: Signal, interpolation: InterpolationMode) -> Signal:
        return signal
    
    def state(self, delay: float, sampling_rate: float, num_samples: int, max_num_taps: int) -> ChannelStateInformation:
        return ChannelStateInformation.Ideal(num_samples, self.receiver_state.antennas.num_receive_antennas, self.transmitter_state.antennas.num_transmit_antennas)


class ChannelRealizationMock(ChannelRealization[ChannelSampleMock]):
    """Implementation of the abstract channel realization base calss for testing purposes"""

    def _sample(
        self,
        state: LinkState,
    ) -> ChannelSampleMock:
        return ChannelSampleMock(state)

    def _reciprocal_sample(self, sample: ChannelSampleMock, state: LinkState) -> ChannelSampleMock:
        return ChannelSampleMock(state)

    def to_HDF(self, group: Group) -> None:
        group.attrs['gain'] = self.gain

    def _propagate(self, signal: Signal, transmitter: Device, receiver: Device, interpolation: InterpolationMode) -> Signal:
        return signal

    def state(self, transmitter: Device, receiver: Device, delay: float, sampling_rate: float, num_samples: int, max_num_taps: int) -> ChannelStateInformation:
        return ChannelStateInformation.Ideal(num_samples=num_samples, num_receive_streams=1, num_transmit_streams=1)


class ChannelMock(Channel[ChannelRealizationMock, ChannelSampleMock]):
    """Implementation of the abstract channel base class for testing purposes only"""

    def _realize(self) -> ChannelRealizationMock:
        return ChannelRealizationMock(self.sample_hooks, self.gain)

    def recall_realization(self, group: Group):
        return ChannelRealizationMock(self.sample_hooks, group.attrs['gain'])


class TestChannelSampleHook(TestCase):
    """Test the channel sample hook class"""
    
    def setUp(self) -> None:
        self.callback = Mock()
        self.transmitter = Mock()
        self.receiver = Mock()
        
        self.hook = ChannelSampleHook(self.callback, self.transmitter, self.receiver)
        
    def test_call(self) -> None:
        """Calling the hook should call the callback with the correct arguments"""
        
        sample = Mock()
        
        self.hook(sample, self.transmitter, Mock())
        self.callback.assert_not_called()
        
        self.hook(sample, Mock(), self.receiver)
        self.callback.assert_not_called()
        
        self.hook(sample, self.transmitter, self.receiver)
        self.callback.assert_called_once_with(sample)
        
        
class TestChannelSample(TestCase):
    """Test the base class for wireless channel samples."""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        self.transmitter = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (2, 1, 1)))
        self.receiver = SimulatedDevice()
        self.carrier_frequency = 1.234e9
        self.bandwidth = 5.678e6
        
        self.sample = ChannelMock().realize().sample(
            self.transmitter,
            self.receiver,
            0.0,
            self.carrier_frequency,
            self.bandwidth
        )
        
    def test_init(self) -> None:
        """Initialization should properly store arguments as properties"""
        
        self.assertEqual(self.carrier_frequency, self.sample.carrier_frequency)
        self.assertEqual(self.bandwidth, self.sample.bandwidth)
        self.assertEqual(0.0, self.sample.time)
        
    def test_properties(self) -> None:
        """Properties should return the correct values"""
        
        self.assertEqual(2, self.sample.num_transmit_antennas)
        self.assertEqual(1, self.sample.num_receive_antennas)
        self.assertEqual(self.carrier_frequency, self.sample.carrier_frequency)
        self.assertEqual(self.bandwidth, self.sample.bandwidth)
    
    def test_propagate_validation(self) -> None:
        """Propagate routine should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = self.sample.propagate(Mock())

    def test_propagate_deviceoutput(self) -> None:
        """Test propgation of device outputs"""

        signal = Signal.Create(self.rng.random((2, 10)), 1.0)
        device_output = Mock(spec=DeviceOutput)
        device_output.mixed_signal = signal

        propagation = self.sample.propagate(device_output)

        self.assertIsInstance(propagation, Signal)

    def test_propagate_signal(self) -> None:
        """Test propagation of signals"""

        signal = Signal.Create(self.rng.random((2, 10)), 1.0)
        propagation = self.sample.propagate(signal)

        self.assertIsInstance(propagation, Signal)


class TestChannelRealization(TestCase):
    """Test base class for channel realizations"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.sample_callback = Mock()
        self.sample_hooks = {ChannelSampleHook(self.sample_callback, None, None),}
        self.gain = 0.9
        
        self.transmitter = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (2, 1, 1)))
        self.receiver = SimulatedDevice()
        self.carrier_frequency = 1.234e9
        self.bandwidth = 5.678e6

        self.realization = ChannelRealizationMock(self.sample_hooks, self.gain)

    def test_initialization(self) -> None:
        """Initialization arguments should be properly stored as properties"""

        self.assertSetEqual(self.sample_hooks, self.realization.sample_hooks)
        self.assertEqual(self.gain, self.realization.gain)
        
    def test_sample(self) -> None:
        """Sampling should call the sample hooks and return the samples"""

        sample = self.realization.sample(self.transmitter, self.receiver, self.carrier_frequency, self.bandwidth)
        self.sample_callback.assert_called_once_with(sample)
        self.assertIsInstance(sample, ChannelSampleMock)

    def test_reciprocal_sample(self) -> None:
        """Reciprocal sampling should call the sample hooks and return the samples"""

        original_sample = self.realization.sample(self.transmitter, self.receiver, self.carrier_frequency, self.bandwidth)
        self.sample_callback.reset_mock()

        reciprocal_sample = self.realization.reciprocal_sample(original_sample, self.transmitter, self.receiver, self.carrier_frequency, self.bandwidth)
        self.sample_callback.assert_called_once_with(reciprocal_sample)
        self.assertIsInstance(reciprocal_sample, ChannelSampleMock)


class TestChannel(TestCase):
    """Test channel base class"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.alpha_device = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (2, 1, 1)))
        self.beta_device = SimulatedDevice()
        self.gain = 0.8

        self.channel = ChannelMock(0.8)

    def test_alpha_device_setget(self) -> None:
        """Alpha device property getter should return setter argument"""

        expected_alpha_device = Mock()
        self.channel.alpha_device = expected_alpha_device

        self.assertIs(expected_alpha_device, self.channel.alpha_device)

    def test_beta_device_setget(self) -> None:
        """Beta device property getter should return setter argument"""

        expected_beta_device = Mock()
        self.channel.beta_device = expected_beta_device

        self.assertIs(expected_beta_device, self.channel.beta_device)

    def test_scenario_setget(self) -> None:
        """Scenario property setter should correctly configure channel"""

        scenario = Mock()
        self.channel.scenario = scenario

        self.assertIs(scenario, self.channel.scenario)
        self.assertIs(scenario, self.channel.random_mother)

    def test_gain_setget(self) -> None:
        """Gain property getter must return setter parameter"""

        gain = 5.0
        self.channel.gain = 5.0

        self.assertIs(gain, self.channel.gain, "Gain property set/get produced unexpected result")

    def test_gain_validation(self) -> None:
        """Gain property setter must raise exception on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.channel.gain = -1.0

        try:
            self.channel.gain = 0.0

        except ValueError:
            self.fail("Gain property set to zero raised unexpected exception")

    def test_interpolation_mode_setget(self) -> None:
        """Interpolation mode property getter should return setter argument"""

        expected_mode = Mock()
        self.channel.interpolation_mode = expected_mode

        self.assertIs(expected_mode, self.channel.interpolation_mode)

    def test_realize(self) -> None:
        """Realizing a channel should generate and cache a new realization"""

        realization = self.channel.realize(cache=True)
        self.assertIs(realization, self.channel.realization)
        self.assertIsInstance(realization, ChannelRealizationMock)

        _ = self.channel.realize(cache=False)
        self.assertIs(realization, self.channel.realization)

    def test_propagate_validation(self) -> None:
        """Propagate routine should raise ValueError on invalid signal stream counts"""

        signal = Signal.Create(self.rng.random((3, 10)), 1.0)

        with self.assertRaises(ValueError):
            self.channel.propagate(signal, self.alpha_device, self.beta_device)

    def test_add_sample_hook(self) -> None:
        """Adding a sample hook should properly store it"""

        callback = Mock()
        hook = self.channel.add_sample_hook(callback)
        self.assertIn(hook, self.channel.sample_hooks)
        
        self.channel.remove_sample_hook(hook)
        self.assertNotIn(hook, self.channel.sample_hooks)
