# -*- coding: utf-8 -*-

from os.path import join
from tempfile import TemporaryDirectory
from typing import Sequence
from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from h5py import File
from numpy.testing import assert_array_equal
from scipy.constants import pi

from hermespy.channel.channel import ChannelRealization, Channel
from hermespy.core import Signal
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ChannelMock(Channel[ChannelRealization]):
    """Implementation of the abstract channel base class for testing purposes only"""
    
    def realize(self, num_samples: int, _: float) -> ChannelRealization:
        
        impulse_response = np.zeros((self.receiver.antennas.num_receive_antennas, self.transmitter.antennas.num_transmit_antennas, num_samples, 1), dtype=np.complex_)
        return ChannelRealization(self, impulse_response)


class TestChannelRealization(TestCase):
    """Test base class for channel realizations"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        
        self.channel = Mock(spec=Channel)
        self.impulse_response = self.rng.standard_normal(size=(2, 2, 10, 3)) + 1j * self.rng.standard_normal(size=(2, 2, 10, 3))
        
        self.realization = ChannelRealization(self.channel, self.impulse_response)
        
    def test_init_validation(self) -> None:
        """Initialization should raise ValueError on invalid impulse resposne"""
        
        with self.assertRaises(ValueError):
            ChannelRealization(self.channel, self.impulse_response[np.newaxis, ::])
        
    def test_properties(self) -> None:
        """Intialization should parameters should be properly stored as class attributes"""
        
        self.assertIs(self.channel, self.realization.channel)
        assert_array_equal(self.impulse_response, self.realization.state)
        
    def test_reciprocal(self) -> None:
        """Reciprocal channel realization should compute the correct impulse response"""
        
        reciprocal = self.realization.reciprocal()
        self.assertSequenceEqual((2, 2, 10, 3), reciprocal.state.shape)

    def test_hdf_serialization(self) -> None:
        """Test HDF roundtrip serialization"""
        
        with TemporaryDirectory() as tempdir:
            
            file_path = join(tempdir, 'test.hdf')

            file = File(file_path, 'w')
            group = file.create_group('g1')
            self.realization.to_HDF(group)
            file.close()
            
            file = File(file_path, 'r')
            recalled_realization = ChannelRealization.from_HDF(file['g1'])
            file.close()
        
        assert_array_equal(self.realization.state, recalled_realization.state)


class TestChannel(TestCase):
    """Test channel base class"""
    
    def setUp(self) -> None:
    
        self.transmitter = SimulatedDevice()
        self.receiver = SimulatedDevice()
        self.scenario = Mock()
        
        self.channel = ChannelMock(transmitter=self.transmitter, receiver=self.receiver)
        
    def test_devices_init_validation(self) -> None:
        """Specifying transmitter / receiver and devices is forbidden"""
        
        with self.assertRaises(ValueError):
            ChannelMock(transmitter=self.transmitter, receiver=self.receiver, devices=(Mock(), Mock()))
            
    def test_devices_init(self) -> None:
        """Specifiying devices insteand of transmitter / receiver should properly initialize channel"""
        
        self.channel = ChannelMock(devices=(self.transmitter, self.receiver))
        
        self.assertIs(self.transmitter, self.channel.transmitter)
        self.assertIs(self.receiver, self.channel.receiver)
        
    def test_active_setget(self) -> None:
        """Active property getter should return setter argument"""
        
        self.channel.active = False
        self.assertFalse(self.channel.active)
        
    def test_transmitter_setget(self) -> None:
        """Transmitter property getter should return setter argument"""
        
        expected_transmitter = Mock()
        self.channel.transmitter = expected_transmitter
        
        self.assertIs(expected_transmitter, self.channel.transmitter)
        
    def test_receiver_setget(self) -> None:
        """Receeiver property getter should return setter argument"""
        
        expected_receiver = Mock()
        self.channel.receiver = expected_receiver
        
        self.assertIs(expected_receiver, self.channel.receiver)
        
    def test_scenario_setget(self) -> None:
        """Scenario property setter should correctly configure channel"""
        
        scenario = Mock()
        self.channel.scenario = scenario
        
        self.assertIs(scenario, self.channel.scenario)
        self.assertIs(scenario, self.channel.random_mother)


    def test_sync_offset_low_setget(self) -> None:
        """Synchronization offset lower bound property getter should return setter argument"""

        expected_sync_offset = 1.2345
        self.channel.sync_offset_low = expected_sync_offset

        self.assertEqual(expected_sync_offset, self.channel.sync_offset_low)

    def test_sync_offset_low_validation(self) -> None:
        """Synchronization offset lower bound property setter should raise ValueError on negative arguments"""

        with self.assertRaises(ValueError):
            self.channel.sync_offset_low = -1.0

        try:
            self.channel.sync_offset_low = 0.

        except ValueError:
            self.fail()

    def test_sync_offset_high_setget(self) -> None:
        """Synchronization offset upper bound property getter should return setter argument"""

        expected_sync_offset = 1.2345
        self.channel.sync_offset_high = expected_sync_offset

        self.assertEqual(expected_sync_offset, self.channel.sync_offset_high)

    def test_sync_offset_high_validation(self) -> None:
        """Synchronization offset upper bound property setter should raise ValueError on negative arguments"""

        with self.assertRaises(ValueError):
            self.channel.sync_offset_high = -1.0

        try:
            self.channel.sync_offset_high = 0.

        except ValueError:
            self.fail()

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

    def test_propagate_validation(self) -> None:
        """Propagation routine must raise errors in case of unsupported scenarios"""

        with self.assertRaises(ValueError):
            self.channel.propagate(Mock())

        with self.assertRaises(RuntimeError):
            ChannelMock().propagate()

        with self.assertRaises(ValueError):
            _ = self.channel.propagate(forwards=Signal(np.array([[1, 2, 3], [4, 5, 6]]), 1.))

        with self.assertRaises(ValueError):
            _ = self.channel.propagate(backwards=Signal(np.array([[1, 2, 3], [4, 5, 6]]), 1.))

    def test_propagate_argument_conversion(self) -> None:
        """Test propagation conversion of input arguments"""
        
        output = self.transmitter.generate_output([])
        output_propagations, _, _ = self.channel.propagate(output)
        
        signal = self.transmitter.transmit().mixed_signal
        signal_propagations, _, _  = self.channel.propagate(signal)
        
        signals = self.transmitter.transmit().emerging_signals
        signals_propagations, _, _  = self.channel.propagate(signals)
        
        self.assertIsInstance(output_propagations, Sequence)
        self.assertIsInstance(signal_propagations, Sequence)
        self.assertIsInstance(signals_propagations, Sequence)
        
    def test_propagate_inactive(self) -> None:
        """Propagation over an inactive channel should resultin empty signals"""
        
        self.channel.active = False
        forwards, backwards, _ = self.channel.propagate()
        
        self.assertEqual(0, forwards[0].num_samples)
        self.assertEqual(0, backwards[0].num_samples)

    def test_min_sampling_rate(self) -> None:
        """Minimum sampling rate property should return zero"""
        
        self.assertEqual(0., self.channel.min_sampling_rate)
