# -*- coding: utf-8 -*-
"""Test channel model for wireless transmission links"""

import unittest
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from h5py import File
from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.random import default_rng

from hermespy.channel import IdealChannel, IdealChannelRealization
from hermespy.core import Signal, UniformArray, IdealAntenna
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestIdealChannelRealization(unittest.TestCase):
    """Test the ideal channel realization"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        self.sampling_rate = 1e6

        self.alpha_device = SimulatedDevice()
        self.beta_device = SimulatedDevice()
        
        self.realization = IdealChannelRealization(self.alpha_device, self.beta_device, 0.9876)

    def __test_propagate_state(self) -> None:
        """Subroutine for testing the propagation of the channel state information"""
        
        test_signal = Signal(self.rng.random((self.alpha_device.antennas.num_transmit_antennas, 100)) + 1j * self.rng.random((self.alpha_device.antennas.num_transmit_antennas, 100)), 1e3)

        signal_propagation = self.realization.propagate(test_signal)
        state_propagation = self.realization.state(self.alpha_device, self.beta_device, 0., self.sampling_rate, test_signal.num_samples, 1 + signal_propagation.signal.num_samples - test_signal.num_samples).propagate(test_signal)
        
        assert_array_almost_equal(signal_propagation.signal.samples, state_propagation.samples)

    def test_propagate_state_miso(self) -> None:
        """Propagation should result in a signal with the correct number of samples in the MISO case"""
        
        self.alpha_device.antennas = UniformArray(IdealAntenna, 1., (2, 1, 1))
        self.__test_propagate_state()
        
    def test_propagate_state_simo(self) -> None:
        """Propagation should result in a signal with the correct number of samples in the SIMO case"""
        
        self.beta_device.antennas = UniformArray(IdealAntenna, 1., (2, 1, 1))
        self.__test_propagate_state()
        
    def test_propagate_state_mimo(self) -> None:
        """Propagation should result in a signal with the correct number of samples in the MIMO case"""
        
        self.alpha_device.antennas = UniformArray(IdealAntenna, 1., (2, 1, 1))
        self.beta_device.antennas = UniformArray(IdealAntenna, 1., (2, 1, 1))
        self.__test_propagate_state()


class TestIdealChannel(unittest.TestCase):
    """Test the channel model base class"""

    def setUp(self) -> None:

        self.alpha_device = SimulatedDevice()
        self.beta_device = SimulatedDevice()
        self.gain = 1.0
        self.random_node = Mock()
        self.random_node._rng = default_rng(42)
        self.sampling_rate = 1e3
        self.channel = IdealChannel(self.alpha_device, self.beta_device, self.gain)
        self.channel.random_mother = self.random_node

        # Number of discrete-time samples generated for baseband_signal propagation testing
        self.propagate_signal_lengths = [1, 10, 100, 1000]
        self.propagate_signal_gains = [1.0, 0.5]

        # Number of discrete-time timestamps generated for realization testing
        self.impulse_response_sampling_rate = self.sampling_rate
        self.impulse_response_lengths = [1, 10, 100, 1000]  # ToDo: Add 0
        self.impulse_response_gains = [1.0, 0.5]

    def test_init(self) -> None:
        """Test that the init properly stores all parameters"""

        self.assertIs(self.alpha_device, self.channel.alpha_device, "Unexpected transmitter parameter initialization")
        self.assertIs(self.beta_device, self.channel.beta_device, "Unexpected receiver parameter initialization")
        self.assertEqual(self.gain, self.channel.gain, "Unexpected gain parameter initialization")

    def test_transmitter_setget(self) -> None:
        """Transmitter property getter must return setter parameter"""

        channel = IdealChannel()
        channel.alpha_device = self.alpha_device

        self.assertIs(self.alpha_device, channel.alpha_device, "Transmitter property set/get produced unexpected result")

    def test_receiver_setget(self) -> None:
        """Receiver property getter must return setter parameter"""

        channel = IdealChannel()
        channel.receiver = self.beta_device

        self.assertIs(self.beta_device, channel.receiver, "Receiver property set/get produced unexpected result")

    def test_propagate_SISO(self) -> None:
        """Test valid propagation for the Single-Input-Single-Output channel"""

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                samples = np.random.rand(1, num_samples) + 1j * np.random.rand(1, num_samples)
                signal = Signal(samples, self.sampling_rate)

                self.channel.gain = gain

                expected_propagated_samples = np.sqrt(gain) * samples
                
                realization = self.channel.realize()
                forwards_signal = realization.propagate(signal, self.alpha_device, self.beta_device).signal
                backwards_signal = realization.propagate(signal, self.beta_device, self.alpha_device).signal

                assert_array_almost_equal(expected_propagated_samples, forwards_signal.samples)
                assert_array_almost_equal(expected_propagated_samples, backwards_signal.samples)

    def test_propagate_SIMO(self) -> None:
        """Test valid propagation for the Single-Input-Multiple-Output channel"""

        self.beta_device.antennas = UniformArray(IdealAntenna, 1., (3, 1, 1))

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                forwards_samples = (np.random.rand(1, num_samples)
                                    + 1j * np.random.rand(1, num_samples))
                backwards_samples = np.random.rand(3, num_samples) + 1j * np.random.rand(3, num_samples)
                forwards_input = Signal(forwards_samples, self.sampling_rate)
                backwards_input = Signal(backwards_samples, self.sampling_rate)

                self.channel.gain = gain

                expected_forwards_samples = np.sqrt(gain) *  np.repeat(forwards_samples, 3, axis=0)
                expected_backwards_samples = np.sqrt(gain) *  np.sum(backwards_samples, axis=0, keepdims=True)

                realization = self.channel.realize()
                forwards_signal = realization.propagate(forwards_input, self.alpha_device, self.beta_device).signal
                backwards_signal = realization.propagate(backwards_input, self.beta_device, self.alpha_device).signal

                assert_array_almost_equal(expected_forwards_samples, forwards_signal.samples)
                assert_array_almost_equal(expected_backwards_samples, backwards_signal.samples)

    def test_propagate_MISO(self) -> None:
        """Test valid propagation for the Multiple-Input-Single-Output channel"""

        self.alpha_device.antennas = UniformArray(IdealAntenna, 1., (3, 1, 1))

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                forwards_samples = (np.random.rand(self.alpha_device.antennas.num_transmit_antennas, num_samples) + 1j * np.random.rand(self.alpha_device.antennas.num_transmit_antennas, num_samples))
                backwards_samples = np.random.rand(1, num_samples) + 1j * np.random.rand(1, num_samples)
                forwards_input = Signal(forwards_samples, self.sampling_rate)
                backwards_input = Signal(backwards_samples, self.sampling_rate)

                self.channel.gain = gain

                expected_forwards_samples = np.sqrt(gain) * np.sum(forwards_samples, axis=0, keepdims=True)
                expected_backwards_samples = np.sqrt(gain) * np.repeat(backwards_samples, self.alpha_device.antennas.num_transmit_antennas, axis=0)

                realization = self.channel.realize()
                forwards_signal = realization.propagate(forwards_input, self.alpha_device, self.beta_device).signal
                backwards_signal = realization.propagate(backwards_input, self.beta_device, self.alpha_device).signal

                assert_array_almost_equal(expected_forwards_samples, forwards_signal.samples)
                assert_array_almost_equal(expected_backwards_samples, backwards_signal.samples)

    def test_propagate_MIMO(self) -> None:
        """Test valid propagation for the Multiple-Input-Multiple-Output channel"""

        self.alpha_device.antennas = UniformArray(IdealAntenna, 1., (3, 1, 1))
        self.beta_device.antennas = UniformArray(IdealAntenna, 1., (2, 1, 1))

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                samples = np.random.rand(3, num_samples) + 1j * np.random.rand(3, num_samples)
                forwards_transmission = Signal(samples, self.sampling_rate)
                backwards_transmission = Signal(samples[:2, :], self.sampling_rate)

                self.channel.gain = gain

                expected_forwards_propagated_samples = np.sqrt(gain) * samples
                expected_backwards_propagated_samples = np.append(np.sqrt(gain) * samples[:2, :], np.zeros((1, samples.shape[1])), axis=0)

                realization = self.channel.realize()
                forwards_signal = realization.propagate(forwards_transmission, self.alpha_device, self.beta_device).signal
                backwards_signal = realization.propagate(backwards_transmission, self.beta_device, self.alpha_device).signal

                assert_array_almost_equal(expected_forwards_propagated_samples, forwards_signal.samples)
                assert_array_almost_equal(expected_backwards_propagated_samples, backwards_signal.samples)

    def test_channel_state_information(self) -> None:
        """Propagating over the linear channel state model should return identical results"""

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                samples = np.random.rand(1, num_samples) + 1j * np.random.rand(1, num_samples)
                signal = Signal(samples, self.sampling_rate)
                self.channel.gain = gain

                realization = self.channel.realize()
                forwards_propagation = realization.propagate(signal, self.alpha_device, self.beta_device)
                backwards_propagation = realization.propagate(signal, self.beta_device, self.alpha_device)

                expected_csi_signal = forwards_propagation.state(delay=0, sampling_rate=self.sampling_rate, num_samples=num_samples, max_num_taps=1).propagate(signal)

                assert_array_equal(forwards_propagation.signal.samples, expected_csi_signal.samples)
                assert_array_equal(backwards_propagation.signal.samples, expected_csi_signal.samples)

    def test_recall_realization(self) -> None:
        """Test realization recall"""

        file = File('test.h5', 'w', driver='core', backing_store=False)
        group = file.create_group('g')
        
        expected_realization = self.channel.realize()
        expected_realization.to_HDF(group)

        recalled_realization = self.channel.recall_realization(group)
        file.close()

        self.assertIs(expected_realization.alpha_device, recalled_realization.alpha_device)
        self.assertIs(expected_realization.beta_device, recalled_realization.beta_device)
        self.assertEqual(expected_realization.gain, recalled_realization.gain)
        
    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.channel.Channel.alpha_device', new_callable=PropertyMock) as transmitter_mock, \
             patch('hermespy.channel.Channel.beta_device', new_callable=PropertyMock) as receiver_mock, \
             patch('hermespy.channel.Channel.random_mother', new_callable=PropertyMock) as random_mock:
            
            transmitter_mock.return_value = None
            receiver_mock.return_value = None
            random_mock.return_value = None
            
            test_yaml_roundtrip_serialization(self, self.channel)
