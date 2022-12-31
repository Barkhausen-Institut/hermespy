# -*- coding: utf-8 -*-
"""Test channel model for wireless transmission links"""

import unittest
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from numpy.testing import assert_array_equal
from numpy.random import default_rng

from hermespy.channel import Channel
from hermespy.core.signal_model import Signal
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization


__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestChannel(unittest.TestCase):
    """Test the channel model base class"""

    def setUp(self) -> None:

        self.transmitter = Mock()
        self.receiver = Mock()
        self.active = True
        self.gain = 1.0
        self.random_node = Mock()
        self.random_node._rng = default_rng(42)
        self.sampling_rate = 1e3
        self.sync_offset_low = 0.
        self.sync_offset_high = 0.
        self.channel = Channel(
            transmitter=self.transmitter,
            receiver=self.receiver,
            active=self.active,
            gain=self.gain,
            sync_offset_low=self.sync_offset_low,
            sync_offset_high=self.sync_offset_high)
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

        self.assertIs(self.transmitter, self.channel.transmitter, "Unexpected transmitter parameter initialization")
        self.assertIs(self.receiver, self.channel.receiver, "Unexpected receiver parameter initialization")
        self.assertEqual(self.active, self.channel.active, "Unexpected active parameter initialization")
        self.assertEqual(self.gain, self.channel.gain, "Unexpected gain parameter initialization")
        self.assertEqual(self.sync_offset_low, self.channel.sync_offset_low)
        self.assertEqual(self.sync_offset_high, self.channel.sync_offset_high)

    def test_active_setget(self) -> None:
        """Active property getter must return setter parameter"""

        active = not self.active
        self.channel.active = active

        self.assertEqual(active, self.channel.active, "Active property set/get produced unexpected result")

    def test_transmitter_setget(self) -> None:
        """Transmitter property getter must return setter parameter"""

        channel = Channel()
        channel.transmitter = self.transmitter

        self.assertIs(self.transmitter, channel.transmitter, "Transmitter property set/get produced unexpected result")

    def test_receiver_setget(self) -> None:
        """Receiver property getter must return setter parameter"""

        channel = Channel()
        channel.receiver = self.receiver

        self.assertIs(self.receiver, channel.receiver, "Receiver property set/get produced unexpected result")

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

    def test_num_inputs_get(self) -> None:
        """Number of inputs property must return number of transmitting antennas"""

        num_inputs = 5
        self.transmitter.antennas.num_antennas = num_inputs

        self.assertEqual(num_inputs, self.channel.num_inputs, "Number of inputs property returned unexpected result")

    def test_num_inputs_validation(self) -> None:
        """Number of inputs property must raise RuntimeError if the channel is currently floating"""

        floating_channel = Channel()
        with self.assertRaises(RuntimeError):
            _ = floating_channel.num_inputs

    def test_num_outputs_get(self) -> None:
        """Number of outputs property must return number of receiving antennas"""

        num_outputs = 5
        self.receiver.antennas.num_antennas = num_outputs

        self.assertEqual(num_outputs, self.channel.num_outputs, "Number of outputs property returned unexpected result")

    def test_num_outputs_validation(self) -> None:
        """Number of outputs property must raise RuntimeError if the channel is currently floating"""

        floating_channel = Channel()
        with self.assertRaises(RuntimeError):
            _ = floating_channel.num_outputs

    def test_propagate_SISO(self) -> None:
        """Test valid propagation for the Single-Input-Single-Output channel"""

        self.transmitter.antennas.num_antennas = 1
        self.receiver.antennas.num_antennas = 1

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                samples = np.random.rand(1, num_samples) + 1j * np.random.rand(1, num_samples)
                signal = Signal(samples, self.sampling_rate)

                self.channel.gain = gain

                expected_propagated_samples = gain * samples
                forwards_signal, backwards_signal, _ = self.channel.propagate(signal, signal)

                assert_array_equal(expected_propagated_samples, forwards_signal[0].samples)
                assert_array_equal(expected_propagated_samples, backwards_signal[0].samples)

    def test_propagate_SIMO(self) -> None:
        """Test valid propagation for the Single-Input-Multiple-Output channel"""

        self.transmitter.antennas.num_antennas = 1
        self.receiver.antennas.num_antennas = 3

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                forwards_samples = (np.random.rand(1, num_samples)
                                    + 1j * np.random.rand(1, num_samples))
                backwards_samples = np.random.rand(3, num_samples) + 1j * np.random.rand(3, num_samples)
                forwards_input = Signal(forwards_samples, self.sampling_rate)
                backwards_input = Signal(backwards_samples, self.sampling_rate)

                self.channel.gain = gain

                expected_forwards_samples = gain * np.repeat(forwards_samples, 3, axis=0)
                expected_backwards_samples = gain * np.sum(backwards_samples, axis=0, keepdims=True)

                forwards_signal, backwards_signal,  _ = self.channel.propagate(forwards_input, backwards_input)

                assert_array_equal(expected_forwards_samples, forwards_signal[0].samples)
                assert_array_equal(expected_backwards_samples, backwards_signal[0].samples)

    def test_propagate_MISO(self) -> None:
        """Test valid propagation for the Multiple-Input-Single-Output channel"""

        num_transmit_antennas = 3
        self.transmitter.antennas.num_antennas = num_transmit_antennas
        self.receiver.antennas.num_antennas = 1

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                forwards_samples = (np.random.rand(num_transmit_antennas, num_samples)
                                    + 1j * np.random.rand(num_transmit_antennas, num_samples))
                backwards_samples = np.random.rand(1, num_samples) + 1j * np.random.rand(1, num_samples)
                forwards_input = Signal(forwards_samples, self.sampling_rate)
                backwards_input = Signal(backwards_samples, self.sampling_rate)

                self.channel.gain = gain

                expected_forwards_samples = gain * np.sum(forwards_samples, axis=0, keepdims=True)
                expected_backwards_samples = gain * np.repeat(backwards_samples, num_transmit_antennas, axis=0)

                forwards_signal, backwards_signal,  _ = self.channel.propagate(forwards_input, backwards_input)

                assert_array_equal(expected_forwards_samples, forwards_signal[0].samples)
                assert_array_equal(expected_backwards_samples, backwards_signal[0].samples)

    def test_propagate_MIMO(self) -> None:
        """Test valid propagation for the Multiple-Input-Multiple-Output channel"""

        num_antennas = 3
        self.transmitter.antennas.num_antennas = num_antennas
        self.receiver.antennas.num_antennas = num_antennas

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                samples = np.random.rand(num_antennas, num_samples) + 1j * np.random.rand(num_antennas,
                                                                                          num_samples)
                signal = Signal(samples, self.sampling_rate)

                self.channel.gain = gain

                expected_propagated_samples = gain * samples
                forwards_signal, backwards_signal,  _ = self.channel.propagate(signal, signal)

                assert_array_equal(expected_propagated_samples, forwards_signal[0].samples)
                assert_array_equal(expected_propagated_samples, backwards_signal[0].samples)

    def test_propagate_validation(self) -> None:
        """Propagation routine must raise errors in case of unsupported scenarios"""

        with self.assertRaises(ValueError):
            _ = self.channel.propagate(Signal(np.array([1, 2, 3]), self.sampling_rate))

        with self.assertRaises(ValueError):

            self.transmitter.num_antennas = 1
            _ = self.channel.propagate(Signal(np.array([[1, 2, 3], [4, 5, 6]]), self.sampling_rate))

        with self.assertRaises(RuntimeError):

            floating_channel = Channel()
            _ = floating_channel.propagate(Signal(np.array([[1, 2, 3]]), self.sampling_rate))

    def test_realize_SISO(self) -> None:
        """Test the realization generation for the Single-Input-Single-Output case"""

        self.transmitter.antennas.num_antennas = 1
        self.receiver.antennas.num_antennas = 1

        for response_length in self.impulse_response_lengths:
            for gain in self.impulse_response_gains:

                self.channel.gain = gain
                expected_state = gain * np.ones((1, 1, response_length, 1), dtype=float)

                realization = self.channel.realize(response_length, self.impulse_response_sampling_rate)
                assert_array_equal(expected_state, realization.state)

    def test_realize_SIMO(self) -> None:
        """Test the realization generation for the Single-Input-Multiple-Output case"""

        self.transmitter.antennas.num_antennas = 1
        self.receiver.antennas.num_antennas = 3

        for response_length in self.impulse_response_lengths:
            for gain in self.impulse_response_gains:

                self.channel.gain = gain
                expected_state = np.zeros((3, 1, response_length, 1), dtype=complex)
                expected_state[:, 0, :, 0] = gain

                realization = self.channel.realize(response_length, self.impulse_response_sampling_rate)
                assert_array_equal(expected_state, realization.state)

    def test_realize_MISO(self) -> None:
        """Test the realization generation for the Multiple-Input-Single-Output case"""

        self.transmitter.antennas.num_antennas = 3
        self.receiver.antennas.num_antennas = 1

        for response_length in self.impulse_response_lengths:
            for gain in self.impulse_response_gains:

                self.channel.gain = gain
                expected_state = np.zeros((1, 3, response_length, 1), dtype=complex)
                expected_state[0, :, :, 0] = gain

                realization = self.channel.realize(response_length, self.impulse_response_sampling_rate)
                assert_array_equal(expected_state, realization.state)

    def test_realize_MIMO(self) -> None:
        """Test the realization generation for the Multiple-Input-Multiple-Output case"""

        num_antennas = 3
        self.transmitter.antennas.num_antennas = num_antennas
        self.receiver.antennas.num_antennas = num_antennas

        for response_length in self.impulse_response_lengths:
            for gain in self.impulse_response_gains:

                self.channel.gain = gain
                expected_state = gain * np.tile(np.eye(num_antennas, num_antennas, dtype=complex)[:, :, None, None],
                                                (1, 1, response_length, 1))

                realization = self.channel.realize(response_length, self.impulse_response_sampling_rate)
                assert_array_equal(expected_state, realization.state)

    def test_realize_validation(self) -> None:
        """realization routine must raise errors in case of unsupported scenarios"""

        with self.assertRaises(RuntimeError):

            floating_channel = Channel()
            floating_channel.realize(np.empty(0, dtype=complex), self.impulse_response_sampling_rate)

    def test_channel_state_information(self) -> None:
        """Propagating over the linear channel state model should return identical results"""

        self.transmitter.antennas.num_antennas = 1
        self.receiver.antennas.num_antennas = 1

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                samples = np.random.rand(1, num_samples) + 1j * np.random.rand(1, num_samples)
                signal = Signal(samples, self.sampling_rate)
                self.channel.gain = gain

                forwards, backwards, csi = self.channel.propagate(signal, signal)
                expected_csi_signal = csi.linear[0, 0, ::].todense() @ samples.T

                assert_array_equal(forwards[0].samples, expected_csi_signal.T)
                assert_array_equal(backwards[0].samples, expected_csi_signal.T)

    def test_synchronization_offset(self) -> None:
        """The synchronization offset should be applied properly by adding a delay to the propgated signal"""

        self.transmitter.antennas.num_antennas = 1
        self.receiver.antennas.num_antennas = 1

        mock_generator = Mock()
        self.random_node._rng = mock_generator

        for num_samples in self.propagate_signal_lengths:
            for offset in [0, 1., -10., 100.]:

                samples = np.random.rand(1, num_samples) + 1j * np.random.rand(1, num_samples)
                signal_delay = 0.11
                signal = Signal(samples, self.sampling_rate, delay=signal_delay)

                mock_generator.uniform.return_value = 0.
                instant_signal, _, instant_channel = self.channel.propagate(signal)

                mock_generator.uniform.return_value = offset
                offset_signal, _, offset_channel = self.channel.propagate(signal)

                # The propagated signal should be delayed by the offset, while the CSI does not change
                assert_array_equal(instant_signal[0].samples, offset_signal[0].samples)
                assert_array_equal(instant_channel.state, offset_channel.state)
                self.assertEqual(signal_delay + offset, offset_signal[0].delay)

    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.channel.Channel.transmitter', new_callable=PropertyMock) as transmitter_mock, \
             patch('hermespy.channel.Channel.receiver', new_callable=PropertyMock) as receiver_mock, \
             patch('hermespy.channel.Channel.random_mother', new_callable=PropertyMock) as random_mock:
            
            transmitter_mock.return_value = None
            receiver_mock.return_value = None
            random_mock.return_value = None
            
            test_yaml_roundtrip_serialization(self, self.channel)
