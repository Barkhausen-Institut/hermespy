# -*- coding: utf-8 -*-
"""Test channel model for wireless transmission links."""

from datetime import time
import unittest
from unittest.mock import Mock
import re
from numpy.core.fromnumeric import size

from numpy.testing import assert_array_equal
from numpy.random import default_rng
import numpy as np
from scipy import stats

from channel import Channel
import scenario
from simulator_core.factory import Factory
from tests.unit_tests.utils import yaml_str_contains_element

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"

class TestChannel(unittest.TestCase):
    """Test the channel model base class."""

    def setUp(self) -> None:

        self.transmitter = Mock()
        self.receiver = Mock()
        self.active = True
        self.gain = 1.0
        self.generator = default_rng(0)
        self.scenario = Mock()
        self.scenario.sampling_rate = 1e3
        self.sync_offset_low = 0
        self.sync_offset_high = 0
        self.channel = Channel(
            transmitter=self.transmitter,
            receiver=self.receiver,
            active=self.active,
            gain=self.gain,
            random_generator=self.generator,
            scenario=self.scenario,
            sync_offset_low=self.sync_offset_low,
            sync_offset_high=self.sync_offset_high)

        # Number of discrete-time samples generated for signal propagation testing
        self.propagate_signal_lengths = [1, 10, 100, 1000]
        self.propagate_signal_gains = [1.0, 0.5]

        # Number of discrete-time timestamps generated for impulse response testing
        self.impulse_response_sampling_rate = self.scenario.sampling_rate
        self.impulse_response_lengths = [1, 10, 100, 1000]  # ToDo: Add 0
        self.impulse_response_gains = [1.0, 0.5]

    def test_init(self) -> None:
        """Test that the init properly stores all parameters."""

        self.assertIs(self.transmitter, self.channel.transmitter, "Unexpected transmitter parameter initialization")
        self.assertIs(self.receiver, self.channel.receiver, "Unexpected receiver parameter initialization")
        self.assertEqual(self.active, self.channel.active, "Unexpected active parameter initialization")
        self.assertEqual(self.gain, self.channel.gain, "Unexpected gain parameter initialization")
        self.assertEqual(self.generator, self.channel.random_generator)
        self.assertEqual(self.scenario, self.channel.scenario)
        self.assertEqual(self.sync_offset_low, self.channel.sync_offset_low)
        self.assertEqual(self.sync_offset_high, self.channel.sync_offset_high)

    def test_active_setget(self) -> None:
        """Active property getter must return setter parameter."""

        active = not self.active
        self.channel.active = active

        self.assertEqual(active, self.channel.active, "Active property set/get produced unexpected result")

    def test_transmitter_setget(self) -> None:
        """Transmitter property getter must return setter parameter."""

        channel = Channel()
        channel.transmitter = self.transmitter

        self.assertIs(self.transmitter, channel.transmitter, "Transmitter property set/get produced unexpected result")

    def test_transmitter_validation(self) -> None:
        """Transmitter property setter must raise exception if already configured."""

        with self.assertRaises(RuntimeError):
            self.channel.transmitter = Mock()

    def test_receiver_setget(self) -> None:
        """Receiver property getter must return setter parameter."""

        channel = Channel()
        channel.receiver = self.receiver

        self.assertIs(self.receiver, channel.receiver, "Receiver property set/get produced unexpected result")

    def test_receiver_validation(self) -> None:
        """Receiver property setter must raise exception if already configured."""

        with self.assertRaises(RuntimeError):
            self.channel.receiver = Mock()

    def test_gain_setget(self) -> None:
        """Gain property getter must return setter parameter."""

        gain = 5.0
        self.channel.gain = 5.0

        self.assertIs(gain, self.channel.gain, "Gain property set/get produced unexpected result")

    def test_gain_validation(self) -> None:
        """Gain property setter must raise exception on arguments smaller than zero."""

        with self.assertRaises(ValueError):
            self.channel.gain = -1.0

        try:
            self.channel.gain = 0.0

        except ValueError:
            self.fail("Gain property set to zero raised unexpected exception")

    def test_random_generator_setget(self) -> None:
        """Random generator property getter should return setter argument."""

        generator = Mock()
        self.channel.random_generator = generator

        self.assertIs(generator, self.channel.random_generator)

    def test_random_generator_get_default(self) -> None:
        """Random generator property getter should return scenario generator if not specified."""

        self.scenario.random_generator = Mock()
        self.channel.random_generator = None

        self.assertIs(self.scenario.random_generator, self.channel.random_generator)

    def test_scenario_setget(self) -> None:
        """Scenario property getter should return setter argument."""

        scenario = Mock()
        self.channel = Channel()
        self.channel.scenario = scenario

        self.assertIs(scenario, self.channel.scenario)

    def test_scenario_get_validation(self) -> None:
        """Scenario property getter should raise RuntimeError if scenario is not set."""

        self.channel = Channel()
        with self.assertRaises(RuntimeError):
            _ = self.channel.scenario

    def test_scenario_set_validation(self) -> None:
        """Scenario property setter should raise RuntimeError if scenario is already set."""

        with self.assertRaises(RuntimeError):
            self.channel.scenario = Mock()

    def test_num_inputs_get(self) -> None:
        """Number of inputs property must return number of transmitting antennas."""

        num_inputs = 5
        self.transmitter.num_antennas = num_inputs

        self.assertEqual(num_inputs, self.channel.num_inputs, "Number of inputs property returned unexpected result")

    def test_num_inputs_validation(self) -> None:
        """Number of inputs property must raise RuntimeError if the channel is currently floating."""

        floating_channel = Channel()
        with self.assertRaises(RuntimeError):
            _ = floating_channel.num_inputs

    def test_num_outputs_get(self) -> None:
        """Number of outputs property must return number of receiving antennas."""

        num_outputs = 5
        self.receiver.num_antennas = num_outputs

        self.assertEqual(num_outputs, self.channel.num_outputs, "Number of outputs property returned unexpected result")

    def test_num_outputs_validation(self) -> None:
        """Number of outputs property must raise RuntimeError if the channel is currently floating."""

        floating_channel = Channel()
        with self.assertRaises(RuntimeError):
            _ = floating_channel.num_outputs

    def test_indices(self) -> None:
        """Indices property must return respective transmitter and receiver indices."""

        expected_indices = (6, 7)
        self.transmitter.index = expected_indices[0]
        self.receiver.index = expected_indices[1]

        self.assertEqual(expected_indices, self.channel.indices, "Channel indices property returned unexpected result")

    def test_propagate_SISO(self) -> None:
        """Test valid propagation for the Single-Input-Single-Output channel."""

        self.transmitter.num_antennas = 1
        self.receiver.num_antennas = 1

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                signal = np.random.rand(1, num_samples) + 1j * np.random.rand(1, num_samples)
                self.channel.gain = gain

                expected_propagated_signal = gain * signal
                propagated_signal, _ = self.channel.propagate(signal)

                assert_array_equal(expected_propagated_signal, propagated_signal)

    def test_propagate_SIMO(self) -> None:
        """Test valid propagation for the Single-Input-Multiple-Output channel."""

        self.transmitter.num_antennas = 1
        self.receiver.num_antennas = 3

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                signal = np.random.rand(1, num_samples) + 1j * np.random.rand(1, num_samples)
                self.channel.gain = gain

                expected_propagated_signal = np.repeat(gain * signal, self.receiver.num_antennas, axis=0)
                propagated_signal, _ = self.channel.propagate(signal)

                assert_array_equal(expected_propagated_signal, propagated_signal)

    def test_propagate_MISO(self) -> None:
        """Test valid propagation for the Multiple-Input-Single-Output channel."""

        num_transmit_antennas = 3
        self.transmitter.num_antennas = num_transmit_antennas
        self.receiver.num_antennas = 1

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                signal = np.random.rand(num_transmit_antennas, num_samples) + 1j * np.random.rand(num_transmit_antennas,
                                                                                                  num_samples)
                self.channel.gain = gain

                expected_propagated_signal = gain * np.sum(signal, axis=0, keepdims=True)
                propagated_signal, _ = self.channel.propagate(signal)

                assert_array_equal(expected_propagated_signal, propagated_signal)

    def test_propagate_MIMO(self) -> None:
        """Test valid propagation for the Multiple-Input-Multiple-Output channel."""

        num_antennas = 3
        self.transmitter.num_antennas = num_antennas
        self.receiver.num_antennas = num_antennas

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:

                signal = np.random.rand(num_antennas, num_samples) + 1j * np.random.rand(num_antennas,
                                                                                         num_samples)
                self.channel.gain = gain

                expected_propagated_signal = gain * signal
                propagated_signal, _ = self.channel.propagate(signal)

                assert_array_equal(expected_propagated_signal, propagated_signal)

    def test_propagate_validation(self) -> None:
        """Propagation routine must raise errors in case of unsupported scenarios."""

        with self.assertRaises(ValueError):
            self.channel.propagate(np.array([1, 2, 3]))

        with self.assertRaises(ValueError):

            self.transmitter.num_antennas = 1
            self.channel.propagate(np.array([[1, 2, 3], [4, 5, 6]]))

        with self.assertRaises(RuntimeError):

            floating_channel = Channel()
            floating_channel.propagate(np.array([[1, 2, 3]]))

    def test_impulse_response_SISO(self) -> None:
        """Test the impulse response generation for the Single-Input-Single-Output case."""

        self.transmitter.num_antennas = 1
        self.receiver.num_antennas = 1

        for response_length in self.impulse_response_lengths:
            for gain in self.impulse_response_gains:

                self.channel.gain = gain
                timestamps = np.arange(response_length) / self.impulse_response_sampling_rate
                expected_impulse_response = gain * np.ones((response_length, 1, 1, 1), dtype=float)

                impulse_response = self.channel.impulse_response(timestamps)
                assert_array_equal(expected_impulse_response, impulse_response)

    def test_impulse_response_SIMO(self) -> None:
        """Test the impulse response generation for the Single-Input-Multiple-Output case."""

        self.transmitter.num_antennas = 1
        self.receiver.num_antennas = 3

        for response_length in self.impulse_response_lengths:
            for gain in self.impulse_response_gains:

                self.channel.gain = gain
                timestamps = np.arange(response_length) / self.impulse_response_sampling_rate
                expected_impulse_response = np.zeros((response_length, 3, 1, 1), dtype=complex)
                expected_impulse_response[:, :, 0, :] = gain

                impulse_response = self.channel.impulse_response(timestamps)
                assert_array_equal(expected_impulse_response, impulse_response)

    def test_impulse_response_MISO(self) -> None:
        """Test the impulse response generation for the Multiple-Input-Single-Output case."""

        self.transmitter.num_antennas = 3
        self.receiver.num_antennas = 1

        for response_length in self.impulse_response_lengths:
            for gain in self.impulse_response_gains:

                self.channel.gain = gain
                timestamps = np.arange(response_length) / self.impulse_response_sampling_rate
                expected_impulse_response = np.zeros((response_length, 1, 3, 1), dtype=complex)
                expected_impulse_response[:, 0, :, :] = gain

                impulse_response = self.channel.impulse_response(timestamps)
                assert_array_equal(expected_impulse_response, impulse_response)

    def test_impulse_response_MIMO(self) -> None:
        """Test the impulse response generation for the Multiple-Input-Multiple-Output case."""

        num_antennas = 3
        self.transmitter.num_antennas = num_antennas
        self.receiver.num_antennas = num_antennas

        for response_length in self.impulse_response_lengths:
            for gain in self.impulse_response_gains:

                self.channel.gain = gain
                timestamps = np.arange(response_length) / self.impulse_response_sampling_rate
                expected_impulse_response = gain * np.tile(np.eye(num_antennas, num_antennas, dtype=complex),
                                                           (response_length, 1, 1))
                expected_impulse_response = np.expand_dims(expected_impulse_response, axis=-1)

                impulse_response = self.channel.impulse_response(timestamps)
                assert_array_equal(expected_impulse_response, impulse_response)

    def test_impulse_response_validation(self) -> None:
        """Impulse response routine must raise errors in case of unsupported scenarios."""

        with self.assertRaises(RuntimeError):

            floating_channel = Channel()
            floating_channel.impulse_response(np.empty(0, dtype=complex))

    def test_to_yaml(self) -> None:
        """Test YAML serialization dump validity."""
        pass

    def test_from_yaml(self) -> None:
        """Test YAML serialization recall validity."""
        pass


class TestChannelTimeOffsetBehavior(unittest.TestCase):
    def setUp(self) -> None:
        self.SEED = 42
        self.rng_default_seed_1 = np.random.default_rng(self.SEED)
        self.rng_default_seed_2 = np.random.default_rng(self.SEED)

        self.scenario = Mock()
        self.scenario.sampling_rate = 1e3
        self.sample_duration = 1/self.scenario.sampling_rate
        self.scenario.random_generator = np.random.default_rng()

        self.mock_transmitter = Mock()
        self.mock_transmitter.num_antennas = 1

        self.mock_receiver = Mock()
        self.mock_receiver.num_antennas = 1

        self.x_one_antenna = (
            np.random.randint(low=1, high=4, size=(1,100))
            + 1j * np.random.randint(low=1, high=4, size=(1,100))
        )

        self.x_two_antenna = (
            np.random.randint(low=1, high=4, size=(2,100))
            + 1j * np.random.randint(low=1, high=4, size=(2,100))
        )

        self.channel_without_delay = self.create_channel(0, 0)

    def test_no_delay_for_default_parameters(self) -> None:
        ch = self.create_channel(None, None)

        np.testing.assert_array_almost_equal(
            ch.propagate(self.x_one_antenna, self.rng_default_seed_1)[0],
            self.propagate_without_delay(self.x_one_antenna, ch, self.rng_default_seed_2)
        )

    def test_one_exact_sample_delay(self) -> None:
        ch = self.create_channel(self.sample_duration, self.sample_duration)
        np.testing.assert_array_almost_equal(
            ch.propagate(self.x_one_antenna, self.rng_default_seed_1)[0],
            np.hstack(
                (np.zeros((1,1), dtype=complex),
                self.propagate_without_delay(self.x_one_antenna, self.rng_default_seed_1)))
        )


    def test_non_int_sample_delay_gets_truncated(self) -> None:
        ch = self.create_channel(1.5*self.sample_duration, 1.5*self.sample_duration)

        np.testing.assert_array_almost_equal(
            ch.propagate(self.x_one_antenna, self.rng_default_seed_1)[0],
            np.hstack(
                (np.zeros((1,1), dtype=complex),
                 self.propagate_without_delay(self.x_one_antenna,
                                              self.rng_default_seed_2)))
        )

    def test_sample_is_picked(self) -> None:
        COMMON_SEED = 42
        SYNC_OFFSET_LOW = 0
        SYNC_OFFSET_HIGH = 10*self.sample_duration

        rng_local = np.random.default_rng(COMMON_SEED)

        delay = rng_local.uniform(low=SYNC_OFFSET_LOW, high=SYNC_OFFSET_HIGH) * self.scenario.sampling_rate
        ch = self.create_channel(SYNC_OFFSET_LOW, SYNC_OFFSET_HIGH, COMMON_SEED)

        np.testing.assert_array_almost_equal(
            ch.propagate(self.x_one_antenna, self.rng_default_seed_1)[0],
            np.hstack(
                (np.zeros((1, int(delay)), dtype=complex),
                 self.propagate_without_delay(self.x_one_antenna,
                                              self.rng_default_seed_2)))
        )

    def propagate_without_delay(self,
                                signal: np.ndarray,
                                ch: Channel = None,
                                rng: np.random.Generator = None) -> np.ndarray:
        if ch is None:
            ch = self.channel_without_delay

        return ch.propagate(signal, rng)[0]

    def create_channel(self, sync_low: float, sync_high: float, seed: int = 42) -> Channel:
        rng = np.random.default_rng(seed)
        return Channel(
            transmitter=self.mock_transmitter,
            receiver=self.mock_receiver,
            scenario=self.scenario,
            active=True,
            gain=1,
            sync_offset_low=sync_low,
            sync_offset_high=sync_high,
            random_generator=rng
        )