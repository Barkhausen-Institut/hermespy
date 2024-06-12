# -*- coding: utf-8 -*-
"""Test channel model for wireless transmission links"""

import unittest
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from h5py import File
from numpy.testing import assert_array_almost_equal
from numpy.random import default_rng

from hermespy.channel import IdealChannel
from hermespy.core import Signal
from hermespy.simulation import SimulatedDevice, SimulatedUniformArray, SimulatedIdealAntenna
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization
from unit_tests.utils import assert_signals_equal

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"



class TestIdealChannel(unittest.TestCase):
    """Test the channel model base class"""

    def setUp(self) -> None:

        self.gain = 1.0
        self.random_node = Mock()
        self.random_node._rng = default_rng(42)
        self.sampling_rate = 1e3
        self.alpha_device = SimulatedDevice(sampling_rate=self.sampling_rate)
        self.beta_device = SimulatedDevice(sampling_rate=self.sampling_rate)
        self.channel = IdealChannel(self.gain)
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

        self.assertEqual(self.gain, self.channel.gain, "Unexpected gain parameter initialization")

    def test_propagate_SISO(self) -> None:
        """Test valid propagation for the Single-Input-Single-Output channel"""

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:
                samples = np.random.rand(1, num_samples) + 1j * np.random.rand(1, num_samples)
                signal = Signal.Create(samples, self.sampling_rate)

                self.channel.gain = gain

                expected_propagated_samples = np.sqrt(gain) * samples

                realization = self.channel.realize()
                forwards_sample = realization.sample(self.alpha_device, self.beta_device)
                backwards_sample = realization.sample(self.beta_device, self.alpha_device)

                forwards_signal = forwards_sample.propagate(signal)
                backwards_signal = backwards_sample.propagate(signal)

                assert_array_almost_equal(expected_propagated_samples, forwards_signal.getitem())
                assert_array_almost_equal(expected_propagated_samples, backwards_signal.getitem())
                
                forwards_state_propagation = forwards_sample.state(num_samples, 1).propagate(signal)
                backwards_state_propagation = backwards_sample.state(num_samples, 1).propagate(signal)

                assert_array_almost_equal(expected_propagated_samples, forwards_state_propagation.getitem())
                assert_array_almost_equal(expected_propagated_samples, backwards_state_propagation.getitem())

    def test_propagate_SIMO(self) -> None:
        """Test valid propagation for the Single-Input-Multiple-Output channel"""

        self.beta_device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1.0, (3, 1, 1))

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:
                forwards_samples = np.random.rand(1, num_samples) + 1j * np.random.rand(1, num_samples)
                backwards_samples = np.random.rand(3, num_samples) + 1j * np.random.rand(3, num_samples)
                forwards_input = Signal.Create(forwards_samples, self.sampling_rate)
                backwards_input = Signal.Create(backwards_samples, self.sampling_rate)

                self.channel.gain = gain

                expected_forwards_samples = np.sqrt(gain) * np.repeat(forwards_samples, 3, axis=0)
                expected_backwards_samples = np.sqrt(gain) * np.sum(backwards_samples, axis=0, keepdims=True)

                realization = self.channel.realize()
                forwards_sample = realization.sample(self.alpha_device, self.beta_device)
                backwards_sample = realization.sample(self.beta_device, self.alpha_device)

                forwards_propagation = forwards_sample.propagate(forwards_input)
                backwards_propagation = backwards_sample.propagate(backwards_input)

                assert_array_almost_equal(expected_forwards_samples, forwards_propagation.getitem())
                assert_array_almost_equal(expected_backwards_samples, backwards_propagation.getitem())

                forwards_state_propagation = forwards_sample.state(num_samples, 1).propagate(forwards_input)
                backwards_state_propagation = backwards_sample.state(num_samples, 1).propagate(backwards_input)

                assert_array_almost_equal(expected_forwards_samples, forwards_state_propagation.getitem())
                assert_array_almost_equal(expected_backwards_samples, backwards_state_propagation.getitem())

    def test_propagate_MISO(self) -> None:
        """Test valid propagation for the Multiple-Input-Single-Output channel"""

        self.alpha_device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1.0, (3, 1, 1))

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:
                forwards_samples = np.random.rand(self.alpha_device.antennas.num_transmit_antennas, num_samples) + 1j * np.random.rand(self.alpha_device.antennas.num_transmit_antennas, num_samples)
                backwards_samples = np.random.rand(1, num_samples) + 1j * np.random.rand(1, num_samples)
                forwards_input = Signal.Create(forwards_samples, self.sampling_rate)
                backwards_input = Signal.Create(backwards_samples, self.sampling_rate)

                self.channel.gain = gain

                expected_forwards_samples = np.sqrt(gain) * np.sum(forwards_samples, axis=0, keepdims=True)
                expected_backwards_samples = np.sqrt(gain) * np.repeat(backwards_samples, self.alpha_device.antennas.num_transmit_antennas, axis=0)

                realization = self.channel.realize()
                forwards_sample = realization.sample(self.alpha_device, self.beta_device)
                backwards_sample = realization.sample(self.beta_device, self.alpha_device)

                forwards_signal = forwards_sample.propagate(forwards_input)
                backwards_signal = backwards_sample.propagate(backwards_input)

                assert_array_almost_equal(expected_forwards_samples, forwards_signal.getitem())
                assert_array_almost_equal(expected_backwards_samples, backwards_signal.getitem())

                forwards_state_propagation = forwards_sample.state(num_samples, 1).propagate(forwards_input)
                backwards_state_propagation = backwards_sample.state(num_samples, 1).propagate(backwards_input)

                assert_array_almost_equal(expected_forwards_samples, forwards_state_propagation.getitem())
                assert_array_almost_equal(expected_backwards_samples, backwards_state_propagation.getitem())

    def test_propagate_MIMO(self) -> None:
        """Test valid propagation for the Multiple-Input-Multiple-Output channel"""

        self.alpha_device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1.0, (2, 1, 1))
        self.beta_device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1.0, (3, 1, 1))

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:
                samples = np.random.rand(3, num_samples) + 1j * np.random.rand(3, num_samples)
                forwards_transmission = Signal.Create(samples[:2, :], self.sampling_rate)
                backwards_transmission = Signal.Create(samples, self.sampling_rate)

                self.channel.gain = gain

                expected_forwards_propagated_samples = np.append(np.sqrt(gain) * samples[:2, :], np.zeros((1, samples.shape[1])), axis=0)
                expected_backwards_propagated_samples = np.sqrt(gain) * samples[:2, :]

                realization = self.channel.realize()
                forwards_sample = realization.sample(self.alpha_device, self.beta_device)
                backwards_sample = realization.sample(self.beta_device, self.alpha_device)

                forwards_signal = forwards_sample.propagate(forwards_transmission)
                backwards_signal = backwards_sample.propagate(backwards_transmission)

                assert_array_almost_equal(expected_forwards_propagated_samples, forwards_signal.getitem())
                assert_array_almost_equal(expected_backwards_propagated_samples, backwards_signal.getitem())

                forwards_state_propagation = forwards_sample.state(num_samples, 1).propagate(forwards_transmission)
                backwards_state_propagation = backwards_sample.state(num_samples, 1).propagate(backwards_transmission)

                #assert_array_almost_equal(expected_forwards_propagated_samples, forwards_state_propagation[:, :])
                #assert_array_almost_equal(expected_backwards_propagated_samples, backwards_state_propagation[:, :])

    def test_propagate_empty(self) -> None:
        """Test propagation with no receive antennas"""
        
        self.alpha_device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1.0, (2, 1, 1))
        self.beta_device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1.0, (0, 0, 0))
        signal = Signal.Create(np.random.rand(self.alpha_device.num_transmit_antennas, 4) + 1j * np.random.rand(self.alpha_device.num_transmit_antennas, 4), self.sampling_rate)
        
        propagation = self.channel.propagate(signal, self.alpha_device, self.beta_device)
        self.assertEqual(0, propagation.num_streams)


    def test_channel_state_information(self) -> None:
        """Propagating over the linear channel state model should return identical results"""

        for num_samples in self.propagate_signal_lengths:
            for gain in self.propagate_signal_gains:
                samples = np.random.rand(1, num_samples) + 1j * np.random.rand(1, num_samples)
                signal = Signal.Create(samples, self.sampling_rate)
                self.channel.gain = gain

                realization = self.channel.realize()
                forwards_sample = realization.sample(self.alpha_device, self.beta_device)
                forwards_propagation = forwards_sample.propagate(signal)
                
                backwards_sample = realization.sample(self.beta_device, self.alpha_device)
                backwards_propagation = backwards_sample.propagate(signal)

                expected_csi_signal = forwards_sample.state(num_samples, max_num_taps=1).propagate(signal)

                assert_signals_equal(self, forwards_propagation, expected_csi_signal)
                assert_signals_equal(self, backwards_propagation, expected_csi_signal)

    def test_recall_realization(self) -> None:
        """Test realization recall"""

        file = File("test.h5", "w", driver="core", backing_store=False)
        group = file.create_group("g")

        expected_realization = self.channel.realize()
        expected_realization.to_HDF(group)

        recalled_realization = self.channel.recall_realization(group)
        file.close()

        self.assertEqual(expected_realization.gain, recalled_realization.gain)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        with patch("hermespy.channel.Channel.random_mother", new_callable=PropertyMock) as random_mock:
            random_mock.return_value = None

            test_yaml_roundtrip_serialization(self, self.channel)
