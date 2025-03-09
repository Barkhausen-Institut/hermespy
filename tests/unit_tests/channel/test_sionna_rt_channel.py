# -*- coding: utf-8 -*-
"""Test Sionna Ray-Tracing Channel"""

from __future__ import annotations
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.constants import speed_of_light

from hermespy.channel.sionna_rt_channel import rt, SionnaRTChannel
from hermespy.core import Signal, Transformation
from hermespy.simulation import SimulatedDevice, SimulatedUniformArray, SimulatedIdealAntenna, StaticTrajectory
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Egor Achkasov"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Egor Achkasov", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSionnaRTChannel(unittest.TestCase):

    def setUp(self) -> None:
        # Properties
        self.seed = 42
        self.rng = np.random.default_rng(self.seed)
        self.sampling_rate = 3*1e9
        self.carrier_freq = 2.14e9

        # Devices
        self.distance_to_target = 10.
        self.device_position = np.asarray([0., 0., self.distance_to_target])
        self.device_trajectory = StaticTrajectory(Transformation.From_Translation(self.device_position))
        self.alpha_device = SimulatedDevice(
            sampling_rate=self.sampling_rate,
            antennas=SimulatedUniformArray(SimulatedIdealAntenna, 3e-3, [3, 1, 1]),
            carrier_frequency=self.carrier_freq)
        self.alpha_device.trajectory = self.device_trajectory
        self.beta_device = SimulatedDevice(
            sampling_rate=self.sampling_rate,
            carrier_frequency=self.carrier_freq)
        self.beta_device.trajectory = self.device_trajectory

        # Channel
        self.scene_file = rt.scene.simple_reflector
        self.channel = SionnaRTChannel(self.scene_file, 0.9876, self.seed)
        # Relization
        self.realization = self.channel.realize()

        # Samples
        self.sample_forward = self.realization.sample(
            self.alpha_device,
            self.beta_device,
            self.carrier_freq,
            self.sampling_rate)
        self.sample_backward = self.realization.sample(
            self.beta_device,
            self.alpha_device,
            self.carrier_freq,
            self.sampling_rate)
        self.channel_samples = [self.sample_forward, self.sample_backward]

    def test_init(self) -> None:
        """Assert consistency between the channel model, realization and samples.
        The scenes must be the same. The sample devices must be the same."""

        # Assert scene
        self.assertEqual(self.channel.scene, self.realization.scene)
        # sample implimentation does not store the scene so it cannot be asserted here

    def test_expected_energy_scale(self) -> None:
        """Channel sample should correctly calculate energy scaling.
        TODO current method implementation is not correct. Fix this test after the method is fixed.
        """

        for sample in self.channel_samples:
            self.assertEqual(sample.expected_energy_scale, np.abs(np.sum(sample._SionnaRTChannelSample__a)))

    def test_propagate(self) -> None:
        """Propagate a signal with a spike and check if the spike is still there"""

        # for each sample in the test case
        for sample in self.channel_samples:
            # Init test signal samples
            signal_shape = (sample.num_transmit_antennas, 150)
            signal_samples = self.rng.random(signal_shape) + 1j * self.rng.random(signal_shape)

            # Add spike to the samples
            x = np.arange(signal_shape[1])
            mu = x[x.size // 2]
            sigma = 0.05
            spike = np.exp(-0.5*((x-mu)/sigma)**2) / (sigma * np.sqrt(2*np.pi))  # normal distribution pdf
            signal_samples *= spike + 1.0
            signal_orig = Signal.Create(signal_samples, self.sampling_rate, self.carrier_freq)

            # Propagate
            signal_prop = sample.propagate(signal_orig)
            num_samples_diff = signal_prop.num_samples - signal_orig.num_samples
            state = sample.state(signal_orig.num_samples, 1 + num_samples_diff)
            signal_state_prop = state.propagate(signal_orig)

            # Propagation and state propagation should be almost the same
            assert_array_almost_equal(signal_prop.getitem(), signal_state_prop.getitem())

            # Assert the delays
            delay_expected = int(2. * self.distance_to_target / speed_of_light * self.sampling_rate)
            delay_actual = np.min(signal_prop.getitem().nonzero()[1])
            self.assertAlmostEqual(delay_actual, delay_expected, delta=1)

            # simple_reflector should cause a phase shift
            samples_expected = np.sum(signal_orig.getitem(), axis=0)
            samples_restored = signal_prop.getitem((0, slice(delay_expected, delay_expected + signal_orig.num_samples)))
            phase_shift = np.angle(samples_expected) - np.angle(samples_restored)
            samples_restored *= np.exp(1.j * phase_shift)

            # Check whether the spike is still on the same place
            self.assertEqual(np.argmax(samples_restored), np.argmax(samples_expected))

    def test_empty_paths(self) -> None:
        """Test state and porpagation when no path hit arrives at a receiver"""

        # Init sample with Rx in a dead zone
        device_position = np.array([0., 0., 0.])
        device_trajectory = StaticTrajectory(Transformation.From_Translation(device_position))
        self.alpha_device.trajectory = device_trajectory
        self.beta_device.trajectory = device_trajectory
        sample = self.realization.sample(
            self.alpha_device,
            self.beta_device,
            self.carrier_freq,
            self.sampling_rate)

        # Init test signal
        signal_shape = (self.alpha_device.num_transmit_antennas, 150)
        samples = np.empty(signal_shape, np.complex128)
        signal_orig = Signal.Create(samples, self.sampling_rate, self.carrier_freq)

        # Test state
        state = sample.state(signal_orig.num_samples, 1)
        self.assertTrue(np.all(state.state == 0.))

        # Test propagate
        signal_prop = sample.propagate(signal_orig)
        self.assertTrue(np.all(signal_prop.getitem() == 0.))
        self.assertEqual(signal_prop.num_streams, self.beta_device.num_receive_antennas)

    def test_reciprocal_sample(self) -> None:
        """Reciptocal sample must set the link state as given"""

        for sample in self.channel_samples:
            rSample = self.realization.reciprocal_sample(
                sample,
                self.alpha_device,
                self.beta_device,
                0,
                self.carrier_freq,
                self.sampling_rate)
            self.assertEqual(rSample.carrier_frequency, self.carrier_freq)
            self.assertEqual(rSample.bandwidth, self.sampling_rate)

    def test_model_serialization(self) -> None:
        """Test Sionna RT channel model serialization"""
        
        test_roundtrip_serialization(self, self.channel)

    def test_realization_serialization(self) -> None:
        """Test Sionna RT channel realization serialization"""
        
        test_roundtrip_serialization(self, self.realization)
