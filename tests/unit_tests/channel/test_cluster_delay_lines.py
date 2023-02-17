# -*- coding: utf-8 -*-
"""
=====================================
3GPP Cluster Delay Line Model Testing
=====================================
"""

from math import sin, cos
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal
from scipy.constants import pi, speed_of_light

from hermespy.core import Direction, Signal, IdealAntenna, Transformation, UniformArray
from hermespy.simulation import SimulatedDevice
from hermespy.channel.cluster_delay_lines import ClusterDelayLine, DelayNormalization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestClusterDelayLine(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.seed = 12345

        self.num_clusters = 3
        self.delay_spread_mean = -7.49
        self.delay_spread_std = 0.55
        self.delay_scaling = 3.8
        self.carrier_frequency = 1e9

        self.transmitter = SimulatedDevice(antennas=UniformArray(IdealAntenna, .5 * speed_of_light / self.carrier_frequency, (2, 2)),
                                           pose=Transformation.From_RPY(pos=np.array([0., 0., 0.]), rpy=np.array([0., 0., 0.])),
                                           carrier_frequency=self.carrier_frequency)
        
        self.receiver = SimulatedDevice(antennas=UniformArray(IdealAntenna, .5 * speed_of_light / self.carrier_frequency, (2, 2)),
                                        pose=Transformation.From_RPY(pos=np.array([100., 0., 0.]), rpy=np.array([0., 0., 0.])),
                                        carrier_frequency=self.carrier_frequency)

        self.channel = ClusterDelayLine(delay_spread_mean=self.delay_spread_mean,
                                        delay_spread_std=self.delay_spread_std,
                                        delay_scaling=self.delay_scaling,
                                        num_clusters=self.num_clusters,
                                        receiver=self.receiver,
                                        transmitter=self.transmitter,
                                        seed=1234)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertEqual(self.delay_spread_mean, self.channel.delay_spread_mean)
        self.assertEqual(self.delay_spread_std, self.channel.delay_spread_std)
        self.assertEqual(self.num_clusters, self.channel.num_clusters)

        self.assertEqual(self.delay_scaling, self.channel.delay_scaling)

    def test_num_clusters_setget(self) -> None:
        """Number of clusters property getter should return setter argument."""

        num_clusters = 123
        self.channel.num_clusters = num_clusters

        self.assertEqual(num_clusters, self.channel.num_clusters)

    def test_num_clusters_validation(self) -> None:
        """Number of clusters property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.channel.num_clusters = -1

        with self.assertRaises(ValueError):
            self.channel.num_clusters = 0

    def test_delay_spread_mean_setget(self) -> None:
        """Delay spread mean property getter should return setter argument."""

        delay_spread = 123
        self.channel.delay_spread_mean = delay_spread

        self.assertEqual(delay_spread, self.channel.delay_spread_mean)

    def test_delay_spread_std_setget(self) -> None:
        """Delay spread mean property getter should return setter argument."""

        std = 123
        self.channel.delay_spread_std = std

        self.assertEqual(std, self.channel.delay_spread_std)

    def test_delay_scaling_setget(self) -> None:
        """Delay scaling property getter should return setter argument."""

        delay_scaling = 123
        self.channel.delay_scaling = delay_scaling

        self.assertEqual(delay_scaling, self.channel.delay_scaling)

    def test_delay_scaling_validation(self) -> None:
        """Delay scaling property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.channel.delay_scaling = -1.

        with self.assertRaises(ValueError):
            self.channel.delay_scaling = 0.5

        try:

            self.channel.delay_scaling = 1.

        except ValueError:
            self.fail()

    def test_rice_factor_mean_setget(self) -> None:
        """Rice factor mean property getter should return setter argument."""

        rice_factor_mean = 123
        self.channel.rice_factor_mean = rice_factor_mean

        self.assertEqual(rice_factor_mean, self.channel.rice_factor_mean)

    def test_rice_factor_mean_validation(self) -> None:
        """Rice factor mean property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.channel.rice_factor_mean = -1.

        try:

            self.channel.rice_factor_mean = 0.

        except ValueError:
            self.fail()
            
    def test_rice_factor_std_setget(self) -> None:
        """Rice factor standard deviation property getter should return setter argument."""

        rice_factor_std = 123
        self.channel.rice_factor_std = rice_factor_std

        self.assertEqual(rice_factor_std, self.channel.rice_factor_std)

    def test_rice_factor_std_validation(self) -> None:
        """Rice factor standard deviation property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.channel.rice_factor_std = -1.

        try:

            self.channel.rice_factor_std = 0.

        except ValueError:
            self.fail()
            
    def test_cluster_shadowing_std_setget(self) -> None:
        """Cluster shadowing standard deviation property getter should return setter argument."""

        cluster_shadowing_std = 123
        self.channel.cluster_shadowing_std = cluster_shadowing_std

        self.assertEqual(cluster_shadowing_std, self.channel.cluster_shadowing_std)

    def test_cluster_shadowing_std_validation(self) -> None:
        """Cluster shadowing standard deviation property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.channel.cluster_shadowing_std = -1.

        try:

            self.channel.cluster_shadowing_std = 0.

        except ValueError:
            self.fail()

    def test_realization_nlos(self):

        self.channel.line_of_sight = False
        num_samples = 100
        sampling_rate = 1e5

        realization = self.channel.realize(num_samples, sampling_rate)

        self.assertFalse(np.any(np.isnan(realization.state)))
        self.assertEqual(num_samples, realization.num_samples)
        self.assertEqual(self.receiver.num_antennas, realization.num_transmit_streams)
        self.assertEqual(self.transmitter.num_antennas, realization.num_receive_streams)

    def test_doppler_shift(self) -> None:
        """A signal being propagated over the channel should be frequency shifted according to the doppler effect"""

        self.channel.num_clusters = 1
        self.receiver.velocity = np.array([10., 0., 0.])

        sampling_rate = 1e3
        num_samples = 400
        signal = np.outer(np.ones(4, dtype=complex), np.exp(2j * pi * .2 * np.arange(num_samples)))

        radial_velocity = (self.transmitter.velocity - self.receiver.velocity) @ Direction.From_Cartesian(self.receiver.global_position - self.transmitter.global_position, True)
        expected_doppler_shift = radial_velocity * self.transmitter.carrier_frequency / speed_of_light
        frequency_resolution = sampling_rate / num_samples

        shifted_signal, _, _ = self.channel.propagate(Signal(signal, sampling_rate))

        input_freq = np.fft.fft(signal[0, :])
        output_freq = np.fft.fft(shifted_signal[0].samples[0, :].flatten())

        self.assertAlmostEqual(expected_doppler_shift, (np.argmax(output_freq) - np.argmax(input_freq)) * frequency_resolution, delta=1)

    def test_time_of_flight_delay_normalization(self) -> None:
        """Time of flight delay normalization should result in an impulse response padded by the appropriate number of samples"""

        sampling_rate = 1e9
        
        self.channel.delay_normalization = DelayNormalization.ZERO
        self.channel.seed = 1
        zero_delay_realization = self.channel.realize(10, sampling_rate)

        self.channel.delay_normalization = DelayNormalization.TOF
        self.channel.seed = 1
        tof_delay_realization = self.channel.realize(10, sampling_rate)

        expected_num_tof_samples = int(np.linalg.norm(self.transmitter.position - self.receiver.position, 2) / speed_of_light * sampling_rate)
        num_tof_samples = tof_delay_realization.num_delay_taps - zero_delay_realization.num_delay_taps

        self.assertEqual(expected_num_tof_samples, num_tof_samples)

    def test_realization_los(self):

        self.channel.line_of_sight = True
        num_samples = 100
        sampling_rate = 1e5

        realization = self.channel.realize(num_samples, sampling_rate)

        self.assertFalse(np.any(np.isnan(realization.state)))
        self.assertEqual(num_samples, realization.num_samples)
        self.assertEqual(self.receiver.num_antennas, realization.num_receive_streams)
        self.assertEqual(self.transmitter.num_antennas, realization.num_transmit_streams)

    def test_pseudo_randomness(self) -> None:
        """Setting the random seed should result in identical impulse responses."""
        
        num_samples = 100
        sampling_rate = 1e5
        
        # Generate first impulse response
        self.channel.seed = 1
        first_realization = self.channel.realize(num_samples, sampling_rate)
        first_number = self.channel._rng.normal()

        # Generate second impulse response with identical initial seed
        self.channel.seed = 1
        second_realization = self.channel.realize(num_samples, sampling_rate)
        second_number = self.channel._rng.normal()

        # Both should be identical
        self.assertEqual(first_number, second_number)
        assert_array_equal(first_realization.state, second_realization.state)
        
    def test_spatial_properties(self) -> None:
        """Direction of arrival estimation should result in the correct angle estimation of impinging devices"""

        self.channel.num_clusters = 1

        self.transmitter.antennas = UniformArray(IdealAntenna, .5 * speed_of_light / self.carrier_frequency, (1,))
        self.transmitter.orientation = np.zeros(3, dtype=float)
        self.receiver.position = np.zeros(3, dtype=float)
        self.receiver.orientation = np.zeros(3, dtype=float)
        self.receiver.antennas = UniformArray(IdealAntenna, .5 * speed_of_light / self.carrier_frequency, (8,8))

        angle_candidates = [(.25 * pi, 0),
                            (.25 * pi, .25 * pi), 
                            (.25 * pi, .5 * pi), 
                            (.5 * pi, 0),
                            (.5 * pi, .25 * pi), 
                            (.5 * pi, .5 * pi), 
                            ]
        range = 1e3
      
        steering_codebook = np.empty((8**2, len(angle_candidates)), dtype=complex)
        for a, (zenith, azimuth) in enumerate(angle_candidates):
            steering_codebook[:, a] = self.receiver.antennas.spherical_phase_response(self.carrier_frequency, azimuth, zenith)


        probing_signal = Signal(np.exp(2j * pi * .25 * np.arange(100)), sampling_rate=1e3, carrier_frequency=self.carrier_frequency)

        for a, (zenith, azimuth) in enumerate(angle_candidates):

            self.channel.seed = 1
            self.transmitter.position = range * np.array([cos(azimuth) * sin(zenith),
                                                          sin(azimuth) * sin(zenith),
                                                          cos(zenith)], dtype=float)

            received_signal, _, _ = self.channel.propagate(probing_signal)

            beamformer = np.linalg.norm(steering_codebook.T.conj() @ received_signal[0].samples, 2, axis=1, keepdims=False)
            self.assertEqual(a, np.argmax(beamformer))
