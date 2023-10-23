# -*- coding: utf-8 -*-
"""
=====================================
3GPP Cluster Delay Line Model Testing
=====================================
"""

from math import sin, cos
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from h5py import File
from numpy.testing import assert_array_equal, assert_array_almost_equal
from scipy.constants import pi, speed_of_light

from hermespy.core import Direction, Signal, IdealAntenna, Transformation, UniformArray
from hermespy.simulation import SimulatedDevice
from hermespy.channel.cluster_delay_lines import ClusterDelayLine, ClusterDelayLineBase, ClusterDelayLineRealization, DelayNormalization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestClusterDelayLineRealization(TestCase):
    """Test the 3GPP Cluster Delay Line Model Realization Implementation"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)

        self.carrier_frequency = 1e9
        self.alpha_device = SimulatedDevice(pose=Transformation.From_Translation(np.array([0, 0, 0])), carrier_frequency=self.carrier_frequency)
        self.beta_device = SimulatedDevice(pose=Transformation.From_Translation(np.array([100, 100, 0])), carrier_frequency=self.carrier_frequency)
        
        num_clusters = 6
        num_rays = ClusterDelayLineBase._ray_offset_angles.size
        
        self.gain = 1.234
        self.los = False
        self.rice_factor = .9
        self.aoa = self.rng.standard_normal((num_clusters, num_rays))
        self.zoa = self.rng.standard_normal((num_clusters, num_rays))
        self.aod = self.rng.standard_normal((num_clusters, num_rays))
        self.zod = self.rng.standard_normal((num_clusters, num_rays))
        self.cluster_delays = np.arange(num_clusters) * 1e-6
        self.cluster_delay_spread = 2e-6
        self.cluster_powers = self.rng.rayleigh(size=num_clusters)
        self.polarization_transformations = self.rng.standard_normal((2, 2, num_clusters, num_rays)) + 1j * self.rng.standard_normal((2, 2, num_clusters, num_rays))
        
        self.realization = ClusterDelayLineRealization(
            self.alpha_device, self.beta_device, self.gain, self.los,
            self.rice_factor, self.aoa, self.zoa, self.aod, self.zod,
            self.cluster_delays, self.cluster_delay_spread, self.cluster_powers,
            self.polarization_transformations
        )
        
    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""
        
        self.assertIs(self.alpha_device, self.realization.alpha_device)
        self.assertIs(self.beta_device, self.realization.beta_device)
        self.assertEqual(self.gain, self.realization.gain)
        self.assertEqual(self.los, self.realization.line_of_sight)
        self.assertEqual(self.rice_factor, self.realization.rice_factor)
        assert_array_equal(self.aoa, self.realization.azimuth_of_arrival)
        assert_array_equal(self.zoa, self.realization.zenith_of_arrival)
        assert_array_equal(self.aod, self.realization.azimuth_of_departure)
        assert_array_equal(self.zod, self.realization.zenith_of_departure)
        assert_array_equal(self.cluster_delays, self.realization.cluster_delays)
        self.assertEqual(self.cluster_delay_spread, self.realization.cluster_delay_spread)
        assert_array_equal(self.cluster_powers, self.realization.cluster_powers)
        assert_array_equal(self.polarization_transformations, self.realization.polarization_transformations)

    def test_propagate_state(self) -> None:
        """Propagation should result in a signal with the correct number of samples"""
        
        sampling_rate = 1e6
        samples = self.rng.standard_normal((self.alpha_device.antennas.num_transmit_antennas, 100)) + 1j* self.rng.standard_normal((self.alpha_device.antennas.num_transmit_antennas, 100))
        signal = Signal(samples, sampling_rate, self.carrier_frequency)
        
        signal_propagation = self.realization.propagate(signal)
        state_propagation = self.realization.state(self.alpha_device, self.beta_device, 0., sampling_rate, signal.num_samples, 1 + signal_propagation.signal.num_samples - signal.num_samples).propagate(signal)
        
        assert_array_almost_equal(signal_propagation.signal.samples, state_propagation.samples)
        
    def test_state_skip_tap(self) -> None:
        """State function should skip taps higher than the max tap index"""
        
        state = self.realization.state(self.alpha_device, self.beta_device, 0., 1e6, 10, 1)
        self.assertEqual(10, state.num_samples)

    def test_plot_angles(self) -> None:
        """Test the angle visualization routine"""
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            self.realization.plot_angles()
            mock_figure.assert_called_once()
            
    def test_plot_power_delay_profile(self) -> None:
        """Test the power delay profile visualization routine"""
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            
            self.realization.plot_power_delay()
            mock_figure.assert_called_once()
 
    def test_plot_rays(self) -> None:
        """Test the ray visualization routine"""
        
        with patch('matplotlib.pyplot.figure') as mock_figure:
            self.realization.plot_rays()
            mock_figure.assert_called_once()
            
    def test_angular_spread_validation(self) -> None:
        """Angular spread routine should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.realization._angular_spread(np.arange(3), np.arange(4))
 
    def test_azimuth_arrival_spread(self) -> None:
        """Test the azimuth of arrival spread calculation"""
        
        spread = self.realization.azimuth_arrival_spread
        
    def test_azimuth_departure_spread(self) -> None:
        """Test the azimuth of departure spread calculation"""
        
        spread = self.realization.azimuth_departure_spread
        
    def test_zenith_arrival_spread(self) -> None:
        """Test the zenith of arrival spread calculation"""
        
        spread = self.realization.zenith_arrival_spread
        
    def test_zenith_departure_spread(self) -> None:
        """Test the zenith of departure spread calculation"""
        
        spread = self.realization.zenith_departure_spread
        
    def test_hdf_serialization(self) -> None:
        """Test serialization from and to HDF"""
        
        file = File('test.h5', 'w', driver='core', backing_store=False)
        group = file.create_group('g')
        
        self.realization.to_HDF(group)
        recalled_realization = ClusterDelayLineRealization.From_HDF(group, self.alpha_device, self.beta_device)
        
        file.close()
        
        assert_array_equal(self.realization.azimuth_of_arrival, recalled_realization.azimuth_of_arrival)
        assert_array_equal(self.realization.azimuth_of_departure, recalled_realization.azimuth_of_departure)
        self.assertIs(self.realization.alpha_device, recalled_realization.alpha_device)
        self.assertIs(self.realization.beta_device, recalled_realization.beta_device)
        self.assertEqual(self.realization.gain, recalled_realization.gain)


class TestClusterDelayLine(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:

        self.seed = 12345

        self.num_clusters = 3
        self.delay_spread_mean = -7.49
        self.delay_spread_std = 0.55
        self.delay_scaling = 3.8
        self.carrier_frequency = 1e9

        self.alpha_device = SimulatedDevice(antennas=UniformArray(IdealAntenna, .5 * speed_of_light / self.carrier_frequency, (2, 2)),
                                            pose=Transformation.From_RPY(pos=np.array([0., 0., 0.]), rpy=np.array([0., 0., 0.])),
                                            carrier_frequency=self.carrier_frequency)
        
        self.beta_device = SimulatedDevice(antennas=UniformArray(IdealAntenna, .5 * speed_of_light / self.carrier_frequency, (2, 2)),
                                           pose=Transformation.From_RPY(pos=np.array([100., 0., 0.]), rpy=np.array([0., 0., 0.])),
                                           carrier_frequency=self.carrier_frequency)

        self.channel = ClusterDelayLine(
            self.alpha_device, self.beta_device,
            delay_spread_mean=self.delay_spread_mean,
            delay_spread_std=self.delay_spread_std,
            delay_scaling=self.delay_scaling,
            num_clusters=self.num_clusters,
            seed=1234
        )

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertEqual(self.delay_spread_mean, self.channel.delay_spread_mean)
        self.assertEqual(self.delay_spread_std, self.channel.delay_spread_std)
        self.assertEqual(self.num_clusters, self.channel.num_clusters)

        self.assertEqual(self.delay_scaling, self.channel.delay_scaling)

    def test_num_clusters_setget(self) -> None:
        """Number of clusters property getter should return setter argument"""

        num_clusters = 123
        self.channel.num_clusters = num_clusters

        self.assertEqual(num_clusters, self.channel.num_clusters)

    def test_num_clusters_validation(self) -> None:
        """Number of clusters property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.channel.num_clusters = -1

        with self.assertRaises(ValueError):
            self.channel.num_clusters = 0

    def test_delay_spread_mean_setget(self) -> None:
        """Delay spread mean property getter should return setter argument"""

        delay_spread = 123
        self.channel.delay_spread_mean = delay_spread

        self.assertEqual(delay_spread, self.channel.delay_spread_mean)

    def test_delay_spread_std_setget(self) -> None:
        """Delay spread mean property getter should return setter argument"""

        std = 123
        self.channel.delay_spread_std = std

        self.assertEqual(std, self.channel.delay_spread_std)

    def test_delay_scaling_setget(self) -> None:
        """Delay scaling property getter should return setter argument"""

        delay_scaling = 123
        self.channel.delay_scaling = delay_scaling

        self.assertEqual(delay_scaling, self.channel.delay_scaling)

    def test_delay_scaling_validation(self) -> None:
        """Delay scaling property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.channel.delay_scaling = -1.

        with self.assertRaises(ValueError):
            self.channel.delay_scaling = 0.5

        try:

            self.channel.delay_scaling = 1.

        except ValueError:
            self.fail()

    def test_rice_factor_mean_setget(self) -> None:
        """Rice factor mean property getter should return setter argument"""

        rice_factor_mean = 123
        self.channel.rice_factor_mean = rice_factor_mean

        self.assertEqual(rice_factor_mean, self.channel.rice_factor_mean)

    def test_rice_factor_mean_validation(self) -> None:
        """Rice factor mean property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.channel.rice_factor_mean = -1.

        try:

            self.channel.rice_factor_mean = 0.

        except ValueError:
            self.fail()
            
    def test_rice_factor_std_setget(self) -> None:
        """Rice factor standard deviation property getter should return setter argument"""

        rice_factor_std = 123
        self.channel.rice_factor_std = rice_factor_std

        self.assertEqual(rice_factor_std, self.channel.rice_factor_std)

    def test_rice_factor_std_validation(self) -> None:
        """Rice factor standard deviation property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.channel.rice_factor_std = -1.

        try:

            self.channel.rice_factor_std = 0.

        except ValueError:
            self.fail()
            
    def test_cluster_shadowing_std_setget(self) -> None:
        """Cluster shadowing standard deviation property getter should return setter argument"""

        cluster_shadowing_std = 123
        self.channel.cluster_shadowing_std = cluster_shadowing_std

        self.assertEqual(cluster_shadowing_std, self.channel.cluster_shadowing_std)

    def test_cluster_shadowing_std_validation(self) -> None:
        """Cluster shadowing standard deviation property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.channel.cluster_shadowing_std = -1.

        try:

            self.channel.cluster_shadowing_std = 0.

        except ValueError:
            self.fail()

    def test_realization_nlos(self):
        """Realizing a non-line-of-sight channel should properly configure the resulting realization"""

        self.channel.line_of_sight = False

        realization = self.channel.realize()
        self.assertFalse(realization.line_of_sight)

    def test_realization_los(self):
        """Realizing a line-of-sight channel should properly configure the resulting realization"""

        self.channel.line_of_sight = True
        
        realization = self.channel.realize()
        self.assertTrue(realization.line_of_sight)

    def test_doppler_shift(self) -> None:
        """A signal being propagated over the channel should be frequency shifted according to the doppler effect"""

        self.channel.num_clusters = 1
        self.beta_device.velocity = np.array([10., 0., 0.])

        sampling_rate = 1e3
        num_samples = 400
        signal = np.outer(np.ones(4, dtype=complex), np.exp(2j * pi * .2 * np.arange(num_samples)))

        radial_velocity = (self.alpha_device.velocity - self.beta_device.velocity) @ Direction.From_Cartesian(self.beta_device.global_position - self.alpha_device.global_position, True).view(np.ndarray)
        expected_doppler_shift = radial_velocity * self.alpha_device.carrier_frequency / speed_of_light
        frequency_resolution = sampling_rate / num_samples

        shifted_propagation = self.channel.propagate(Signal(signal, sampling_rate, self.carrier_frequency))

        input_freq = np.abs(np.fft.fft(signal[0, :]))
        output_freq = np.abs(np.fft.fft(shifted_propagation.signal.samples[0, :].flatten()))

        self.assertAlmostEqual(expected_doppler_shift, (np.argmax(output_freq) - np.argmax(input_freq)) * frequency_resolution, delta=1*frequency_resolution)

    def test_time_of_flight_delay_normalization(self) -> None:
        """Time of flight delay normalization should result in an impulse response padded by the appropriate number of samples"""

        sampling_rate = 1e9
        
        self.channel.delay_normalization = DelayNormalization.ZERO
        self.channel.seed = 1
        zero_delay_realization = self.channel.realize()

        self.channel.delay_normalization = DelayNormalization.TOF
        self.channel.seed = 1
        tof_delay_realization = self.channel.realize()
        
        signal = Signal(np.ones((self.alpha_device.antennas.num_transmit_antennas, 1), dtype=np.complex_), sampling_rate, self.carrier_frequency)
        zero_propagation = zero_delay_realization.propagate(signal)
        tof_propagation = tof_delay_realization.propagate(signal)

        expected_num_tof_samples = int(np.linalg.norm(self.alpha_device.position - self.beta_device.position, 2) / speed_of_light * sampling_rate)
        num_tof_samples = tof_propagation.signal.num_samples - zero_propagation.signal.num_samples
        self.assertEqual(expected_num_tof_samples, num_tof_samples)

    def test_realization_position_validation(self) -> None:
        """Realization should raise RuntimeError if connected devices collide"""

        self.alpha_device.position = np.zeros(3)
        self.beta_device.position = np.zeros(3)
        
        with self.assertRaises(RuntimeError):
            self.channel.realize()

    def test_pseudo_randomness(self) -> None:
        """Setting the random seed should result in identical impulse responses"""
        
        # Generate first impulse response
        self.channel.seed = 1
        first_realization = self.channel.realize()
        first_number = self.channel._rng.normal()

        # Generate second impulse response with identical initial seed
        self.channel.seed = 1
        second_realization = self.channel.realize()
        second_number = self.channel._rng.normal()

        # Both should be identical
        self.assertEqual(first_number, second_number)
        assert_array_equal(first_realization.azimuth_of_arrival, second_realization.azimuth_of_arrival)
        assert_array_equal(first_realization.azimuth_of_departure, second_realization.azimuth_of_departure)

    def test_spatial_properties(self) -> None:
        """Direction of arrival estimation should result in the correct angle estimation of impinging devices"""

        self.channel.num_clusters = 1

        self.alpha_device.antennas = UniformArray(IdealAntenna, .5 * speed_of_light / self.carrier_frequency, (1,))
        self.alpha_device.orientation = np.zeros(3, dtype=float)
        self.beta_device.position = np.zeros(3, dtype=float)
        self.beta_device.orientation = np.zeros(3, dtype=float)
        self.beta_device.antennas = UniformArray(IdealAntenna, .5 * speed_of_light / self.carrier_frequency, (8,8))

        angle_candidates = [(0, 0),
                            (.25 * pi, .5 * pi),
                            (.5 * pi, 0),
                            (.5 * pi, .25 * pi),
                            (.5 * pi, .5 * pi),
                            ]
        range = 1e3
      
        steering_codebook = np.empty((8**2, len(angle_candidates)), dtype=complex)
        for a, (zenith, azimuth) in enumerate(angle_candidates):
            steering_codebook[:, a] = self.beta_device.antennas.spherical_phase_response(self.carrier_frequency, azimuth, zenith)

        probing_signal = Signal(np.exp(2j * pi * .25 * np.arange(100)), sampling_rate=1e3, carrier_frequency=self.carrier_frequency)

        for a, (zenith, azimuth) in enumerate(angle_candidates):

            self.channel.seed = 1
            self.alpha_device.position = range * np.array([cos(azimuth) * sin(zenith),
                                                          sin(azimuth) * sin(zenith),
                                                          cos(zenith)], dtype=float)

            received_propagation = self.channel.propagate(probing_signal)

            beamformer = np.linalg.norm(steering_codebook.T.conj() @ received_propagation.signal.samples, 2, axis=1, keepdims=False)
            self.assertEqual(a, np.argmax(beamformer))

    def test_delay_spread_std_validation(self) -> None:
        """Delay spread standard deviation property setter should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.channel.delay_spread_std = -1.

        try:
            self.channel.delay_spread_std = 0.

        except ValueError:
            self.fail()

    def test_aod_spread_std_validation(self) -> None:
        """AOD spread standard deviation property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.channel.aod_spread_std = -1.

        try:
            self.channel.aod_spread_std = 0.

        except ValueError:
            self.fail()
            
    def test_aoa_spread_std_validation(self) -> None:
        """AOA spread standard deviation property setter should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.channel.aoa_spread_std = -1.

        try:
            self.channel.aoa_spread_std = 0.

        except ValueError:
            self.fail()
            
    def test_zoa_spread_std_validation(self) -> None:
        """ZOA spread standard deviation property setter should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.channel.zoa_spread_std = -1.

        try:
            self.channel.zoa_spread_std = 0.

        except ValueError:
            self.fail()
            
    def test_zod_spread_std_validation(self) -> None:
        """ZOD spread standard deviation property setter should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.channel.zod_spread_std = -1.

        try:
            self.channel.zod_spread_std = 0.

        except ValueError:
            self.fail()
            
    def test_cross_polarization_power_std_validation(self) -> None:
        """Cross polarization power standard deviation property setter should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.channel.cross_polarization_power_std = -1.

        try:
            self.channel.cross_polarization_power_std = 0.

        except ValueError:
            self.fail()
            
    def test_num_rays_validation(self) -> None:
        """Number of rays property setter should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.channel.num_rays = -1

        with self.assertRaises(ValueError):
            self.channel.num_rays = 0
            
    def test_cluster_delay_spread_validation(self) -> None:
        """Cluster delay spread property setter should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.channel.cluster_delay_spread = -1.

        try:
            self.channel.cluster_delay_spread = 0.

        except ValueError:
            self.fail()
            
    def test_cluster_aod_spread_validation(self) -> None:
        """Cluster AOD spread property setter should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.channel.cluster_aod_spread = -1.

        try:
            self.channel.cluster_aod_spread = 0.

        except ValueError:
            self.fail()
            
    def test_cluster_aoa_spread_validation(self) -> None:
        """Cluster AOA spread property setter should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.channel.cluster_aoa_spread = -1.

        try:
            self.channel.cluster_aoa_spread = 0.

        except ValueError:
            self.fail()
            
    def test_cluster_zoa_spread_validation(self) -> None:
        """Cluster ZOA spread property setter should raise ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            self.channel.cluster_zoa_spread = -1.

        try:
            self.channel.cluster_zoa_spread = 0.

        except ValueError:
            self.fail()

    def test_recall_realization(self) -> None:
        """Test realization recall"""

        file = File('test.h5', 'w', driver='core', backing_store=False)
        group = file.create_group('group')
        
        expected_realization = self.channel.realize()
        expected_realization.to_HDF(group)

        recalled_realization = self.channel.recall_realization(group)
        file.close()

        assert_array_equal(expected_realization.azimuth_of_arrival, recalled_realization.azimuth_of_arrival)
