# -*- coding: utf-8 -*-
"""Test Multipath Fading Channel Model"""

import unittest
from copy import deepcopy
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
import numpy.random as rand
import numpy.testing as npt
from numpy import exp
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import stats
from scipy.constants import pi

from hermespy.channel import MultipathFadingChannel, AntennaCorrelation, CustomAntennaCorrelation
from hermespy.core import Signal, UniformArray, IdealAntenna
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockAntennaCorrelation(AntennaCorrelation):
    """Mock antenna correlation implementation for test purposes"""

    @property
    def covariance(self) -> np.ndarray:

        return np.identity(2, dtype=complex)


class TestAntennaCorrelation(unittest.TestCase):
    """Test antenna correlation model base class"""

    def setUp(self) -> None:

        self.correlation = MockAntennaCorrelation()
    
    def test_channel_setget(self) -> None:
        """Channel property getter should return setter argument"""

        channel = Mock()
        self.correlation.channel = channel
        
        self.assertIs(channel, self.correlation.channel)

    def test_device_setget(self) -> None:
        """Device property getter should return setter argument"""

        device = Mock()
        self.correlation.device = device

        self.assertIs(device, self.correlation.device)


class TestCustomAntennaCorrelation(unittest.TestCase):
    """Test custom antenna correlation model"""

    def setUp(self) -> None:

        self.device = Mock()
        self.device.num_antennas = 2
        self.covariance = np.identity(2, dtype=complex)

        self.correlation = CustomAntennaCorrelation(covariance=self.covariance)
        self.correlation.device = self.device

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        assert_array_equal(self.covariance, self.correlation.covariance)

    def test_covariance_setget(self) -> None:
        """Covariance property getter should return setter argument"""

        covariance = 5 * np.identity(2, dtype=complex)
        self.correlation.covariance = covariance

        assert_array_equal(covariance, self.correlation.covariance)

    def test_covariance_set_validation(self) -> None:
        """Covariance property setter should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.correlation.covariance = np.diag(np.array([1, 2, 3]), 2)

        with self.assertRaises(ValueError):
            self.correlation.covariance = np.zeros((2, 2), dtype=complex)

    def test_covariance_get_validation(self) -> None:
        """Covariance property should raise a RuntimeError if the number of device antennas does not match"""

        self.device.num_antennas = 4

        with self.assertRaises(RuntimeError):
            _ = self.correlation.covariance


class TestMultipathFadingChannel(unittest.TestCase):
    """Test the multipath fading channel implementation"""

    def setUp(self) -> None:

        self.active = True
        self.gain = 1.0

        self.delays = np.zeros(1, dtype=float)
        self.power_profile = np.ones(1, dtype=float)
        self.rice_factors = np.zeros(1, dtype=float)

        self.sampling_rate = 1e6
        self.transmit_frequency = pi * self.sampling_rate
        self.num_sinusoids = 40
        self.doppler_frequency = 0.0
        self.los_doppler_frequency = 0.

        self.transmitter = SimulatedDevice(sampling_rate=self.sampling_rate)
        self.receiver = SimulatedDevice(sampling_rate=self.sampling_rate)
        self.sync_offset_low = 1e-6
        self.sync_offset_high = 3e-6

        self.channel_params = {
            'delays': self.delays,
            'power_profile': self.power_profile,
            'rice_factors': self.rice_factors,
            'transmitter': self.transmitter,
            'receiver': self.receiver,
            'active': self.active,
            'gain': self.gain,
            'num_sinusoids': self.num_sinusoids,
            'los_angle': None,
            'doppler_frequency': self.doppler_frequency,
            'los_doppler_frequency': self.los_doppler_frequency,
            'sync_offset_low': self.sync_offset_low,
            'sync_offset_high': self.sync_offset_high,
            'seed': 42,
        }

        self.num_samples = 100

        self.min_number_samples = 1000
        self.max_number_samples = 500000
        self.max_doppler_frequency = 100
        self.max_number_paths = 20
        self.max_delay_in_samples = 30

    def test_init(self) -> None:
        """The object initialization should properly store all parameters"""

        channel = MultipathFadingChannel(**self.channel_params)

        self.assertIs(self.transmitter, channel.transmitter, "Unexpected transmitter parameter initialization")
        self.assertIs(self.receiver, channel.receiver, "Unexpected receiver parameter initialization")
        self.assertEqual(self.active, channel.active, "Unexpected active parameter initialization")
        self.assertEqual(self.gain, channel.gain, "Unexpected gain parameter initialization")
        self.assertEqual(self.num_sinusoids, channel.num_sinusoids)
        self.assertEqual(self.doppler_frequency, channel.doppler_frequency)
        self.assertEqual(self.sync_offset_low, channel.sync_offset_low)
        self.assertEqual(self.sync_offset_high, channel.sync_offset_high)

    def test_init_validation(self) -> None:
        """Object initialization should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params['delays'] = np.array([1, 2])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params['power_profile'] = np.array([1, 2])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params['rice_factors'] = np.array([1, 2])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params['delays'] = np.array([-1.0])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params['power_profile'] = np.array([-1.0])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params['rice_factors'] = np.array([-1.0])
            _ = MultipathFadingChannel(**params)
            
        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params['delays'] = np.ones((1, 2))
            _ = MultipathFadingChannel(**params)
        
        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params['delays'] = np.empty((0,))
            _ = MultipathFadingChannel(**params)

    def test_delays_get(self) -> None:
        """Delays getter should return init param"""

        channel = MultipathFadingChannel(**self.channel_params)
        np.testing.assert_array_almost_equal(self.delays, channel.delays)

    def test_power_profiles_get(self) -> None:
        """Power profiles getter should return init param"""

        channel = MultipathFadingChannel(**self.channel_params)
        np.testing.assert_array_almost_equal(self.power_profile, channel.power_profile)

    def test_rice_factors_get(self) -> None:
        """Rice factors getter should return init param"""

        channel = MultipathFadingChannel(**self.channel_params)
        np.testing.assert_array_almost_equal(self.rice_factors, channel.rice_factors)

    def test_doppler_frequency_setget(self) -> None:
        """Doppler frequency property getter should return setter argument"""

        channel = MultipathFadingChannel(**self.channel_params)

        doppler_frequency = 5
        channel.doppler_frequency = doppler_frequency

        self.assertEqual(doppler_frequency, channel.doppler_frequency)

    def test_los_doppler_frequency_setget(self) -> None:
        """Line-of-Sight Doppler frequency property getter should return setter argument,
        alternatively the global Doppler"""

        channel = MultipathFadingChannel(**self.channel_params)

        los_doppler_frequency = 5
        channel.los_doppler_frequency = los_doppler_frequency
        self.assertEqual(los_doppler_frequency, channel.los_doppler_frequency)

        doppler_frequency = 4
        channel.doppler_frequency = doppler_frequency
        channel.los_doppler_frequency = None
        self.assertEqual(doppler_frequency, channel.los_doppler_frequency)

    def test_max_delay_get(self) -> None:
        """Max delay property should return maximum of delays"""

        self.channel_params['delays'] = np.array([1, 2, 3])
        self.channel_params['power_profile'] = np.zeros(3)
        self.channel_params['rice_factors'] = np.ones(3)

        channel = MultipathFadingChannel(**self.channel_params)
        self.assertEqual(max(self.channel_params['delays']), channel.max_delay)

    def test_num_sequences_get(self) -> None:
        """Number of fading sequences property should return core parameter lengths"""

        self.channel_params['delays'] = np.array([1, 2, 3])
        self.channel_params['power_profile'] = np.zeros(3)
        self.channel_params['rice_factors'] = np.ones(3)

        channel = MultipathFadingChannel(**self.channel_params)
        self.assertEqual(len(self.channel_params['delays']), channel.num_resolvable_paths)

    def test_num_sinusoids_setget(self) -> None:
        """Number of sinusoids property getter should return setter argument"""

        channel = MultipathFadingChannel(**self.channel_params)

        num_sinusoids = 100
        channel.num_sinusoids = num_sinusoids

        self.assertEqual(num_sinusoids, channel.num_sinusoids)

    def test_num_sinusoids_validation(self) -> None:
        """Number of sinusoids property setter should raise ValueError on invalid arguments"""

        channel = MultipathFadingChannel(**self.channel_params)

        with self.assertRaises(ValueError):
            channel.num_sinusoids = -1

    def test_los_angle_setget(self) -> None:
        """Line of sight angle property getter should return setter argument"""

        channel = MultipathFadingChannel(**self.channel_params)

        los_angle = 15
        channel.los_angle = los_angle
        self.assertEqual(los_angle, channel.los_angle)

        channel.los_angle = None
        self.assertEqual(None, channel.los_angle)

    def test_realization_seed(self) -> None:
        """Re-setting the random rng seed should result in identical impulse responses"""

        channel = MultipathFadingChannel(**self.channel_params)

        channel.seed = 100
        first_draw = channel.realize(self.num_samples, self.sampling_rate)

        channel.seed = 100
        second_draw = channel.realize(self.num_samples, self.sampling_rate)

        assert_array_almost_equal(first_draw.state, second_draw.state)
        
    def test_realization_no_interpolation(self) -> None:
        """Test impulse response generation without interpolation"""
        
        self.channel_params['impulse_response_interpolation'] = False
        channel = MultipathFadingChannel(**self.channel_params)
        
        realization = channel.realize(self.num_samples, self.sampling_rate)
        self.assertEqual(self.num_samples, realization.num_samples)
        
    def test_realization_fixed_los_angle(self) -> None:
        """Test impulse response generation with a fixed line of sight angle"""
        
        self.channel_params['los_angle'] = 0.
        channel = MultipathFadingChannel(**self.channel_params)
        
        realization = channel.realize(self.num_samples, self.sampling_rate)
        self.assertEqual(self.num_samples, realization.num_samples)

    def test_propagation_siso_no_fading(self) -> None:
        """
        Test the propagation through a SISO multipath channel model without fading
        Check if the output sizes are consistent
        Check the output of a SISO multipath channel model without fading (K factor of Rice distribution = inf)
        """

        self.rice_factors[0] = float('inf')
        self.delays[0] = 10 / self.sampling_rate
        channel = MultipathFadingChannel(**self.channel_params)

        timestamps = np.arange(self.num_samples) / self.sampling_rate
        transmission = exp(1j * timestamps * self.transmit_frequency).reshape(1, self.num_samples)
        output, _, _ = channel.propagate(Signal(transmission, self.sampling_rate))

        self.assertEqual(10, output[0].num_samples - transmission.shape[1],
                         "Propagation impulse response has unexpected length")

    def test_propagation_fading(self) -> None:
        """Test the propagation through a SISO multipath channel with fading"""

        test_delays = np.array([1., 2., 3., 4.], dtype=float) / self.sampling_rate

        reference_params = self.channel_params.copy()
        delayed_params = self.channel_params.copy()

        reference_params['delays'] = np.array([0.0])
        reference_channel = MultipathFadingChannel(**reference_params)

        timestamps = np.arange(self.num_samples) / self.sampling_rate
        transmit_samples = np.exp(2j * pi * timestamps * self.transmit_frequency).reshape((1, self.num_samples))
        transmit_signal = Signal(transmit_samples, self.sampling_rate)

        for d, delay in enumerate(test_delays):

            delayed_params['delays'] = reference_params['delays'] + delay
            delayed_channel = MultipathFadingChannel(**delayed_params)

            reference_channel.seed = d
            reference_propagation, _, _ = reference_channel.propagate(transmit_signal)

            delayed_channel.seed = d
            delayed_propagation, _, _ = delayed_channel.propagate(transmit_signal)

            zero_pads = int(self.sampling_rate * float(delay))
            npt.assert_array_almost_equal(reference_propagation[0].samples, delayed_propagation[0].samples[:, zero_pads:])

    def test_rayleigh(self) -> None:
        """
        Test if the amplitude of a path is Rayleigh distributed.
        Verify that both real and imaginary components are zero-mean normal random variables with the right variance and
        uncorrelated.
        """
        
        max_number_of_drops = 200
        samples_per_drop = 1000
        self.doppler_frequency = 200

        self.channel_params['delays'][0] = 0.
        self.channel_params['power_profile'][0] = 1.
        self.channel_params['rice_factors'][0] = 0.
        self.channel_params['doppler_frequency'] = self.doppler_frequency

        channel = MultipathFadingChannel(**self.channel_params)

        samples = np.array([])

        is_rayleigh = False
        alpha = .05
        max_corr = 0.05

        number_of_drops = 0
        while not is_rayleigh and number_of_drops < max_number_of_drops:

            realization = channel.realize(samples_per_drop, self.doppler_frequency)
            samples = np.append(samples, realization.state.ravel())

            _, p_real = stats.kstest(np.real(samples), 'norm', args=(0, 1 / np.sqrt(2)))
            _, p_imag = stats.kstest(np.imag(samples), 'norm', args=(0, 1 / np.sqrt(2)))

            corr = np.corrcoef(np.real(samples), np.imag(samples))
            corr = corr[0, 1]

            if p_real > alpha and p_imag > alpha and abs(corr) < max_corr:
                is_rayleigh = True

            number_of_drops += 1

        self.assertTrue(is_rayleigh)

    def test_rice(self) -> None:
        """
        Test if the amplitude of a path is Ricean distributed.
        """
        
        max_number_of_drops = 100
        doppler_frequency = 200
        samples_per_drop = 1000

        self.channel_params['delays'][0] = 0.
        self.channel_params['power_profile'][0] = 1.
        self.channel_params['rice_factors'][0] = 1.
        self.channel_params['doppler_frequency'] = doppler_frequency

        channel = MultipathFadingChannel(**self.channel_params)
        samples = np.array([])

        is_rice = False
        alpha = .05

        number_of_drops = 0
        while not is_rice and number_of_drops < max_number_of_drops:

            realization = channel.realize(samples_per_drop, doppler_frequency)
            samples = np.append(samples, realization.state.ravel())

            dummy, p_real = stats.kstest(np.abs(samples), 'rice', args=(np.sqrt(2), 0, 1 / 2))

            if p_real > alpha:
                is_rice = True

            number_of_drops += 1

        self.assertTrue(is_rice)

    def test_power_delay_profile(self) -> None:
        """
        Test if the resulting power delay profile matches with the one specified in the parameters.
        Test also an interpolated channel (should have the same rms delay spread)
        """
        
        max_number_of_drops = 100
        samples_per_drop = 1000
        max_delay_spread_dev = 12 / self.sampling_rate  # Check what is acceptable here

        self.doppler_frequency = 50
        self.channel_params['doppler_frequency'] = self.doppler_frequency
        self.channel_params['delays'] = np.zeros(5)
        self.channel_params['power_profile'] = np.ones(5)
        self.channel_params['rice_factors'] = np.zeros(5)
        self.channel_params['delays'] = np.array([0, 3, 6, 7, 8]) / self.sampling_rate

        mean_delay = np.mean(self.channel_params['delays'])
        config_delay_spread = np.mean((self.channel_params['delays'] - mean_delay) ** 2)

        delayed_channel = MultipathFadingChannel(**self.channel_params)

        for s in range(max_number_of_drops):

            delayed_channel.random_generator = np.random.default_rng(s+10)
            delayed_realization = delayed_channel.realize(samples_per_drop, self.sampling_rate)

            delayed_time = np.arange(delayed_realization.state.shape[-1]) / self.sampling_rate
            delay_diff = (delayed_time - np.mean(delayed_time)) ** 2
            delayed_power = delayed_realization.state.real ** 2 + delayed_realization.state.imag ** 2
            delay_spread = np.sqrt(np.mean(delayed_power @ delay_diff) / np.mean(delayed_power))

            spread_delta = abs(config_delay_spread - delay_spread)
            self.assertTrue(spread_delta < max_delay_spread_dev,
                            msg=f"{spread_delta} larger than max {max_delay_spread_dev}")

    def test_channel_gain(self) -> None:
        """
        Test if channel gain is applied correctly on both propagation and channel impulse response
        """
        
        gain = 10

        doppler_frequency = 200
        signal_length = 1000

        self.channel_params['delays'][0] = 0.
        self.channel_params['power_profile'][0] = 1.
        self.channel_params['rice_factors'][0] = 0.
        self.channel_params['doppler_frequency'] = doppler_frequency

        channel_no_gain = MultipathFadingChannel(**self.channel_params)

        self.channel_params['gain'] = gain
        channel_gain = MultipathFadingChannel(**self.channel_params)

        frame_size = (1, signal_length)
        tx_samples = rand.normal(0, 1, frame_size) + 1j * rand.normal(0, 1, frame_size)
        tx_signal = Signal(tx_samples, self.sampling_rate)

        channel_no_gain.random_generator = np.random.default_rng(42)  # Reset random number rng
        signal_out_no_gain, _, _ = channel_no_gain.propagate(tx_signal)

        channel_gain.random_generator = np.random.default_rng(42)   # Reset random number rng
        signal_out_gain, _, _ = channel_gain.propagate(tx_signal)

        assert_array_almost_equal(signal_out_gain[0].samples, signal_out_no_gain[0].samples * np.sqrt(gain))

        timestamps = np.array([0, 100, 500]) / self.sampling_rate

        channel_no_gain.random_generator = np.random.default_rng(50)  # Reset random number rng
        realization_no_gain = channel_no_gain.realize(len(timestamps), self.sampling_rate)

        channel_gain.random_generator = np.random.default_rng(50)   # Reset random number rng
        realization_gain = channel_gain.realize(len(timestamps), self.sampling_rate)

        npt.assert_array_almost_equal(realization_gain.state, realization_no_gain.state * np.sqrt(gain))

    def test_antenna_correlation(self) -> None:
        """Test channel simulation with antenna correlation modeling"""
        
        self.transmitter.antennas = UniformArray(IdealAntenna, 1e-2, (2, 1, 1))
        self.receiver.antennas = UniformArray(IdealAntenna, 1e-2, (2, 1, 1))
        
        uncorrelated_channel = MultipathFadingChannel(**self.channel_params)
        
        self.channel_params['alpha_correlation'] = MockAntennaCorrelation()
        self.channel_params['beta_correlation'] = MockAntennaCorrelation()
        
        correlated_channel = MultipathFadingChannel(**self.channel_params)
        
        uncorrelated_realization = uncorrelated_channel.realize(self.num_samples, self.sampling_rate)
        correlated_realization = correlated_channel.realize(self.num_samples, self.sampling_rate)

        # Since the correlation mock is just an identity, both channel states should be identical
        assert_array_almost_equal(uncorrelated_realization.state, correlated_realization.state)
        
    def test_alpha_correlation_setget(self) -> None:
        """Alpha correlation property getter should return setter argument"""
        
        channel = MultipathFadingChannel(**self.channel_params)
        expected_correlation = Mock()
        
        channel.alpha_correlation = expected_correlation
        
        self.assertIs(expected_correlation, channel.alpha_correlation)
        self.assertIs(self.transmitter, channel.alpha_correlation.device)
        
    def test_beta_correlation_setget(self) -> None:
        """Beta correlation property getter should return setter argument"""
        
        channel = MultipathFadingChannel(**self.channel_params)
        expected_correlation = Mock()
        
        channel.beta_correlation = expected_correlation
        
        self.assertIs(expected_correlation, channel.beta_correlation)
        self.assertIs(self.receiver, channel.beta_correlation.device)
        
    def test_transmitter_setget(self) -> None:
        """Setting the transmitter property should update the correlation configuration"""
        
        channel = MultipathFadingChannel(**self.channel_params)
        channel.alpha_correlation = Mock()
        expected_device = Mock()
        
        channel.transmitter = expected_device
        
        self.assertIs(expected_device, channel.transmitter)
        self.assertIs(expected_device, channel.alpha_correlation.device)
        
    def test_receiver_setget(self) -> None:
        """Setting the receiver property should update the correlation configuration"""
        
        channel = MultipathFadingChannel(**self.channel_params)
        channel.beta_correlation = Mock()
        expected_device = Mock()
        
        channel.receiver = expected_device
        
        self.assertIs(expected_device, channel.receiver)
        self.assertIs(expected_device, channel.beta_correlation.device)

    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.channel.multipath_fading_channel.MultipathFadingChannel.transmitter', new=PropertyMock) as transmitter, \
             patch('hermespy.channel.multipath_fading_channel.MultipathFadingChannel.receiver', new=PropertyMock) as receiver:

            transmitter.return_value = self.transmitter
            receiver.return_value = self.receiver

            test_yaml_roundtrip_serialization(self, MultipathFadingChannel(**self.channel_params), {'num_outputs', 'num_inputs'})
