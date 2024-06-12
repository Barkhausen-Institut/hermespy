# -*- coding: utf-8 -*-
"""Test Multipath Fading Channel Model"""

import unittest
from copy import deepcopy
from itertools import product
from unittest.mock import Mock

import numpy as np
import numpy.random as rand
from h5py import File
from numpy import exp
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import stats
from scipy.constants import pi

from hermespy.channel import MultipathFadingChannel, AntennaCorrelation, CustomAntennaCorrelation, TDL, Exponential, Cost259, StandardAntennaCorrelation, CorrelationType, Cost259Type, TDLType
from hermespy.channel.channel import LinkState
from hermespy.channel.fading.fading import MultipathFadingSample
from hermespy.core import AntennaMode, Signal, DenseSignal, Transformation
from hermespy.simulation import StaticTrajectory, SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization
from unit_tests.utils import SimulationTestContext
from unit_tests.utils import assert_signals_equal

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
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
        self.device = SimulatedDevice()
        self.device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1e-2, (2, 1, 1))
        self.covariance = np.identity(2, dtype=complex)

        self.correlation = CustomAntennaCorrelation(covariance=self.covariance)

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

        self.device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1e-2, (4, 1, 1))

        with self.assertRaises(ValueError):
            _ = self.correlation.sample_covariance(self.device.state(0).antennas, AntennaMode.TX)


class TestMultipathFadingSample(unittest.TestCase):
    """Test the multipath fading channel realization"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.sampling_rate = 1e9

        self.tx_device = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-2, (2, 1, 1)), sampling_rate=self.sampling_rate)
        self.rx_device = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-2, (2, 1, 1)), sampling_rate=self.sampling_rate)

        self.gain = 0.987
        self.num_paths = 10
        self.num_sinusoids = 5
        self.power_profile = self.rng.uniform(0, 1, size=(self.num_paths))
        self.delays = self.rng.uniform(0, 100 / self.sampling_rate, size=(self.num_paths))
        self.los_angles = self.rng.uniform(0, 2 * pi, size=(self.num_paths))
        self.nlos_angles = self.rng.uniform(0, 2 * pi, size=(self.num_paths, self.num_sinusoids))
        self.los_phases = self.rng.uniform(0, 2 * pi, size=(self.num_paths))
        self.nlos_phases = self.rng.uniform(0, 2 * pi, size=(self.num_paths, self.num_sinusoids))
        self.los_gains = self.rng.uniform(0, 1, size=(self.num_paths))
        self.nlos_gains = self.rng.uniform(0, 1, size=(self.num_paths))
        self.los_doppler = self.rng.uniform(0, 50 * self.sampling_rate)
        self.nlos_doppler = self.rng.uniform(0, 50 * self.sampling_rate)

        self.sample = MultipathFadingSample(
            self.power_profile,
            self.delays,
            self.los_angles,
            self.nlos_angles,
            self.los_phases,
            self.nlos_phases,
            self.los_gains,
            self.nlos_gains,
            self.los_doppler,
            self.nlos_doppler,
            np.eye(2),
            1.1,
            LinkState(
                self.tx_device.state(0.0),
                self.rx_device.state(0.0),
                self.tx_device.carrier_frequency,
                self.tx_device.sampling_rate,
                0.0,
            )
        )

    def test_propagate_state(self) -> None:
        """Propagation should result in a signal with the correct number of samples"""

        num_samples = 100
        signal = Signal.Create(self.rng.normal(0, 1, size=(2, num_samples)) + 1j * self.rng.normal(0, 1, size=(2, num_samples)), self.sampling_rate)

        signal_propagation = self.sample.propagate(signal)
        state_propagation = self.sample.state(signal.num_samples, 1 + signal_propagation.num_samples - signal.num_samples).propagate(signal)

        assert_signals_equal(self, signal_propagation, state_propagation)

    def test_plot_power_delay(self) -> None:
        """Plotting power delay profile should not raise any errors"""

        with SimulationTestContext(patch_plot=True):
            
            figure, axes = self.sample.plot_power_delay()
            axes[0, 0].stem.assert_called()
            
            axes[0, 0].stem.reset_mock()
            figure, axes = self.sample.plot_power_delay(axes=axes)
            axes[0, 0].get_figure.assert_called()
            axes[0, 0].stem.assert_called()


class TestMultipathFadingChannel(unittest.TestCase):
    """Test the multipath fading channel implementation"""

    def setUp(self) -> None:
        self.gain = 0.9876

        self.delays = np.zeros(1, dtype=float)
        self.power_profile = np.ones(1, dtype=float)
        self.rice_factors = np.zeros(1, dtype=float)

        self.sampling_rate = 1e6
        self.transmit_frequency = pi * self.sampling_rate
        self.num_sinusoids = 40
        self.doppler_frequency = 0.0
        self.los_doppler_frequency = 0.0

        self.alpha_device = SimulatedDevice(sampling_rate=self.sampling_rate)
        self.beta_device = SimulatedDevice(sampling_rate=self.sampling_rate)

        self.channel_params = {"gain": self.gain, "delays": self.delays, "power_profile": self.power_profile, "rice_factors": self.rice_factors, "num_sinusoids": self.num_sinusoids, "los_angle": None, "doppler_frequency": self.doppler_frequency, "los_doppler_frequency": self.los_doppler_frequency, "seed": 42}

        self.num_samples = 100

        self.min_number_samples = 1000
        self.max_number_samples = 500000
        self.max_doppler_frequency = 100
        self.max_number_paths = 20
        self.max_delay_in_samples = 30

    def test_init(self) -> None:
        """The object initialization should properly store all parameters"""

        channel = MultipathFadingChannel(**self.channel_params)

        self.assertEqual(self.gain, channel.gain, "Unexpected gain parameter initialization")
        self.assertEqual(self.num_sinusoids, channel.num_sinusoids)
        self.assertEqual(self.doppler_frequency, channel.doppler_frequency)

    def test_init_validation(self) -> None:
        """Object initialization should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params["delays"] = np.array([1, 2])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params["power_profile"] = np.array([1, 2])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params["rice_factors"] = np.array([1, 2])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params["delays"] = np.array([-1.0])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params["power_profile"] = np.array([-1.0])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params["rice_factors"] = np.array([-1.0])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params["delays"] = np.ones((1, 2))
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = deepcopy(self.channel_params)
            params["delays"] = np.empty((0,))
            _ = MultipathFadingChannel(**params)
            
    def test_correlation_distance_setget(self) -> None:
        """Correlation distance property getter should return setter argument"""

        channel = MultipathFadingChannel(**self.channel_params)

        correlation_distance = 5
        channel.correlation_distance = correlation_distance

        self.assertEqual(correlation_distance, channel.correlation_distance)
        
    def test_correlation_distance_validation(self) -> None:
        """Correlation distance property setter should raise ValueError on invalid arguments"""

        channel = MultipathFadingChannel(**self.channel_params)

        with self.assertRaises(ValueError):
            channel.correlation_distance = -1

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

        self.channel_params["delays"] = np.array([1, 2, 3])
        self.channel_params["power_profile"] = np.zeros(3)
        self.channel_params["rice_factors"] = np.ones(3)

        channel = MultipathFadingChannel(**self.channel_params)
        self.assertEqual(max(self.channel_params["delays"]), channel.max_delay)

    def test_num_sequences_get(self) -> None:
        """Number of fading sequences property should return core parameter lengths"""

        self.channel_params["delays"] = np.array([1, 2, 3])
        self.channel_params["power_profile"] = np.zeros(3)
        self.channel_params["rice_factors"] = np.ones(3)

        channel = MultipathFadingChannel(**self.channel_params)
        self.assertEqual(len(self.channel_params["delays"]), channel.num_resolvable_paths)

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
        first_draw = channel.realize().sample(self.alpha_device, self.beta_device)

        channel.seed = 100
        second_draw = channel.realize().sample(self.alpha_device, self.beta_device)

        assert_array_almost_equal(first_draw.nlos_angles, second_draw.nlos_angles)

    def test_propagation_siso_no_fading(self) -> None:
        """
        Test the propagation through a SISO multipath channel model without fading
        Check if the output sizes are consistent
        Check the output of a SISO multipath channel model without fading (K factor of Rice distribution = inf)
        """

        self.rice_factors[0] = float("inf")
        self.delays[0] = 10 / self.sampling_rate
        channel = MultipathFadingChannel(**self.channel_params)

        timestamps = np.arange(self.num_samples) / self.sampling_rate
        transmission = exp(1j * timestamps * self.transmit_frequency).reshape(1, self.num_samples)
        propagation = channel.propagate(Signal.Create(transmission, self.sampling_rate), self.alpha_device, self.beta_device)

        self.assertEqual(10, propagation.num_samples - self.num_samples, "Propagation impulse response has unexpected length")

    def test_propagation_fading(self) -> None:
        """Test the propagation through a SISO multipath channel with fading"""

        test_delays = np.array([1.0, 2.0, 3.0, 4.0], dtype=float) / self.sampling_rate

        reference_params = self.channel_params.copy()
        delayed_params = self.channel_params.copy()

        reference_params["delays"] = np.array([0.0])
        reference_channel = MultipathFadingChannel(**reference_params)

        timestamps = np.arange(self.num_samples) / self.sampling_rate
        transmit_samples = np.exp(2j * pi * timestamps * self.transmit_frequency).reshape((1, self.num_samples))
        transmit_signal = Signal.Create(transmit_samples, self.sampling_rate)

        for d, delay in enumerate(test_delays):
            delayed_params["delays"] = reference_params["delays"] + delay
            delayed_channel = MultipathFadingChannel(**delayed_params)

            reference_channel.seed = d
            reference_propagation = reference_channel.propagate(transmit_signal, self.alpha_device, self.beta_device)

            delayed_channel.seed = d
            delayed_propagation = delayed_channel.propagate(transmit_signal, self.alpha_device, self.beta_device)

            zero_pads = int(self.sampling_rate * float(delay))
            assert_array_almost_equal(reference_propagation.getitem(), delayed_propagation.getitem((slice(None, None), slice(zero_pads, None))))

    def test_rayleigh(self) -> None:
        """
        Test if the amplitude of a path is Rayleigh distributed.
        Verify that both real and imaginary components are zero-mean normal random variables with the right variance and
        uncorrelated.
        """

        max_number_of_drops = 200
        samples_per_drop = 1
        self.doppler_frequency = 1e3

        self.channel_params["delays"][0] = 0.0
        self.channel_params["power_profile"][0] = 1.0
        self.channel_params["rice_factors"][0] = 0.0
        self.channel_params["doppler_frequency"] = self.doppler_frequency

        channel = MultipathFadingChannel(**self.channel_params)

        samples = np.array([])

        is_rayleigh = False
        alpha = 0.05
        max_corr = 0.05

        number_of_drops = 0
        while not is_rayleigh and number_of_drops < max_number_of_drops:
            sample = channel.realize().sample(self.alpha_device, self.beta_device)
            state = sample.state(samples_per_drop, 1)
            samples = np.append(samples, state.dense_state().ravel())

            _, p_real = stats.kstest(np.real(samples), "norm", args=(0, 1 / np.sqrt(2)))
            _, p_imag = stats.kstest(np.imag(samples), "norm", args=(0, 1 / np.sqrt(2)))

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
        doppler_frequency = 0.5 * self.sampling_rate
        samples_per_drop = 1

        self.channel_params["delays"][0] = 0.0
        self.channel_params["power_profile"][0] = 1.0
        self.channel_params["rice_factors"][0] = 1.0
        self.channel_params["doppler_frequency"] = doppler_frequency

        channel = MultipathFadingChannel(**self.channel_params)
        samples = np.array([])

        is_rice = False
        alpha = 0.05

        number_of_drops = 0
        while not is_rice and number_of_drops < max_number_of_drops:
            sample = channel.realize().sample(self.alpha_device, self.beta_device)
            state = sample.state(samples_per_drop, 1)
            samples = np.append(samples, state.dense_state().ravel())

            _, p_real = stats.kstest(np.abs(samples), "rice", args=(np.sqrt(2), 0, 1 / 2))

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
        self.channel_params["doppler_frequency"] = self.doppler_frequency
        self.channel_params["delays"] = np.zeros(5)
        self.channel_params["power_profile"] = np.ones(5)
        self.channel_params["rice_factors"] = np.zeros(5)
        self.channel_params["delays"] = np.array([0, 3, 6, 7, 8]) / self.sampling_rate

        mean_delay = np.mean(self.channel_params["delays"])
        config_delay_spread = np.mean((self.channel_params["delays"] - mean_delay) ** 2)

        delayed_channel = MultipathFadingChannel(**self.channel_params)

        for s in range(max_number_of_drops):
            delayed_channel._rng = np.random.default_rng(s + 10)

            realization = delayed_channel.realize()
            sample = realization.sample(self.alpha_device, self.beta_device)
            delayed_state = sample.state(samples_per_drop, 1).dense_state()

            delayed_time = np.arange(delayed_state.shape[-1]) / self.sampling_rate
            delay_diff = (delayed_time - np.mean(delayed_time)) ** 2
            delayed_power = delayed_state.real**2 + delayed_state.imag**2
            delay_spread = np.sqrt(np.mean(delayed_power @ delay_diff) / np.mean(delayed_power))

            spread_delta = abs(config_delay_spread - delay_spread)
            self.assertTrue(spread_delta < max_delay_spread_dev, msg=f"{spread_delta} larger than max {max_delay_spread_dev}")

    def test_channel_gain(self) -> None:
        """
        Test if channel gain is applied correctly on both propagation and channel impulse response
        """

        gain = 10

        doppler_frequency = 200
        signal_length = 1000

        self.channel_params["gain"] = 1.0
        self.channel_params["delays"][0] = 0.0
        self.channel_params["power_profile"][0] = 1.0
        self.channel_params["rice_factors"][0] = 0.0
        self.channel_params["doppler_frequency"] = doppler_frequency

        channel_no_gain = MultipathFadingChannel(**self.channel_params)

        self.channel_params["gain"] = gain
        channel_gain = MultipathFadingChannel(**self.channel_params)

        frame_size = (1, signal_length)
        tx_samples = rand.normal(0, 1, frame_size) + 1j * rand.normal(0, 1, frame_size)
        tx_signal = Signal.Create(tx_samples, self.sampling_rate)

        channel_no_gain._rng = np.random.default_rng(42)  # Reset random number rng
        propagation_no_gain = channel_no_gain.propagate(tx_signal, self.alpha_device, self.beta_device)

        channel_gain._rng = np.random.default_rng(42)  # Reset random number rng
        propagation_gain = channel_gain.propagate(tx_signal, self.alpha_device, self.beta_device)

        assert_array_almost_equal(propagation_no_gain.getitem() * gain ** .5, propagation_gain.getitem())

    def test_antenna_correlation(self) -> None:
        """Test channel simulation with antenna correlation modeling"""

        self.alpha_device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1e-2, (2, 1, 1))
        self.beta_device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1e-2, (2, 1, 1))

        uncorrelated_channel = MultipathFadingChannel(**self.channel_params)

        self.channel_params["antenna_correlation"] = MockAntennaCorrelation()
        correlated_channel = MultipathFadingChannel(**self.channel_params)

        uncorrelated_realization = uncorrelated_channel.realize()
        correlated_realization = correlated_channel.realize()
        
        uncorrelated_sample = uncorrelated_realization.sample(self.alpha_device, self.beta_device)
        correlated_sample = correlated_realization.sample(self.alpha_device, self.beta_device)
        
        uncorrelated_state = uncorrelated_sample.state(self.num_samples, 1).dense_state()
        correlated_state = correlated_sample.state(self.num_samples, 1).dense_state()

        # Since the correlation mock is just an identity, both channel states should be identical
        assert_array_almost_equal(uncorrelated_state, correlated_state)

    def test_alpha_correlation_setget(self) -> None:
        """Alpha correlation property getter should return setter argument"""

        channel = MultipathFadingChannel(**self.channel_params)
        expected_correlation = Mock()

        channel.alpha_correlation = expected_correlation

        self.assertIs(expected_correlation, channel.alpha_correlation)

    def test_beta_correlation_setget(self) -> None:
        """Beta correlation property getter should return setter argument"""

        channel = MultipathFadingChannel(**self.channel_params)
        expected_correlation = Mock()

        channel.beta_correlation = expected_correlation

        self.assertIs(expected_correlation, channel.beta_correlation)
        
    def test_realization_reciprocal_sample(self) -> None:
        """Test reciprocal channel realization"""

        channel = MultipathFadingChannel(**self.channel_params)

        realization = channel.realize()
        sample = realization.sample(self.alpha_device, self.beta_device)
        reciprocal_sample = realization.reciprocal_sample(sample, self.beta_device, self.alpha_device)

        assert_array_almost_equal(sample.nlos_angles, reciprocal_sample.nlos_angles)

    def test_recall_realization(self) -> None:
        """Test realization recall"""

        channel = MultipathFadingChannel(**self.channel_params)

        file = File("test.h5", "w", driver="core", backing_store=False)
        group = file.create_group("g")

        expected_realization = channel.realize()
        expected_realization.to_HDF(group)

        recalled_realization = channel.recall_realization(group)
        file.close()

        assert_array_almost_equal(expected_realization.gain, recalled_realization.gain)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, MultipathFadingChannel(**self.channel_params), {"num_outputs", "num_inputs"})


class TestStandardAntennaCorrelation(unittest.TestCase):
    """Test standard antenna correlation models"""

    def setUp(self) -> None:
        self.device = SimulatedDevice()
        self.num_antennas = [1, 2, 4]

        self.correlation = StandardAntennaCorrelation(CorrelationType.LOW)


    def test_correlation_setget(self) -> None:
        """Correlation type property getter should return setter argument"""

        self.correlation.correlation = CorrelationType.HIGH
        self.assertIs(CorrelationType.HIGH, self.correlation.correlation)

        self.correlation.correlation = CorrelationType.MEDIUM
        self.assertIs(CorrelationType.MEDIUM, self.correlation.correlation)

    def test_sample_covariance(self) -> None:
        """Test covariance matrix generation"""

        for correlation_type, num_antennas in product(CorrelationType, self.num_antennas):
            self.device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1e-2, (num_antennas, 1, 1))
            self.correlation.correlation = correlation_type

            covariance = self.correlation.sample_covariance(self.device, AntennaMode.TX)

            self.assertCountEqual([num_antennas, num_antennas], covariance.shape)
            self.assertTrue(np.allclose(covariance, covariance.T.conj()))  # Hermitian check

    def test_covariance_validation(self) -> None:
        """Covariance matrix generation should exceptions on invalid parameters"""

        with self.assertRaises(RuntimeError):
            self.device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1e-2, (5, 1, 1))
            _ = self.correlation.sample_covariance(self.device, AntennaMode.TX)


class TestCost259(unittest.TestCase):
    """Test the Cost256 template for the multipath fading channel model."""

    def setUp(self) -> None:
        self.sampling_rate = 10e6
        self.carrier_frequency = 2.4e8
        self.alpha_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, sampling_rate=self.sampling_rate, pose=StaticTrajectory(Transformation.From_Translation(np.array([100, 0, 0]))))
        self.beta_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, sampling_rate=self.sampling_rate, pose=StaticTrajectory(Transformation.From_Translation(np.array([100, 0, 0]))))
    
    def test_init(self) -> None:
        """Test the template initializations."""

        for model_type in Cost259Type:
            channel = Cost259(model_type=model_type)
            self.assertIs(model_type, channel.model_type)

    def test_init_validation(self) -> None:
        """Template initialization should raise ValueError on invalid model type."""

        with self.assertRaises(ValueError):
            _ = Cost259(100000)

        with self.assertRaises(ValueError):
            _ = Cost259(Cost259Type.HILLY, los_angle=0.0)

    def test_model_type(self) -> None:
        """The model type property should return"""

        for model_type in Cost259Type:
            channel = Cost259(model_type)
            self.assertEqual(model_type, channel.model_type)
            
    def test_expected_scale(self) -> None:
        """Test the expected amplitude scaling"""
        
        model = Cost259(Cost259Type.HILLY, doppler_frequency=10.0, seed=42)
        unit_energy_signal = DenseSignal(np.ones((self.alpha_device.num_transmit_antennas, 100)) / 10, self.sampling_rate, self.carrier_frequency)
        num_attempts = 1000
        
        cumulated_propagated_energy = np.zeros((self.beta_device.num_receive_antennas), dtype=np.float_)
        cumulated_expected_scale = 0.0
        for _ in range(num_attempts):
            realization = model.realize()
            sample = realization.sample(self.alpha_device, self.beta_device)
            propagated_signal = sample.propagate(unit_energy_signal)
            cumulated_propagated_energy += propagated_signal.energy
            cumulated_expected_scale += sample.expected_energy_scale
            
        mean_propagated_energy = cumulated_propagated_energy / num_attempts
        mean_expected_energy = (cumulated_expected_scale / num_attempts) ** 2
        
        self.assertAlmostEqual(mean_propagated_energy, mean_expected_energy, delta=1e-1)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, Cost259(Cost259Type.HILLY), {"num_outputs", "num_inputs"})


class Test5GTDL(unittest.TestCase):
    """Test the 5GTDL template for the multipath fading channel model."""

    def setUp(self) -> None:
        self.rms_delay = 1e-6
        self.sampling_rate = 10e6
        self.carrier_frequency = 2.4e8
        self.alpha_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, sampling_rate=self.sampling_rate, pose=StaticTrajectory(Transformation.From_Translation(np.array([100, 0, 0]))))
        self.beta_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, sampling_rate=self.sampling_rate, pose=StaticTrajectory(Transformation.From_Translation(np.array([100, 0, 0]))))
    

    def test_init(self) -> None:
        """Test the template initializations."""

        for model_type in TDLType:
            channel = TDL(model_type=model_type)
            self.assertIs(model_type, channel.model_type)

    def test_init_validation(self) -> None:
        """Template initialization should raise ValueError on invalid model type."""

        with self.assertRaises(ValueError):
            _ = TDL(model_type=100000)

        with self.assertRaises(ValueError):
            _ = TDL(rms_delay=-1.0)

        with self.assertRaises(ValueError):
            _ = TDL(model_type=TDLType.D, los_doppler_frequency=0.0)

        with self.assertRaises(ValueError):
            _ = TDL(model_type=TDLType.E, los_doppler_frequency=0.0)

    def test_model_type(self) -> None:
        """The model type property should return the proper model type."""

        for model_type in TDLType:
            channel = TDL(model_type=model_type)
            self.assertEqual(model_type, channel.model_type)
            
    def test_expected_scale(self) -> None:
        """Test the expected amplitude scaling"""
        
        model = TDL(model_type=TDLType.E, doppler_frequency=10.0, seed=42)
        unit_energy_signal = DenseSignal(np.ones((self.alpha_device.num_transmit_antennas, 100)) / 10, self.sampling_rate, self.carrier_frequency)
        num_attempts = 1000
        
        cumulated_propagated_energy = np.zeros((self.beta_device.num_receive_antennas), dtype=np.float_)
        cumulated_expected_scale = 0.0
        for _ in range(num_attempts):
            realization = model.realize()
            sample = realization.sample(self.alpha_device, self.beta_device)
            propagated_signal = sample.propagate(unit_energy_signal)
            cumulated_propagated_energy += propagated_signal.energy
            cumulated_expected_scale += sample.expected_energy_scale
            
        mean_propagated_energy = cumulated_propagated_energy / num_attempts
        mean_expected_energy = (cumulated_expected_scale / num_attempts) ** 2
        
        self.assertAlmostEqual(mean_propagated_energy, mean_expected_energy, delta=1e-1)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        channel = TDL(model_type=TDLType.B)
        test_yaml_roundtrip_serialization(self, channel, {"num_outputs", "num_inputs"})


class TestExponential(unittest.TestCase):
    """Test the exponential template for the multipath fading channel model."""

    def setUp(self) -> None:
        self.tap_interval = 1e-5
        self.rms_delay = 1e-8
        self.channel = Exponential(tap_interval=self.tap_interval, rms_delay=self.rms_delay)

        self.sampling_rate = 10e6
        self.carrier_frequency = 2.4e8
        self.alpha_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, sampling_rate=self.sampling_rate, pose=StaticTrajectory(Transformation.From_Translation(np.array([100, 0, 0]))))
        self.beta_device = SimulatedDevice(carrier_frequency=self.carrier_frequency, sampling_rate=self.sampling_rate, pose=StaticTrajectory(Transformation.From_Translation(np.array([100, 0, 0]))))

    def test_init(self) -> None:
        """Initialization arguments should be properly parsed."""
        pass

    def test_init_validation(self) -> None:
        """Object initialization should raise ValueErrors on negative tap intervals and rms delays."""

        with self.assertRaises(ValueError):
            _ = Exponential(0.0, 1.0)

        with self.assertRaises(ValueError):
            _ = Exponential(1.0, 0.0)
            
    def test_expected_scale(self) -> None:
        """Test the expected amplitude scaling"""
        
        unit_energy_signal = DenseSignal(np.ones((self.alpha_device.num_transmit_antennas, 100)) / 10, self.sampling_rate, self.carrier_frequency)
        num_attempts = 1000
        
        cumulated_propagated_energy = np.zeros((self.beta_device.num_receive_antennas), dtype=np.float_)
        cumulated_expected_scale = 0.0
        for _ in range(num_attempts):
            realization = self.channel.realize()
            sample = realization.sample(self.alpha_device, self.beta_device)
            propagated_signal = sample.propagate(unit_energy_signal)
            cumulated_propagated_energy += propagated_signal.energy
            cumulated_expected_scale += sample.expected_energy_scale
            
        mean_propagated_energy = cumulated_propagated_energy / num_attempts
        mean_expected_energy = (cumulated_expected_scale / num_attempts) ** 2
        
        self.assertAlmostEqual(mean_propagated_energy, mean_expected_energy, delta=1e-1)


    def test_serialization(self) -> None:
        """Test YAML serialization"""

        test_yaml_roundtrip_serialization(self, self.channel, {"num_outputs", "num_inputs"})
