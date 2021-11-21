# -*- coding: utf-8 -*-
"""Test Multipath Fading Channel Model."""

import copy
import unittest
from unittest.mock import Mock

import numpy as np
import numpy.random as rand
import numpy.testing as npt
import scipy
from numpy import exp
from scipy import stats
from scipy.constants import pi

from hermespy.channel import MultipathFadingChannel

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMultipathFadingChannel(unittest.TestCase):
    """Test the multipath fading channel implementation."""

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

        self.scenario = Mock()
        self.transmitter = Mock()
        self.receiver = Mock()
        self.scenario.sampling_rate = self.sampling_rate
        self.transmitter.sampling_rate = self.sampling_rate
        self.receiver.sampling_rate = self.sampling_rate
        self.transmitter.num_antennas = 1
        self.receiver.num_antennas = 1

        self.channel_params = {
            'delays': self.delays,
            'power_profile': self.power_profile,
            'rice_factors': self.rice_factors,
            'transmitter': self.transmitter,
            'receiver': self.receiver,
            'scenario': self.scenario,
            'active': self.active,
            'gain': self.gain,
            'num_sinusoids': self.num_sinusoids,
            'los_angle': None,
            'doppler_frequency': self.doppler_frequency,
            'random_generator': np.random.default_rng(0)
        }

        self.num_samples = 100

        self.min_number_samples = 1000
        self.max_number_samples = 500000
        self.max_doppler_frequency = 100
        self.max_number_paths = 20
        self.max_delay_in_samples = 30

    def test_init(self) -> None:
        """The object initialization should properly store all parameters."""

        channel = MultipathFadingChannel(**self.channel_params)

        self.assertIs(self.transmitter, channel.transmitter, "Unexpected transmitter parameter initialization")
        self.assertIs(self.receiver, channel.receiver, "Unexpected receiver parameter initialization")
        self.assertEqual(self.active, channel.active, "Unexpected active parameter initialization")
        self.assertEqual(self.gain, channel.gain, "Unexpected gain parameter initialization")
        self.assertEqual(self.num_sinusoids, channel.num_sinusoids)
        self.assertEqual(self.doppler_frequency, channel.doppler_frequency)

    def test_init_validation(self) -> None:
        """Object initialization should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            params = copy.deepcopy(self.channel_params)
            params['delays'] = np.array([1, 2])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = copy.deepcopy(self.channel_params)
            params['power_profile'] = np.array([1, 2])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = copy.deepcopy(self.channel_params)
            params['rice_factors'] = np.array([1, 2])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = copy.deepcopy(self.channel_params)
            params['delays'] = np.array([-1.0])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = copy.deepcopy(self.channel_params)
            params['power_profile'] = np.array([-1.0])
            _ = MultipathFadingChannel(**params)

        with self.assertRaises(ValueError):
            params = copy.deepcopy(self.channel_params)
            params['rice_factors'] = np.array([-1.0])
            _ = MultipathFadingChannel(**params)

    def test_delays_get(self) -> None:
        """Delays getter should return init param."""

        channel = MultipathFadingChannel(**self.channel_params)
        np.testing.assert_array_equal(self.delays, channel.delays)

    def test_power_profiles_get(self) -> None:
        """Power profiles getter should return init param."""

        channel = MultipathFadingChannel(**self.channel_params)
        np.testing.assert_array_equal(self.power_profile, channel.power_profile)

    def test_rice_factors_get(self) -> None:
        """Rice factors getter should return init param."""

        channel = MultipathFadingChannel(**self.channel_params)
        np.testing.assert_array_equal(self.rice_factors, channel.rice_factors)

    def test_doppler_frequency_setget(self) -> None:
        """Doppler frequency property getter should return setter argument."""

        channel = MultipathFadingChannel(**self.channel_params)

        doppler_frequency = 5
        channel.doppler_frequency = doppler_frequency

        self.assertEqual(doppler_frequency, channel.doppler_frequency)

    def test_los_doppler_frequency_setget(self) -> None:
        """Line-of-Sight Doppler frequency property getter should return setter argument,
        alternatively the global Doppler."""

        channel = MultipathFadingChannel(**self.channel_params)

        los_doppler_frequency = 5
        channel.los_doppler_frequency = los_doppler_frequency
        self.assertEqual(los_doppler_frequency, channel.los_doppler_frequency)

        doppler_frequency = 4
        channel.doppler_frequency = doppler_frequency
        channel.los_doppler_frequency = None
        self.assertEqual(doppler_frequency, channel.los_doppler_frequency)

    def test_transmit_precoding_setget(self) -> None:
        """Transmit precoding property getter should return setter argument."""

        channel = MultipathFadingChannel(**self.channel_params)

        precoding = np.identity(5)
        channel.transmit_precoding = precoding
        npt.assert_array_equal(precoding, channel.transmit_precoding)

        channel.transmit_precoding = None
        self.assertEqual(None, channel.transmit_precoding)

    def test_transmit_precoding_validation(self) -> None:
        """Doppler frequency setter should raise ValueError on invalid arguments."""

        channel = MultipathFadingChannel(**self.channel_params)

        with self.assertRaises(ValueError):
            precoding = np.array([1])
            channel.transmit_precoding = precoding

        with self.assertRaises(ValueError):
            precoding = np.array([[1, 0], [1, 1]])
            channel.transmit_precoding = precoding

        with self.assertRaises(ValueError):
            precoding = np.ones((2, 2))
            channel.transmit_precoding = precoding
            
    def test_receive_postcoding_setget(self) -> None:
        """Transmit postcoding property getter should return setter argument."""

        channel = MultipathFadingChannel(**self.channel_params)

        postcoding = np.identity(5)
        channel.receive_postcoding = postcoding
        npt.assert_array_equal(postcoding, channel.receive_postcoding)

        channel.receive_postcoding = None
        self.assertEqual(None, channel.receive_postcoding)

    def test_receive_postcoding_validation(self) -> None:
        """Doppler frequency setter should raise ValueError on invalid arguments."""

        channel = MultipathFadingChannel(**self.channel_params)

        with self.assertRaises(ValueError):
            postcoding = np.array([1])
            channel.receive_postcoding = postcoding

        with self.assertRaises(ValueError):
            postcoding = np.array([[1, 0], [1, 1]])
            channel.receive_postcoding = postcoding

        with self.assertRaises(ValueError):
            postcoding = np.ones((2, 2))
            channel.receive_postcoding = postcoding

    def test_max_delay_get(self) -> None:
        """Max delay property should return maximum of delays."""

        self.channel_params['delays'] = np.array([1, 2, 3])
        self.channel_params['power_profile'] = np.zeros(3)
        self.channel_params['rice_factors'] = np.ones(3)

        channel = MultipathFadingChannel(**self.channel_params)
        self.assertEqual(max(self.channel_params['delays']), channel.max_delay)

    def test_num_sequences_get(self) -> None:
        """Number of fading sequences property should return core parameter lengths."""

        self.channel_params['delays'] = np.array([1, 2, 3])
        self.channel_params['power_profile'] = np.zeros(3)
        self.channel_params['rice_factors'] = np.ones(3)

        channel = MultipathFadingChannel(**self.channel_params)
        self.assertEqual(len(self.channel_params['delays']), channel.num_resolvable_paths)

    def test_num_sinusoids_setget(self) -> None:
        """Number of sinusoids property getter should return setter argument."""

        channel = MultipathFadingChannel(**self.channel_params)

        num_sinusoids = 100
        channel.num_sinusoids = num_sinusoids

        self.assertEqual(num_sinusoids, channel.num_sinusoids)

    def test_num_sinusoids_validation(self) -> None:
        """Number of sinusoids property setter should raise ValueError on invalid arguments."""

        channel = MultipathFadingChannel(**self.channel_params)

        with self.assertRaises(ValueError):
            channel.num_sinusoids = -1

    def test_los_angle_setget(self) -> None:
        """Line of sight angle property getter should return setter argument."""

        channel = MultipathFadingChannel(**self.channel_params)

        los_angle = 15
        channel.los_angle = los_angle
        self.assertEqual(los_angle, channel.los_angle)

        channel.los_angle = None
        self.assertEqual(None, channel.los_angle)

    def test_impulse_response_seed(self) -> None:
        """Re-setting the random generator seed should result in identical impulse responses."""

        channel = MultipathFadingChannel(**self.channel_params)
        timestamps = np.arange(self.num_samples) / self.sampling_rate

        channel.random_generator = np.random.default_rng(100)
        first_draw = channel.impulse_response(timestamps)

        channel.random_generator = np.random.default_rng(100)
        second_draw = channel.impulse_response(timestamps)

        np.testing.assert_almost_equal(first_draw, second_draw)

    def test_antenna_correlation(self) -> None:
        """Test the antenna correlation."""

        NO_ANTENNAS = 2

        self.transmitter.num_antennas = NO_ANTENNAS
        self.receiver.num_antennas = NO_ANTENNAS

        transmit_precoding = scipy.linalg.sqrtm(np.identity(NO_ANTENNAS) * 4)
        receive_postcoding = scipy.linalg.sqrtm(np.identity(NO_ANTENNAS) * 8)
        self.channel_params['transmit_precoding'] = transmit_precoding
        self.channel_params['receive_postcoding'] = receive_postcoding

        correlated_channel = MultipathFadingChannel(**self.channel_params)

        self.channel_params.pop('transmit_precoding')
        self.channel_params.pop('receive_postcoding')
        uncorrelated_channel = MultipathFadingChannel(**self.channel_params)

        tx_signal = np.exp(2j * pi * np.arange(NO_ANTENNAS * self.num_samples) *
                           self.transmit_frequency / self.sampling_rate).reshape((NO_ANTENNAS, self.num_samples))

        correlated_channel.random_generator = np.random.default_rng(22)
        rx_signal_correlated_channel, correlated_channel_state = correlated_channel.propagate(tx_signal)

        uncorrelated_channel.random_generator = np.random.default_rng(22)
        rx_signal_uncorrelated_channel, uncorrelated_channel_state = uncorrelated_channel.propagate(transmit_precoding @ tx_signal)

        # Make sure both channels generated the same impulse response for propagation
        np.testing.assert_array_equal(correlated_channel_state.state, uncorrelated_channel_state.state)

        expected_rx_signal_correlated_channel = (receive_postcoding  @ rx_signal_uncorrelated_channel)
        np.testing.assert_array_almost_equal(expected_rx_signal_correlated_channel, rx_signal_correlated_channel)

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
        output, _ = channel.propagate(transmission)

        self.assertEqual(10, output.shape[1] - transmission.shape[1],
                         "Propagation impulse response has unexpected length")

    def test_propagation_fading(self) -> None:
        """
        Test the propagation through a SISO multipath channel with fading.
        """

        test_delays = np.array([1., 2., 3., 4.]) / self.sampling_rate

        reference_params = self.channel_params.copy()
        delayed_params = self.channel_params.copy()

        reference_params['delays'] = np.array([0.0])
        reference_channel = MultipathFadingChannel(**reference_params)

        timestamps = np.arange(self.num_samples) / self.sampling_rate
        transmit_signal = np.exp(2j * pi * timestamps * self.transmit_frequency).reshape((1, self.num_samples))

        for d in range(len(test_delays)):
            delayed_params['delays'] = reference_params['delays'] + test_delays[d]
            delayed_channel = MultipathFadingChannel(**delayed_params)

            reference_channel.random_generator = np.random.default_rng(d)
            reference_propagation, _ = reference_channel.propagate(transmit_signal)

            delayed_channel.random_generator = np.random.default_rng(d)
            delayed_propagation, _ = delayed_channel.propagate(transmit_signal)

            zero_pads = int(test_delays[d] * self.sampling_rate)
            npt.assert_array_almost_equal(reference_propagation, delayed_propagation[:, zero_pads:])

#    def test_propagation_fading_interpolation(self) -> None:
#        """Test the propagation when path delays do not match sampling instants.
#        The delayed signals are expected to be zero-padded and slightly phase shifted.
#        """
#
#        test_delays = np.array([.75, 1.25, 15]) / self.sampling_rate
#
#        reference_params = self.channel_params.copy()
#        delayed_params = self.channel_params.copy()
#
#        reference_params['delays'] = np.array([0]) / self.sampling_rate
#        reference_channel = MultipathFadingChannel(**reference_params)
#
#        timestamps = np.arange(self.num_samples) / self.sampling_rate
#        transmit_signal = np.exp(2j * pi * timestamps * self.transmit_frequency).reshape((1, self.num_samples))
#
#        for d, test_delay in enumerate(test_delays):
#
#            delayed_params['delays'] = reference_params['delays'] + test_delays[d]
#            delayed_channel = MultipathFadingChannel(**delayed_params)
#
#            reference_channel.random_generator = np.random.default_rng(d)
#            reference_propagation, reference_response = reference_channel.propagate(transmit_signal)
#
#            reference_channel.random_generator = np.random.default_rng(d)
#            delayed_propagation, delayed_response = delayed_channel.propagate(transmit_signal)
#
#            # Make sure both powers are comparable
#            reference_power = np.linalg.norm(reference_propagation)
#            delayed_power = np.linalg.norm(delayed_propagation)
#
#            self.assertAlmostEqual(reference_power, delayed_power, msg=f"Test delay #{d} power divergence")

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

        time_interval = 1 / self.doppler_frequency  # get uncorrelated samples
        timestamps = np.arange(samples_per_drop) * time_interval

        samples = np.array([])

        is_rayleigh = False
        alpha = .05
        max_corr = 0.05

        number_of_drops = 0
        while not is_rayleigh and number_of_drops < max_number_of_drops:

            channel_gains = channel.impulse_response(timestamps)
            samples = np.append(samples, channel_gains.ravel())

            _, p_real = stats.kstest(np.real(samples), 'norm', args=(0, 1 / np.sqrt(2)))
            _, p_imag = stats.kstest(np.real(samples), 'norm', args=(0, 1 / np.sqrt(2)))

            corr = np.corrcoef(np.real(samples), np.imag(samples))
            corr = corr[0, 1]

            if p_real > alpha and p_imag > alpha and np.abs(corr) < max_corr:
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

        time_interval = 1 / doppler_frequency  # get uncorrelated samples
        timestamps = np.arange(samples_per_drop) * time_interval

        samples = np.array([])

        is_rice = False
        alpha = .05

        number_of_drops = 0
        while not is_rice and number_of_drops < max_number_of_drops:

            channel_gains = channel.impulse_response(timestamps)
            samples = np.append(samples, channel_gains.ravel())

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

        timestamps = np.arange(samples_per_drop) / self.sampling_rate

        for s in range(max_number_of_drops):

            delayed_channel.random_generator = np.random.default_rng(s+10)
            delayed_response = delayed_channel.impulse_response(timestamps)

            delayed_time = np.arange(delayed_response.shape[-1]) / self.sampling_rate
            delay_diff = (delayed_time - np.mean(delayed_time)) ** 2
            delayed_power = delayed_response.real ** 2 + delayed_response.imag ** 2
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
        sampling_rate = 1e6
        signal_length = 1000

        self.channel_params['delays'][0] = 0.
        self.channel_params['power_profile'][0] = 1.
        self.channel_params['rice_factors'][0] = 0.
        self.channel_params['doppler_frequency'] = doppler_frequency

        channel_no_gain = MultipathFadingChannel(**self.channel_params)

        self.channel_params['gain'] = gain
        channel_gain = MultipathFadingChannel(**self.channel_params)

        frame_size = (1, signal_length)
        tx_signal = rand.normal(size=frame_size) + 1j * rand.normal(size=frame_size)

        channel_no_gain.random_generator = np.random.default_rng(42)  # Reset random number generator
        signal_out_no_gain, _ = channel_no_gain.propagate(tx_signal)

        channel_gain.random_generator = np.random.default_rng(42)   # Reset random number generator
        signal_out_gain, _ = channel_gain.propagate(tx_signal)

        npt.assert_array_almost_equal(signal_out_gain, signal_out_no_gain * gain)

        timestamps = np.asarray([0, 100, 500]) / sampling_rate

        channel_no_gain.random_generator = np.random.default_rng(50)  # Reset random number generator
        channel_state_info_no_gain = channel_no_gain.impulse_response(timestamps)

        channel_gain.random_generator = np.random.default_rng(50)   # Reset random number generator
        channel_state_info_gain = channel_gain.impulse_response(timestamps)

        npt.assert_array_almost_equal(channel_state_info_gain, channel_state_info_no_gain * gain)

    def test_to_yaml(self) -> None:
        """Test YAML serialization dump validity."""
        pass

    def test_from_yaml(self) -> None:
        """Test YAML serialization recall validity."""
        pass

    def test_mimo(self) -> None:
        """
        Test the MIMO channel
        The following tests are performed

        1 - check output sizes
        2 - test MIMO output with known channel
        3 - check the channel impulse response
        4 - test antenna correlation
        :return:
        """
        pass


"""if __name__ == '__main__':
    '''
    If this file is called, initially plot some graphs to that indicate if multipath channel is working correctly, and
    then perform all unit tests.
    '''

    rnd = np.random.RandomState(44)
    rx_modem_params = ParametersRxModem()
    tx_modem_params = ParametersTxModem()
    param = ParametersChannel(rx_modem_params, tx_modem_params)
    param.multipath_model = 'STOCHASTIC'

    ###################################
    # plot Rayleigh/Rice distribution
    max_num_drops = 100
    doppler_frequency = 200
    sampling_rate = 1e6
    samples_per_drop = 1000

    param.delays = np.array([0])
    param.power_delay_profile = np.array([1])

    param.k_factor_rice = np.asarray([0])
    channel_rayleigh = MultipathFadingChannel(
        param, rnd, sampling_rate, doppler_frequency)

    param.k_factor_rice = np.asarray([1])
    channel_rice = MultipathFadingChannel(
        param, rnd, sampling_rate, doppler_frequency)

    time_interval = 1 / doppler_frequency  # get uncorrelated samples
    timestamps = np.arange(samples_per_drop) * time_interval

    samples_rayleigh = np.array([])
    samples_rice = np.array([])

    aux_samples = np.array([])

    for drop in range(max_num_drops):
        channel_rayleigh.init_drop()
        channel_rice.init_drop()

        channel_gains = channel_rayleigh.get_impulse_response(timestamps)
        samples_rayleigh = np.append(samples_rayleigh, channel_gains.ravel())

        channel_gains = channel_rice.get_impulse_response(timestamps)
        samples_rice = np.append(samples_rice, channel_gains.ravel())

    # plot distribution of real part
    plt.figure()
    plt.subplot(131)
    hist, bin_edges, dummy = plt.hist(
        np.real(samples_rayleigh), density=True, bins=40, label='histogram')
    plt.plot(
        bin_edges,
        stats.norm.pdf(
            bin_edges,
            scale=1 /
            np.sqrt(2)),
        label='normal distribution')
    plt.title('Histogram of real part of channel gain')
    plt.legend()

    # plot distribution of absolute value (Rayleigh fading)
    plt.subplot(132)
    hist, bin_edges, dummy = plt.hist(
        np.abs(samples_rayleigh), density=True, bins=40, label='histogram')
    plt.plot(
        bin_edges,
        stats.rayleigh.pdf(
            bin_edges,
            scale=1 /
            np.sqrt(2)),
        label='Rayleigh distribution')
    plt.title('Histogram of channel gain amplitude - Rayleigh fading')
    plt.legend()

    # plot distribution of absolute value (Rice fading)
    plt.subplot(133)
    hist, bin_edges, dummy = plt.hist(
        np.abs(samples_rice), density=True, bins=40, label='histogram')
    plt.plot(
        bin_edges,
        stats.rice.pdf(
            bin_edges,
            b=np.sqrt(2),
            scale=1 / 2),
        label='Rice distribution')
    plt.title('Histogram of channel gain amplitude - Rice fading')
    plt.legend()

    ##########################################################################
    # plot power delay/profile
    max_number_paths = 20
    max_delay_in_samples = 30
    max_doppler_frequency = 100

    number_of_paths = rnd.randint(2, np.minimum(
        max_number_paths, max_delay_in_samples))
    path_delay_in_samples = rnd.choice(
        max_delay_in_samples,
        number_of_paths,
        replace=False)
    path_delay_in_samples = np.sort(
        path_delay_in_samples) - path_delay_in_samples.min()
    max_delay_in_samples = path_delay_in_samples.max()
    path_delay = path_delay_in_samples / sampling_rate

    param.delays = path_delay

    power_delay_profile = np.abs(rnd.random_sample(param.delays.shape))
    param.power_delay_profile = power_delay_profile / \
        np.sum(power_delay_profile)

    doppler_frequency = rnd.random_sample() * max_doppler_frequency

    param.k_factor_rice = np.zeros(path_delay.size)  # Rayleigh fading

    # create channel with no need for interpolation
    channel = MultipathFadingChannel(
        param, rnd, sampling_rate, doppler_frequency)

    # create channel with interpolation (delays are not multiples of sampling
    # interval)
    sampling_rate_interp = sampling_rate * (1 + rnd.random_sample() * 0.5)
    channel_interp = MultipathFadingChannel(
        param, rnd, sampling_rate_interp, doppler_frequency)

    time_interval = 1 / doppler_frequency  # get uncorrelated samples
    timestamps = np.arange(samples_per_drop) * time_interval

    all_channels = np.array([])
    all_channels_interp = np.array([])

    for drop in range(max_num_drops):
        channel.init_drop()
        channel_interp.init_drop()

        channel_gains = np.squeeze(channel.get_impulse_response(timestamps))
        if all_channels.size:
            all_channels = np.append(all_channels, channel_gains, axis=0)
        else:
            all_channels = channel_gains

        channel_gains = np.squeeze(
            channel_interp.get_impulse_response(timestamps))
        if all_channels_interp.size:
            all_channels_interp = np.append(
                all_channels_interp, channel_gains, axis=0)
        else:
            all_channels_interp = channel_gains

    mean_power = np.mean(np.abs(all_channels) ** 2, axis=0)
    mean_power_interp = np.mean(np.abs(all_channels_interp) ** 2, axis=0)

    rms_delay_spread = np.sqrt(np.sum((np.arange(channel.max_delay_in_samples + 1) / channel.sampling_rate)**2
                                      * mean_power))
    rms_delay_spread_interp = np.sqrt(np.sum((np.arange(channel_interp.max_delay_in_samples + 1)
                                              / channel_interp.sampling_rate)**2 * mean_power_interp))

    plt.figure()
    plt.subplot(121)
    plt.stem(
        param.delays * 1e6,
        param.power_delay_profile,
        label='desired profile',
        linefmt='r',
        markerfmt='or')
    marker = plt.stem(np.arange(max_delay_in_samples + 1) / sampling_rate * 1e6, mean_power, label='actual profile',
                      linefmt='b', markerfmt='Db')
    marker[0].set_markerfacecolor('none')
    plt.xlabel('delay (ns)')
    plt.title(
        'Power Delay Profile (rms delay spread = {:f}ns)'.format(
            rms_delay_spread * 1e6))
    plt.legend()

    plt.subplot(122)

    plt.stem(
        param.delays * 1e6,
        param.power_delay_profile,
        label='desired profile',
        linefmt='r',
        markerfmt='or')
    marker = plt.stem(np.arange(channel_interp.max_delay_in_samples + 1) / sampling_rate_interp * 1e6, mean_power_interp,
                      label='actual profile', linefmt='b', markerfmt='Db')
    marker[0].set_markerfacecolor('none')
    plt.xlabel('delay (ns)')
    plt.title(
        'Power Delay Profile - Interpolated (rms delay spread = {:f}ns)'.format(
            rms_delay_spread_interp * 1e6))
    plt.legend()

    ##########################################################################
    # plot autocorrelation and Doppler spectrum
    max_num_drops = 1000
    doppler_frequency = 200
    sampling_rate = 1e6
    samples_per_drop = 1000
    channel_oversampling_rate = 100  # in relation to doppler_frequency

    param.delays = np.array([0])
    param.power_delay_profile = np.array([1])

    param.k_factor_rice = np.asarray([0])
    channel = MultipathFadingChannel(
        param, rnd, sampling_rate, doppler_frequency)

    time_interval = 1 / doppler_frequency / channel_oversampling_rate
    timestamps = np.arange(samples_per_drop) * time_interval

    autocorr = np.zeros(2 * samples_per_drop - 1, dtype=complex)

    for drop in range(max_num_drops):
        channel.init_drop()

        channel_gains = np.squeeze(
            channel_rayleigh.get_impulse_response(timestamps))
        autocorr += np.correlate(channel_gains,
                                 channel_gains,
                                 'full') / samples_per_drop

    autocorr = autocorr / max_num_drops

    # plot distribution of real part
    dt = np.arange(-samples_per_drop + 1, samples_per_drop) * time_interval
    plt.figure()
    plt.subplot(121)
    plt.plot(dt * doppler_frequency, autocorr, label='simulation')
    plt.plot(
        dt *
        doppler_frequency,
        special.j0(
            2 *
            np.pi *
            doppler_frequency *
            dt),
        label='theory')
    plt.xlabel('time (doppler x delay)')
    plt.title('Rayleigh Fading - autocorrelation')
    plt.xlim(-5, 5)
    plt.legend()

    plt.subplot(122)
    doppler_spectrum = np.fft.fftshift(np.fft.fft(autocorr))
    freq = np.fft.fftshift(
        np.fft.fftfreq(
            autocorr.size,
            time_interval)) / doppler_frequency
    plt.plot(freq, np.abs(doppler_spectrum), label='simulation')
    plt.title('Doppler Spectrum')
    plt.xlabel('normalized frequency (f/fd)')
    plt.ylabel('power spectrum density')
    plt.xlim(-doppler_frequency * 1.5, doppler_frequency * 1.5)

    plt.show()

    unittest.main()"""
