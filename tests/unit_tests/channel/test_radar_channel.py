# -*- coding: utf-8 -*-
"""Test Radar Channel"""

import unittest
from unittest.mock import patch, Mock, PropertyMock

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal
from scipy.constants import pi, speed_of_light

from hermespy.channel import RadarChannel
from hermespy.core import FloatingError, Signal
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization


__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRadarChannel(unittest.TestCase):

    def setUp(self) -> None:

        self.range = 100.
        self.radar_cross_section = 1.

        self.random_generator = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.random_generator

        self.transmitter = Mock()
        self.transmitter.carrier_frequency = 1e9
        self.transmitter.sampling_rate = 1e6
        self.transmitter.antennas.num_antennas = 1
        self.transmitter.antennas.spherical_response.return_value = np.array([1.], dtype=complex)
        self.transmitter.velocity = np.zeros(3, dtype=float)
        self.receiver = self.transmitter

        self.target_exists = True

        self.losses_db = 0

        self.channel = RadarChannel(self.range, self.radar_cross_section,
                                    transmitter=self.transmitter,
                                    receiver=self.receiver,
                                    target_exists=self.target_exists,
                                    losses_db=self.losses_db)
        self.channel.random_mother = self.random_node

        self.expected_delay = 2 * self.range / speed_of_light

    def test_init(self) -> None:
        """The object initialization should properly store all parameters"""

        self.assertEqual(self.range, self.channel.target_range)
        self.assertIs(self.radar_cross_section, self.channel.radar_cross_section)
        self.assertIs(self.transmitter, self.channel.transmitter)
        self.assertIs(self.receiver, self.channel.receiver)
        self.assertIs(self.target_exists, self.channel.target_exists)
        self.assertIs(self.losses_db, self.channel.losses_db)

    def test_target_range_setget(self) -> None:
        """Target range property getter should return setter argument"""

        new_range = 500
        self.channel.target_range = new_range

        self.assertEqual(new_range, self.channel.target_range)

    def test_target_range_validation(self) -> None:
        """Target range property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.channel.target_range = -1.

    def test_target_exists_setget(self) -> None:
        """Target exists flag getter should return setter argument"""

        new_target_exists = False
        self.channel.target_exists = new_target_exists
        self.assertEqual(new_target_exists, self.channel.target_exists)

    def test_radar_cross_section_get(self) -> None:
        """Radar cross section getter should return init param"""

        self.assertEqual(self.radar_cross_section, self.channel.radar_cross_section)

    def test_cross_section_validation(self) -> None:
        """Radar cross section property should raise ValueError on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.channel.radar_cross_section = -1.12345

        try:
            self.channel.radar_cross_section = 0.

        except ValueError:
            self.fail()

    def test_losses_db_get(self) -> None:
        """Losses getter should return init param"""

        self.assertEqual(self.losses_db, self.channel.losses_db)

    def test_velocity_setget(self) -> None:
        """Velocity getter should return setter argument"""

        new_velocity = 20

        self.channel.target_velocity = new_velocity
        self.assertEqual(new_velocity, self.channel.target_velocity)

    def test_impulse_response_anchored_validation(self) -> None:
        """Impulse response should raise FloatingError if not anchored to a device"""

        with patch.object(RadarChannel, 'transmitter', None), self.assertRaises(FloatingError):
            _ = self.channel.impulse_response(0, 1.)

    def test_impulse_response_carrier_frequency_validation(self) -> None:
        """Impulse response should raise RuntimeError if device carrier frequencies are smaller or equal to zero"""

        self.transmitter.carrier_frequency = 0.

        with self.assertRaises(RuntimeError):
            _ = self.channel.impulse_response(0, 1.)

    def test_impulse_response_interference_validation(self) -> None:
        """Impulse response should raise RuntimeError if not configured as a self-interference channel"""

        with patch.object(RadarChannel, 'receiver', None), self.assertRaises(RuntimeError):
            _ = self.channel.impulse_response(0, 1.)

    def _create_impulse_train(self, interval_in_samples: int, number_of_pulses: int):

        interval = interval_in_samples / self.transmitter.sampling_rate

        number_of_samples = int(np.ceil(interval * self.transmitter.sampling_rate * number_of_pulses))
        output_signal = np.zeros((1, number_of_samples), dtype=complex)

        interval_in_samples = int(np.around(interval * self.transmitter.sampling_rate))

        output_signal[:, :number_of_samples:interval_in_samples] = 1.0

        return output_signal

    def test_propagation_delay_integer_num_samples(self) -> None:
        """
        Test if the received signal corresponds to the expected delayed version, given that the delay is a multiple
        of the sampling interval
        """

        samples_per_symbol = 1000
        num_pulses = 10
        delay_in_samples = 507

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        expected_range = speed_of_light * delay_in_samples / self.transmitter.sampling_rate / 2
        expected_amplitude = ((speed_of_light / self.transmitter.carrier_frequency) ** 2 *
                              self.radar_cross_section / (4 * pi) ** 3 / expected_range ** 4)

        self.channel.target_range = expected_range

        output, _, _ = self.channel.propagate(Signal(input_signal, self.transmitter.sampling_rate))

        expected_output = np.hstack((np.zeros((1, delay_in_samples)), input_signal)) * expected_amplitude
        assert_array_almost_equal(abs(expected_output), np.abs(output[0].samples[:, :expected_output.size]))

    def test_propagation_delay_noninteger_num_samples(self) -> None:
        """
        Test if the received signal corresponds to the expected delayed version, given that the delay falls in the
        middle of two sampling instants.
        """
        samples_per_symbol = 800
        num_pulses = 20
        delay_in_samples = 312

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        expected_range = speed_of_light * (delay_in_samples + .5) / self.transmitter.sampling_rate / 2
        expected_amplitude = ((speed_of_light / self.transmitter.carrier_frequency) ** 2 *
                              self.radar_cross_section / (4 * pi) ** 3 / expected_range ** 4)

        self.channel.target_position = expected_range

        output, _, _ = self.channel.propagate(Signal(input_signal, self.transmitter.sampling_rate))

        straddle_loss = np.sinc(.5)
        peaks = np.abs(output[0].samples[:, delay_in_samples:input_signal.size:samples_per_symbol])

        assert_array_almost_equal(peaks, expected_amplitude * straddle_loss * np.ones(peaks.shape))

    def test_propagation_delay_doppler(self) -> None:
        """
        Test if the received signal corresponds to a frequency-shifted version of the transmitted signal with the
        expected Doppler shift
        """

        samples_per_symbol = 50
        num_pulses = 100
        initial_delay_in_samples = 1000
        expected_range = speed_of_light * initial_delay_in_samples / self.transmitter.sampling_rate / 2
        velocity = 10
        expected_amplitude = ((speed_of_light / self.transmitter.carrier_frequency) ** 2 *
                              self.radar_cross_section / (4 * pi) ** 3 / expected_range ** 4)

        initial_delay = initial_delay_in_samples / self.transmitter.sampling_rate

        timestamps_impulses = np.arange(num_pulses) * samples_per_symbol / self.transmitter.sampling_rate
        traveled_distances = velocity * timestamps_impulses
        delays = initial_delay + 2 * traveled_distances / speed_of_light
        expected_peaks = timestamps_impulses + delays
        peaks_in_samples = np.around(expected_peaks * self.transmitter.sampling_rate).astype(int)
        straddle_delay = expected_peaks - peaks_in_samples / self.transmitter.sampling_rate
        relative_straddle_delay = straddle_delay * self.transmitter.sampling_rate
        expected_straddle_amplitude = np.sinc(relative_straddle_delay) * expected_amplitude

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        self.channel.target_range = expected_range
        self.channel.velocity = velocity

        output, _, _ = self.channel.propagate(Signal(input_signal, self.transmitter.sampling_rate))

        assert_array_almost_equal(np.abs(output[0].samples[0, peaks_in_samples].flatten()), expected_straddle_amplitude)

    def test_doppler_shift(self) -> None:
        """
        Test if the received signal corresponds to the expected delayed version, given time variant delays on account of
        movement
        """

        velocity = 100
        self.channel.target_velocity = velocity

        num_samples = 100000
        sinewave_frequency = .25 * self.transmitter.sampling_rate
        doppler_shift = 2 * velocity / speed_of_light * self.transmitter.carrier_frequency

        time = np.arange(num_samples) / self.transmitter.sampling_rate

        input_signal = np.sin(2 * np.pi * sinewave_frequency * time)
        output, _, _ = self.channel.propagate(Signal(input_signal[np.newaxis, :], self.transmitter.sampling_rate))

        input_freq = np.fft.fft(input_signal)
        output_freq = np.fft.fft(output[0].samples.flatten()[-num_samples:])

        freq_resolution = self.transmitter.sampling_rate / num_samples

        freq_in = np.argmax(np.abs(input_freq[:int(num_samples/2)])) * freq_resolution
        freq_out = np.argmax(np.abs(output_freq[:int(num_samples/2)])) * freq_resolution

        self.assertAlmostEqual(freq_out - freq_in, doppler_shift, delta=np.abs(doppler_shift)*.01)

    def test_no_echo(self) -> None:
        """
        Test if no echos are observed if target_exists is set to False
        """
        samples_per_symbol = 500
        num_pulses = 15

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        self.channel.target_exists = False
        output, _, _ = self.channel.propagate(Signal(input_signal, self.transmitter.sampling_rate))

        assert_array_almost_equal(output[0].samples, np.zeros(output[0].samples.shape))

    def test_serialization(self) -> None:
        """Test YAML serialization"""
        
        with patch('hermespy.channel.Channel.transmitter', new_callable=PropertyMock) as transmitter_mock, \
             patch('hermespy.channel.Channel.receiver', new_callable=PropertyMock) as receiver_mock, \
             patch('hermespy.channel.Channel.random_mother', new_callable=PropertyMock) as random_mock:
            
            transmitter_mock.return_value = None
            receiver_mock.return_value = None
            random_mock.return_value = None
            
            test_yaml_roundtrip_serialization(self, self.channel)
