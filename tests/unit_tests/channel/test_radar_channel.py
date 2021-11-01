import unittest
import numpy as np
from unittest.mock import Mock
from copy import deepcopy

from channel.radar_channel import RadarChannel
from scipy import constants
from tools.math import db2lin, lin2db

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRadarChannel(unittest.TestCase):

    def setUp(self) -> None:
        self.range = 100
        self.radar_cross_section = 1

        self.random_number_gen = np.random.RandomState(42)

        self.transmitter = Mock()
        self.transmitter.carrier_frequency = 60e9
        self.receiver = Mock()
        self.transmitter.sampling_rate = 1e9
        self.transmitter.num_antennas = 1
        self.receiver.num_antennas = 1

        self.target_exists = True
        self.tx_rx_isolation_db = float("inf")
        self.tx_antenna_gain_db = 0
        self.rx_antenna_gain_db = 0
        self.losses_db = 0
        self.velocity = 0
        self.filter_response_in_samples = 21

        self.channel = RadarChannel(self.range, self.radar_cross_section,
                                    random_number_gen=self.random_number_gen,
                                    transmitter=self.transmitter,
                                    receiver=self.receiver,
                                    target_exists=self.target_exists,
                                    tx_rx_isolation_db=self.tx_rx_isolation_db,
                                    tx_antenna_gain_db=self.tx_antenna_gain_db,
                                    rx_antenna_gain_db=self.rx_antenna_gain_db,
                                    losses_db=self.losses_db,
                                    velocity=self.velocity,
                                    filter_response_in_samples=self.filter_response_in_samples)

        self.expected_delay = 2 * self.range / constants.speed_of_light

    def test_init(self) -> None:
        """The object initialization should properly store all parameters."""

        channel = RadarChannel(self.range, self.radar_cross_section,
                               random_number_gen=self.random_number_gen,
                               transmitter=self.transmitter,
                               receiver=self.receiver,
                               target_exists=self.target_exists,
                               tx_rx_isolation_db=self.tx_rx_isolation_db,
                               tx_antenna_gain_db=self.tx_antenna_gain_db,
                               rx_antenna_gain_db=self.rx_antenna_gain_db,
                               losses_db=self.losses_db,
                               velocity=self.velocity,
                               filter_response_in_samples=self.filter_response_in_samples)

        self.assertIs(self.range, channel.target_range)
        self.assertIs(self.radar_cross_section, channel.radar_cross_section)
        self.assertIs(self.transmitter, channel.transmitter)
        self.assertIs(self.receiver, channel.receiver)
        self.assertIs(self.target_exists, channel.target_exists)
        self.assertIs(self.tx_rx_isolation_db, channel.tx_rx_isolation_db)
        self.assertIs(self.tx_antenna_gain_db, channel.tx_antenna_gain_db)
        self.assertIs(self.rx_antenna_gain_db, channel.rx_antenna_gain_db)
        self.assertIs(self.losses_db, channel.losses_db)
        self.assertIs(self.velocity, channel.velocity)
        self.assertIs(self.filter_response_in_samples, channel.filter_response_in_samples)

    def test_target_range_setget(self) -> None:
        """Target range property getter should return setter argument."""
        new_range = 500

        channel = deepcopy(self.channel)
        channel.target_range = new_range
        self.assertEqual(new_range, channel.target_range)

    def test_target_exists_setget(self) -> None:
        """Target exists flag getter should return setter argument."""
        new_target_exists = False

        channel = deepcopy(self.channel)
        channel.target_exists = new_target_exists
        self.assertEqual(new_target_exists, channel.target_exists)

    def test_radar_cross_section_get(self) -> None:
        """Radar cross section getter should return init param."""
        self.assertEqual(self.radar_cross_section, self.channel.radar_cross_section)

    def test_tx_rx_isolation_db_setget(self) -> None:
        """tx/rx isolation getter should return setter argument."""
        new_tx_rx_isolation_db = 100

        channel = deepcopy(self.channel)
        channel.tx_rx_isolation_db = new_tx_rx_isolation_db
        self.assertEqual(new_tx_rx_isolation_db, channel.tx_rx_isolation_db)

    def test_tx_antenna_gain_db_get(self) -> None:
        """tx antenna gain getter should return init param."""
        self.assertEqual(self.tx_antenna_gain_db, self.channel.tx_antenna_gain_db)

    def test_rx_antenna_gain_db_get(self) -> None:
        """rx antenna gain getter should return init param."""
        self.assertEqual(self.rx_antenna_gain_db, self.channel.rx_antenna_gain_db)

    def test_losses_db_get(self) -> None:
        """losses getter should return init param."""
        self.assertEqual(self.losses_db, self.channel.losses_db)

    def test_velocity_setget(self) -> None:
        """velocity getter should return setter argument."""
        new_velocity = 20

        channel = deepcopy(self.channel)
        channel.velocity = new_velocity
        self.assertEqual(new_velocity, channel.velocity)

    def test_filter_response_in_samples_get(self) -> None:
        """losses getter should return init param."""
        self.assertEqual(self.filter_response_in_samples, self.channel.filter_response_in_samples)

    def _create_impulse_train(self, interval_in_samples: int, number_of_pulses: int):

        interval = interval_in_samples / self.transmitter.sampling_rate

        number_of_samples = int(np.ceil(interval * self.transmitter.sampling_rate * number_of_pulses))
        output_signal = np.zeros((1, number_of_samples), dtype=complex)

        interval_in_samples = int(np.around(interval * self.transmitter.sampling_rate))

        output_signal[:, :number_of_samples:interval_in_samples] = 1.0

        return output_signal

    def test_propagation_delay_integer_num_samples(self):
        """
        Test if the received signal corresponds to the expected delayed version, given that the delay is a multiple
        of the sampling interval.
        """
        samples_per_symbol = 1000
        num_pulses = 10
        delay_in_samples = 507

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        expected_range = constants.speed_of_light * delay_in_samples / self.transmitter.sampling_rate / 2

        channel = deepcopy(self.channel)
        channel.target_range = expected_range

        channel.init_drop()
        output = channel.propagate(input_signal)

        expected_output = np.hstack((np.zeros((1, delay_in_samples)), input_signal))

        np.testing.assert_array_almost_equal(expected_output, np.abs(output[:, :expected_output.size]))

    def test_propagation_delay_noninteger_num_samples(self):
        """
        Test if the received signal corresponds to the expected delayed version, given that the delay falls in the
        middle of two sampling instants.
        """
        samples_per_symbol = 800
        num_pulses = 20
        delay_in_samples = 312

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        expected_range = constants.speed_of_light * (delay_in_samples + .5) / self.transmitter.sampling_rate / 2

        channel = deepcopy(self.channel)
        channel.target_range = expected_range

        channel.init_drop()
        output = channel.propagate(input_signal)

        straddle_loss = np.sinc(.5)
        peaks = np.abs(output[:, delay_in_samples:input_signal.size:samples_per_symbol])

        np.testing.assert_array_almost_equal(peaks, straddle_loss * np.ones(peaks.shape))

    def test_propagation_delay_doppler(self):
        """
        Test if the received signal corresponds to a frequency-shifted version of the transmitted signal with the
        expected Doppler shift
        """

        samples_per_symbol = 5000
        num_pulses = 100
        initial_delay_in_samples = 1000
        velocity = 500

        initial_delay = initial_delay_in_samples / self.transmitter.sampling_rate

        timestamps_impulses = np.arange(num_pulses) * samples_per_symbol / self.transmitter.sampling_rate
        traveled_distances = velocity * timestamps_impulses
        delays = initial_delay + 2 * traveled_distances / constants.speed_of_light
        expected_peaks = timestamps_impulses + delays
        peaks_in_samples = np.around(expected_peaks * self.transmitter.sampling_rate).astype(int)
        straddle_delay = expected_peaks - peaks_in_samples / self.transmitter.sampling_rate
        relative_straddle_delay = straddle_delay * self.transmitter.sampling_rate
        expected_straddle_amplitude = np.sinc(relative_straddle_delay)

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        expected_range = constants.speed_of_light * initial_delay_in_samples / self.transmitter.sampling_rate / 2

        channel = deepcopy(self.channel)
        channel.target_range = expected_range
        channel.velocity = velocity

        channel.init_drop()

        output = channel.propagate(input_signal)

        np.testing.assert_array_almost_equal(np.abs(output[0, peaks_in_samples].flatten()), expected_straddle_amplitude)

    def test_propagation_leakage(self):
        """
        Test if the leakage between transmitter and receiver is correctly modelled
        """
        isolation_db = 10

        samples_per_symbol = 800
        num_pulses = 20

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        self.channel.init_drop()
        output_ideal_isolation = self.channel.propagate(input_signal)

        channel = deepcopy(self.channel)
        channel.tx_rx_isolation_db = isolation_db

        output = channel.propagate(input_signal)

        self_interference = output - output_ideal_isolation
        norm_factor = db2lin(-self.channel.attenuation_db - isolation_db, conversion_type='amplitude')

        np.testing.assert_array_almost_equal(np.abs(self_interference[0, :input_signal.size]),
                                             np.abs(input_signal[0, :]) * norm_factor)

    def test_doppler_shift(self):
        """
        Test if the received signal corresponds to the expected delayed version, given time variant delays on account of
        movement
        """
        velocity = -100
        num_samples = 100000
        sinewave_frequency = 100e6
        doppler_shift = 2 * velocity / constants.speed_of_light * self.transmitter.carrier_frequency

        time = np.arange(num_samples) / self.transmitter.sampling_rate

        input_signal = np.sin(2 * np.pi * sinewave_frequency * time)

        channel = deepcopy(self.channel)
        channel.velocity = velocity

        output = channel.propagate(input_signal[np.newaxis, :])

        input_freq = np.fft.fft(input_signal)
        output_freq = np.fft.fft(output.flatten()[-num_samples:])

        freq_resolution = self.transmitter.sampling_rate / num_samples

        freq_in = np.argmax(np.abs(input_freq[:int(num_samples/2)])) * freq_resolution
        freq_out = np.argmax(np.abs(output_freq[:int(num_samples/2)])) * freq_resolution

        self.assertAlmostEqual(freq_in - freq_out, doppler_shift, delta=np.abs(doppler_shift)*.01)

    def test_no_echo(self):
        """
        Test if no echos are observed if target_exists is set to False
        """
        samples_per_symbol = 500
        num_pulses = 15

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        channel = deepcopy(self.channel)
        channel.target_exists = False

        channel.init_drop()
        output = channel.propagate(input_signal)

        np.testing.assert_array_equal(output, np.zeros(output.shape))

    def test_get_impulse_response(self):
        samples_per_symbol = 1200
        num_pulses = 20

        isolation_db = 150
        velocity = 20

        input_signal = self._create_impulse_train(samples_per_symbol, num_pulses)

        channel = deepcopy(self.channel)
        channel.velocity = velocity
        channel.tx_rx_isolation_db = isolation_db

        channel.init_drop()

        output = channel.propagate(input_signal)

        sample_idx_last_symbol = (num_pulses - 1) * samples_per_symbol
        timestamps = np.array([0, sample_idx_last_symbol / self.transmitter.sampling_rate])
        channel_state_info = channel.get_impulse_response(timestamps).squeeze()

        num_samples_in_csi = channel_state_info.shape[1]

        observed_csi = np.vstack((output[0, :num_samples_in_csi],
                                  output[0, sample_idx_last_symbol: sample_idx_last_symbol + num_samples_in_csi]))

        # interpolation filter for propagation will be truncated, must be taken into account in absolute tolerance
        atol = np.maximum(np.abs(np.sinc(channel.filter_response_in_samples/2)),
                          np.abs(np.sinc(np.floor(channel.filter_response_in_samples/2))))
        np.testing.assert_allclose(channel_state_info, observed_csi, atol=atol)

        pass
