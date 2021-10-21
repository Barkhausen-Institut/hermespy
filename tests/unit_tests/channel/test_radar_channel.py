import unittest
import numpy as np
from unittest.mock import Mock

from channel.radar_channel import RadarChannel
from simulator_core.tools import constants


class TestRadarChannel(unittest.TestCase):

    def setUp(self) -> None:
        self.range = 100
        self.radar_cross_section = 1
        self.carrier_frequency = 10e9

        self.random_number_gen = np.random.RandomState(42)

        self.transmitter = Mock()
        self.receiver = Mock()
        self.transmitter.sampling_rate = 1e9
        self.transmitter.num_antennas = 1
        self.receiver.num_antennas = 1

        self.channel = RadarChannel(self.range, self.radar_cross_section, self.carrier_frequency,
                                    random_number_gen=self.random_number_gen,
                                    transmitter=self.transmitter,
                                    receiver=self.receiver)
        self.expected_delay = 2 * self.range / constants.speed_of_light

    def _create_impulse_train(self, interval: float, number_of_pulses: int):

        number_of_samples = int(np.ceil(interval * self.transmitter.sampling_rate * number_of_pulses))
        output_signal = np.zeros(number_of_samples, dtype=complex)

        interval_in_samples = int(np.around(interval / self.transmitter.sampling_rate))

        output_signal[:number_of_samples:interval_in_samples] = 1.0

        return output_signal

    def test_propagation_delay_integer_num_samples(self):
        """
        Test if the received signal corresponds to the expected delayed version, given that the delay is a multiple
        of the sampling interval.
        """
        samples_per_symbol = 1000
        num_pulses = 10
        delay_in_samples = 507

        interval = samples_per_symbol / self.transmitter.sampling_rate
        input_signal = self._create_impulse_train(interval, num_pulses)

        expected_range = constants.speed_of_light * delay_in_samples / self.transmitter.sampling_rate / 2
        old_range = self.channel.target_range
        self.channel.target_range = expected_range

        self.channel.init_drop()
        output = self.channel.propagate(input_signal)

        expected_output = np.concatenate((np.zeros(delay_in_samples), input_signal))

        np.testing.assert_array_almost_equal(expected_output, output.flatten()[:expected_output.size])

    def test_propagation_delay_noninteger_num_samples(self):
        pass

    def test_propagation_doppler(self):
        pass


    def test_propagation_leakage(self):
        pass


if __name__ == '__main__':
    unittest.main()

