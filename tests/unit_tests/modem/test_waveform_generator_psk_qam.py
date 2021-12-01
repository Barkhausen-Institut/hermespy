# -*- coding: utf-8 -*-
"""Waveform Generation for Phase-Shift-Keying Quadrature Amplitude Modulation Testing."""

import unittest
from unittest.mock import Mock

import numpy as np

from hermespy.modem.waveform_generator_psk_qam import WaveformGeneratorPskQam
from hermespy.modem.modem import Modem

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestWaveformGeneratorPskQam(unittest.TestCase):
    """Test the Phase-Shift-Keying / Quadrature Amplitude Modulation Waveform Generator."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)

        self.modem = Mock()
        self.filter_type = 'ROOT_RAISED_COSINE'
        self.symbol_rate = 125e3
        self.oversampling_factor = 8
        self.modulation_order = 16
        self.guard_interval = 1e-3
        self.chirp_bandwidth = 100e6
        self.chirp_duration = 1e-6

        self.generator = WaveformGeneratorPskQam(modem=self.modem, symbol_rate=self.symbol_rate,
                                                 oversampling_factor=self.oversampling_factor,
                                                 modulation_order=self.modulation_order,
                                                 guard_interval=self.guard_interval,
                                                 chirp_duration=self.chirp_duration,
                                                 chirp_bandwidth=self.chirp_bandwidth)


    def test_proper_samples_in_frame_calculation(self) -> None:
        frame_duration = self.params_qpsk.number_data_symbols / self.params_qpsk.symbol_rate
        samples_in_frame_expected = int(np.ceil((frame_duration + self.params_qpsk.guard_interval)
                                                * self.params_qpsk.sampling_rate))
        self.assertEqual(
            self.waveform_generator_qpsk._samples_in_frame,
            samples_in_frame_expected)

    def test_proper_frame_length(self) -> None:
        self.modem_qpsk.generator._filter_tx.delay_in_samples = 0
        bits_in_frame = utils.flatten_blocks(
            self.source_qpsk.get_bits(
                self.params_qpsk.bits_in_frame))

        output_signal, _, _ = self.waveform_generator_qpsk.create_frame(
            self.timestamp, bits_in_frame)

        self.assertEqual(
            output_signal.size,
            self.waveform_generator_qpsk._samples_in_frame)

    def test_tx_filter_time_delay(self) -> None:
        self.waveform_generator_qpsk._filter_tx.delay_in_samples = 5
        bits_in_frame = utils.flatten_blocks(
            self.source_qpsk.get_bits(
                self.params_qpsk.bits_in_frame))

        _, _, initial_sample_num = self.waveform_generator_qpsk.create_frame(
            self.timestamp, bits_in_frame)

        self.assertEqual(
            initial_sample_num,
            self.timestamp - self.waveform_generator_qpsk._filter_tx.delay_in_samples
        )

    def test_timestamp_on_frame_creation(self) -> None:
        self.waveform_generator_qpsk._filter_tx.delay_in_samples = 5
        bits_in_frame = utils.flatten_blocks(
            self.source_qpsk.get_bits(
                self.params_qpsk.bits_in_frame))

        _, new_timestamp, _ = self.waveform_generator_qpsk.create_frame(
            self.timestamp, bits_in_frame)

        self.assertEqual(
            new_timestamp,
            self.timestamp + self.waveform_generator_qpsk._samples_in_frame
        )

    def test_too_short_signal_length_for_demodulation(self) -> None:
        # define input parameters
        timestamp_in_samples = 0
        rx_signal = np.ones((1, 3))  # empty baseband_signal cannot be demodulated

        demodulated_bits, left_over_rx_signal = self.waveform_generator_qpsk.receive_frame(
            rx_signal, timestamp_in_samples, 0)

        self.assertIsNone(demodulated_bits[0])
        np.testing.assert_array_equal(left_over_rx_signal, np.array([]))

    def test_demodulating_signal_length_one_sample_longer_than_frame(
            self) -> None:
        # define received baseband_signal. length of baseband_signal is 1 sample longer than the
        # length of a frame
        frame_overlap = 1
        rx_signal = np.random.randint(
            1,
            size=(
                1,
                self.waveform_generator_qpsk._samples_in_frame +
                frame_overlap))

        timestamp_in_samples = 0

        # impulse response is the same as its input
        self.waveform_generator_qpsk._channel = Mock()
        symbols_in_frame = self.waveform_generator_qpsk.param.number_data_symbols
        self.waveform_generator_qpsk._channel.get_impulse_response = lambda timestamps: np.ones(
            (symbols_in_frame, 1, 1, 1))

        bits, remaining_rx_signal = self.waveform_generator_qpsk.receive_frame(
            rx_signal, timestamp_in_samples, 0)

        self.assertEqual(len(bits[0]),
                         self.waveform_generator_qpsk.param.bits_in_frame)
        self.assertEqual(remaining_rx_signal.shape[1], frame_overlap)

    def test_proper_bit_energy_calculation(self) -> None:
        """Tests if theoretical bit energy is calculated correctly"""

        # define test parameters
        number_of_drops = 5
        number_of_frames = 2
        relative_difference = .05  # relative difference between desired and measured power

        modem = self.modem_qpsk
        bit_energy = self.estimate_energy(modem, number_of_drops,
                                          number_of_frames, self.source_qam.get_bits(
                                              self.params_qam.bits_in_frame),
                                          'bit_energy')

        # compare the measured energy with the expected values
        self.assertAlmostEqual(
            bit_energy,
            modem.get_bit_energy(),
            delta=bit_energy * relative_difference
        )

    def test_proper_symbol_energy_calculation(self) -> None:
        """Tests if theoretical symbol energy is calculated correctly"""

        # define test parameters
        number_of_drops = 4
        number_of_frames = 3
        relative_difference = .05  # relative difference between desired and measured power

        modem = self.modem_qam

        symbol_energy = self.estimate_energy(modem, number_of_drops,
                                             number_of_frames, self.source_qam.get_bits(
                                                 self.params_qam.bits_in_frame),
                                             'symbol_energy')

        # compare the measured energy with the expected values
        self.assertAlmostEqual(
            symbol_energy,
            modem.get_symbol_energy(),
            delta=symbol_energy *
            relative_difference)

    def test_proper_power_calculation(self) -> None:
        """Tests if theoretical baseband_signal power is calculated correctly"""

        # define test parameters
        number_of_drops = 5
        number_of_frames = 3
        relative_difference = .05  # relative difference between desired and measured power

        modem = self.modem_qam

        power = self.estimate_energy(modem, number_of_drops,
                                     number_of_frames, self.source_qam.get_bits(
                                         self.params_qam.bits_in_frame),
                                     'power')

        # compare the measured energy with the expected values
        self.assertAlmostEqual(
            power,
            modem.generator.get_power(),
            delta=power *
            relative_difference)

    @staticmethod
    def estimate_energy(modem: Modem, number_of_drops: int, number_of_frames: int,
                        data_bits: np.array, energy_type: str) -> float:
        """Generates a baseband_signal with a few drops and frames and measures average energy or power.
        In this method a baseband_signal is generated over several drops, and the average power or bit/symbol energy is
        calculated.

        Args:
            modem(Modem): modem for which transmit energy is calculated
            number_of_drops(int): number of drops for which baseband_signal is generated
            number_of_frames(int): number of frames generated in each drop
            data_bits(np.array): the data bits to be sent created by the BitsSource
            energy_type(str): what type of energy is to be returned.
                              Allowed values are 'bit_energy', 'symbol_energy' and 'power'

        Returns:
            power_or_energy(float): estimated power or bit/symbol-energy, depending on the value of `energy_type`
        """

        energy_sum = 0

        frame_duration = modem.param.technology.number_data_symbols / \
            modem.param.technology.symbol_rate
        number_pilot_symbols = modem.param.technology.number_preamble_symbols + \
            modem.param.technology.number_postamble_symbols
        if number_pilot_symbols > 0:
            frame_duration += number_pilot_symbols / \
                modem.param.technology.pilot_symbol_rate

        for idx in range(number_of_drops):
            signal = modem.send(frame_duration * number_of_frames)
            energy = np.sum(np.real(signal)**2 + np.imag(signal)**2)

            energy_sum += energy

        energy_avg = energy_sum / number_of_drops
        if energy_type == 'bit_energy':
            number_of_bits = modem.param.technology.bits_in_frame * number_of_frames
            power_or_energy = energy_avg / number_of_bits
        elif energy_type == 'symbol_energy':
            symbols_in_frame = number_pilot_symbols + \
                modem.param.technology.number_data_symbols
            number_of_symbols = symbols_in_frame * number_of_frames
            power_or_energy = energy_avg / number_of_symbols
        elif energy_type == 'power':
            number_of_data_samples = number_of_frames * \
                int(np.ceil(frame_duration * modem.param.technology.sampling_rate))
            power_or_energy = energy_avg / number_of_data_samples
        else:
            raise ValueError("invalid 'energy_type'")

        return power_or_energy


if __name__ == '__main__':
    unittest.main()
