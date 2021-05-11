import unittest
from unittest.mock import Mock

import numpy as np

from parameters_parser.parameters_psk_qam import ParametersPskQam
from parameters_parser.parameters_tx_modem import ParametersTxModem
from modem.waveform_generator_psk_qam import WaveformGeneratorPskQam
from modem.modem import Modem
from source.bits_source import BitsSource
import tests.unit_tests.modem.utils as utils


class TestWaveformGeneratorPskQam(unittest.TestCase):

    def setUp(self) -> None:

        ########################################
        # create a QPSK modem with method stubs

        # create input parameters for QPSK modem creation
        rng = np.random.RandomState(42)
        self.source_qpsk = BitsSource(rng)

        # define parameters
        self.params_qpsk = ParametersPskQam()
        self.params_qpsk.modulation_order = 4

        self.params_qpsk.guard_interval = 1e-3

        self.params_qpsk.filter_type = 'ROOT_RAISED_COSINE'
        self.params_qpsk.symbol_rate = 125e3
        self.params_qpsk.bandwidth = self.params_qpsk.symbol_rate
        self.params_qpsk.oversampling_factor = 8
        self.params_qpsk.sampling_rate = self.params_qpsk.symbol_rate * \
            self.params_qpsk.oversampling_factor

        self.params_qpsk.filter_length_in_symbols = 16
        self.params_qpsk.roll_off_factor = .5

        frame_duration = 9e-3
        self.params_qpsk.number_preamble_symbols = 0
        self.params_qpsk.number_postamble_symbols = 0
        self.params_qpsk.number_data_symbols = int(
            frame_duration * self.params_qpsk.symbol_rate)

        self.params_qpsk.bits_per_symbol = int(
            np.log2(self.params_qpsk.modulation_order))
        self.params_qpsk.bits_in_frame = self.params_qpsk.bits_per_symbol * \
            self.params_qpsk.number_data_symbols

        self.modem_qpsk_params = ParametersTxModem()
        self.modem_qpsk_params.technology = self.params_qpsk

        self.waveform_generator_qpsk = WaveformGeneratorPskQam(self.params_qpsk)
        self.modem_qpsk = Modem(self.modem_qpsk_params, self.source_qpsk, rng)

        # create (method) stubs for dependencies
        self.waveform_generator_qpsk._mapping.get_symbols = lambda bits: bits[::int(  # type: ignore
            np.log2(self.params_qpsk.modulation_order))]
        self.waveform_generator_qpsk._filter_tx.filter = lambda frame: frame  # type: ignore
        self.waveform_generator_qpsk._filter_rx.filter = lambda frame: frame  # type: ignore
        self.waveform_generator_qpsk._filter_rx.delay_in_samples = 0

        self.timestamp = 0
        ########################################

        ########################################
        # create a full QAM modem

        # create input parameters for QAM modem creation
        rng = np.random.RandomState(42)
        self.source_qam = BitsSource(rng)

        # define parameters
        self.params_qam = ParametersPskQam()
        self.params_qam.modulation_order = 64
        self.params_qam.modulation_is_complex = True

        self.params_qam.symbol_rate = 250e3
        self.params_qam.bandwidth = self.params_qam.symbol_rate
        self.params_qam.oversampling_factor = 16
        self.params_qam.sampling_rate = self.params_qam.symbol_rate * \
            self.params_qam.oversampling_factor
        self.params_qam.filter_type = "ROOT_RAISED_COSINE"

        self.params_qam.filter_length_in_symbols = 16
        self.params_qam.roll_off_factor = .2

        frame_duration = 5e-3
        self.params_qam.number_preamble_symbols = 0
        self.params_qam.number_postamble_symbols = 0
        self.params_qam.number_data_symbols = int(
            frame_duration * self.params_qam.symbol_rate)
        self.params_qam.guard_interval = .5e-3
        self.params_qam.bits_per_symbol = int(
            np.log2(self.params_qam.modulation_order))
        self.params_qam.bits_in_frame = self.params_qam.bits_per_symbol * \
            self.params_qam.number_data_symbols

        self.modem_qam_params = ParametersTxModem()
        self.modem_qam_params.technology = self.params_qam

        self.waveform_generator_qam = WaveformGeneratorPskQam(self.params_qam)
        self.modem_qam = Modem(self.modem_qam_params, self.source_qam, rng)
        ########################################

    def test_proper_samples_in_frame_calculation(self) -> None:
        frame_duration = self.params_qpsk.number_data_symbols / self.params_qpsk.symbol_rate
        samples_in_frame_expected = int(np.ceil((frame_duration + self.params_qpsk.guard_interval)
                                                * self.params_qpsk.sampling_rate))
        self.assertEqual(
            self.waveform_generator_qpsk._samples_in_frame,
            samples_in_frame_expected)

    def test_proper_frame_length(self) -> None:
        self.modem_qpsk.waveform_generator._filter_tx.delay_in_samples = 0
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
        rx_signal = np.ones((1, 3))  # empty signal cannot be demodulated

        demodulated_bits, left_over_rx_signal = self.waveform_generator_qpsk.receive_frame(
            rx_signal, timestamp_in_samples, 0)

        self.assertIsNone(demodulated_bits[0])
        np.testing.assert_array_equal(left_over_rx_signal, np.array([]))

    def test_demodulating_signal_length_one_sample_longer_than_frame(
            self) -> None:
        # define received signal. length of signal is 1 sample longer than the
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
        """Tests if theoretical signal power is calculated correctly"""

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
            modem.waveform_generator.get_power(),
            delta=power *
            relative_difference)

    @staticmethod
    def estimate_energy(modem: Modem, number_of_drops: int, number_of_frames: int,
                        data_bits: np.array, energy_type: str) -> float:
        """Generates a signal with a few drops and frames and measures average energy or power.
        In this method a signal is generated over several drops, and the average power or bit/symbol energy is
        calculated.

        Args:
            modem(Modem): modem for which transmit energy is calculated
            number_of_drops(int): number of drops for which signal is generated
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
