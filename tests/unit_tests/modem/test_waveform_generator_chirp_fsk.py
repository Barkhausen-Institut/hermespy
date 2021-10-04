from parameters_parser.parameters_repetition_encoder import ParametersRepetitionEncoder
import unittest
import os

import numpy as np
from scipy import integrate

from modem.waveform_generator_chirp_fsk import WaveformGeneratorChirpFsk
from modem.modem import Modem
from parameters_parser.parameters_chirp_fsk import ParametersChirpFsk
from parameters_parser.parameters_tx_modem import ParametersTxModem
from parameters_parser.parameters_repetition_encoder import ParametersRepetitionEncoder
from source.bits_source import BitsSource


class TestWaveformGeneratorChirpFsk(unittest.TestCase):
    def setUp(self) -> None:
        self.params_cfsk = ParametersChirpFsk()
        self.params_cfsk.modulation_order = 32
        self.params_cfsk.chirp_duration = 4e-6
        self.params_cfsk.chirp_bandwidth = 200e6
        self.params_cfsk.freq_difference = 5e6
        self.params_cfsk.oversampling_factor = 4
        self.params_cfsk.number_data_chirps = 20
        self.params_cfsk.number_pilot_chirps = 2
        self.params_cfsk.guard_interval = 4e-6
        self.params_cfsk.sampling_rate = (
            self.params_cfsk.chirp_bandwidth * self.params_cfsk.oversampling_factor
        )
        self.params_cfsk.bits_per_symbol = 5
        self.params_cfsk.bits_in_frame = 100

        self.modem_params = ParametersTxModem()
        self.modem_params.number_of_antennas = 1
        self.modem_params.technology = self.params_cfsk
        self.modem_params.encoding_params = [ParametersRepetitionEncoder()]

        self.source = BitsSource(np.random.RandomState(42))
        self.waveform_generator_chirp_fsk = WaveformGeneratorChirpFsk(self.params_cfsk)
        self.modem_chirp_fsk = Modem(
            self.modem_params,
            self.source,
            np.random.RandomState(42),
            np.random.RandomState(43))

        self.parent_dir = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)), 'res')

    def test_proper_samples_in_chirp_calculation(self) -> None:
        samples_in_chirp_expected = int(
            np.around(
                self.params_cfsk.chirp_duration *
                self.params_cfsk.sampling_rate)
        )
        self.assertEqual(
            samples_in_chirp_expected, self.waveform_generator_chirp_fsk._samples_in_chirp
        )

    def test_proper_samples_in_frame_calculation(self) -> None:
        samples_in_frame_expected = (
            self.waveform_generator_chirp_fsk._samples_in_chirp
            * self.waveform_generator_chirp_fsk._chirps_in_frame
            + int((np.around(self.params_cfsk.guard_interval *
                             self.params_cfsk.sampling_rate)))
        )

        self.assertEqual(
            samples_in_frame_expected, self.waveform_generator_chirp_fsk._samples_in_frame
        )

    def test_proper_chirps_in_frame_calculation(self) -> None:
        chirps_in_frame_expected = (
            self.params_cfsk.number_pilot_chirps + self.params_cfsk.number_data_chirps
        )
        self.assertEqual(
            chirps_in_frame_expected, self.waveform_generator_chirp_fsk._chirps_in_frame
        )

    def test_proper_chirp_init(self) -> None:
        f0 = -self.params_cfsk.chirp_bandwidth / 2
        f1 = -f0

        slope = self.params_cfsk.chirp_bandwidth / self.params_cfsk.chirp_duration
        self.assertEqual(f0, self.waveform_generator_chirp_fsk._f0)
        self.assertEqual(f1, self.waveform_generator_chirp_fsk._f1)
        self.assertEqual(slope, self.waveform_generator_chirp_fsk._slope)

    def test_proper_chirp_offset_calculation_at_frame_creation(self) -> None:
        offset_expected = np.array(
            [8, 17, 1, 14, 23, 31, 7, 8, 7, 27, 10,
                24, 0, 27, 26, 29, 10, 11, 31, 30]
        )
        offset_calculated = self.waveform_generator_chirp_fsk._calculate_frequency_offsets(
            self.source.get_bits(self.params_cfsk.bits_in_frame)
        )

        np.testing.assert_array_equal(offset_expected, offset_calculated)

    def test_frequency_chirp_calculation(self) -> None:
        initial_frequencies = np.array([1, 10])
        samples_in_chirp = self.waveform_generator_chirp_fsk._samples_in_chirp
        slope = self.waveform_generator_chirp_fsk._slope
        chirp_time = self.waveform_generator_chirp_fsk._chirp_time
        f1 = self.waveform_generator_chirp_fsk._f1

        no_samples = initial_frequencies.size * samples_in_chirp

        f_expected = np.zeros(no_samples, dtype=complex)
        f_expected[:samples_in_chirp] = initial_frequencies[0] + \
            slope * chirp_time
        f_expected[samples_in_chirp:] = initial_frequencies[1] + \
            slope * chirp_time
        f_expected[f_expected > f1] -= self.params_cfsk.chirp_bandwidth

        a_expected = np.ones(no_samples, dtype=complex)

        f, a = self.waveform_generator_chirp_fsk._calculate_chirp_frequencies(
            initial_frequencies
        )

        np.testing.assert_array_almost_equal(f_expected, f[:no_samples])
        np.testing.assert_array_almost_equal(a_expected, a[:no_samples])

    def test_proper_new_timestamp_after_frame_creation(self) -> None:
        timestamp = 0
        new_timestamp_expected = (
            timestamp + self.waveform_generator_chirp_fsk._samples_in_frame
        )

        _, new_timestamp, initial_sample_num = self.waveform_generator_chirp_fsk.create_frame(
            timestamp, self.source.get_bits(self.params_cfsk.bits_in_frame)
        )

        self.assertEqual(new_timestamp_expected, new_timestamp)
        self.assertEqual(timestamp, initial_sample_num)

    def test_proper_time_signal_creation(self) -> None:
        initial_frequencies = np.array(
            [
                -1.0e08,
                -1.0e08,
                -6.0e07,
                -1.5e07,
                -9.5e07,
                -3.0e07,
                1.5e07,
                5.5e07,
                -6.5e07,
                -6.0e07,
                -6.5e07,
                3.5e07,
                -5.0e07,
                2.0e07,
                -1.0e08,
                3.5e07,
                3.0e07,
                4.5e07,
                -5.0e07,
                -4.5e07,
                5.5e07,
                5.0e07,
            ]
        )

        f, a = self.waveform_generator_chirp_fsk._calculate_chirp_frequencies(
            initial_frequencies
        )

        phase = (
            2
            * np.pi
            * integrate.cumtrapz(f, dx=1 / self.params_cfsk.sampling_rate, initial=0)
        )

        output_signal_expected = a * np.exp(1j * phase)

        output_signal, _, _ = self.waveform_generator_chirp_fsk.create_frame(
            0, self.source.get_bits(self.params_cfsk.bits_in_frame)
        )

        np.testing.assert_array_almost_equal(
            output_signal_expected[np.newaxis, :], output_signal)

    def test_prototype_chirps_for_modulation_symbols(self) -> None:
        cos_signal_expected = self.read_saved_results_from_file(
            'cos_signal.npy')
        sin_signal_expected = self.read_saved_results_from_file(
            'sin_signal.npy')

        np.testing.assert_array_almost_equal(
            cos_signal_expected, self.waveform_generator_chirp_fsk._prototype_function["cos"]
        )

        np.testing.assert_array_almost_equal(
            sin_signal_expected, self.waveform_generator_chirp_fsk._prototype_function["sin"]
        )

    def test_bit_energy_calculation(self) -> None:
        cos_signal_expected = self.read_saved_results_from_file(
            'cos_signal.npy')
        symbol_energy = sum(abs(cos_signal_expected[0, :]) ** 2)

        bit_energy_expected = symbol_energy / self.params_cfsk.bits_per_symbol
        bit_energy = self.waveform_generator_chirp_fsk.get_bit_energy()
        self.assertAlmostEqual(bit_energy_expected, bit_energy)

    def test_symbol_energy_calculation(self) -> None:
        cos_signal_expected = self.read_saved_results_from_file(
            'cos_signal.npy')
        symbol_energy_expected = sum(abs(cos_signal_expected[0, :]) ** 2)

        symbol_energy = self.waveform_generator_chirp_fsk.get_symbol_energy()
        self.assertAlmostEqual(symbol_energy_expected, symbol_energy)

    def read_saved_results_from_file(self, file_name: str) -> np.ndarray:
        if not os.path.exists(os.path.join(self.parent_dir, file_name)):
            raise FileNotFoundError(
                f"{file_name} must be in same folder as this file.")
        else:
            results = np.load(os.path.join(self.parent_dir, file_name))

        return results

    def test_rx_signal_properly_demodulated(self) -> None:
        rx_signal = self.read_saved_results_from_file('rx_signal.npy')
        rx_signal = np.reshape(rx_signal, (1, rx_signal.shape[0]))

        received_bits, _ = self.waveform_generator_chirp_fsk.receive_frame(
            rx_signal, 0, 0)
        received_bits_expected = self.read_saved_results_from_file(
            'received_bits.npy').ravel()

        np.testing.assert_array_equal(received_bits[0], received_bits_expected)

    def test_rx_signal_demodulation_long_signal(self) -> None:
        signal_overlength = 3
        rx_signal = self.read_saved_results_from_file('rx_signal.npy')
        rx_signal = np.append(rx_signal, np.ones(signal_overlength))
        rx_signal = np.reshape(rx_signal, (1, rx_signal.shape[0]))

        _, left_over_rx_signal = self.waveform_generator_chirp_fsk.receive_frame(
            rx_signal, 0, 0)

        self.assertEqual(left_over_rx_signal.shape[1], signal_overlength)

    def test_proper_bit_energy_calculation(self) -> None:
        """Tests if theoretical bit energy is calculated correctly"""

        # define test parameters
        number_of_drops = 5
        number_of_frames = 2
        relative_difference = .01  # relative difference between desired and measured power

        modem = self.modem_chirp_fsk

        bit_energy = self.estimate_energy(modem, number_of_drops,
                                          number_of_frames, self.source.get_bits(
                                              self.params_cfsk.bits_in_frame),
                                          'bit_energy')

        # compare the measured energy with the expected values
        self.assertAlmostEqual(
            bit_energy,
            modem.get_bit_energy(),
            delta=bit_energy *
            relative_difference)

    def test_proper_symbol_energy_calculation(self) -> None:
        """Tests if theoretical symbol energy is calculated correctly"""

        # define test parameters
        number_of_drops = 4
        number_of_frames = 3
        relative_difference = .01  # relative difference between desired and measured power

        modem = self.modem_chirp_fsk

        symbol_energy = self.estimate_energy(modem, number_of_drops,
                                             number_of_frames, self.source.get_bits(
                                                 self.params_cfsk.bits_in_frame),
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
        relative_difference = .01  # relative difference between desired and measured power

        modem = self.modem_chirp_fsk

        power = self.estimate_energy(modem, number_of_drops,
                                     number_of_frames, self.source.get_bits(
                                         self.params_cfsk.bits_in_frame),
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
            data_bits(np.array): Data bits to send
            energy_type(str): what type of energy is to be returned.
                              Allowed values are 'bit_energy', 'symbol_energy' and 'power'

        Returns:
            power_or_energy(float): estimated power or bit/symbol-energy, depending on the value of `energy_type`
        """

        energy_sum = 0
        preamble_energy_sum = 0

        # create an index for the preamble samples
        number_preamble_samples = modem.param.technology.number_pilot_chirps * \
            modem.waveform_generator._samples_in_chirp
        preamble_samples = np.asarray([], dtype=int)
        for idx in range(number_of_frames):
            preamble_samples = np.append(preamble_samples, np.arange(number_preamble_samples) +
                                         idx * modem.waveform_generator.samples_in_frame)

        # calculate average energy
        for idx in range(number_of_drops):
            signal = modem.send(
                modem.waveform_generator.max_frame_length *
                number_of_frames)
            energy = np.sum(np.real(signal)**2 + np.imag(signal)**2)
            preamble_energy = np.sum(
                np.real(signal[:, preamble_samples])**2 +
                np.imag(signal[:, preamble_samples])**2)

            energy_sum += energy
            preamble_energy_sum += preamble_energy

        energy_avg = energy_sum / number_of_drops
        preamble_energy_avg = preamble_energy_sum / number_of_drops
        data_energy = energy_avg - preamble_energy_avg

        power_or_energy = None
        if energy_type == 'bit_energy':
            number_of_bits = modem.param.technology.bits_in_frame * number_of_frames
            power_or_energy = data_energy / number_of_bits
        elif energy_type == 'symbol_energy':
            number_of_symbols = modem.waveform_generator._chirps_in_frame * number_of_frames
            power_or_energy = energy_avg / number_of_symbols
        elif energy_type == 'power':
            number_of_data_samples = number_of_frames * \
                int(modem.param.technology.number_data_chirps *
                    modem.waveform_generator._samples_in_chirp)
            power_or_energy = data_energy / number_of_data_samples
        else:
            raise ValueError("invalid 'energy_type'")

        return power_or_energy


if __name__ == '__main__':
    unittest.main()
