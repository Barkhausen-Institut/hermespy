from coding import EncoderFactory
import numpy as np
import unittest.mock
from unittest.mock import patch, Mock

from numpy import random as rnd
import copy

from modem.modem import Modem
from parameters_parser.parameters_tx_modem import ParametersTxModem
from parameters_parser.parameters_psk_qam import ParametersPskQam
from source.bits_source import BitsSource
from coding import EncoderManager
from parameters_parser.parameters_repetition_encoder import ParametersRepetitionEncoder


class TestModem(unittest.TestCase):

    def setUp(self) -> None:
        # do some general setup that is required for creating the Modem
        rng_src = rnd.RandomState(42)
        rng_hw = rnd.RandomState(43)
        self.params_psk_am = ParametersPskQam()
        self.params_psk_am.modulation_order = 2
        self.params_psk_am.symbol_rate = 125e3
        self.params_psk_am.bandwidth = self.params_psk_am.symbol_rate
        self.params_psk_am.filter_type = 'ROOT_RAISED_COSINE'
        self.params_psk_am.roll_off_factor = .5
        self.params_psk_am.oversampling_factor = 8
        self.params_psk_am.filter_length_in_symbols = 16
        self.params_psk_am.sampling_rate = 3e3
        self.params_psk_am.number_preamble_symbols = 0
        self.params_psk_am.number_postamble_symbols = 0
        self.params_psk_am.number_data_symbols = 100
        self.params_psk_am.guard_interval = 1e-3
        self.params_psk_am.bits_per_symbol = int(
            np.log2(self.params_psk_am.modulation_order))
        self.params_psk_am.bits_in_frame = self.params_psk_am.number_data_symbols * \
            self.params_psk_am.bits_per_symbol
        self.source = BitsSource(rng_src)

        # generate patch for ParametersModem
        self.patch_parameters_modem = patch(
            'parameters_parser.parameters_modem.ParametersModem')
        MockParametersModem = self.patch_parameters_modem.start()
        mock_parameters_modem = MockParametersModem()

        # create modem; ParametersModem needs to be mocked since it is abstract
        mock_parameters_modem.encoding_type = ["REPETITION"]
        mock_parameters_modem.encoding_params = [ParametersRepetitionEncoder()]
        mock_parameters_modem.technology = self.params_psk_am
        mock_parameters_modem.crc_bits = 0

        self.modem = Modem(mock_parameters_modem, self.source, rng_hw, rng_src)

        # assign necessary mocks
        self.mock_rf_chain = Mock()
        self.mock_waveform_generator = Mock()
        self.mock_waveform_generator.get_bit_energy.return_value = 1
        self.mock_waveform_generator.get_symbol_energy.return_value = 1
        self.modem.rf_chain = self.mock_rf_chain
        self.modem.waveform_generator = self.mock_waveform_generator

    def tearDown(self) -> None:
        self.patch_parameters_modem.stop()

    def test_scaling_of_energy_with_code_ratio(self) -> None:
        N = 2
        K = 1
        params_encoder = ParametersRepetitionEncoder()
        params_encoder.encoded_bits_n = N
        params_encoder.data_bits_k = K

        encoder_factory = EncoderFactory()
        encoder = encoder_factory.get_encoder(
            params_encoder, "repetition", self.params_psk_am.bits_in_frame, np.random.RandomState())
        encoder_manager = EncoderManager()
        encoder_manager.add_encoder(encoder)
        self.modem.encoder_manager = encoder_manager

        symbol_energy_n2k1 = self.modem.get_symbol_energy()
        bit_energy_n2k1 = self.modem.get_bit_energy()

        params_encoder.encoded_bits_n = 1
        symbol_energy_n1k1 = self.modem.get_symbol_energy()
        bit_energy_n1k1 = self.modem.get_bit_energy()

        self.assertEqual(symbol_energy_n1k1 * N / K, symbol_energy_n2k1)
        self.assertEqual(bit_energy_n1k1 * N / K, bit_energy_n2k1)

    def test_send(self):
        """Tests if RfChain.send and waveform_generator.create_frame are properly called."""

        # define method parameters
        DROP_LENGTH = 0.1

        TIMESTAMP = 30000
        INITIAL_SAMPLE_NUM = 0
        FRAME = np.ones((1, int(np.ceil(DROP_LENGTH * self.params_psk_am.sampling_rate))))

        # define return values
        self.mock_waveform_generator.create_frame = Mock(
            return_value=(FRAME, TIMESTAMP, INITIAL_SAMPLE_NUM)
        )

        _ = self.modem.send(DROP_LENGTH)
        np.testing.assert_array_equal(
            self.mock_waveform_generator.create_frame.call_args[0][1],
            self.source.bits_in_drop[0]
        )
        self.assertEqual(
            self.mock_waveform_generator.create_frame.call_args[0][0],
            0
        )
        np.testing.assert_array_equal(
            self.mock_rf_chain.send.call_args[0][0],
            FRAME
        )

    def test_receive(self) -> None:
        """Tests if RfChain.receive and WaveformGenerator.receive are properly called."""

        # define method parameters
        SIGNAL = np.zeros((1, 3))
        RET_VAL_RF_CHAIN = SIGNAL
        NOISE = 42
        # define return values
        self.mock_rf_chain.receive = Mock(return_value=RET_VAL_RF_CHAIN)
        self.mock_waveform_generator.receive_frame = Mock(
            return_value=([np.array([0, 0, 0])], np.zeros(0))
        )
        self.mock_waveform_generator.db_to_linear = Mock(return_value=NOISE)

        other_modem = copy.copy(self.modem)
        self.modem.paired_tx_modem = other_modem
        _ = self.modem.receive(SIGNAL, NOISE)
        np.testing.assert_array_equal(
            self.mock_rf_chain.receive.call_args[0][0],
            SIGNAL
        )
        np.testing.assert_array_equal(
            self.mock_waveform_generator.receive_frame.call_args[0][0],
            SIGNAL
        )
        self.assertEqual(
            self.mock_waveform_generator.receive_frame.call_args[0][1], 0)
        self.assertEqual(
            self.mock_waveform_generator.receive_frame.call_args[0][2], NOISE)

    def test_get_bit_energy(self) -> None:
        _ = self.modem.get_bit_energy()
        self.mock_waveform_generator.get_bit_energy.assert_called_once()

    def test_get_symbol_energy(self) -> None:
        _ = self.modem.get_symbol_energy()
        self.mock_waveform_generator.get_symbol_energy.assert_called_once()

    def test_set_channel(self) -> None:
        mock_channel = Mock()
        self.modem.set_channel(mock_channel)
        self.mock_waveform_generator.set_channel.assert_called_once_with(
            mock_channel)

    def test_tx_power(self) -> None:
        """
        Tests if transmit power is set up correctly.
        In this test a modem is created with a given desired power, and it is checked whether the resulting signal has
        the expected power.
        """

        desired_power_db = 5

        number_of_drops = 5
        number_of_frames = 2

        relative_difference = .01  # relative difference between desired and measured power

        param_tech = ParametersPskQam()

        frame_duration = 1e-3
        param_tech.modulation_order = 16
        param_tech.symbol_rate = 125e6
        param_tech.bandwidth = param_tech.symbol_rate
        param_tech.filter_type = 'ROOT_RAISED_COSINE'
        param_tech.roll_off_factor = .3
        param_tech.oversampling_factor = 8
        param_tech.filter_length_in_symbols = 16
        param_tech.number_preamble_symbols = 0
        param_tech.number_postamble_symbols = 0
        param_tech.number_data_symbols = int(
            frame_duration * param_tech.symbol_rate)
        param_tech.guard_interval = .1e-3
        param_tech.bits_per_symbol = int(np.log2(param_tech.modulation_order))
        param_tech.bits_in_frame = param_tech.bits_per_symbol * \
            param_tech.number_data_symbols
        param_tech.sampling_rate = param_tech.symbol_rate * param_tech.oversampling_factor

        param = ParametersTxModem()

        param.technology = param_tech
        param.position = np.asarray([0, 0, 0])
        param.velocity = np.asarray([0, 0, 0])
        param.number_of_antennas = 1
        param.carrier_frequency = 1e9
        param.tx_power = 10 ** (desired_power_db / 10)
        param.encoding_type = ["REPETITION"]
        param.encoding_params = [ParametersRepetitionEncoder()]

        source = BitsSource(np.random.RandomState())

        modem = Modem(param, source, np.random.RandomState(), np.random.RandomState(), tx_modem=None)

        power_sum = 0

        for idx in range(number_of_drops):
            signal = modem.send(
                frame_duration * number_of_frames
            )
            power = np.sum(
                np.real(signal)**2 + np.imag(signal)**2) / signal.size

            power_sum += power

        power_avg = power_sum / number_of_drops

        # compensate for guard interval
        power_avg = power_avg * \
            (frame_duration + param_tech.guard_interval) / frame_duration

        self.assertAlmostEqual(
            power_avg,
            param.tx_power,
            delta=param.tx_power *
            relative_difference)


if __name__ == '__main__':
    unittest.main()
