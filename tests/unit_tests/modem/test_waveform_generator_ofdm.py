import unittest
import shutil
import os
from typing import List

import numpy as np
from scipy.io import loadmat

import hermes
from channel.multipath_fading_channel import MultipathFadingChannel
from source.bits_source import BitsSource
from parameters_parser.parameters_channel import ParametersChannel
from parameters_parser.parameters_ofdm import (
    ParametersOfdm, ResourceType, OfdmSymbolConfig, GuardInterval,
    ResourcePattern, MultipleRes
)
from modem.waveform_generator_ofdm import WaveformGeneratorOfdm
from parameters_parser.parameters_rx_modem import ParametersRxModem
from parameters_parser.parameters_tx_modem import ParametersTxModem


class TestWaveformGeneratorOfdm(unittest.TestCase):

    def setUp(self) -> None:
        self.rnd = np.random.RandomState(42)
        self.source = BitsSource(self.rnd)

        self.FFT_SIZE = 2048
        self.NO_TX_ANTENNAS = 1

        self.params = ParametersOfdm()
        self.params.fft_size = self.FFT_SIZE
        self.params.precoding = "NONE"
        self.params.channel_estimation = "IDEAL"
        self.params.equalization = "ZF"
        self.params.number_occupied_subcarriers = 1200
        self.params.frame_guard_interval = .001
        self.params.subcarrier_spacing = 15e3
        self.params.cp_ratio = np.array([0.07, 0.073])
        self.params.number_tx_antennas = self.NO_TX_ANTENNAS
        self.params._ofdm_symbol_res_str = "100(r11d), 100(6r6d)"
        self.params._frame_structure_str = "g, 10(12),1"
        self.params._cp_lengths_str = "c1,c2"
        self.params.modulation_order = 4
        self.params.sampling_rate = self.FFT_SIZE * self.params.subcarrier_spacing
        self.params.mimo_scheme = "NONE"
        self.params.dc_suppression = True
        self.params._check_params()
        self.params.reference_symbols = np.array([1, 2, 3, 4])

        self.O = WaveformGeneratorOfdm(self.params)

    @classmethod
    def tearDownClass(cls) -> None:
        os.environ["MPLBACKEND"] = ""

    def test_dc_suppression_activated(self) -> None:
        initial_index = 1448
        resource_element_mapping = np.arange(1448, 2048)
        final_index = 600
        resource_element_mapping = np.append(resource_element_mapping, np.arange(1, 601))

        np.testing.assert_array_equal(
            self.O._resource_element_mapping,
            resource_element_mapping
        )

    def test_dc_suppresion_deactivated(self) -> None:
        self.params.dc_suppression = False
        O = WaveformGeneratorOfdm(self.params)

        initial_index = 1448
        resource_element_mapping = np.arange(1448, 2048)
        final_index = 600
        resource_element_mapping = np.append(resource_element_mapping, np.arange(600))

        np.testing.assert_array_equal(
            O._resource_element_mapping,
            resource_element_mapping
        )

    def test_samples_in_frame_calculation(self) -> None:
        cp_samples = [
            int(np.around(self.params.cp_ratio[0]*self.params.fft_size)),
            int(np.around(self.params.cp_ratio[1]*self.params.fft_size))
        ]
        gi_samples = int(np.around(
            self.params.frame_guard_interval
            * self.params.subcarrier_spacing
            * self.params.fft_size))
        expected_samples = (
            10 * (
                    cp_samples[0] + self.params.fft_size
                    + cp_samples[1] + self.params.fft_size
                 )
               + cp_samples[0] + self.params.fft_size + gi_samples
        )

        O = WaveformGeneratorOfdm(self.params)
        self.assertEqual(expected_samples, O.samples_in_frame)

    def test_resource_elements_allocated_correctly(self) -> None:
        """ test if reference and data symbols are allocated in the right positions"""

        # Resource-element grid of dimension N_symb x N_sc
        time_frequency_grid_ref = np.zeros((21, 1200, 1))
        time_frequency_grid_data = np.zeros((21, 1200, 1))
        ref_idx_symbol_1 = np.tile(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 100)
        data_idx_symbol_1 = np.logical_not(ref_idx_symbol_1)
        ref_idx_symbol_2 = np.tile(np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]), 100)
        data_idx_symbol_2 = np.logical_not(ref_idx_symbol_2)

        for idx in range(10):
            time_frequency_grid_ref[2*idx, :, 0] = ref_idx_symbol_1
            time_frequency_grid_ref[2*idx + 1, :, 0] = ref_idx_symbol_2
            time_frequency_grid_data[2*idx, :, 0] = data_idx_symbol_1
            time_frequency_grid_data[2*idx + 1, :, 0] = data_idx_symbol_2
        time_frequency_grid_ref[20, :, 0] = ref_idx_symbol_1
        time_frequency_grid_data[20, :, 0] = data_idx_symbol_1

        np.testing.assert_array_almost_equal(time_frequency_grid_ref, self.O.reference_frame != 0)
        np.testing.assert_array_almost_equal(time_frequency_grid_data, self.O.data_frame_indices)

    def test_time_indices_allocated_correctly(self) -> None:

        all_indices = np.concatenate((self.O.prefix_time_indices, self.O.guard_time_indices, self.O.data_time_indices))
        all_indices = np.sort(all_indices)
        np.testing.assert_equal(all_indices, np.arange(self.O._samples_in_frame_no_oversampling, dtype=int))

    def test_ofdmSymbolCreation_timeDomain_noSamples(self) -> None:
        frame = np.random.rand(self.O._number_ofdm_symbols, self.O.param.number_occupied_subcarriers,
                               self.O.param.number_tx_antennas)
        number_of_samples_ofdm_symbols = self.O.param.fft_size + np.around(self.O.param.fft_size * self.params.cp_ratio)
        number_of_samples_guard = int(np.around(self.O.param.frame_guard_interval * self.O.sampling_rate))

        expected_number_of_samples = (11 * int(number_of_samples_ofdm_symbols[0]) +
                                      10 * int(number_of_samples_ofdm_symbols[1]) + number_of_samples_guard)

        time_domain_frame = self.O.create_ofdm_frame_time_domain(frame)

        self.assertEqual(expected_number_of_samples, time_domain_frame.size)

    def test_frameCreation_startsWithGuardIntervalZeros(self) -> None:
        data_bits = np.random.randint(2, size=self.O.bits_in_frame)
        signal, _, _ = self.O.create_frame(0, data_bits)
        np.testing.assert_array_almost_equal(
            np.zeros(
                (self.NO_TX_ANTENNAS, self.params.frame_structure[0].no_samples),
                dtype=complex),
            signal[:, :self.params.frame_structure[0].no_samples])

    def test_discardingReferenceSymbols(self) -> None:
        res_types: List[ResourcePattern] = [
            ResourcePattern(
                MultipleRes=[MultipleRes(ResourceType.REFERENCE, 1)],
                number=1),
            ResourcePattern(
                MultipleRes=[MultipleRes(ResourceType.DATA, 1)],
                number=2
            )]

        ofdm_symbol_config = OfdmSymbolConfig(resource_types=res_types)
        ofdm_symbol_resources = np.array([[0, 1, 2],
                                          [0, 1, 2]])

        np.testing.assert_array_almost_equal(
            ofdm_symbol_resources[:, 1:],
            self.O.discard_reference_symbols(ofdm_symbol_config, ofdm_symbol_resources)
        )

    def test_MMSE_lower_BER_than_ZF(self) -> None:
        """Checks if MMSE is actually performed by checking if BER is lower for
        low SNRs than for ZF equalization. """
        settings_dir_ZF = os.path.join(
            "tests", "unit_tests", "modem", "res", "settings_ZF")
        settings_dir_MMSE = os.path.join(
            "tests", "unit_tests", "modem", "res", "settings_MMSE")

        results_dir = os.path.join(settings_dir_ZF, "..", "results")
        arguments_hermes = ["-p", settings_dir_ZF, "-o", results_dir]

        import matplotlib.pyplot as plt
        plt.switch_backend("agg")
        hermes.hermes(arguments_hermes)

        # get ZF results at first
        results_simulation_ZF = loadmat(
            os.path.join(results_dir, "statistics.mat"))
        shutil.rmtree(results_dir)

        # now MMSE results
        arguments_hermes = ["-p", settings_dir_MMSE, "-o", results_dir]
        hermes.hermes(arguments_hermes)
        results_simulation_MMSE = loadmat(
            os.path.join(results_dir, "statistics.mat"))
        shutil.rmtree(results_dir)

        # BER of MMSE should be lower for low SNRs than ZF
        ber_mean_ZF = results_simulation_ZF["ber_mean"]
        ber_mean_MMSE = results_simulation_MMSE["ber_mean"]

        np.testing.assert_array_less(ber_mean_MMSE, ber_mean_ZF)

    def test_channel_estimation_ideal_noOversampling(self) -> None:
        """
        Test ideal channel estimation for a SISO system.
        In this test we verify if the 'WaveformGeneratorOfdm.channel_estimation' method returns the expected frequency
        response from a known channel.
        """
        # create a channel
        rx_modem_params = ParametersRxModem()
        tx_modem_params = ParametersTxModem()

        channel_param = ParametersChannel(rx_modem_params, tx_modem_params)
        channel_param.multipath_model = "EXPONENTIAL"
        channel_param.tap_interval = 1 / self.params.sampling_rate
        channel_param.rms_delay = 1 / self.params.subcarrier_spacing * \
            self.params.cp_ratio[1] / 30
        channel_param.velocity = np.asarray([10, 0, 0])
        channel_param.attenuation_db = 0

        channel_param.check_params()

        channel = MultipathFadingChannel(
            channel_param,
            np.random.RandomState(),
            self.params.sampling_rate,
            50)
        channel.init_drop()

        frame_symbols_idxs = np.asarray([])
        idx = 0
        for element in self.params.frame_structure:
            if isinstance(element, OfdmSymbolConfig):
                frame_symbols_idxs = np.append(frame_symbols_idxs, idx)
                idx += element.no_samples + element.cyclic_prefix_samples
            else:
                idx += element.no_samples

        frame_symbols_idxs *= self.params.oversampling_factor

        sampled_channel = channel.get_impulse_response(
            frame_symbols_idxs / self.params.sampling_rate)

        expected_channel_in_frequency = np.fft.fft(
            sampled_channel, n=self.params.fft_size)
        remove_idx = np.arange(int(self.O.param.fft_size / 2),
                               int(self.params.fft_size - self.O.param.fft_size / 2))
        expected_channel_in_frequency = np.transpose(
            np.squeeze(expected_channel_in_frequency))
        expected_channel_in_frequency = np.delete(
            expected_channel_in_frequency, remove_idx, 0)

        self.O.set_channel(channel)
        self.O.param.channel_estimation = 'IDEAL'
        estimated_channel = self.O.channel_estimation(
            None, 0)
        np.testing.assert_allclose(
            expected_channel_in_frequency,
            np.squeeze(estimated_channel))

    def test_channel_estimation_ideal_preamble(self) -> None:
        """
        Test ideal preamble-based channel estimation for a SISO system.
        In this test we verify if the 'WaveformGeneratorOfdm.channel_estimation' method returns the expected frequency
        response from a known channel at the beginning of a frame.
        """
        self._test_channel_estimation_ideal_reference("IDEAL_PREAMBLE")

    def test_channel_estimation_ideal_postamble(self) -> None:
        """
        Test ideal preamble-based channel estimation for a SISO system.
        In this test we verify if the 'WaveformGeneratorOfdm.channel_estimation' method returns the expected frequency
        response from a known channel at the end of a frame.
        """
        self._test_channel_estimation_ideal_reference("IDEAL_POSTAMBLE")

    def test_channel_estimation_ideal_midamble(self) -> None:
        """
        Test ideal preamble-based channel estimation for a SISO system.
        In this test we verify if the 'WaveformGeneratorOfdm.channel_estimation' method returns the expected frequency
        response from a known channel in the middle of a frame.
        """
        self._test_channel_estimation_ideal_reference("IDEAL_MIDAMBLE")

    def _test_channel_estimation_ideal_reference(
            self, position_in_frame: str) -> None:
        """
        Test ideal reference-signal-based channel estimation for a single reference in an UL frame.
        In this test we verify if the 'WaveformGeneratorOfdm.channel_estimation' method returns the expected frequency
        response from a known channel at a given position in the frame.

        Args:
            position_in_frame(str): indicates at which point the reference signal was considered. The following values
                are accepted:
                    "IDEAL_PREAMBLE" - reference signal is at the beginning of the frame
                    "IDEAL_POSTAMBLE" - reference signal is at the end of the frame
                    "IDEAL_MIDAMBLE" - reference signal is exactly in thevmiddle of the frame
        """

        # create a channel
        rx_modem_params = ParametersRxModem()
        tx_modem_params = ParametersTxModem()

        channel_param = ParametersChannel(rx_modem_params, tx_modem_params)
        channel_param.multipath_model = "COST259"
        channel_param.cost_259_type = "TYPICAL_URBAN"
        channel_param.attenuation_db = 3
        channel_param.velocity = np.asarray([np.random.random() * 20, 0, 0])

        channel_param.check_params()

        channel = MultipathFadingChannel(
            channel_param,
            np.random.RandomState(),
            self.params.sampling_rate,
            50)
        channel.init_drop()

        if position_in_frame == "IDEAL_PREAMBLE":
            timestamp = 0.
        elif position_in_frame == "IDEAL_POSTAMBLE":
            timestamp = self.O.samples_in_frame / \
                self.O.param.sampling_rate
        elif position_in_frame == "IDEAL_MIDAMBLE":
            timestamp = self.O.samples_in_frame / \
                self.O.param.sampling_rate * .5
        else:
            raise ValueError("invalid 'position_in_frame'")

        sampled_channel = channel.get_impulse_response(np.asarray([timestamp]))

        long_fft_size = self.O.param.fft_size * \
            self.params.oversampling_factor
        expected_channel_in_frequency = np.fft.fft(
            sampled_channel, n=long_fft_size)
        remove_idx = np.arange(int(self.O.param.fft_size / 2),
                               int(long_fft_size - self.O.param.fft_size / 2))
        expected_channel_in_frequency = np.transpose(
            np.squeeze(expected_channel_in_frequency, axis=(1, 2)))
        expected_channel_in_frequency = np.delete(
            expected_channel_in_frequency, remove_idx, 0)

        self.O.set_channel(channel)
        self.O.param.channel_estimation = position_in_frame
        estimated_channel = self.O.channel_estimation(None, 0)
        number_of_channel_samples = estimated_channel.shape[3]
        expected_channel_in_frequency = np.tile(
            expected_channel_in_frequency, number_of_channel_samples)
        np.testing.assert_allclose(
            expected_channel_in_frequency,
            np.squeeze(estimated_channel))


if __name__ == '__main__':
    unittest.main()




