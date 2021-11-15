import unittest.mock
from unittest.mock import patch, Mock
from typing import List

import numpy as np

from hermespy.simulator_core import DropLoop
from hermespy.simulator_core import Statistics


class TestDropLoop(unittest.TestCase):

    def setUp(self) -> None:
        # initialize drop loop with mocks
        self.patcher = patch.object(
            Statistics, '__init__', lambda a, b, c, d, e: None)
        _ = self.patcher.start()

        mock_params_general = Mock()
        mock_params_general.max_num_simulation_iterations = 3
        mock_params_general.drop_length = 0

        mock_scenario = Mock()
        mock_scenario.rx_modems = [Mock]
        mock_scenario.tx_modems = [Mock]
        mock_scenario.tx_modems[0].waveform_generator = Mock()
        mock_scenario.tx_modems[0].waveform_generator.max_frame_length = .1

        self.drop_loop = DropLoop(mock_params_general, mock_scenario)
        self.drop_loop._statistics = Mock()
        self.drop_loop._statistics.num_drops = 2

    def tearDown(self) -> None:
        self.patcher.stop()

    def test_drop_loop_end_command_on_zero_snr_list(self) -> None:
        self.drop_loop._statistics.get_snr_list.return_value = np.array([])  # type: ignore

        self.assertTrue(self.drop_loop.verify_drop_loop())

    def test_drop_loop_end_command_on_exceeding_num_drops(self) -> None:
        self.drop_loop._scenario.rx_modems = []

        self.drop_loop.params.max_num_simulation_iterations = self.drop_loop._statistics.__num_drops - 1

        self.assertTrue(self.drop_loop.verify_drop_loop())

    def test_drop_loop_continue_command(self) -> None:
        self.drop_loop._statistics.get_snr_list.return_value = np.array([  # type: ignore
                                                                        1, 2, 3])

        self.assertFalse(self.drop_loop.verify_drop_loop())

    def test_length_of_transmitted_signals(self) -> None:
        no_tx_modems = 2

        _ = self.setup_transmit_signals(no_tx_modems)

        # perform actual test
        tx_signal = self.drop_loop._transmit_signals()

        self.assertEqual(no_tx_modems, len(tx_signal))

    def test_if_tx_signal_is_actually_sent(self) -> None:
        no_tx_modems = 2
        mock_tx_modem = self.setup_transmit_signals(no_tx_modems)

        _ = self.drop_loop._transmit_signals()
        self.assertEqual(
            no_tx_modems,
            mock_tx_modem.send.call_count,
            "The amount of send method calls must be equal to the number of sending modems."
        )

    def test_tx_spectrum_update(self) -> None:
        _ = self.setup_transmit_signals(2)

        tx_signal = self.drop_loop._transmit_signals()

        self.drop_loop._statistics.update_tx_spectrum.assert_called_once_with(  # type: ignore
            tx_signal)

    def test_noise_addition_proper_method_call(self) -> None:
        # define input parameters
        self.drop_loop.params.snr_type = 'EB/N0(DB)'
        self.drop_loop._statistics.get_snr_list.return_value = [1, 2, 3]  # type: ignore
        rx_signal = [np.ones((10,))]

        mock_rx_modem = Mock()
        mock_rx_modem.paired_tx_modem.get_bit_energy.return_value = 3
        mock_rx_modem.receive.return_value = rx_signal

        mock_noise = Mock()
        mock_noise.add_noise.return_value = (rx_signal, 1)

        output: List[np.ndarray] = []

        self.drop_loop._receive(mock_rx_modem, rx_signal, mock_noise, output, 0)

        # check method calls
        mock_rx_modem.paired_tx_modem.get_bit_energy.assert_called_once()
        mock_noise.add_noise.assert_any_call(rx_signal, 1, 3)

    def test_proper_method_calls_receive_signals(self) -> None:
        no_rx_modems = 3
        no_tx_signals = 2
        no_channels = no_tx_signals * no_rx_modems

        # create mocks
        mock_channel = Mock()
        mock_channel.propagate.return_value = np.ones((10,))
        self.drop_loop._scenario.channels = [
            [mock_channel] * no_tx_signals] * no_rx_modems

        mock_rx_sampler = Mock()
        mock_rx_sampler.resample.return_value = np.ones((10,))
        self.drop_loop._scenario.rx_samplers = [mock_rx_sampler] * no_rx_modems

        self.drop_loop._scenario.rx_modems = [Mock()] * no_rx_modems

        tx_signal = [np.ones((10, 1))] * no_tx_signals
        self.drop_loop._receive = Mock()  # type: ignore
        self.drop_loop._receive.return_value = None

        self.drop_loop._scenario.noise = [Mock()] * no_rx_modems

        # call actual function
        _ = self.drop_loop._capture_signals(tx_signal)

        self.assertEqual(mock_channel.propagate.call_count, no_channels)
        self.assertEqual(mock_rx_sampler.resample.call_count, no_rx_modems)
        self.assertEqual(self.drop_loop._receive.call_count, no_rx_modems)

    def setup_transmit_signals(self, no_tx_modems: int) -> Mock:
        # mock some stuff
        self.drop_loop._statistics = Mock()
        self.drop_loop._statistics.update_tx_spectrum.return_value = None

        self.drop_loop._scenario.sources = [Mock()] * no_tx_modems

        mock_tx_modem = Mock()
        mock_tx_modem.send.return_value = np.ndarray((3, 2))
        self.drop_loop._scenario.tx_modems = [mock_tx_modem] * no_tx_modems

        return mock_tx_modem


if __name__ == "__main__":
    #os.chdir("../../..")
    unittest.main()
