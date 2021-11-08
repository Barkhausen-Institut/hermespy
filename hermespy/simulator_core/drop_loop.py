from typing import List

import numpy as np

from parameters_parser.parameters_general import ParametersGeneral
from scenario.scenario import Scenario
from simulator_core.statistics import Statistics
from modem.modem import Modem
from noise.noise import Noise

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "André Noll Barreto"
__email__ = "andre.nollbarreto@barkhauseninstitut.org"
__status__ = "Prototype"


class DropLoop:
    """Implements the main simulation loop.

    Given a certain simulation scenario an object of this class will run the
    simulation drops over all desired SNR values until the requirements
    ( confidence interval, number of drops) are satisfied.
    """

    def __init__(self, parameters: ParametersGeneral,
                 scenario: Scenario) -> None:
        self.params = parameters
        self._scenario = scenario
        self._statistics = Statistics(
            parameters,
            scenario.params,
            scenario.tx_modems,
            scenario.rx_modems)

        # update simulation drop length to consider at least one frame of all
        # modems
        drop_lengths = [
            modem.waveform_generator.max_frame_length for modem in scenario.tx_modems]
        drop_lengths.append(self.params.drop_length)
        self.params.drop_length = max(drop_lengths)

    def run_loop(self) -> Statistics:
        """Simulate the required number of drops over all the desired SNR values

        Returns:
            Statistics: simulation results
        """
        done = False
        while not done:

            self._scenario.init_drop()

            tx_signal = self._transmit_signals()
            rx_signal = self._capture_signals(tx_signal)

            self._statistics.update_error_rate(
                self._scenario.sources, rx_signal)
            done = self.verify_drop_loop()

        return self._statistics

    def _capture_signals(
            self, tx_signal: List[np.ndarray]) -> List[np.ndarray]:
        """Captures transmitted signals at all receivers

        Signal is propagated through channel, resampled, depending on the sampling rate of each receiver, and
        deteriorated with receiver noise.

        Args:
            tx_signal (List[np.ndarray]): List of transmitted signals.

        Returns:
            list[np.ndarray]: Received signals.
        """
        output: List[np.ndarray] = []

        for idx_rx, modem in enumerate(self._scenario.rx_modems):

            rx_signal = np.empty(len(tx_signal), dtype='object')

            for idx_tx, signal in enumerate(tx_signal):
                rx_signal[idx_tx] = self._scenario.channels[idx_rx][idx_tx].propagate(
                    signal)

            rx_signal = self._scenario.rx_samplers[idx_rx].resample(rx_signal)
            self._statistics.update_rx_spectrum(rx_signal, idx_rx)

            self._receive(modem, rx_signal, self._scenario.noise[idx_rx],
                          output, idx_rx)
        return output

    def _receive(self, rx_modem: Modem, rx_signal: List[np.array], noise: Noise, output: List[np.ndarray],
                 idx_rx: int) -> None:
        """Implement a single receiver
        In this method (thermal) noise is added to the received signal, which is then processed to detect the
        transmitted bits.

        Args:
            rx_modem (Modem): Modem to calculate the SNR for.
            rx_signal (List[np.array]): Signal that needs to be noisied.
            noise (Noise): Noise source to add to the signal.
            output (List[np.ndarray]):
                List that contains the 'rx_signal' added with noise given
                different 'snr_values'. Each list element contains the nosied
                signal for modem 'idx_rx'. This list element in turn contains a
                list as well, each item corresponding to a frame. The frame itself
                is a '#snr x #bits' np.ndarray.
            idx_rx (int): Index of modex, required for snr.
        """
        energy: float = None
        if self.params.snr_type == 'EB/N0(DB)':
            energy = rx_modem.paired_tx_modem.get_bit_energy()
        elif self.params.snr_type == 'ES/N0(DB)':
            energy = rx_modem.paired_tx_modem.get_symbol_energy()
        elif not self.params.snr_type == 'CUSTOM':
            raise ValueError('invalid "params.snr_type"')

        rx_signal_snr_added: List[np.ndarray] = []

        for snr in self._statistics.get_snr_list(idx_rx):
            rx_signal_noisy, noise_var = noise.add_noise(rx_signal, snr, energy)
            rx_signal_noisy_in_frames = rx_modem.receive(
                rx_signal_noisy, noise_var)  # a list with `#frames` elements

            if snr != self._statistics.get_snr_list(idx_rx)[0]:
                for idx_frame, frame in enumerate(rx_signal_noisy_in_frames):
                    rx_signal_snr_added[idx_frame] = np.vstack(
                        (rx_signal_snr_added[idx_frame], frame))
            else:
                # ensure that a frame is always in the shape of #no_snrs x #samples
                # if #no_snrs==1, then the shape needs to be (1, #no_samples)
                # this is guaranteed by the reshape
                rx_signal_snr_added = [np.reshape(frame, (1, -1))
                                       for frame in rx_signal_noisy_in_frames]

        output.extend([rx_signal_snr_added])

    def _transmit_signals(self) -> List[np.ndarray]:
        """Transmits signals.

        Returns:
            list(np.ndarray):
                List of length `#tx_modems` containing np.ndarray of sent signals.
        """

        tx_signal = self._scenario.transmit(self.params.drop_length)

        for source, modem in zip(self._scenario.sources,
                                 self._scenario.tx_modems):
            tx_signal.append(modem.send(self.params.drop_length))

        self._statistics.update_tx_spectrum(tx_signal)

        return tx_signal

    def verify_drop_loop(self) -> bool:
        """Verifies if drop loop is completed.

        Returns:
            boolean: True: Done, false: not done.
        """
        done = False

        for rx_modem_idx in range(len(self._scenario.rx_modems)):
            done = done | np.all(
                self._statistics.get_snr_list(rx_modem_idx) == 0)

        if self._statistics.__num_drops >= self.params.max_num_drops:
            done = True

        return done
