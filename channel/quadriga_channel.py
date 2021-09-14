import numpy as np

from modem import Modem
from channel.channel import Channel
from parameters_parser.parameters_channel import ParametersChannel
from channel.quadriga_interface import QuadrigaInterface


class QuadrigaChannel(Channel):
    def __init__(self, modem_tx: Modem, modem_rx: Modem, sampling_rate: float,
                 rng: np.random.RandomState, quadriga_interface: QuadrigaInterface) -> None:
        """Maps the output of the QuadrigaInterface to fit into hermes software architecture.

        Args:
            modem_tx(Modem): Transmitter modem.
            modem_rx(Modem): Receiver Modem.
            sampling_rate(float): Sampling rate in Hertz.
            rng(np.random.RandomState): Random number generator.
            quadriga_interface(QuadrigaInterface): Interface to the actual
                quadriga backend.
        """
        self.params = ParametersChannel(modem_rx.param, modem_tx.param)
        self.params.gain = 1
        super().__init__(self.params, rng, sampling_rate)

        self._quadriga_interface = quadriga_interface
        self._quadriga_interface.update_quadriga_parameters(modem_tx, modem_rx)

        self._modem_tx = modem_tx
        self._modem_rx = modem_rx

    def init_drop(self) -> None:
        """Initializes random channel parameters for each drop"""
        seed = self.random.random_sample(1)
        self._quadriga_interface.init_drop(seed)

    def propagate(self, tx_signal: np.ndarray) -> np.ndarray:
        """Modifies the input signal and returns it after channel propagation.

        Args:
            tx_signal(np.ndarray):
                array of size `number_tx_antennas` X `number_of_samples`.

        Returns:
            np.ndarray:
                Output signal after propagation through quadriga channel
                with `number_rx_antennas` rows and `samples` columns.
        """
        cir, tau = self._quadriga_interface.get_impulse_response(
            self._modem_tx, self._modem_rx)

        if tx_signal.ndim == 1:
            tx_signal = np.reshape(tx_signal, (1, -1))

        number_of_samples_in = tx_signal.shape[1]
        self.max_delay_in_samples = np.around(
            np.max(tau) * self.sampling_rate).astype(int)
        number_of_samples_out = number_of_samples_in + self.max_delay_in_samples

        rx_signal = np.zeros(
            (self.number_rx_antennas, number_of_samples_out), dtype=complex)

        for rx_antenna in range(self.number_rx_antennas):
            rx_signal_ant = np.zeros(number_of_samples_out, dtype=complex)

            for tx_antenna in range(self.number_tx_antennas):
                tx_signal_ant = tx_signal[tx_antenna, :]

                # of dimension, #paths x #snap_shots, along the third dimension are the samples
                # choose first snapshot, i.e. assume static
                cir_txa_rxa = cir[rx_antenna, tx_antenna, :, 0]
                tau_txa_rxa = tau[rx_antenna, tx_antenna, :, 0]

                # cir from quadriga corresponds to impulse_response_siso of
                # MultiPathFadingChannel

                time_delay_in_samples_vec = np.around(
                    tau_txa_rxa * self.sampling_rate).astype(int)

                for delay_idx, delay_in_samples in enumerate(
                        time_delay_in_samples_vec):
                    padding = self.max_delay_in_samples - delay_in_samples

                    path_response_at_delay = cir_txa_rxa[delay_idx]

                    signal_path = tx_signal_ant * path_response_at_delay
                    signal_path = np.concatenate((
                        np.zeros(delay_in_samples),
                        signal_path,
                        np.zeros(padding)))

                    rx_signal_ant += signal_path

            rx_signal[rx_antenna, :] = rx_signal_ant

        return rx_signal * self.param.gain

    def get_impulse_response(self, timestamps: np.array) -> np.ndarray:
        """Calculates the channel impulse response.

        This method can be used for instance by the transceivers to obtain the
        channel state information.

        Args:
            timestamps (np.array):
                Time instants with length T to calculate the response for.

        Returns:
            np.ndarray:
                Impulse response in all 'number_rx_antennas' x 'number_tx_antennas'
                channels at the time instants given in vector 'timestamps'.
                `impulse_response` is a 4D-array, with the following dimensions:
                1- sampling instants, 2 - Rx antennas, 3 - Tx antennas, 4 - delays
                (in samples)
                The shape is T x number_rx_antennas x number_tx_antennas x (L+1)
        """
        cir, tau = self._quadriga_interface.get_impulse_response(
            self._modem_tx, self._modem_rx)
        self.max_delay_in_samples = np.around(
            np.max(tau) * self.sampling_rate).astype(int)

        impulse_response = np.zeros((timestamps.size,
                                     self.number_rx_antennas,
                                     self.number_tx_antennas,
                                     self.max_delay_in_samples + 1), dtype=complex)

        for tx_antenna in range(self.number_tx_antennas):
            for rx_antenna in range(self.number_rx_antennas):
                # of dimension, #paths x #snap_shots, along the third dimension are the samples
                # choose first snapshot, i.e. assume static
                cir_txa_rxa = cir[rx_antenna, tx_antenna, :, 0]
                tau_txa_rxa = tau[rx_antenna, tx_antenna, :, 0]

                time_delay_in_samples_vec = np.around(
                    tau_txa_rxa * self.sampling_rate).astype(int)

                for delay_idx, delay_in_samples in enumerate(
                        time_delay_in_samples_vec):

                    impulse_response[:, rx_antenna, tx_antenna, delay_in_samples] += (
                        cir_txa_rxa[delay_idx]
                    )
        return impulse_response
