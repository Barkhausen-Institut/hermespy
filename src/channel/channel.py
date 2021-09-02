import numpy as np

from parameters_parser.parameters_channel import ParametersChannel


class Channel:
    """Implements an ideal distortion-less channel.

    It also serves as a base class for all other channel models.

    For MIMO systems, the received signal is the addition of the signal transmitted at all
    antennas.
    The channel will provide `number_rx_antennas` outputs to a signal
    consisting of `number_tx_antennas` inputs. Depending on the channel model,
    a random number generator, given by `rnd` may be needed. The sampling rate is
    the same at both input and output of the channel, and is given by `sampling_rate`
    samples/second.

    Attributes:
        param (ParametersChannel): Channel-related parameters.
        number_tx_antennas(int): Number of transmitting antennas.
        number_rx_antennas(int): Number of receiving antennas.
        random_number_gen(np.random.RandomState):
            Random number generator for random noise etc.
        sampling_rate(float): Sampling rate in Hz.
        ideal_mimo_response(np.ndarray):
            Rows denoting the receiving antennas, columns denoting the samples.
    """

    def __init__(
            self,
            param: ParametersChannel,
            random_number_gen: np.random.RandomState,
            sampling_rate: float) -> None:
        self.param = param

        self.sampling_rate = sampling_rate

        self.number_tx_antennas = param.params_tx_modem.number_of_antennas
        self.number_rx_antennas = param.params_rx_modem.number_of_antennas

        self.random: np.random.RandomState = random_number_gen

        self.ideal_mimo_response = np.ones((self.number_rx_antennas, self.number_tx_antennas))

    def init_drop(self) -> None:
        """Initializes random channel parameters for each drop, if required by model."""
        pass

    def propagate(self, tx_signal: np.ndarray) -> np.ndarray:
        """Modifies the input signal and returns it after channel propagation.

        For the ideal channel in the base class, the MIMO channel is modeled as a matrix of one's.

        If 'tx_signal' is an array of size `number_tx_antennas` X `number_of_samples`,
        then the output `rx_signal` will be an array of size
        `number_rx_antennas` X `number_of_samples`.

        Args:
            tx_signal (np.ndarray): Input signal.

        Returns:
            np.ndarray:
                The distorted signal after propagation. The output depends
                on the channel model employed.
        """
        if tx_signal.ndim == 1:
            tx_signal = np.reshape(tx_signal, (1, -1))

        if tx_signal.ndim != 2 or tx_signal.shape[0] != self.number_tx_antennas:
            raise ValueError(
                'tx_signal must be an array with {:d} rows'.format(
                    self.number_tx_antennas))

        rx_signal = self.ideal_mimo_response @ tx_signal

        return rx_signal * self.param.gain

    def get_impulse_response(self, timestamps: np.array) -> np.ndarray:
        """Calculate the channel impulse responses.

        This method can be used for instance by the transceivers to obtain the channel state
        information.

        Args:
            timestamps (np.ndarray):
                Time instants with length `T` to calculate the response for.

        Returns:
            np.ndarray:
                Impulse response in all `number_rx_antennas` x `number_tx_antennas`.
                4-dimensional array of size `T x number_rx_antennas x number_tx_antennas x (L+1)`
                where `L` is the maximum path delay (in samples). For the ideal
                channell in the base class, `L = 0`.
        """
        impulse_responses = np.tile(
            self.ideal_mimo_response, (timestamps.size, 1, 1))
        impulse_responses = np.expand_dims(impulse_responses, axis=3)

        return impulse_responses * self.param.gain
