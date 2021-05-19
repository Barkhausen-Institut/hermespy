import numpy as np
import scipy

from channel.channel import Channel
from parameters_parser.parameters_channel import ParametersChannel


class MultipathFadingChannel(Channel):
    """Implements a stochastic fading multipath channel.

    For MIMO systems, the received signal is the addition of the signal transmitted
    at all antennas.
    The channel model is defined in the parameters, which should have the following fields:
    - param.delays - numpy.ndarray containing the delays of all significant paths (in s)
    - param.power_delay_profile - numpy.ndarray containing the average power of each path
    (in linear scale). It must have the same number of elements as 'param.delays'
    - param.k_factor_rice: numpy.ndarray containing the K-factor of the Ricean
    distribution for each path (in linear scale). It must have the same number of
    elements as 'param.delays'

    The channel is time-variant, with the auto-correlation depending on  its maximum
    'doppler_frequency' (in Hz).
    Realizations in different drops are independent.

    The model supports multiple antennas, and the correlation among different antennas can be
    specified in parameters 'tx_cov_matrix' and 'rx_cov_matrix', for the both transmitter and
    at the receiver side, respectively.
    They both must be Hermitian, positive definite square matrices.

    Rayleigh/Rice fading and uncorrelated scattering is considered. Fading follows Jakes'
    Doppler spectrum, using the simulation approach from:

        `C. Xiao, Zheng, Y. and Beaulieu, N. "Novel sum-of-sinusoids simulation models for
        rayleigh and rician fading channels. IEEE Trans. Wireless Comm., 5(12), 2006`

    which is based on the sum of sinusoids with random phases.

    If the delays are not multiple of the sampling interval, then sinc-based interpolation is
    considered.

    Antenna correlation considers the Kronecker model, as described in:

        K. Yu, M. Bengtsson, B. Ottersten, D. McNamara, P. Karlsson, and M. Beach, “A wideband
        statistical model for NLOS indoor MIMO channels,” IEEE Veh. Technol. Conf.
        (VTC - Spring), 2002

    The channel will provide 'number_rx_antennas' outputs to a signal
    consisting of 'number_tx_antennas' inputs. A random number generator,
    given by 'rnd' may be needed. The sampling rate is the same at both input and output
    of the channel, and is given by 'sampling_rate' samples/second.
    `tx_cov_matrix` and `rx_cov_matrix` are covariance matrices for transmitter
    and receiver.

    Attributes:
        number_of_paths(int): Number of paths in the multipath channel.
        los_phi(np.array): initial phase of line-of-sight (LOS) components
        los_theta(np.array): angle of arrival of LOS component(default=0)
        los_gain(np.array): path gains for LOS
        non_los_gain(np.array): path gains for NLOS
        doppler_frequency(float): doppler frequency in Hz.
        postcoding_matrix(np.ndarray): for receive antenna correlation.
        precoding_matrix(np.ndarray): transmit antenna correlation.
        phi (np.array): random phases of paths in Jake's model
        theta(np.array): random angles of paths of sinusoids in Jake's model
        interpolation_filter(np.array): Filter to apply if sampling instants are not
            at time delay instants.
        max_delay_in_samples(int): Maximum delay of all paths in samples.
    """

    number_of_sinewaves = 20  # to be used in Jake's model

    def __init__(self, parameters: ParametersChannel, random_number_gen: np.random.RandomState,
                 sampling_rate: float, doppler_frequency: float = None) -> None:
        super().__init__(parameters,
                         random_number_gen,
                         sampling_rate)

        self.number_of_paths = parameters.delays.size
        self.los_phi = np.array([])
        self.los_theta = np.array([])

        self.los_gain: np.array = np.sqrt(
            self.param.k_factor_rice) / np.sqrt(1 + self.param.k_factor_rice)
        self.los_gain[np.isposinf(self.param.k_factor_rice)] = 1.

        self.non_los_gain: np.array = 1 / np.sqrt(1 + self.param.k_factor_rice)

        self.doppler_frequency = doppler_frequency

        # normalization
        self.param.power_delay_profile = np.abs(
            self.param.power_delay_profile) / np.sum(self.param.power_delay_profile)

        # postcoding matrix for receive antenna correlation
        if self.param.rx_cov_matrix.size == 0:
            self.postcoding_matrix = np.array([])
        else:
            if not np.array_equal(self.param.rx_cov_matrix, np.transpose(self.param.rx_cov_matrix)):
                raise ValueError('rx covariance matrix must be Hermitian')
            if np.any(np.linalg.eigvals(self.param.rx_cov_matrix) <= 0):
                raise ValueError(
                    'rx covariance matrix must be positive definite')

            self.postcoding_matrix = scipy.linalg.sqrtm(self.param.rx_cov_matrix)

        # precoding matrix for transmit antenna correlation
        if self.param.tx_cov_matrix.size == 0:
            self.precoding_matrix = np.array([])
        else:
            if not np.array_equal(self.param.tx_cov_matrix, np.transpose(self.param.tx_cov_matrix)):
                raise ValueError('tx covariance matrix must be Hermitian')
            if np.any(np.linalg.eigvals(self.param.tx_cov_matrix) <= 0):
                raise ValueError(
                    'tx covariance matrix must be positive definite')

            self.precoding_matrix = scipy.linalg.sqrtm(self.param.tx_cov_matrix)

        self._number_of_sinewaves = [self.number_of_paths,
                                     MultipathFadingChannel.number_of_sinewaves,
                                     self.number_rx_antennas, self.number_tx_antennas]
        self._number_of_los_components = [
            self.number_of_paths, self.number_rx_antennas, self.number_tx_antennas]

        self.phi = np.array([])
        self.theta = np.array([])

        # check if filtering is needed (if delays are not in sampling instants)
        self.interpolation_filter = np.array([])
        delay_sample = np.around(self.param.delays * self.sampling_rate)
        self.max_delay_in_samples = int(
            np.ceil(self.param.delays.max() * self.sampling_rate))

        # if discrete delays are not very close to the desired delay, use filters for
        # intermediate delays. Close means that they ar within 5% of the minimum delay
        # difference
        if self.param.delays.size > 1:
            atol = 0.05 * np.min(np.diff(self.param.delays))
        else:
            atol = 0.05 * self.param.delays
        if not np.allclose(delay_sample / self.sampling_rate,
                           self.param.delays, atol=atol):
            self.max_delay_in_samples += 1
            self.interpolation_filter = np.zeros(
                (self.max_delay_in_samples + 1, self.number_of_paths))
            delays = np.arange(self.max_delay_in_samples +
                               1) / self.sampling_rate
            for path in range(self.number_of_paths):
                interp_filter = np.sinc(
                    (delays - self.param.delays[path]) * self.sampling_rate)
                interp_filter = interp_filter / \
                    np.sqrt(np.sum(interp_filter ** 2))
                self.interpolation_filter[:, path] = interp_filter

    def init_drop(self) -> None:
        """Initializes random channel parameters for each drop, by selecting random phases."""
        # initial phase of sinusoids
        self.phi = self.random.random_sample(
            self._number_of_sinewaves) * 2 * np.pi - np.pi
        self.theta = self.random.random_sample(
            self._number_of_sinewaves) * 2 * np.pi - np.pi
        self.los_phi = self.random.random_sample(
            self._number_of_los_components) * 2 * np.pi - np.pi
        self.los_theta = self.random.random_sample(
            self._number_of_los_components) * 2 * np.pi - np.pi

        if self.param.los_theta_0 != 0.:
            self.los_theta[0, :, :] = self.param.los_theta_0

    def propagate(self, tx_signal: np.ndarray) -> np.ndarray:
        """Modifies the input signal and returns it after channel propagation.

        Args:
            tx_signal (np.ndarray):
                Input signal to channel with shape of `N_tx_antennas x n_samples`.

        Returns:
            np.ndarray:
                Distorted signal after channel propagation. If input is
                an array of size 'number_tx_antennas' X 'number_of_samples',
                then the output is of size 'number_rx_antennas' X 'number_of_samples'
                + L, with L denoting the maximum excess delay of the power
                delay profile, in samples.

        """
        if tx_signal.ndim == 1:
            if self.number_tx_antennas > 1:
                raise ValueError(f'tx signal has 1 spatial dimension, {self.number_tx_antennas} expected')
            tx_signal = np.reshape(tx_signal, (1, -1))

        if (tx_signal.ndim > 2) or (tx_signal.ndim == 2 and
                                    tx_signal.shape[0] != self.number_tx_antennas):
            raise ValueError('input vector has wrong number of antennas')

        # calculate number of samples and time reference (output may have more samples on
        # account of delayed paths
        number_of_samples_in = tx_signal.shape[1]
        number_of_samples_out = number_of_samples_in + self.max_delay_in_samples

        rx_signal = np.zeros(
            (self.number_rx_antennas, number_of_samples_out), dtype=complex)

        # transmit antenna correlation
        if self.precoding_matrix.size != 0:
            tx_signal = self.precoding_matrix @ tx_signal

        # Add paths to form received signal
        for rx_antenna in range(self.number_rx_antennas):
            rx_signal_ant = np.zeros(number_of_samples_out, dtype=complex)

            for tx_antenna in range(self.number_tx_antennas):
                tx_signal_antenna = tx_signal[tx_antenna, :]

                impulse_response_siso = (
                    self._get_path_responses(rx_antenna, tx_antenna,
                                             np.arange(number_of_samples_in)
                                             / self.sampling_rate)
                )
                if self.interpolation_filter.size == 0:
                    delay_in_samples_vec = np.around(
                        self.param.delays * self.sampling_rate).astype(int)
                else:
                    delay_in_samples_vec = np.arange(
                        self.max_delay_in_samples + 1)

                for delay_idx, delay_in_samples in enumerate(
                        delay_in_samples_vec):
                    padding = self.max_delay_in_samples - delay_in_samples

                    # add faded path to received signal
                    if self.interpolation_filter.size == 0:
                        path_response_at_delay = impulse_response_siso[delay_idx, :]
                    else:
                        path_response_at_delay = (
                            self.interpolation_filter[delay_idx,
                                                      :] @ impulse_response_siso
                        )
                    signal_path = tx_signal_antenna * path_response_at_delay
                    signal_path = np.concatenate(
                        (np.zeros(delay_in_samples), signal_path, np.zeros(padding)))

                    rx_signal_ant += signal_path

            rx_signal[rx_antenna, :] = rx_signal_ant

        # receive antenna correlation
        if self.postcoding_matrix.size != 0:
            rx_signal = np.matmul(self.postcoding_matrix, rx_signal)

        return rx_signal * self.param.gain

    def get_impulse_response(self, timestamps: np.array) -> np.ndarray:
        """Calculates the channel impulse response.

        This method can be used for instance by the transceivers to obtain the
        channel state information.

        Args:
            timestamps (np.array): Time instants with length T to calculate the
                response for.

        Returns:
            np.ndarray:
                Impulse response in all 'number_rx_antennas' x 'number_tx_antennas' channels
                at the time instants given in vector 'timestamps'.
                `impulse_response` is a 4D-array, with the following dimensions:
                1- sampling instants, 2 - Rx antennas, 3 - Tx antennas, 4 - delays
                (in samples)
                The shape is T x number_rx_antennas x number_tx_antennas x (L+1)
        """

        impulse_response = np.zeros((timestamps.size,
                                     self.number_rx_antennas,
                                     self.number_tx_antennas,
                                     self.max_delay_in_samples + 1), dtype=complex)

        if self.interpolation_filter.size == 0:
            delay_in_samples_vec = np.around(
                self.param.delays * self.sampling_rate).astype(int)
        else:
            delay_in_samples_vec = np.arange(self.max_delay_in_samples + 1)

        for tx_antenna in range(self.number_tx_antennas):
            for rx_antenna in range(self.number_rx_antennas):
                path_gains = self._get_path_responses(
                    rx_antenna, tx_antenna, timestamps)

                for delay_idx, delay_in_samples in enumerate(
                        delay_in_samples_vec):
                    # add faded path to received signal
                    # impulse response dimensions are explained in method
                    # header
                    if self.interpolation_filter.size == 0:
                        impulse_response[:, rx_antenna, tx_antenna, delay_in_samples] = \
                            impulse_response[:, rx_antenna, tx_antenna,
                                             delay_in_samples] + path_gains[delay_idx, :]
                    else:
                        impulse_response[:, rx_antenna, tx_antenna, delay_in_samples] += \
                            (impulse_response[:, rx_antenna, tx_antenna, delay_in_samples] +
                             self.interpolation_filter[delay_idx, :] @ path_gains)

        if self.precoding_matrix.size > 0 or self.postcoding_matrix.size > 0:
            for instant in range(timestamps.size):
                for delay in delay_in_samples_vec:
                    mimo_channel = impulse_response[instant, :, :, delay]
                    mimo_channel = (
                        self.postcoding_matrix
                        @ mimo_channel
                        @ self.precoding_matrix
                    )
                    impulse_response[instant, :, :, delay] = mimo_channel

        return impulse_response * self.param.gain

    def _get_path_responses(
            self, rx_antenna: int, tx_antenna: int, time: np.array) -> np.ndarray:
        """Returns the channel gain of all relevant paths.

        Between transmitting and receiving antennas. antenna 'tx_antenna' and receive antenna

        Args:
            rx_antenna(int): Receiving antenna.
            tx_antenna(int): Transmitting antenna.
            time(np.array): Time instants (in s) of size T to calculate the responses for.

        Returns:
            np.ndarray:
                Gains vector with shape L x T, L+1 being the number of
                relevant paths in the model, .
        """

        path_gain = np.zeros((self.number_of_paths, time.size), dtype=complex)

        # Rayleigh-fading component
        norm_factor = np.sqrt(1 / MultipathFadingChannel.number_of_sinewaves)

        for path in range(self.number_of_paths):
            for sinewave_idx in range(
                    MultipathFadingChannel.number_of_sinewaves):
                # add random sinewaves to generate Rayleigh fading
                phi = self.phi[path, sinewave_idx, rx_antenna, tx_antenna]
                theta = self.theta[path, sinewave_idx, rx_antenna, tx_antenna]
                alpha = (2 * np.pi * (sinewave_idx + 1) + theta) / \
                    MultipathFadingChannel.number_of_sinewaves

                path_gain_real = np.cos(
                    2 * np.pi * self.doppler_frequency * time * np.cos(alpha) + phi)
                path_gain_imag = np.sin(
                    2 * np.pi * self.doppler_frequency * time * np.cos(alpha) + phi)
                path_gain[path, :] += (path_gain_real +
                                       1j * path_gain_imag) * norm_factor

            # add line-of-sight component and adjust power according to power
            # delay profile
            los_component = np.exp(1j * (2 * np.pi
                                         * self.param.los_doppler_factor
                                         * self.doppler_frequency
                                         * np.cos(
                                             self.los_theta[path, rx_antenna, tx_antenna])
                                         * time
                                         + self.los_phi[path, rx_antenna, tx_antenna]))
            path_gain[path, :] = ((los_component
                                   * self.los_gain[path]
                                   + path_gain[path, :]
                                   * self.non_los_gain[path])
                                  * np.sqrt(self.param.power_delay_profile[path]))

        return path_gain
