import math

import numpy as np
from scipy import signal

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronaue@barkhauseninstitut.org"
__status__ = "Prototype"


class RxSampler:
    """Defines a resampler at the receiver input.

    This is required if the signal originates from different transmitters using
    different sampling rates and/or carrier frequencies.

    Attributes:
        rx_sampling_rate (float): Sampling rate of receiver.
        rx_center_freq (float): Center, i.e. carrier frequency of receiver.
    """

    def __init__(self, rx_sampling_rate: float, rx_center_freq: float) -> None:
        self.rx_sampling_rate = rx_sampling_rate
        self.rx_center_freq = rx_center_freq
        self._tx_sampling_rate = 0
        self._tx_center_freq = 0

        self._oversample_rate: np.array = 0
        self._interpolate_factor_rx: float = 0
        self._interpolate_factors_tx: np.array = 0
        self._decimate_factor_rx: float = 0
        self._decimate_factor_tx: float = 0

    ###################################
    # property definitions

    @property
    def oversample_rate(self) -> np.array:
        """np.array: Returns read-only oversample rate."""
        return self._oversample_rate

    @property
    def interpolate_factor_rx(self) -> float:
        """float: Returns read-only interpolate factor for receiver."""
        return self._interpolate_factor_rx

    @property
    def interpolate_factors_tx(self) -> np.array:
        """np.array: Returns read-only interpolate factors for transmitter."""
        return self._interpolate_factors_tx

    @property
    def decimate_factor_rx(self) -> float:
        """float: Returns read-only decimate factor for receiver."""
        return self._decimate_factor_rx

    @property
    def decimate_factor_tx(self) -> float:
        """float: Returns read-only decimate factor for transmitter."""
        return self._decimate_factor_tx

    @property
    def rx_sampling_rate(self) -> float:
        """float: Sampling rate of receiver."""
        return self._rx_sampling_rate

    @rx_sampling_rate.setter
    def rx_sampling_rate(self, rx_sampling_rate: float) -> None:
        if rx_sampling_rate < 0:
            self._rx_sampling_rate = 0.0
        else:
            self._rx_sampling_rate = rx_sampling_rate

    @property
    def rx_center_freq(self) -> float:
        """float: Carrier frequency of receiver."""
        return self._rx_center_freq

    @rx_center_freq.setter
    def rx_center_freq(self, rx_center_freq: float) -> None:
        self._rx_center_freq = rx_center_freq

    @property
    def tx_sampling_rate(self) -> np.array:
        """np.array: Returns read-only sampling rate of transmitter."""
        return self._tx_sampling_rate

    @property
    def tx_center_freq(self) -> np.array:
        """np.array: Returns read-only carrier frequency of transmitter."""
        return self._tx_center_freq

    # property definitions END
    #############################################

    def set_tx_sampling_rate(
        self, tx_sampling_rate: np.array, tx_center_freq: np.array
    ) -> None:
        """Setter method for sampling rate of transmitter.

        Beside setting the sampling rate and the carrier frequency for the
        transmitter, factors for resampling are internally calulcated by calling
        `computeFactors()`.

        Args:
            tx_sampling_rate (np.array): Sampling rates of transmitting modems.
            tx_center_freq (np.array): Center frequency of transmitting modems.
        """
        self._tx_sampling_rate = tx_sampling_rate
        self._tx_center_freq = tx_center_freq
        self.computeFactors()

    def resample(self, input_signal: np.ndarray) -> np.ndarray:
        """Resamples input signal.

        Interpolation and decimate factors are internally calculcated. As a
        filter, the Kaiser window is being used with beta being 5 and n being 10.

        Args:
            input_signal(np.ndarray): Input signal.

        Returns:
            np.ndarray: Resampled signal.
        """
        no_tx_signals = len(input_signal)

        # get maximum number of samples of all input signals
        no_samples = 0
        for signal_index in range(no_tx_signals):
            if input_signal[signal_index].shape[1] > no_samples:
                no_samples = input_signal[signal_index].shape[1]

        no_rx_antennas = input_signal[0].shape[0]

        # center and sampling frequencies are the same - everything can be
        # stupidly added
        if np.all(self._tx_sampling_rate == self._rx_sampling_rate) and np.all(
            self._tx_center_freq == self._rx_center_freq
        ):

            # add up all received signals
            # note: different signal sizes need to be handled
            delta_fillup = np.zeros(
                (no_rx_antennas, no_samples - input_signal[0].shape[1])
            )

            signal_out = np.hstack((input_signal[0], delta_fillup))

            for signal_index in range(1, no_tx_signals):
                delta_fillup = np.zeros(
                    (no_rx_antennas, no_samples -
                     input_signal[signal_index].shape[1])
                )
                signal_out = signal_out + np.hstack(
                    (input_signal[signal_index], delta_fillup)
                )
        else:
            # resample to higher sample rate
            upsampled_tx = np.empty(no_tx_signals, dtype="object")
            length_upsampled_tx = np.zeros(no_tx_signals)

            for signal_index in range(no_tx_signals):
                no_samples = input_signal[signal_index].shape[1]

                # perform actual upsampling
                for antenna_idx in range(no_rx_antennas):
                    upsampled_signal = signal.resample_poly(
                        x=input_signal[signal_index][antenna_idx, :],
                        up=self._interpolate_factors_tx[signal_index],
                        down=self._decimate_factor_tx,
                    ).T

                    if antenna_idx == 0:
                        upsampled_tx[signal_index] = np.zeros(
                            (no_rx_antennas,
                             upsampled_signal.shape[0]), dtype="complex"
                        )

                    upsampled_tx[signal_index][antenna_idx, :] = upsampled_signal
                length_upsampled_tx[signal_index] = upsampled_tx[signal_index].shape[1]

            # Truncate to smaller signal size considering that it respects the
            # expected number of samples of the receiver
            for signal_index in range(no_tx_signals):
                upsampled_tx[signal_index] = upsampled_tx[signal_index][
                    :, : int(min(length_upsampled_tx))
                ]

            # shift to respective center frequencies
            freq_shift = self._tx_center_freq - self._rx_center_freq
            rad_freq_shift = (2 * math.pi * freq_shift) / self._oversample_rate

            for signal_index in range(no_tx_signals):
                if rad_freq_shift[signal_index] != 0:
                    samples_index = np.arange(
                        upsampled_tx[signal_index].shape[1]) + 1
                    exp_shift = np.exp(
                        1j * rad_freq_shift[signal_index] * samples_index
                    )

                    # perform shift in time domain
                    for rx_index in range(no_rx_antennas):
                        upsampled_tx[signal_index][rx_index, :] = (
                            exp_shift * upsampled_tx[signal_index][rx_index, :]
                        )

            # sum all tx signals to form the aggregate rx signal
            summed_rx = np.zeros(
                (no_rx_antennas, upsampled_tx[0].shape[1]), dtype="complex"
            )

            for rx_index in range(no_rx_antennas):
                summed_rx[rx_index, :] = upsampled_tx[0][rx_index, :]

                for signal_index in range(1, no_tx_signals):
                    summed_rx[rx_index, :] = (
                        summed_rx[rx_index, :] +
                        upsampled_tx[signal_index][rx_index, :]
                    )

            # resample to receiver frequency
            lrx_samples = math.ceil(
                summed_rx.shape[1]
                * self._interpolate_factor_rx
                / self._decimate_factor_rx
            )
            signal_out = np.zeros(
                (no_rx_antennas, lrx_samples), dtype="complex")
            for rx_index in range(no_rx_antennas):
                signal_out[rx_index, :] = signal.resample_poly(
                    summed_rx[rx_index, :],
                    self._interpolate_factor_rx,
                    self._decimate_factor_rx,
                )

        return signal_out

    def computeFactors(self) -> None:
        """This function calculates the interpolation factor (I) and decimation
            factor (D) for resampling the transmitted signals, the factors to
            resample the combined received signals and the oversampling rate.
            The function updates the respective private properties of the class. """

        if np.any(self._tx_sampling_rate != self._rx_sampling_rate) or np.any(
            self._tx_center_freq != self._rx_center_freq
        ):

            # compute interpolation and decimation factors for resampling
            center_freqs = np.append(
                self._tx_center_freq, self._rx_center_freq)
            sampling_rates = np.append(
                self._tx_sampling_rate, self._rx_sampling_rate)

            min_freq = min(center_freqs - sampling_rates / 2.0)
            max_freq = max(center_freqs + sampling_rates / 2.0)

            # sampling frequency needs to two times the frequency of the signal
            upsample_rate_min = 2 * max(
                max_freq - self._rx_center_freq, self._rx_center_freq - min_freq
            )

            # least common multiple of sampling rates and the needed upsample
            # rate
            least_common_multiple_of_rates = np.lcm.reduce(
                sampling_rates.astype(np.int64))

            # compute interpolation (I) and decimation (D) factors
            min_over_common_rate = upsample_rate_min / least_common_multiple_of_rates

            if min_over_common_rate <= 1:
                upsample_rate = least_common_multiple_of_rates
                self._interpolate_factors_tx = upsample_rate / self._tx_sampling_rate
                self._decimate_factor_tx = math.floor(1 / min_over_common_rate)
            else:
                upsample_rate = least_common_multiple_of_rates * math.ceil(
                    min_over_common_rate
                )
                self._interpolate_factors_tx = upsample_rate / self._tx_sampling_rate
                self._decimate_factor_tx = 1

            self._oversample_rate = (
                self._tx_sampling_rate
                * self._interpolate_factors_tx
                / self._decimate_factor_tx
            )

            assert not np.any(
                self._oversample_rate < upsample_rate_min
            ), "Error on oversampling rate calculation. It should not be lower than the minimum oversampling rate!"

            if np.all(self._oversample_rate == self._oversample_rate[0]):
                self._oversample_rate = self._oversample_rate[0]
            else:
                assert (
                    False
                ), "Error on oversampling rate calculation. All values should be the same!"

            self._interpolate_factor_rx = self._decimate_factor_tx
            self._decimate_factor_rx = round(
                self._oversample_rate
                * self._decimate_factor_tx
                / self._rx_sampling_rate
            )
