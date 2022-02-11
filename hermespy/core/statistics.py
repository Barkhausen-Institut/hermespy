# -*- coding: utf-8 -*-
"""Transmission statistics computation."""

import os
from typing import List, Tuple, Optional
from matplotlib import pyplot as plt
import scipy.io as sio
from scipy.fft import fftshift
from enum import Enum
import scipy.fft as fft
from scipy import stats
import numpy as np

from hermespy.core.scenario import Scenario
from hermespy.core.drop import Drop
from hermespy.core.tools.theoretical_results import TheoreticalResults

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronauer@barkhauseninstitut.org"
__status__ = "Prototype"


class ConfidenceMetric(Enum):
    """Confidence metric for stopping criteria during simulation execution. """

    DISABLED = 0        # No stopping criterion
    BER = 1             # Bit Error Rate
    BLER = 2            # Block Error Rate
#    FLER = 3            # Frame error rate


class SNRType(Enum):
    """Supported types of signal-to-noise ratios."""

    EBN0 = 0
    """Bit energy to noise power ratio."""

    ESN0 = 1
    """Symbol energy to noise power ratio."""

    CUSTOM = 2


class Statistics:
    """Generates, saves and plots the result statistics of a given simulation.

    Attributes:

        __scenario (Scenario): The scenario for which statistics should be generated.
        __calc_transmit_spectrum (bool): Compute the transmitted signals frequency domain spectra.
        __calc_receive_spectrum (bool): Compute the received signals frequency domain spectra.
        __calc_transmit_stft (bool): Compute the short time Fourier transform of transmitted signals.
        __calc_receive_stft (bool): Compute the short time Fourier transform of received signals.
        __spectrum_fft_size (int); Number of discrete frequency bins computed within the Fast Fourier Transforms.
        __num_drops (np.array): SNR-Specific Number of drops already added to the statistics.
        __confidence_margin (float): Margin for the stopping criteria check.
        __confidence_level (float): Probability to find interval for that metric is within.
        __confidence_metric (ConfidenceMetric): Metric that the stopping criteria is to be calculated for.
        snr_loop (List[float]): List of (linear) signal to noise ratios.
        __num_snr_loops (int): Different number of snrs to perform simulation for.
        snr_type (SNRType): Type of SNR to be used for noise calculation.
        __run_flag_matrix (np.ndarray): Determines if next simulation shall be run.
    """

    __scenario: Scenario
    __calc_transmit_spectrum: bool
    __calc_receive_spectrum: bool
    __calc_transmit_stft: bool
    __calc_receive_stft: bool
    __spectrum_fft_size: int
    __num_drops: np.array
    __confidence_margin: float
    __confidence_level: float
    __confidence_metric: ConfidenceMetric
    snr_loop: List[float]
    __num_snr_loops: int
    snr_type: SNRType
    __theory: TheoreticalResults = TheoreticalResults()
    __run_flag_matrix: np.ndarray

    def __init__(self,
                 scenario: Scenario,
                 snr_loop: List[float],
                 calc_transmit_spectrum: bool = True,
                 calc_receive_spectrum: bool = True,
                 calc_transmit_stft: bool = True,
                 calc_receive_stft: bool = True,
                 spectrum_fft_size: int = 0,
                 snr_type: SNRType = SNRType.EBN0,
                 calc_theory: bool = True,
                 confidence_margin: float = 0.1,
                 confidence_level: float = 0.99,
                 confidence_metric: ConfidenceMetric = ConfidenceMetric.DISABLED,
                 min_num_drops: int = 0,
                 max_num_drops: int = 1) -> None:
        """Transmission statistics object initialization.

        Args:
            scenario (Scenario): The scenario for which to generate statistics.
            snr_loop (List[float]): The (linear) signal to noise ratios for which to generate statistics.
            calc_transmit_spectrum (bool): Compute the transmitted signals frequency domain spectra.
            calc_receive_spectrum (bool): Compute the received signals frequency domain spectra.
            calc_transmit_stft (bool): Compute the short time Fourier transform of transmitted signals.
            calc_receive_stft (bool): Compute the short time Fourier transform of received signals.
            spectrum_fft_size (int): Number of discrete frequency bins computed within the Fast Fourier Transforms.
            snr_type (SNRTYpe): Type of SNR to be used for noise calculation.
            calc_theory (bool, optional): Calculate theoretical results, if possible.
            confidence_margin (float): Margin for the stopping criteria check.
            confidence_level (float): Probability to find interval for that metric is within.
            confidence_metric (ConfidenceMetric): Metric that the stopping crtieria is to be calculated for.
            min_num_drops (int): Minimum number of simulation drops to calculate.
            max_num_drops (int): Maximum numbr of simulation drops to calculate.
        """

        self.__scenario = scenario
        self.snr_loop = snr_loop
        self.__calc_transmit_spectrum = calc_transmit_spectrum
        self.__calc_receive_spectrum = calc_receive_spectrum
        self.__calc_transmit_stft = calc_transmit_stft
        self.__calc_receive_stft = calc_receive_stft
        self.__spectrum_fft_size = spectrum_fft_size
        self.__calc_theory = calc_theory
        self.__confidence_margin = confidence_margin
        self.__confidence_level = confidence_level
        self.__confidence_metric = confidence_metric
        self.__min_num_drops = min_num_drops
        self.__max_num_drops = max_num_drops
        self.snr_type = snr_type
        
        # Inferred attributes
        self.__num_snr_loops = len(snr_loop)
        self.__num_drops = np.zeros(self.__num_snr_loops, dtype=int)
        self._no_simulation_iterations = 0
        self._drop_updates = np.zeros(
            (self.__num_snr_loops,
             self.__scenario.num_transmitters,
             self.__scenario.num_receivers),
            dtype=bool
        )

        self.bit_error_num_drops = np.zeros((self.__num_snr_loops, scenario.num_transmitters, scenario.num_receivers))
        self.block_error_num_drops = np.zeros((self.__num_snr_loops, scenario.num_transmitters, scenario.num_receivers))

        self.bit_error_sum = np.zeros((self.__num_snr_loops, scenario.num_transmitters, scenario.num_receivers))
        self.block_error_sum = np.zeros((self.__num_snr_loops, scenario.num_transmitters, scenario.num_receivers))

        self.bit_error_min = np.ones((self.__num_snr_loops, scenario.num_transmitters, scenario.num_receivers))
        self.block_error_min = np.ones((self.__num_snr_loops, scenario.num_transmitters, scenario.num_receivers))

        self.bit_error_max = np.zeros((self.__num_snr_loops, scenario.num_transmitters, scenario.num_receivers))
        self.block_error_max = np.zeros((self.__num_snr_loops, scenario.num_transmitters, scenario.num_receivers))

        self.bit_error_mean = np.zeros((self.__num_snr_loops, scenario.num_transmitters, scenario.num_receivers))
        self.block_error_mean = np.zeros((self.__num_snr_loops, scenario.num_transmitters, scenario.num_receivers))

        self.bit_errors = np.zeros(
            (self.__max_num_drops,
             self.__num_snr_loops,
             scenario.num_transmitters,
             scenario.num_receivers),
        )
        self.block_errors = np.zeros(
            (self.__max_num_drops,
             self.__num_snr_loops,
             scenario.num_transmitters,
             scenario.num_receivers),
        )

        self._frequency_range_tx = [
            np.zeros(self.__spectrum_fft_size)
            for _ in range(self.__scenario.num_transmitters)
        ]

        self._frequency_range_rx = [np.array([]) for _ in range(self.__scenario.num_transmitters)]

        self._periodogram_tx: List[np.ndarray] = [np.empty(0, dtype=complex) for _ in
                                                  range(self.__scenario.num_transmitters)]
        self._periodogram_rx: List[np.ndarray] = [np.empty(0, dtype=complex) for _ in
                                                  range(self.__scenario.num_receivers)]

        self._stft_tx: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self._stft_rx: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        if self.__calc_theory:
            self.theoretical_results = self.__theory.theory(scenario, np.array(snr_loop))

        else:
            self.theoretical_results = None

        self.__run_flag_matrix = np.ones(
            (self.__scenario.num_transmitters,
             self.__scenario.num_receivers,
             self.__num_snr_loops),
            dtype=bool
        )

    @property
    def no_simulation_iterations(self) -> int:
        """Returns the number of simulation iterations."""
        return self._no_simulation_iterations

    def add_drop(self, drop: Drop, snr_index: int) -> None:
        """Add a new transmission drop to the statistics.

        Args:
            drop (Drop): The drop to be added.
            snr_index (int): Index of the SNR within the configured loop.
        """

        if self.__calc_transmit_spectrum:
            self.__add_transmit_spectrum(drop, snr_index)
                
        if self.__calc_receive_spectrum:
            self.__add_receive_spectrum(drop, snr_index)

        if self.__calc_transmit_stft:
            self._stft_tx = drop.transmit_stft

        if self.__calc_receive_stft:
            self._stft_rx = drop.receive_stft

        self.__add_bit_error_rate(drop, snr_index)
        if not self.__confidence_metric == ConfidenceMetric.DISABLED:
            self.update_stopping_criteria(snr_index)

    def add_drops(self, drops: List[Drop], snr_index: int) -> None:
        """Add multiple transmission drops to the statistics.

        Args:
            drops (List[Drop]): List of drops to be added.
            snr_index (int): Index of SNR tap.
        """

        for drop in drops:
            self.add_drop(drop, snr_index)

    @property
    def run_flag_matrix(self) -> np.ndarray:
        """Returns run_flag matrix of last drop."""
        return self.__run_flag_matrix

    @property
    def num_drops(self) -> np.ndarray:
        """Access the number of drops already added to this statistics.

        Returns:
            np.array: SNR-specific number of drops.
        """

        return self.__num_drops

    def __add_transmit_spectrum(self, drop: Drop, snr_index: int) -> None:
        """Subroutine to add a new transmit spectral information from a new drop.

        Args:
            drop (Drop): The new drop.
            snr_index (int): Respective snr index.
        """

        # Fetch transmit spectra from drop
        transmit_spectra = drop.transmit_spectrum

        # Initialize containers during first drop
        if self.__num_drops[snr_index] < 1:
            for transmitter_index in range(self.__scenario.num_transmitters):
                self._frequency_range_tx[transmitter_index] = transmit_spectra[transmitter_index][0]
                self._periodogram_tx[transmitter_index] = np.zeros(len(transmit_spectra[transmitter_index][0]),
                                                                   dtype=float)

        # Add new periodogram to the sum of periodograms
        for periodogram, (_, new_periodogram) in zip(self._periodogram_tx, transmit_spectra):
            if new_periodogram is not None:
                periodogram += new_periodogram

    def __add_receive_spectrum(self, drop: Drop, snr_index: int) -> None:
        """Subroutine to add a new receive spectral information from a new drop.

        Args:
            drop (Drop): The new drop.
            snr_index (int): Respective SNR index.
        """

        # Fetch receive spectra from drop
        receive_spectra = drop.receive_spectrum

        # Initialize containers during first drop
        if self.__num_drops[snr_index] < 1:
            for receiver_index in range(self.__scenario.num_receivers):
                self._frequency_range_rx[receiver_index] = receive_spectra[receiver_index][0]
                self._periodogram_rx[receiver_index] = np.zeros(len(receive_spectra[receiver_index][0]),
                                                                dtype=float)

        # Add new periodogram to the sum of periodograms
        for periodogram, (_, new_periodogram) in zip(self._periodogram_rx, receive_spectra):
            if new_periodogram is not None:
                periodogram += new_periodogram

    def __add_bit_error_rate(self, drop: Drop, snr_index: int) -> None:
        """Calculates error rate between received signal in bits and source bits.

        It is also checked whether the stopping criteria need to be updated.

        Args:
            drop (Drop): The drop to be added.
            snr_index (int): Index of the SNR within the configured loop.
        """

        # Fetch bit and block errors grids from drop object
        bit_error_rates: List[List[Optional[float]]] = drop.bit_error_rates
        block_error_rates: List[List[Optional[np.ndarray]]] = drop.block_error_rates

        # Enumerate over bit and block error grids simultaneously.
        # Each respective grid entry may be None if no error rate computation is feasible.
        for tx_modem, (bit_error_row, block_error_row) in enumerate(zip(bit_error_rates, block_error_rates)):
            for rx_modem, (bit_error, block_error) in enumerate(zip(bit_error_row, block_error_row)):

                if bit_error is not None:
                    # Increase drop counter for the specific grid field
                    self.bit_error_num_drops[snr_index, tx_modem, rx_modem] += 1

                    # Sum up all error rates
                    self.bit_error_sum[snr_index, tx_modem, rx_modem] += bit_error

                    # Update minimal bit error over all drops
                    bit_error_min = self.bit_error_min[snr_index, tx_modem, rx_modem]
                    self.bit_error_min[snr_index, tx_modem, rx_modem] = min(bit_error_min, bit_error)

                    # Update maximal bit error over all drops
                    bit_error_max = self.bit_error_max[snr_index, tx_modem, rx_modem]
                    self.bit_error_max[snr_index, tx_modem, rx_modem] = max(bit_error_max, bit_error)

                    bit_error_mean = self.bit_error_mean[snr_index, tx_modem, rx_modem]
                    self.bit_error_mean[snr_index, tx_modem, rx_modem] = self.__update_mean(
                        old_mean=bit_error_mean,
                        no_old_samples=self.bit_error_num_drops[snr_index, tx_modem, rx_modem] - 1,
                        new_sample=bit_error
                    )
                    self.bit_errors[self.num_drops[snr_index], snr_index, tx_modem, rx_modem] = bit_error

                if block_error is not None:

                    # Increase drop counter for the specific grid field
                    self.block_error_num_drops[snr_index, tx_modem, rx_modem] += 1
                    
                    # Sum up all error rates
                    self.block_error_sum[snr_index, tx_modem, rx_modem] += block_error

                    # Update minimal block error over all drops
                    block_error_min = self.block_error_min[snr_index, tx_modem, rx_modem]
                    self.block_error_min[snr_index, tx_modem, rx_modem] = min(block_error_min, block_error)

                    # Update maximal block error over all drops
                    block_error_max = self.block_error_max[snr_index, tx_modem, rx_modem]
                    self.block_error_max[snr_index, tx_modem, rx_modem] = max(block_error_max, block_error)

                    block_error_mean = self.block_error_mean[snr_index, tx_modem, rx_modem]
                    self.block_error_mean[snr_index, tx_modem, rx_modem] = self.__update_mean(
                        old_mean=block_error_mean,
                        no_old_samples=self.block_error_num_drops[snr_index, tx_modem, rx_modem] - 1,
                        new_sample=block_error
                    )
                    self.block_errors[self.num_drops[snr_index], snr_index, tx_modem, rx_modem] = block_error

        self.__num_drops[snr_index] += 1

    def __update_mean(self, old_mean: float,
                          no_old_samples: int,
                          new_sample: float) -> float:
        """Updates mean iteratively.

        Args:
            old_mean (float): Mean to be updated.
            no_old_sampls (int): Number of samples that were used for old_mean calculation.
            new_sample (float): New sample.

        Returns:
            float: New mean.
        """
        return no_old_samples / (no_old_samples + 1) * old_mean + 1 / (no_old_samples+1) * new_sample

    def update_stopping_criteria(self, snr_index: int) -> None:
        """Updates the stopping criteria.

        Args:
            snr_index (int): SNR-specific drop that stopping criteria shall be updated for.
        """
        for rx_modem_idx in range(self.__scenario.num_receivers):
            for tx_modem_idx in range(self.__scenario.num_transmitters):
                if self.__num_drops[snr_index] >= self.__min_num_drops:
                    if self.__run_flag_matrix[tx_modem_idx, rx_modem_idx, snr_index]:

                        if self.__confidence_metric == ConfidenceMetric.BER:
                            self.__update_flag_matrix_ber(snr_index, tx_modem_idx, rx_modem_idx)
                        elif self.__confidence_metric == ConfidenceMetric.BLER:
                            self.__update_flag_matrix_bler(snr_index, tx_modem_idx, rx_modem_idx)

    def __update_flag_matrix_ber(self, snr_index: int, tx_modem_idx: int, rx_modem_idx: int) -> None:
        errors = self.bit_errors[:self.__num_drops[snr_index], snr_index, tx_modem_idx, rx_modem_idx]

        mean_lower_bound, mean_upper_bound = self.estimate_confidence_intervals_mean(
            errors, self.__confidence_level)

        self.bit_error_min[snr_index, tx_modem_idx, rx_modem_idx] = mean_lower_bound
        self.bit_error_max[snr_index, tx_modem_idx, rx_modem_idx] = mean_upper_bound
        confidence_margin = self.get_confidence_margin(
                    upper=mean_upper_bound,
                    lower=mean_lower_bound,
                    mean=self.bit_error_mean[snr_index, tx_modem_idx, rx_modem_idx]
        )
        self.__run_flag_matrix[tx_modem_idx, rx_modem_idx, snr_index] = (
            confidence_margin > self.__confidence_margin
        )

    def __update_flag_matrix_bler(self, snr_index: int, tx_modem_idx: int, rx_modem_idx: int) -> None:
        errors = self.bit_errors[:self.__num_drops[snr_index], snr_index, tx_modem_idx, rx_modem_idx]

        mean_lower_bound, mean_upper_bound = self.estimate_confidence_intervals_mean(
            errors, self.__confidence_level)

        self.block_error_min[snr_index, tx_modem_idx, rx_modem_idx] = mean_lower_bound
        self.block_error_max[snr_index, tx_modem_idx, rx_modem_idx] = mean_upper_bound
        confidence_margin = self.get_confidence_margin(
                    upper=mean_upper_bound,
                    lower=mean_lower_bound,
                    mean=self.block_error_mean[snr_index, tx_modem_idx, rx_modem_idx]
        )
        self.__run_flag_matrix[tx_modem_idx, rx_modem_idx, snr_index] = (
            confidence_margin > self.__confidence_margin
        )

    def get_confidence_margin(self, upper: float,
                                    lower: float,
                                    mean: float) -> float:
        """Calculates current confidence margin for confidence metric mean.

        Args:
            upper (float): Upper boundary.
            lower (float): Lower boundary.
            mean (float): Mean value-
        Returns:
            float: Confidence_margin.
        """
        old_settings = np.seterr(divide="ignore", invalid="ignore")
        confidence_margin = np.float64((upper - lower)) / np.float64(mean)
        np.seterr(**old_settings)
        return confidence_margin

    def estimate_confidence_intervals_mean(self,
                                           data: np.ndarray,
                                           alpha: float) -> Tuple[float, float]:
        """Estimates bayesian confidence intervals for the mean.

        Args:
            data (np.ndarray): Data samples.
            alpha (float): Probability that return confidence interval contains true parameter.

        Returns:
            (float, float): Lower and upper bound of estimated mean.
        """

        lower_bound = min(data)
        upper_bound = max(data)

        if lower_bound != upper_bound:

            estimates = stats.bayes_mvs(data=data, alpha=alpha)

            lower_bound = estimates[0][1][0]
            upper_bound = estimates[0][1][1]

        return lower_bound, upper_bound

    def save(self, results_dir: str) -> None:
        """averages out the stored statistics from all the drops and store them in a matlab file.

        Theoretical values (if available) are stored as well. If 'plot' is True,
        then plot results.

        Args:
            results_dir (str): the desired directory to save matlab file in.
        """

        filename = os.path.join(results_dir, "statistics.mat")

        """for rx_modem_idx in range(self.__scenario.num_transmitters):
            print(f"\n\tResults for Rx {rx_modem_idx}")

            for snr, ber, fer in zip(
                self.snr_loop, self.bit_error_sum[rx_modem_idx], self.block_error_sum[rx_modem_idx]
            ):
                print(f"\t{self.snr_type.value} = {snr}dB\tBER = {ber:e}, \tfer = {fer:e}")"""

        mat_dict = {
            "snr_type": self.snr_type.name,
            "snr_vector": self.snr_loop,
            "ber_mean": self.average_bit_error_rate,
            "fer_mean": self.average_block_error_rate,
            "ber_lower": self.bit_error_min,
            "ber_upper": self.bit_error_max,
            "fer_lower": self.block_error_min,
            "fer_upper": self.block_error_max,
        }

        if self.__calc_transmit_spectrum:
            for idx, (periodogram, frequency) in enumerate(zip(self._periodogram_tx, self._frequency_range_tx)):
                if periodogram is not None and frequency is not None:
                    mat_dict["frequency_tx_" + str(idx)] = fft.fftshift(frequency)
                    mat_dict["power_spectral_density_tx_" + str(idx)] = fft.fftshift(periodogram) / np.amax(periodogram)

        if self.__calc_transmit_stft:
            for idx, (time, freq, power) in enumerate(self._stft_tx):
                if time is not None and freq is not None and power is not None:
                    mat_dict["stft_time_tx_" + str(idx)] = time
                    mat_dict["stft_frequency_tx" + str(idx)] = freq
                    mat_dict["stft_power_tx" + str(idx)] = power

        if self.__calc_receive_spectrum:
            for idx, (periodogram, frequency) in enumerate(zip(self._periodogram_rx, self._frequency_range_rx)):

                mat_dict["frequency_rx_" + str(idx)] = fft.fftshift(frequency)
                mat_dict["power_spectral_density_rx_" + str(idx)] = fft.fftshift(periodogram) / np.amax(periodogram)

        if self.__calc_receive_stft:
            for idx, (time, freq, power) in enumerate(self._stft_rx):
                if time is not None and freq is not None and power is not None:
                    mat_dict["stft_time_rx_" + str(idx)] = time
                    mat_dict["stft_frequency_rx_" + str(idx)] = freq
                    mat_dict["stft_power_rx_" + str(idx)] = power

        ber_theory = np.nan * np.ones((self.__scenario.num_transmitters,
                                      self.__scenario.num_receivers,
                                      self.__num_snr_loops), dtype=float)
        fer_theory = np.nan * np.ones((self.__scenario.num_transmitters,
                                      self.__scenario.num_receivers,
                                      self.__num_snr_loops), dtype=float)
        theory_notes = [[np.nan for _ in self.__scenario.receivers] for _ in self.__scenario.transmitters]

        if self.theoretical_results is not None:

            for tx_idx, rx_idx in zip(range(self.__scenario.num_transmitters), range(self.__scenario.num_receivers)):

                link_theory = self.theoretical_results[tx_idx, rx_idx]
                if link_theory is not None:

                    if 'ber' in link_theory:
                        ber_theory[tx_idx, rx_idx, :] = link_theory['ber']

                    if 'fer' in link_theory:
                        fer_theory[tx_idx, rx_idx, :] = link_theory['fer']

                    if 'notes' in link_theory:
                        theory_notes[tx_idx][rx_idx] = link_theory['notes']

            mat_dict["ber_theory"] = ber_theory
            mat_dict["fer_theory"] = fer_theory
            mat_dict["theory_notes"] = theory_notes

        # Save results in matlab file
        sio.savemat(filename, mat_dict)

    @property
    def average_bit_error_rate(self) -> np.ndarray:
        """The average bit error rate over all drops and SNRs.

        Returns:
            np.ndarray: A matrix of dimension `num_snr_loops`x`num_transmitters`x`num_receivers`.
        """

        return self.bit_error_sum / self.bit_error_num_drops

    @property
    def average_block_error_rate(self) -> np.ndarray:
        """The average bit block error rate over all drops and SNRs.

        Returns:
            np.ndarray: A matrix of dimension `num_snr_loops`x`num_transmitters`x`num_receivers`.
        """

        return self.block_error_sum / self.bit_error_num_drops

    def plot_transmit_spectrum(self) -> None:
        """Plot the transmit spectral estimations."""

        for transmitter, (frequency, periodogram) in enumerate(zip(self._frequency_range_tx, self._periodogram_tx)):

            power = 10 * np.log10(periodogram / np.amax(periodogram))

            figure, axes = plt.subplots()
            figure.suptitle("Average PSD of TX #{}".format(transmitter))

            axes.plot(fftshift(frequency), fftshift(power))
            axes.set(xlabel="Frequency [Hz]")
            axes.set(ylabel="Power [dB]")
            
    def plot_receive_spectrum(self) -> None:
        """Plot the receive spectral estimations."""

        for receiver, (frequency, periodogram) in enumerate(zip(self._frequency_range_rx, self._periodogram_rx)):

            power = 10 * np.log10(periodogram / np.amax(periodogram))

            figure, axes = plt.subplots()
            figure.suptitle("Average PSD of RX #{}".format(receiver))

            axes.plot(fftshift(frequency), fftshift(power))
            axes.set(xlabel="Frequency [Hz]")
            axes.set(ylabel="Power [dB]")

    def plot_bit_error_rates(self) -> None:
        """Plot the bit error rates."""

        # Fetch bit error rates
        bit_error_rates = self.average_bit_error_rate
        snr = 10 * np.log10(self.snr_loop)

        # Initialize plot window
        plot, axes = plt.subplots(self.__scenario.num_transmitters, self.__scenario.num_receivers, squeeze=False)
        plot.suptitle("Bit Error Rate")

        for tx_idx, rx_idx in zip(range(self.__scenario.num_transmitters),
                                  range(self.__scenario.num_receivers)):

            # Skip the link between this transmitter and receiver if it seems invalid
            if np.any(self.bit_error_num_drops[:, tx_idx, rx_idx] < 1):

                axes[tx_idx, rx_idx].text(0, 0, "Link invalid, no bit error rate available")

            else:

                lower = bit_error_rates[:, tx_idx, rx_idx] - self.bit_error_min[:, tx_idx, rx_idx]
                upper = self.bit_error_max[:, tx_idx, rx_idx] - bit_error_rates[:, tx_idx, rx_idx]
                error = np.vstack((lower, upper))

                # Plot error-bar representation with upper and lower error limits
                axes[tx_idx, rx_idx].errorbar(snr, bit_error_rates[:, tx_idx, rx_idx], error,
                                              label="Simulation")

                # Plot theory, if available
                if self.theoretical_results is not None and self.theoretical_results[tx_idx, rx_idx] is not None:

                    ber = self.theoretical_results[tx_idx, rx_idx]['ber']
                    axes[tx_idx, rx_idx].plot(snr, ber, label="Theory")

                axes[tx_idx, rx_idx].grid()
                axes[tx_idx, rx_idx].legend()

                # Scale to log if possible
                if np.any(bit_error_rates[:, tx_idx, rx_idx] > 0.0):
                    axes[tx_idx, rx_idx].set_yscale("log", nonpositive="mask")

        # Add outer labeling
        for tx_idx in range(self.__scenario.num_transmitters):
            axes[tx_idx, 0].set(ylabel="BER Tx #{}".format(tx_idx))

        for rx_idx in range(self.__scenario.num_receivers):
            axes[0, rx_idx].set(xlabel="{} [dB] Rx #{}, ".format(self.snr_type.name, rx_idx))

            """if ber_theory[transmitter_index] is not np.nan:
                plt.plot(
                    self.param_general.snr_vector,
                    ber_theory[transmitter_index],
                    label="theory")"""

    def plot_block_error_rates(self) -> None:
        """Plot the block error rates."""

        # Fetch block error rates
        block_error_rates = self.average_block_error_rate
        snr = 10 * np.log10(self.snr_loop)

        # Initialize plot window
        plot, axes = plt.subplots(self.__scenario.num_transmitters, self.__scenario.num_receivers, squeeze=False)
        plot.suptitle("Block Error Rate")

        for tx_idx, rx_idx in zip(range(self.__scenario.num_transmitters),
                                  range(self.__scenario.num_receivers)):

            # Skip the link between this transmitter and receiver if it seems invalid
            if np.any(self.block_error_num_drops[:, tx_idx, rx_idx] < 1):

                axes[tx_idx, rx_idx].text(0, 0, "Link invalid, no block error rate available")

            else:

                lower = block_error_rates[:, tx_idx, rx_idx] - self.block_error_min[:, tx_idx, rx_idx]
                upper = self.block_error_max[:, tx_idx, rx_idx] - block_error_rates[:, tx_idx, rx_idx]
                error = np.vstack((lower, upper))

                # Plot error-bar representation with upper and lower error limits
                axes[tx_idx, rx_idx].errorbar(snr, block_error_rates[:, tx_idx, rx_idx], error,
                                              label="Simulation")

                # Plot theory, if available
                if self.theoretical_results is not None and self.theoretical_results[tx_idx, rx_idx] is not None:

                    ber = self.theoretical_results[tx_idx, rx_idx]['ber']
                    axes[tx_idx, rx_idx].plot(snr, ber, label="Theory")

                axes[tx_idx, rx_idx].grid()
                axes[tx_idx, rx_idx].legend()

                # Scale to log if possible
                if np.any(block_error_rates[:, tx_idx, rx_idx] > 0.0):
                    axes[tx_idx, rx_idx].set_yscale("log", nonpositive="mask")

        # Add outer labeling
        for tx_idx in range(self.__scenario.num_transmitters):
            axes[tx_idx, 0].set(ylabel="BLER Tx #{}".format(tx_idx))

        for rx_idx in range(self.__scenario.num_receivers):
            axes[0, rx_idx].set(xlabel="{} [dB] Rx #{}, ".format(self.snr_type.name, rx_idx))
