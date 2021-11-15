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

from hermespy.scenario import Scenario
from .drop import Drop
from hermespy.simulator_core.tools.theoretical_results import TheoreticalResults

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronauer@barkhauseninstitut.org"
__status__ = "Prototype"


class SNRType(Enum):
    """Supported signal-to-noise ratio types."""

    EBN0 = 0
    ESN0 = 1
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
        snr_loop (List[float]): List of (linear) signal to noise ratios.
        __num_snr_loops (int): Different number of snrs to perform simulation for.
        run_flag[List[np.array]]:
            Each list item corresponds to one receiver modem.
            Each list item is a np.array stating if results are to be calculated
            for the corresponding SNR value.
    """

    __scenario: Scenario
    __calc_transmit_spectrum: bool
    __calc_receive_spectrum: bool
    __calc_transmit_stft: bool
    __calc_receive_stft: bool
    __spectrum_fft_size: int
    __num_drops: int
    snr_loop: List[float]
    __num_snr_loops: int
    snr_type: SNRType
    __theory: TheoreticalResults = TheoreticalResults()

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
                 confidence_margin: float = 0.0,
                 min_num_drops: int = 0) -> None:
        """Transmission statistics object initialization.

        Args:
            scenario (Scenario): The scenario for which to generate statistics.
            snr_loop (List[float]): The (linear) signal to noise ratios for which to generate statistics.
            calc_transmit_spectrum (bool): Compute the transmitted signals frequency domain spectra.
            calc_receive_spectrum (bool): Compute the received signals frequency domain spectra.
            calc_transmit_stft (bool): Compute the short time Fourier transform of transmitted signals.
            calc_receive_stft (bool): Compute the short time Fourier transform of received signals.
            spectrum_fft_size (int): Number of discrete frequency bins computed within the Fast Fourier Transforms.
            calc_theory (bool, optional): Calculate theoretical results, if possible.
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
        self.__min_num_drops = min_num_drops
        self.snr_type = snr_type
        
        # Inferred attributes
        self.__num_snr_loops = len(snr_loop)
        self.__num_drops = np.zeros(self.__num_snr_loops)
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

        self.bit_errors = []
        self.block_errors = []

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

        self.__flag_matrix = np.ones(
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
        self.update_stopping_criteria(snr_index)
        # Increase internal drop counter
        self.__num_drops[snr_index] += 1

    def add_drops(self, drops: List[Drop], snr_index: int) -> None:
        """Add multiple transmission drops to the statistics.

        Args:
            drops (List[Drop]): List of drops to be added.
            snr_index (int): Index of SNR tap.
        """

        for drop in drops:
            self.add_drop(drop, snr_index)
    @property
    def flag_matrix(self) -> np.ndarray:
        """Returns flag matrix of last drop."""
        return self.__flag_matrix

    @property
    def num_drops(self) -> np.array:
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
        current_ber = np.zeros(
            (self.__num_snr_loops,
             self.__scenario.num_transmitters,
             self.__scenario.num_receivers)
        )
        current_bler = np.zeros(current_ber.shape)
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
                    self.bit_error_mean[snr_index, tx_modem, rx_modem] = self.update_mean(
                        old_mean=bit_error_mean,
                        no_old_samples=self.bit_error_num_drops[snr_index, tx_modem, rx_modem] - 1,
                        new_sample=bit_error
                    )

                    current_ber[snr_index, tx_modem, rx_modem] = bit_error

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
                    self.block_error_mean[snr_index, tx_modem, rx_modem] = self.update_mean(
                        old_mean=block_error_mean,
                        no_old_samples=self.block_error_num_drops[snr_index, tx_modem, rx_modem] - 1,
                        new_sample=block_error
                    )

                    current_bler[snr_index, tx_modem, rx_modem] = block_error
        self.bit_errors.append(current_ber)
        self.block_errors.append(current_bler)

    def update_mean(self, old_mean: float,
                             no_old_samples: float,
                             new_sample: float) -> float:
        return no_old_samples / (no_old_samples + 1) * old_mean + 1 / (no_old_samples+1) * new_sample


    def update_stopping_criteria(self, snr_index: int) -> None:
        # TODO: What to do if ber is none?
        for rx_modem_idx in range(self.__scenario.num_receivers):
            for tx_modem_idx in range(self.__scenario.num_transmitters):
                if self.__num_drops[snr_index] >= self.__min_num_drops:
                    if self.__flag_matrix[tx_modem_idx, rx_modem_idx, snr_index] == True:
                        mean_lower_bound, mean_upper_bound = self.estimate_confidence_intervals_mean(
                            np.array(self.bit_errors), self.__confidence_margin
                        )

                        self.bit_error_min[snr_index, tx_modem_idx, rx_modem_idx] = mean_lower_bound
                        self.bit_error_max[snr_index, tx_modem_idx, rx_modem_idx] = mean_upper_bound

                        confidence_margin = self.get_confidence_margin_ber_mean(
                            tx_modem_idx, rx_modem_idx, snr_index
                        )
                        self.__flag_matrix[tx_modem_idx, rx_modem_idx, snr_index] = (
                            confidence_margin > self.__confidence_margin
                        )

    def get_confidence_margin_ber_mean(self, tx_modem_idx: int,
                                             rx_modem_idx: int,
                                             snr_idx: int) -> float:
        """Calculates current confidence margin for BER mean."""
        old_settings = np.seterr(divide="ignore", invalid="ignore")
        confidence_margin = (
                (self.bit_error_max[snr_idx, tx_modem_idx, rx_modem_idx]
                 - self.bit_error_min[snr_idx, tx_modem_idx, rx_modem_idx])
                 / self.bit_error_mean[snr_idx, tx_modem_idx, rx_modem_idx]
        )
        np.seterr(**old_settings)
        return confidence_margin

    def estimate_confidence_intervals_mean(self, data: np.array, 
                                                 alpha: float) -> Tuple[float, float]:
        """Estimates bayesian confidence intervals for the mean.

        Args:
            data (np.array): Data samples.
            alpha (float): Probability that return confidence interval contains true parameter.

        Returns:
            (float, float): Lower and upper bound of estimated mean.
        """
        lower_bound = data[0]
        upper_bound = data[0]
        if len(data) > 1:
            estimates = stats.bayes_mvs(data=data, alpha=alpha)

            if not np.isnan(estimates[0][1][0]):
                lower_bound = estimates[0][1][0]
                if lower_bound < 0:
                    lower_bound = 0
                upper_bound = estimates[0][1][1]

        return lower_bound, upper_bound
        """for rx_modem_idx, received_signals in enumerate(received_bits):

            # get respective snr indices
            snr_indices = list(np.where(self.run_flag[rx_modem_idx])[0])

            # get respective tx_modem
            tx_modem = self.param_scenario.rx_modem_params[rx_modem_idx].tx_modem

            # since there are different SNR values added to the received signal,
            # we have multiple (#snr) signals for one receiver!

            for signal_idx, snr_idx in enumerate(snr_indices):
                rx_signal = [frame[signal_idx, :]
                             for frame in received_signals]
                error_stats = sources[tx_modem].get_number_of_errors(rx_signal)
                num_bits = error_stats.number_of_bits
                number_of_bit_errors = error_stats.number_of_bit_errors
                num_blocks = error_stats.number_of_blocks
                number_of_block_errors = error_stats.number_of_block_errors

                self.ber[rx_modem_idx][snr_idx] = np.append(
                    self.ber[rx_modem_idx][snr_idx], number_of_bit_errors / num_bits
                )
                self.fer[rx_modem_idx][snr_idx] = np.append(
                    self.fer[rx_modem_idx][snr_idx],
                    number_of_block_errors / num_blocks,
                )

                # update b(l)er statistics
                self.bit_error_sum[rx_modem_idx][snr_idx] = np.mean(
                    self.ber[rx_modem_idx][snr_idx]
                )
                self.bit_error_min[rx_modem_idx][snr_idx] = np.min(
                    self.ber[rx_modem_idx][snr_idx]
                )
                self.bit_error_max[rx_modem_idx][snr_idx] = np.max(
                    self.ber[rx_modem_idx][snr_idx]
                )

                self.block_error_sum[rx_modem_idx][snr_idx] = np.mean(
                    self.fer[rx_modem_idx][snr_idx]
                )
                self.block_error_min[rx_modem_idx][snr_idx] = np.min(
                    self.fer[rx_modem_idx][snr_idx]
                )
                self.block_error_max[rx_modem_idx][snr_idx] = np.max(
                    self.fer[rx_modem_idx][snr_idx]
                )

                # calculate confidence margins
                if self.__num_drops >= self.param_general.min_num_drops:
                    # define those to save typing
                    ber_lower = self.bit_error_min[rx_modem_idx][snr_idx]
                    fer_lower = self.block_error_min[rx_modem_idx][snr_idx]
                    ber_upper = self.bit_error_max[rx_modem_idx][snr_idx]
                    fer_upper = self.block_error_max[rx_modem_idx][snr_idx]

                    # start with BER calculation
                    ber_rx = self.ber[rx_modem_idx][snr_idx]

                    if self.__num_drops > 1:
                        ber_stats = stats.bayes_mvs(
                            ber_rx, alpha=self.param_general.confidence_level
                        )
                        if not np.isnan(ber_stats[0][1][0]):
                            ber_lower = ber_stats[0][1][0]
                            if ber_lower < 0:
                                ber_lower = 0
                            ber_upper = ber_stats[0][1][1]
                    else:
                        ber_upper = self.ber[rx_modem_idx][snr_idx]
                        ber_lower = self.ber[rx_modem_idx][snr_idx]

                    # do the same stuff for fer
                    fer_rx = self.fer[rx_modem_idx][snr_idx]

                    if self.__num_drops > 1:
                        fer_stats = stats.bayes_mvs(
                            fer_rx, alpha=self.param_general.confidence_level
                        )

                        if not np.isnan(fer_stats[0][1][0]):
                            fer_lower = fer_stats[0][1][0]
                            if fer_lower < 0:
                                fer_lower = 0
                            fer_upper = fer_stats[0][1][1]
                    else:
                        fer_lower = self.fer[rx_modem_idx][snr_idx]
                        fer_upper = self.fer[rx_modem_idx][snr_idx]

                    self.bit_error_min[rx_modem_idx][snr_idx] = ber_lower
                    self.block_error_min[rx_modem_idx][snr_idx] = fer_lower
                    self.bit_error_max[rx_modem_idx][snr_idx] = ber_upper
                    self.block_error_max[rx_modem_idx][snr_idx] = fer_upper

            # update stopping criteria
            if (
                self.param_general.confidence_margin > 0
                and self.__num_drops >= self.param_general.min_num_drops
            ):
                old_settings = np.seterr(divide="ignore", invalid="ignore")

                if self.param_general.confidence_metric == "BER":
                    confidence_margin = (
                                                self.bit_error_max[rx_modem_idx] -
                                                self.bit_error_min[rx_modem_idx]
                    ) / self.bit_error_sum[rx_modem_idx]
                elif self.param_general.confidence_metric == "fer":
                    confidence_margin = (
                                                self.block_error_max[rx_modem_idx] -
                                                self.block_error_min[rx_modem_idx]
                    ) / self.block_error_sum[rx_modem_idx]

                self.run_flag[rx_modem_idx] = np.logical_or(
                    confidence_margin > self.param_general.confidence_margin,
                    np.isnan(confidence_margin),
                )
                np.seterr(**old_settings)

            if self.param_general.verbose:
                print(f"Drop {self.__num_drops} (Rx {rx_modem_idx + 1}):")
                for snr, idx in zip(self.get_snr_list(
                        rx_modem_idx), snr_indices):
                    print(
                        "\tSNR = {:f} dB, BER = {:f}, fer = {:f}".format(
                            snr,
                            self.ber[rx_modem_idx][idx][-1],
                            self.fer[rx_modem_idx][idx][-1],
                        )
                    )"""

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

                mat_dict["frequency_tx_" + str(idx)] = fft.fftshift(frequency)
                mat_dict["power_spectral_density_tx_" + str(idx)] = fft.fftshift(periodogram) / np.amax(periodogram)

        if self.__calc_transmit_stft:
            for idx, (time, freq, power) in enumerate(self._stft_tx):

                mat_dict["stft_time_tx_" + str(idx)] = time
                mat_dict["stft_frequency_tx" + str(idx)] = freq
                mat_dict["stft_power_tx" + str(idx)] = power

        if self.__calc_receive_spectrum:
            for idx, (periodogram, frequency) in enumerate(zip(self._periodogram_rx, self._frequency_range_rx)):

                mat_dict["frequency_rx_" + str(idx)] = fft.fftshift(frequency)
                mat_dict["power_spectral_density_rx_" + str(idx)] = fft.fftshift(periodogram) / np.amax(periodogram)

        if self.__calc_receive_stft:
            for idx, (time, freq, power) in enumerate(self._stft_rx):

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
