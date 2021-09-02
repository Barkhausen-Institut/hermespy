import os
from typing import List

from matplotlib import pyplot as plt
import scipy.io as sio
from scipy import signal
from scipy import stats
import numpy as np

from source.bits_source import BitsSource
from parameters_parser.parameters_general import ParametersGeneral
from parameters_parser.parameters_scenario import ParametersScenario
from modem.modem import Modem
from simulator_core.tools.theoretical_results import TheoreticalResults


class Statistics:
    """Generates, saves and plots the result statistics of a given simulation.

    Attributes:
        param_general(ParametersGeneral): Parameters related to simulation.
        param_scenario(ParametersScenario): Scenario-specific settings.
        num_drops(int): Number of drops already performed.
        num_snrs(int): Different number of snrs to perform simulation for.
        number_of_tx_signals(int):
        run_flag[List[np.array]]:
            Each list item corresponds to one receiver modem.
            Each list item is a np.array stating if results are to be calculated
            for the corresponding SNR value.
    """

    def __init__(
        self,
        parameters: ParametersGeneral,
        scenario: ParametersScenario,
        tx_modems: List[Modem],
        rx_modems: List[Modem]
    ) -> None:

        self.param_general = parameters
        self.param_scenario = scenario
        self.num_drops = 0

        self.num_snrs = self.param_general.snr_vector.size

        self.number_of_tx_signals = len(self.param_scenario.tx_modem_params)
        self.no_rx_modems = len(self.param_scenario.rx_modem_params)

        self.run_flag = [
            np.ones(self.num_snrs, dtype=bool)
            for rx_modem_idx in range(self.no_rx_modems)
        ]
        self.ber = [
            [np.empty(0) for idx in range(self.num_snrs)]
            for idx in range(self.no_rx_modems)
        ]

        self.fer = [
            [np.empty(0) for idx in range(self.num_snrs)]
            for idx in range(self.no_rx_modems)
        ]

        self.ber_mean = [
            np.ones(self.num_snrs) * np.nan for idx in range(self.no_rx_modems)
        ]
        self.ber_lower = [
            np.ones(self.num_snrs) * np.nan for idx in range(self.no_rx_modems)
        ]
        self.ber_upper = [
            np.ones(self.num_snrs) * np.nan for idx in range(self.no_rx_modems)
        ]

        self.fer_mean = [
            np.ones(self.num_snrs) * np.nan for idx in range(self.no_rx_modems)
        ]
        self.fer_lower = [
            np.ones(self.num_snrs) * np.nan for idx in range(self.no_rx_modems)
        ]
        self.fer_upper = [
            np.ones(self.num_snrs) * np.nan for idx in range(self.no_rx_modems)
        ]

        self._tx_sampling_rate = [
            modem.waveform_generator.sampling_rate for modem in tx_modems
        ]

        self._rx_sampling_rate = [
            modem.waveform_generator.sampling_rate for modem in rx_modems
        ]

        self._frequency_range_tx = [
            np.zeros(self.param_general.spectrum_fft_size)
            for idx in range(self.number_of_tx_signals)
        ]

        self._frequency_range_rx = [np.array([]) for idx in range(self.no_rx_modems)]

        self._periodogram_tx = [
            np.zeros(self.param_general.spectrum_fft_size)
            for idx in range(self.number_of_tx_signals)
        ]

        self._periodogram_rx = [np.array([]) for idx in range(self.no_rx_modems)]

        self._stft_freq_tx: List[np.array] = [np.array([])]
        self._stft_time_tx: List[np.array] = [np.array([])]
        self._stft_power_tx: List[np.array] = [np.array([])]

        self._stft_freq_rx = [np.array([]) for idx in range(self.no_rx_modems)]
        self._stft_time_rx = [np.array([]) for idx in range(self.no_rx_modems)]
        self._stft_power_rx = [np.array([]) for idx in range(self.no_rx_modems)]

        if parameters.calc_theory:
            self.theoretical_results = TheoreticalResults(scenario)
        else:
            self.theoretical_results = None

    def update_tx_spectrum(self, all_tx_signals: List[np.ndarray]) -> None:
        """updates spectrum analysis for all the transmit signals in "all_tx_signals" at a drop.

        Welch's method is employed for spectral analysis. For multiple antennas, onl the first antenna is considered.
        """
        if self.param_general.calc_spectrum_tx:
            for sampling_rate, frequency_range, periodogram, tx_signal in zip(
                self._tx_sampling_rate,
                self._frequency_range_tx,
                self._periodogram_tx,
                all_tx_signals,
            ):
                # make sure that signal is at least as long as FFT and pad it
                # with zeros is needed
                if tx_signal.shape[1] < self.param_general.spectrum_fft_size:
                    number_of_antennas = tx_signal.shape[0]
                    tx_signal = np.concatenate((tx_signal,
                                                np.zeros((number_of_antennas, self.param_general.spectrum_fft_size -
                                                         tx_signal.size))))

                freq, new_periodogram = signal.welch(
                    tx_signal[0, :],
                    fs=sampling_rate,
                    nperseg=self.param_general.spectrum_fft_size,
                    noverlap=self.param_general.spectrum_fft_size / 2,
                    return_onesided=False,
                )
                frequency_range[:] = freq
                periodogram += new_periodogram

        if self.param_general.calc_stft_tx and self._stft_freq_tx[0].size == 0:
            self._stft_freq_tx = []
            self._stft_time_tx = []
            self._stft_power_tx = []

            for sampling_rate, tx_signal in zip(
                    self._tx_sampling_rate, all_tx_signals):
                # make sure that signal is at least as long as FFT and pad it
                # with zeros is needed
                if tx_signal.shape[1] < self.param_general.spectrum_fft_size:
                    number_of_antennas = tx_signal.shape[0]
                    tx_signal = np.concatenate((tx_signal,
                                                np.zeros(number_of_antennas, self.param_general.spectrum_fft_size -
                                                         tx_signal.size)))

                f, t, Zxx = signal.stft(tx_signal[0, :], sampling_rate,
                                        nperseg=self.param_general.spectrum_fft_size,
                                        noverlap=self.param_general.spectrum_fft_size / 2,
                                        return_onesided=False)
                self._stft_freq_tx.append(np.fft.fftshift(f))
                self._stft_time_tx.append(t)
                self._stft_power_tx.append(np.fft.fftshift(Zxx, axes=0))

    def update_rx_spectrum(self, rx_signal: np.ndarray, rx_idx: int) -> None:
        """updates spectrum analysis for a given received signal

        Welch's method is employed for spectral analysis. For MIMO, only first antenna is considered.

        Args:
                rx_signal(numpy.ndarray): received signal

                rx_idx(int): modem index of the received signal
        """

        rx_signal = rx_signal[0, :].ravel()
        # make sure that signal is at least as long as FFT and pad it with
        # zeros is needed
        if rx_signal.size < self.param_general.spectrum_fft_size:
            rx_signal = np.concatenate((rx_signal, np.zeros(self.param_general.spectrum_fft_size -
                                                            rx_signal.size)))

        if self.param_general.calc_spectrum_rx and self._periodogram_rx[rx_idx].size == 0:

            self._frequency_range_rx[rx_idx], self._periodogram_rx[rx_idx] = signal.welch(
                rx_signal, fs=self._rx_sampling_rate[rx_idx],
                nperseg=self.param_general.spectrum_fft_size, return_onesided=False,
            )

        if self.param_general.calc_stft_rx and self._stft_freq_rx[rx_idx].size == 0:
            f, self._stft_time_rx[rx_idx], Zxx = signal.stft(rx_signal.ravel(), self._rx_sampling_rate[rx_idx],
                                                             return_onesided=False)
            self._stft_freq_rx[rx_idx] = np.fft.fftshift(f)
            self._stft_power_rx[rx_idx] = np.fft.fftshift(Zxx, axes=0)

    def update_error_rate(
        self, sources: List[BitsSource], detected_bits: List[List[np.array]]
    ) -> None:
        """Calculates error rate between received signal in bits and source bits.

        It is also checked whether the stopping crtieria need to be updated.

        Args:
            sources (List[BitsSource]): list of sources for the tx modems
            detected_bits (List[List[np.array]]):
                List of bits received by the receiver. Each list item corresponds
                to a receiving modem. A list item contains a list, which in turn
                represents a frame. This np.array of the frame is of size
                `self.num_snrs x no_blocks x bits_length`
                with the detoriated signals.
        """
        for rx_modem_idx in range(len(detected_bits)):
            if detected_bits[rx_modem_idx][0].shape[0] != sum(
                    self.run_flag[rx_modem_idx]):
                raise ValueError(
                    "'detected_bits' and 'snr_vector' must have the same length"
                )
        self.num_drops += 1

        # iterate over receivers and its signals received
        for rx_modem_idx, received_signals in enumerate(detected_bits):
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
                self.ber_mean[rx_modem_idx][snr_idx] = np.mean(
                    self.ber[rx_modem_idx][snr_idx]
                )
                self.ber_lower[rx_modem_idx][snr_idx] = np.min(
                    self.ber[rx_modem_idx][snr_idx]
                )
                self.ber_upper[rx_modem_idx][snr_idx] = np.max(
                    self.ber[rx_modem_idx][snr_idx]
                )

                self.fer_mean[rx_modem_idx][snr_idx] = np.mean(
                    self.fer[rx_modem_idx][snr_idx]
                )
                self.fer_lower[rx_modem_idx][snr_idx] = np.min(
                    self.fer[rx_modem_idx][snr_idx]
                )
                self.fer_upper[rx_modem_idx][snr_idx] = np.max(
                    self.fer[rx_modem_idx][snr_idx]
                )

                # calculate confidence margins
                if self.num_drops >= self.param_general.min_num_drops:
                    # define those to save typing
                    ber_lower = self.ber_lower[rx_modem_idx][snr_idx]
                    fer_lower = self.fer_lower[rx_modem_idx][snr_idx]
                    ber_upper = self.ber_upper[rx_modem_idx][snr_idx]
                    fer_upper = self.fer_upper[rx_modem_idx][snr_idx]

                    # start with BER calculation
                    ber_rx = self.ber[rx_modem_idx][snr_idx]

                    if self.num_drops > 1:
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

                    if self.num_drops > 1:
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

                    self.ber_lower[rx_modem_idx][snr_idx] = ber_lower
                    self.fer_lower[rx_modem_idx][snr_idx] = fer_lower
                    self.ber_upper[rx_modem_idx][snr_idx] = ber_upper
                    self.fer_upper[rx_modem_idx][snr_idx] = fer_upper

            # update stopping criteria
            if (
                self.param_general.confidence_margin > 0
                and self.num_drops >= self.param_general.min_num_drops
            ):
                old_settings = np.seterr(divide="ignore", invalid="ignore")

                if self.param_general.confidence_metric == "BER":
                    confidence_margin = (
                        self.ber_upper[rx_modem_idx] -
                        self.ber_lower[rx_modem_idx]
                    ) / self.ber_mean[rx_modem_idx]
                elif self.param_general.confidence_metric == "fer":
                    confidence_margin = (
                        self.fer_upper[rx_modem_idx] -
                        self.fer_lower[rx_modem_idx]
                    ) / self.fer_mean[rx_modem_idx]

                self.run_flag[rx_modem_idx] = np.logical_or(
                    confidence_margin > self.param_general.confidence_margin,
                    np.isnan(confidence_margin),
                )
                np.seterr(**old_settings)

            if self.param_general.verbose:
                print(f"Drop {self.num_drops} (Rx {rx_modem_idx+1}):")
                for snr, idx in zip(self.get_snr_list(
                        rx_modem_idx), snr_indices):
                    print(
                        "\tSNR = {:f} dB, BER = {:f}, fer = {:f}".format(
                            snr,
                            self.ber[rx_modem_idx][idx][-1],
                            self.fer[rx_modem_idx][idx][-1],
                        )
                    )

    def get_snr_list(self, rx_modem_idx: int) -> List[int]:
        return self.param_general.snr_vector[self.run_flag[rx_modem_idx]]

    def save(self, results_dir: str) -> None:
        """averages out the stored statistics from all the drops and store them in a matlab file.

        Theoretical values (if available) are stored as well. If 'plot' is True,
        then plot results.

        Args:
            results_dir (str): the desired directory to save matlab file in.
        """
        filename = os.path.join(results_dir, "statistics.mat")

        if self.param_general.snr_type == 'EB/N0(DB)':
            snr_str = 'Eb/N0'
        elif self.param_general.snr_type == 'ES/N0(DB)':
            snr_str = 'Es/N0'
        elif self.param_general.snr_type == 'CUSTOM':
            snr_str = 'Custom'
        else:
            raise ValueError('invalid SNR type')

        print("Simulation Finished")
        for rx_modem_idx in range(self.param_scenario.number_of_rx_modems):
            print(f"\n\tResults for Rx {rx_modem_idx}")

            for snr, ber, fer in zip(
                self.param_general.snr_vector, self.ber_mean[rx_modem_idx], self.fer_mean[rx_modem_idx]
            ):
                print(f"\t{snr_str} = {snr}dB\tBER = {ber:e}, \tfer = {fer:e}")

        mat_dict = {
            "snr_type": self.param_general.snr_type,
            "snr_vector": self.param_general.snr_vector,
            "ber_mean": self.ber_mean,
            "fer_mean": self.fer_mean,
            "ber_lower": self.ber_lower,
            "ber_upper": self.ber_upper,
            "fer_lower": self.fer_lower,
            "fer_upper": self.fer_upper,
        }
        if self.param_general.calc_spectrum_tx:
            for idx, (periodogram, frequency) in enumerate(
                zip(self._periodogram_tx, self._frequency_range_tx)
            ):
                periodogram[:] = np.fft.fftshift(
                    periodogram) / np.amax(periodogram)
                frequency[:] = np.fft.fftshift(frequency)
                mat_dict["frequency_tx_" + str(idx)] = frequency
                mat_dict["power_spectral_density_tx_" + str(idx)] = periodogram

        if self.param_general.calc_stft_tx:
            for idx, (time, freq, power) in enumerate(
                zip(self._stft_time_tx, self._stft_freq_tx, self._stft_power_tx)
            ):
                mat_dict["stft_time_tx_" + str(idx)] = time
                mat_dict["stft_frequency_tx" + str(idx)] = freq
                mat_dict["stft_power_tx" + str(idx)] = power

        if self.param_general.calc_spectrum_rx:
            for idx, (periodogram, frequency) in enumerate(
                zip(self._periodogram_rx, self._frequency_range_rx)
            ):
                periodogram[:] = np.fft.fftshift(
                    periodogram) / np.amax(periodogram)
                frequency[:] = np.fft.fftshift(frequency)
                mat_dict["frequency_rx_" + str(idx)] = frequency
                mat_dict["power_spectral_density_rx_" + str(idx)] = periodogram

        if self.param_general.calc_stft_rx:
            for idx, (time, freq, power) in enumerate(
                zip(self._stft_time_rx, self._stft_freq_rx, self._stft_power_rx)
            ):
                mat_dict["stft_time_rx_" + str(idx)] = time
                mat_dict["stft_frequency_rx_" + str(idx)] = freq
                mat_dict["stft_power_rx_" + str(idx)] = power

        theory = None
        if self.theoretical_results is not None:
            theory = self.theoretical_results.get_results(
                self.param_general.snr_type, self.param_general.snr_vector
            )

        ber_theory = []
        fer_theory = []
        theory_notes = []

        if theory is not None:
            for rx_modem_idx in range(self.param_scenario.number_of_rx_modems):
                if not theory[rx_modem_idx]:
                    ber_theory.append(np.nan)
                    fer_theory.append(np.nan)
                    theory_notes.append("")
                else:
                    ber_theory.append(theory[rx_modem_idx]["ber"])
                    if "fer" in theory[rx_modem_idx].keys():
                        fer_theory.append(theory[rx_modem_idx]["fer"])
                    else:
                        fer_theory.append(np.nan)
                    theory_notes.append(theory[rx_modem_idx]["notes"])

            mat_dict["ber_theory"] = ber_theory
            mat_dict["fer_theory"] = fer_theory
            mat_dict["theory_notes"] = theory_notes

        sio.savemat(filename, mat_dict)

        if self.param_general.plot:

            for rx_modem_idx in range(self.param_scenario.number_of_rx_modems):
                plt.figure()
                if ber_theory[rx_modem_idx] is not np.nan:
                    plt.plot(
                        self.param_general.snr_vector,
                        ber_theory[rx_modem_idx],
                        label="theory")

                error = np.vstack(
                    (self.ber_mean[rx_modem_idx] - self.ber_lower[rx_modem_idx],
                     self.ber_upper[rx_modem_idx] - self.ber_mean[rx_modem_idx])
                )
                plt.errorbar(
                    self.param_general.snr_vector, self.ber_mean[rx_modem_idx], error, label="simulation"
                )
                plt.yscale("log", nonposy="mask")
                plt.title("Rx" + str(rx_modem_idx))
                plt.xlabel(snr_str + "(dB)")
                plt.ylabel("BER")
                plt.grid()
                plt.legend()

                filename = os.path.join(
                    results_dir, "BER_Rx" + str(rx_modem_idx) + ".png")
                plt.savefig(filename)

            for rx_modem_idx in range(self.param_scenario.number_of_rx_modems):
                plt.figure()

                if fer_theory[rx_modem_idx] is not np.nan:
                    plt.plot(
                        self.param_general.snr_vector,
                        fer_theory[rx_modem_idx],
                        label="theory")

                error = np.vstack(
                    (self.fer_mean[rx_modem_idx] - self.fer_lower[rx_modem_idx],
                     self.fer_upper[rx_modem_idx] - self.fer_mean[rx_modem_idx])
                )
                plt.errorbar(
                    self.param_general.snr_vector, self.fer_mean[rx_modem_idx], error, label="simulation"
                )
                plt.yscale("log", nonposy="mask")
                plt.title("Rx" + str(rx_modem_idx))
                plt.xlabel(snr_str + "(dB)")
                plt.ylabel("fer")
                plt.grid()
                plt.legend()

                filename = os.path.join(
                    results_dir, "fer_" + str(rx_modem_idx) + ".png")
                plt.savefig(filename)

            if self.param_general.calc_spectrum_tx:
                for idx, (periodogram, frequency) in enumerate(
                    zip(self._periodogram_tx, self._frequency_range_tx)
                ):
                    plt.figure()
                    plt.plot(frequency, 10 * np.log10(periodogram))
                    plt.xlabel("frequency (Hz)")
                    plt.ylabel("dB")
                    plt.title(f"Power Spectral density of TX {idx} Signal")

                    filename = os.path.join(
                        results_dir, "PSD_TX_" + str(idx) + ".png")
                    plt.savefig(filename)

            if self.param_general.calc_stft_tx:
                for idx, (t, f, Zxx) in enumerate(
                    zip(self._stft_time_tx, self._stft_freq_tx, self._stft_power_tx)
                ):
                    plt.figure()
                    plt.pcolormesh(t, f, np.abs(Zxx))
                    plt.title(f"STFT of TX {idx} signal")
                    plt.ylabel("Frequency [Hz]")
                    plt.xlabel("Time [sec]")

                    filename = os.path.join(
                        results_dir, "STFT_TX_" + str(idx) + ".png")
                    plt.savefig(filename)

            if self.param_general.calc_spectrum_rx:
                for idx, (periodogram, frequency) in enumerate(
                    zip(self._periodogram_rx, self._frequency_range_rx)
                ):
                    plt.figure()
                    plt.plot(frequency, 10 * np.log10(periodogram.ravel()))
                    plt.xlabel("frequency (Hz)")
                    plt.ylabel("dB")
                    plt.title(f"Power Spectral density of RX {idx} Signal")

                    filename = os.path.join(
                        results_dir, "PSD_RX_" + str(idx) + ".png")
                    plt.savefig(filename)

            if self.param_general.calc_stft_rx:
                for idx, (t, f, Zxx) in enumerate(
                    zip(self._stft_time_rx, self._stft_freq_rx, self._stft_power_rx)
                ):
                    plt.figure()
                    plt.pcolormesh(t, f, np.abs(Zxx))
                    plt.title(f"STFT of RX {idx} Signal")
                    plt.ylabel("Frequency [Hz]")
                    plt.xlabel("Time [sec]")

                    filename = os.path.join(
                        results_dir, "STFT_RX_" + str(idx) + ".png")
                    plt.savefig(filename)

            plt.show()
