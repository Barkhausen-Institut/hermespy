from typing import List, Tuple

import numpy as np

from modem.waveform_generator import WaveformGenerator
from parameters_parser.parameters_psk_qam import ParametersPskQam
from modem.tools.shaping_filter import ShapingFilter
from modem.tools.psk_qam_mapping import PskQamMapping


class WaveformGeneratorPskQam(WaveformGenerator):
    """This method provides a class for a generic PSK/QAM modem.

    The modem has the following characteristics:
    - root-raised cosine filter with arbitrary roll-off factor
    - arbitrary constellation, as defined in modem.tools.psk_qam_mapping:PskQamMapping

    This implementation has currently the following limitations:
    - SISO only
    - hard output only (no LLR)
    - no reference signal
    - ideal channel estimation
    - equalization of ISI with FMCW in AWGN channel only
    - no equalization (only amplitude and phase of first propagation path is compensated)
    """

    def __init__(self, param: ParametersPskQam) -> None:
        """Creates a modem object

        Args:
            param(ParametersPskQam): object containing all the relevant parameters
            source(BitsSource): bits source for transmitter
        """
        super().__init__(param)
        self.param = param
        self._set_frame_derived_parameters()
        self._set_filters()
        self._set_sampling_indices()
        self._set_pulse_correlation_matrix()

    def create_frame(self, timestamp: int,
                     data_bits: np.array) -> Tuple[np.ndarray, int, int]:
        frame = np.zeros(self._samples_in_frame, dtype=complex)
        frame[self._symbol_idx[:self.param.number_preamble_symbols]] = 1
        start_index_data = self.param.number_preamble_symbols
        end_index_data = self.param.number_preamble_symbols + self.param.number_data_symbols
        frame[self._symbol_idx[start_index_data: end_index_data]
              ] = self._mapping.get_symbols(data_bits)
        frame[self._symbol_idx[end_index_data:]] = 1

        output_signal = self._filter_tx.filter(frame)

        initial_sample_num = timestamp - self._filter_tx.delay_in_samples
        timestamp += self._samples_in_frame

        return output_signal[np.newaxis, :], timestamp, initial_sample_num

    def receive_frame(self,
                      rx_signal: np.ndarray,
                      timestamp_in_samples: int,
                      noise_var: float) -> Tuple[List[np.ndarray], np.ndarray]:

        useful_signal_length = self._samples_in_frame + self._filter_rx.delay_in_samples

        if rx_signal.shape[1] < useful_signal_length:
            bits = None
            rx_signal = np.array([])
        else:
            frame_signal = rx_signal[0, :useful_signal_length].ravel()
            symbol_idx = self._data_symbol_idx + self._filter_rx.delay_in_samples + self._filter_tx.delay_in_samples

            frame_signal = self._filter_rx.filter(frame_signal)

            # get channel gains (first tap only)
            timestamps = (timestamp_in_samples + symbol_idx) / self.param.sampling_rate
            channel = self._channel.get_impulse_response(timestamps)
            channel = channel[:, :, :, 0].ravel()

            # equalize
            rx_symbols = self._equalizer(frame_signal[symbol_idx], channel, noise_var)

            # detect
            bits = self._mapping.detect_bits(rx_symbols)

            rx_signal = rx_signal[:, self._samples_in_frame:]

        return list([bits]), rx_signal

    def _equalizer(self, data_symbols: np.ndarray, channel: np.ndarray, noise_var) -> np.ndarray:
        """Equalize the received data symbols

        This method applies a linear block equalizer to the received data symbols to compensate for intersymbol
        interference in case of non-orthogonal transmission pulses.
        The equalizer can be either NONE, ZF or MMSE, depending on parameter `self.param.equalizer`

        Note that currently  this is not a channel equalization, but it equalizes only the ISI in an AWGN channel.
        Only the amplitude and phase of the first path of the propagation channel is compensated.

        Args:
            data_symbols(np.ndarray): received data symbols after matched filtering
            channel(np.ndarray): one-path complex channel gain at the sampling instants of the data symbols
            noise_var(float): noise variance (for MMSE equalizer)

        Returns:
            equalized_signal(np.ndarray): data symbols after ISI equalization and channel compensation
        """

        if self._pulse_correlation_matrix.size:
            snr_factor = 0  # ZF
            h_matrix = self._pulse_correlation_matrix
            h_matrix_hermitian = h_matrix.conjugate().T

            if self.param.equalizer == "MMSE":
                snr_factor = noise_var * h_matrix

            isi_equalizer = np.matmul(h_matrix_hermitian,
                                      np.linalg.inv(np.matmul(h_matrix_hermitian, h_matrix) + snr_factor))

            equalized_symbols = np.matmul(isi_equalizer, data_symbols[:, np.newaxis]).flatten()
        else:
            equalized_symbols = data_symbols

        # compensate channel phase and amplitude
        equalized_symbols = equalized_symbols / channel

        return equalized_symbols

    def _set_frame_derived_parameters(self) -> None:
        """ Derives local frame-specific parameter based on parameter class
        """
        # derived parameters
        if self.param.number_preamble_symbols > 0 or self.param.number_postamble_symbols > 0:
            self._samples_in_pilot = int(
                np.round(
                    self.param.sampling_rate /
                    self.param.pilot_symbol_rate))
        else:
            self._samples_in_pilot = 0

        self._samples_in_guard = int(
            np.round(
                self.param.guard_interval *
                self.param.sampling_rate))

        self._samples_in_frame = int(
            (self.param.number_preamble_symbols +
             self.param.number_postamble_symbols) *
            self._samples_in_pilot +
            self._samples_in_guard +
            self.param.oversampling_factor *
            self.param.number_data_symbols)
        self._mapping = PskQamMapping(self.param.modulation_order, is_complex=self.param.modulation_is_complex)

    def _set_filters(self) -> None:
        """ Initializes transmit and reception filters based on parameter class
        """
        self._filter_tx = ShapingFilter(
            self.param.filter_type,
            self.param.oversampling_factor,
            is_matched=False,
            length_in_symbols=self.param.filter_length_in_symbols,
            roll_off=self.param.roll_off_factor,
            bandwidth_factor=self.param.bandwidth /
            self.param.symbol_rate)

        if self.param.filter_type == "RAISED_COSINE":
            # for raised cosine, receive filter is a low-pass filter with
            # bandwidth Rs(1+roll-off)/2
            self._filter_rx = ShapingFilter(
                "RAISED_COSINE",
                self.param.oversampling_factor,
                self.param.filter_length_in_symbols,
                0,
                1. + self.param.roll_off_factor)
        else:
            # for all other filter types, receive filter is a matched filter
            self._filter_rx = ShapingFilter(
                self.param.filter_type,
                self.param.oversampling_factor,
                is_matched=True,
                length_in_symbols=self.param.filter_length_in_symbols,
                roll_off=self.param.roll_off_factor,
                bandwidth_factor=self.param.bandwidth /
                self.param.symbol_rate)

        self._samples_overhead_in_frame = self._filter_rx.delay_in_samples

    def _set_sampling_indices(self) -> None:
        """ Determines the sampling instants for pilots and data at a given frame
        """
        # create a vector with the position of every pilot and data symbol in a
        # frame
        preamble_symbol_idx = np.arange(
            self.param.number_preamble_symbols) * self._samples_in_pilot
        start_idx = self.param.number_preamble_symbols * self._samples_in_pilot
        self._data_symbol_idx = start_idx + \
            np.arange(self.param.number_data_symbols) * \
            self.param.oversampling_factor
        start_idx += self.param.number_data_symbols * self.param.oversampling_factor
        postamble_symbol_idx = start_idx + \
            np.arange(self.param.number_postamble_symbols) * \
            self._samples_in_pilot
        self._symbol_idx = np.concatenate(
            (preamble_symbol_idx, self._data_symbol_idx, postamble_symbol_idx))

        self._data_symbol_idx += int(self.param.oversampling_factor / 2)
        self._symbol_idx += int(self.param.oversampling_factor / 2)

    def _set_pulse_correlation_matrix(self):
        """ Creates a matrix with autocorrelation among pulses at different instants
        """

        if self.param.filter_type == 'FMCW' and self.param.equalizer != "NONE":
            ######################################################################################
            # calculate the correlation matrix between data symbols sampled at different instants
            ######################################################################################

            # generate an NxN matrix with the time differences between the sampling instants of the N symbols
            # i.e., time_delay_matrix(i,j) = T_i - T_j, with T_i the sampling instant of the i-th symbol
            time_delay_matrix = np.zeros((self.param.number_data_symbols, self.param.number_data_symbols))
            for row in range(self.param.number_data_symbols):
                time_delay_matrix[row, :] = np.arange(row, row - self.param.number_data_symbols, -1)
            time_delay_matrix = time_delay_matrix / self.param.symbol_rate

            # the correlation between two symbols r_i and r_j is obtained as a known function of the difference between
            # their sampling instants
            non_zero_idx = np.nonzero(time_delay_matrix)
            isi_matrix = np.ones((self.param.number_data_symbols, self.param.number_data_symbols))
            isi_matrix[non_zero_idx] = (np.sin(np.pi * self.param.chirp_bandwidth * time_delay_matrix[non_zero_idx] *
                                               (1 - np.abs(time_delay_matrix[non_zero_idx]
                                                           / self.param.chirp_duration))) /
                                        (np.pi * self.param.chirp_bandwidth * time_delay_matrix[non_zero_idx]))

            time_idx = np.nonzero(np.abs(time_delay_matrix) > self.param.chirp_duration)
            isi_matrix[time_idx] = 0

            self._pulse_correlation_matrix = isi_matrix
        else:
            self._pulse_correlation_matrix = np.array([])

    def get_bit_energy(self) -> float:
        return 1 / self.param.bits_per_symbol

    def get_symbol_energy(self) -> float:
        return 1.0

    def get_power(self) -> float:
        return 1 / self.param.oversampling_factor
