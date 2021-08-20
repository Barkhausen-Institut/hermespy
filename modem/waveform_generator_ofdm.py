from typing import List, Tuple
from copy import copy

import numpy as np
from scipy import signal
from scipy import interpolate

from modem.waveform_generator import WaveformGenerator
from parameters_parser.parameters_ofdm import (
    ParametersOfdm, OfdmSymbolConfig, GuardInterval, ResourceType, ResourcePattern)
from modem.tools.psk_qam_mapping import PskQamMapping
from modem.tools.mimo import Mimo


class WaveformGeneratorOfdm(WaveformGenerator):
    """This module provides a class for a generic OFDM modem, with a flexible frame configuration.

    The following features are supported:
    - The modem can transmit or receive custom-defined frames. (see class ParametersOfdm). The frame may contain UL/DL data
    symbols, null carriers, pilot subcarriers, reference signals and guard intervals.
    - SC-FDMA can also be implemented with a precoder.
    - Subcarriers can be modulated with BPSK/QPSK/16-/64-/256-QAM.
    - cyclic-prefix OFDM are supported.

    This implementation has currently the following limitations:
    - all subcarriers use the same modulation scheme
    - ideal channel estimation assumed
    
    Attributes:
    param (ParametersOfdm): OFDM-specific parameters.
    reference_frame (numpy.ndarray): a 3D array containing the reference symbols in frequency domain. The array is of
        size N_symb x K_sc x M_tx, with N_symb the number of OFDM symbols, K_sc the number of occupied subcarriers and
        N_tx the number of transmit antennas
    data_frame_indices (numpy.ndarray): a 3D boolean array (N_symb x K_sc x M_tx )indicating the position of all data
        subcarriers
    guard_time_indices (numpy.ndarray):
    prefix_time_indices (numpy.ndarray):
    data_time_indices (numpy.ndarray): vectors containing the indices of the guard intervals, prefixes and data in time
        samples, considering sampling at the FFT rate
    channel_sampling_timestamps (numpy.ndarray): vector containing the timestamps (in terms of nor obersampled samples)
        of each OFDM symbol
    """
    def __init__(self, param: ParametersOfdm) -> None:
        super().__init__(param)
        self.param = param
        self._samples_in_frame_no_oversampling = 0

        self._mapping = PskQamMapping(self.param.modulation_order)

        self._mimo = Mimo(mimo_method=self.param.mimo_scheme,
                          number_tx_antennas=self.param.number_tx_antennas,
                          number_of_streams=self.param.number_streams)
        self._resource_element_mapping: np.array = self._calculate_resource_element_mapping()
        self._samples_in_frame_no_oversampling, self._cyclic_prefix_overhead = (
            self._calculate_samples_in_frame()
        )

        self._number_ofdm_symbols = sum(isinstance(frame_element, OfdmSymbolConfig)
                                        for frame_element in self.param.frame_structure)
         
        self.reference_frame = np.zeros((self._number_ofdm_symbols, self.param.number_occupied_subcarriers,
                                         self.param.number_tx_antennas), dtype=complex)
        self.data_frame_indices = np.zeros((self._number_ofdm_symbols, self.param.number_occupied_subcarriers,
                                           self.param.number_tx_antennas), dtype=bool)
        self.guard_time_indices = np.array([], dtype=int)
        self.prefix_time_indices = np.array([], dtype=int)
        self.data_time_indices = np.array([], dtype=int)
        self.channel_sampling_timestamps = np.array([], dtype=int)

        # derived variables for precoding
        self._data_resource_elements_per_symbol = np.array([])

        self._generate_frame_structure()

    def _generate_frame_structure(self):
        """Creates the OFDM frame structure in time, frequency and space.

        This method interprets the OFDM parameters in 'self.param' that describe the OFDM frame and generates matrices
        with the allocation of all resource elements in a time/frequency/antenna grid

        """
        ofdm_symbol_idx = 0
        sample_idx = 0
        self.channel_sampling_timestamps = np.array([], dtype=int)

        for frame_element in self.param.frame_structure:
            if isinstance(frame_element, GuardInterval):
                self.guard_time_indices = np.append(self.guard_time_indices,
                                                    np.arange(sample_idx, sample_idx + frame_element.no_samples))
                sample_idx += frame_element.no_samples

            elif isinstance(frame_element, OfdmSymbolConfig):
                ref_idxs = self._get_subcarrier_indices(frame_element, ResourceType.REFERENCE)
                self.channel_sampling_timestamps = np.append(self.channel_sampling_timestamps, sample_idx)

                # fill out resource elements with pilot symbols
                ref_symbols = np.tile(self.param.reference_symbols,
                                      int(np.ceil(ref_idxs.size / self.param.reference_symbols.size)))
                ref_symbols = ref_symbols[:ref_idxs.size]

                self.reference_frame[ofdm_symbol_idx, ref_idxs, 0] = ref_symbols

                # update indices for data and (cyclic) prefix
                data_idxs = self._get_subcarrier_indices(frame_element, ResourceType.DATA)
                self.data_frame_indices[ofdm_symbol_idx, data_idxs, :] = True

                self.prefix_time_indices = np.append(self.prefix_time_indices,
                                                     np.arange(sample_idx, sample_idx +
                                                               frame_element.cyclic_prefix_samples))
                sample_idx += frame_element.cyclic_prefix_samples
                self.data_time_indices = np.append(self.data_time_indices,
                                                   np.arange(sample_idx, sample_idx + frame_element.no_samples))
                sample_idx += frame_element.no_samples

                ofdm_symbol_idx += 1

        if self.param.precoding != "NONE":
            # check if all symbols have the same number of data REs
            self._data_resource_elements_per_symbol = np.sum(self.data_frame_indices[:, :, 0], axis=1)

    def _get_subcarrier_indices(self, frame_element, resource_type):
        #############################################################################
        # calculate indices for data and pilot resource elements in this OFDM symbol
        subcarrier_idx = 0
        resource_idxs: np.array = np.array([], dtype=int)

        for res_pattern in frame_element.resource_types:
            for pattern_el_idx in range(res_pattern.number):
                for res in res_pattern.MultipleRes:
                    if res.ResourceType == resource_type:
                        resource_idxs = np.append(resource_idxs, np.arange(subcarrier_idx, subcarrier_idx + res.number))
                    subcarrier_idx += res.number
        return resource_idxs

    def _calculate_samples_in_frame(self) -> Tuple[int, float]:
        samples_in_frame_no_oversampling = 0
        number_cyclic_prefix_samples = 0
        number_of_data_samples = 0

        for frame_element in self.param.frame_structure:
            if isinstance(frame_element, GuardInterval):
                samples_in_frame_no_oversampling += frame_element.no_samples
            else:
                samples_in_frame_no_oversampling += frame_element.cyclic_prefix_samples
                number_cyclic_prefix_samples += frame_element.cyclic_prefix_samples

                samples_in_frame_no_oversampling += frame_element.no_samples
                number_of_data_samples += frame_element.no_samples

        cyclic_prefix_overhead = (number_of_data_samples + number_cyclic_prefix_samples) / number_of_data_samples

        return samples_in_frame_no_oversampling, cyclic_prefix_overhead

    def _calculate_resource_element_mapping(self) -> np.array:
        initial_index = self.param.fft_size - \
            int(np.ceil(self.param.number_occupied_subcarriers / 2))
        resource_element_mapping: np.array = np.arange(
            initial_index, self.param.fft_size)
        final_index = int(np.floor(self.param.number_occupied_subcarriers / 2))
        resource_element_mapping = np.append(
            resource_element_mapping, np.arange(
                self.param.dc_suppression, final_index + self.param.dc_suppression))
        return resource_element_mapping

    ###################################
    # property definitions
    @property
    def samples_in_frame(self) -> int:
        """int: Returns read-only samples_in_frame"""
        return self._samples_in_frame_no_oversampling * self.param.oversampling_factor

    @property
    def bits_in_frame(self) -> int:
        """int: Returns read-only bits_in_frame"""
        return self.param.bits_in_frame

    @property
    def cyclic_prefix_overhead(self) -> float:
        """int: Returns read-only cyclic_prefix_overhead"""
        return self._cyclic_prefix_overhead

    # property definitions END
    #############################################

    def create_frame(self, timestamp: int, data_bits: np.array) -> Tuple[np.ndarray, int, int]:
        """Creates a modulated complex baseband signal for a whole transmit frame.

        The signal will be modulated based on the bits generated by "self.source".

        Args:
            timestamp(int): timestamp (in samples) of initial sample in frame
            data_bits (np.array):
                Flattened blocks, whose bits are supposed to fit into this frame.

        Returns:
            (np.ndarray, int, int):
            
            `output_signal(numpy.ndarray)`: 2D array containing the transmitted signal with
            (self.param.number_tx_antennas x self.samples_in_frame) elements

            `timestamp(int)`: current timestamp (in samples) of the following frame

            `initial_sample_num(int)`: sample in which this frame starts (equal to initial timestamp)
        """
        output_signal: np.ndarray = np.zeros(
            (self.param.number_tx_antennas, self._samples_in_frame_no_oversampling),
            dtype=complex)

        # fill time-frequency grid with reference and data symbols
        data_symbols_in_frame = self._mapping.get_symbols(data_bits)

        # MIMO if needed
        data_symbols_in_frame = self._mimo.encode(data_symbols_in_frame)
        data_symbols_in_frame = data_symbols_in_frame.flatten('F')

        # data is mapped across all frequencies first
        full_frame = copy(self.reference_frame)
        full_frame[np.where(self.data_frame_indices)] = data_symbols_in_frame

        full_frame = self._precode(full_frame)

        output_signal = self.create_ofdm_frame_time_domain(full_frame)

        initial_sample_num = timestamp
        timestamp += self.samples_in_frame

        if self.param.oversampling_factor > 1:
            output_signal = signal.resample_poly(
                output_signal, self.param.oversampling_factor, 1, axis=1)
        return output_signal, timestamp, initial_sample_num

    def create_ofdm_frame_time_domain(self, frame: np.ndarray):
        """Creates one OFDM frame in time domain.

        Args:
            frame(numpy.array): a 3D array containing the symbols in frequency domain.
                The array is of size N_symb x K_sc x M_tx, with N_symb the number of OFDM symbols, K_sc the number of
                occupied subcarriers and N_tx the number of transmit antennas

        Returns:
            frame_in_time_domain(numpy.array): an M_tx x N_samp array containing the time-domain OFDM frame.
                Note that the samples are at the FFT sampling rate, not considering any oversampling factor.
        """
        frame_in_freq_domain = np.zeros((self._number_ofdm_symbols, self.param.fft_size, self.param.number_tx_antennas),
                                        dtype=complex)
        frame_in_freq_domain[:, self._resource_element_mapping, :] = frame

        frame_in_time_domain = np.fft.ifft(frame_in_freq_domain, norm='ortho', axis=1)
        frame_in_time_domain = self._add_guard_intervals(frame_in_time_domain)

        return frame_in_time_domain

    def _add_guard_intervals(self, frame):
        """Adds guard intervals and cyclic prefixes to a time-domain OFDM frame.

        The position of the null guard intervals and the length of the cyclic prefixes are defined in
        self.param.frame_structure.

        Args:
            frame(numpy.array): a 2D array containing the raw OFDM symbols in time domain. It is of size
                N_symb x N_fft x M_tx, with M_tx the number of transmit antennas and N_symb the number of symbols.

        Returns:
            output_signal(numpy.array): an M_tx x N_samp array containing the time-domain OFDM frame.
        """
        output_signal: np.ndarray = np.zeros((self.param.number_tx_antennas, self._samples_in_frame_no_oversampling),
                                             dtype=complex)

        data_symbols = np.reshape(frame, (self._number_ofdm_symbols * self.param.fft_size,
                                          self.param.number_tx_antennas))
        data_symbols = data_symbols.transpose()
        output_signal[:, self.data_time_indices] = data_symbols
        output_signal[:, self.prefix_time_indices] = output_signal[:, self.prefix_time_indices + self.param.fft_size]

        return output_signal

    def _precode(self, frame):
        """Precode the frequemcy-domain OFDM frame

        The precoding algorithm is defined in 'self.param.precoding'. Currently, only DFT-spread precoding is supported

        Args:
            frame(numpy.array): a 3D array(N_symb x K_sc x M_tx) containing the OFDM resource elements

        Returns:
            frame(numpy.array): the precoded frame
        """
        if self.param.precoding == 'DFT':
            # iterate over all symbols as they may have different number of data REs
            for symbol in range(self._number_ofdm_symbols):
                data_indices = self.data_frame_indices[symbol, :, 0]
                if np.any(data_indices):
                    data_symbols = frame[symbol, data_indices, :]
                    data_symbols = np.fft.fft(data_symbols, axis=0, norm='ortho')
                    frame[symbol, data_indices, :] = data_symbols

        return frame

    def receive_frame(self,
                      rx_signal: np.ndarray,
                      timestamp_in_samples: int,
                      noise_var: float) -> Tuple[List[np.ndarray], np.ndarray]:
        """Demodulates the signal for a whole received frame.

        This method extracts a signal frame from 'rx_signal' and demodulates it according to
        the frame and modulation parameters.

        Args:
            rx_signal(numpy.ndarray):
                N x S array containg the received signal, with N the number of receive antennas
                and S the number of samples left in the drop.
            timestamp_in_samples(int):
                timestamp of initial sample in received signal, relative to the first sample in
                the simulation drop.
            noise_var (float): noise variance (for equalization).

        Returns:
            (list[np.ndarray], np.ndarray):
                `list[numpy.ndarray]`: 
                    list of detected blocks of bits.
                `numpy.ndarray`:
                    N x S' array containing the remaining part of the signal, after this frame was
                    demodulated.  S' = S - self.samples_in_frame
        """
        if rx_signal.shape[1] < self.samples_in_frame:
            bits = None
            rx_signal = np.array([])
        else:
            bits = np.array([])
            frame_signal = rx_signal[:, :self.samples_in_frame]
            rx_signal = rx_signal[:, self.samples_in_frame:]

            if self.param.oversampling_factor > 1:
                frame_signal = signal.decimate(frame_signal, self.param.oversampling_factor)

            frame_in_freq_domain = self._get_frame_in_freq_domain(copy(frame_signal))
            channel_estimation = self.channel_estimation(frame_in_freq_domain, timestamp_in_samples)

            frame_symbols, noise_var = self._equalize(frame_in_freq_domain, channel_estimation, noise_var)

            frame_symbols, noise_var = self._decode(frame_symbols, noise_var)
            bits = self._mapping.detect_bits(frame_symbols.flatten('F'), noise_var.flatten('F'))

        return list([bits]), rx_signal

    def _get_frame_in_freq_domain(self, frame_in_time_domain: np.ndarray):
        """Converts a frame from time to frequency domain

        This method removes all the guard intervals and prefixes, and converts time domain to frequency domain.

        Args:
            frame_in_time_domain(numpy.array): a M_rx x N_samp array containing the time-domain frame, with M_rx the
                number of receive antennas and N_samp the number of samples in a frame (without oversampling)

        Returns:
            frame_freq_domain(numpy.array): the frequemcy-domain frame, of size M_rx x N_symb x N_fft, with N_symb the
                number of OFDM symbols in the frame and N_fft the FFT length.
        """
        # remove guard intervals and cyclic prefixes
        frame_in_time_domain = frame_in_time_domain[:, self.data_time_indices]

        # convert to frequency domain
        frame_in_time_domain = np.reshape(frame_in_time_domain, (self.param.number_rx_antennas,
                                                                 self._number_ofdm_symbols,
                                                                 self.param.fft_size))

        frame_in_freq_domain = np.fft.fft(frame_in_time_domain, norm='ortho')

        return frame_in_freq_domain

    def _equalize(self, frame_in_freq_domain, channel_in_freq_domain, noise_var):
        """Equalize the frequency-domain symbols

        Perform linear frequency-domain equalization according to estimated channel.
        Both ZF or MMSE are supported, as defined in 'self.param.equalization'

        Args:
            frame_in_freq_domain(numpy.ndarray): a 3D array (M_rx x N_symb x N_fft) containing the frequency-domain frame,
                with M_rx the number of receive antennas, N_symb the number of OFDM symbols in a frame and N_fft the FFT
                length.
            channel_in_freq_domain(numpy.ndarray): a 4D array (M_rx x N_symb x N_fft) containing the channel estimates
            noise_var(float): estimated noise variance

        Returns:
            data_symbols(numpy.ndarray): M_rx x N_re array with data symbols after equalization, with N_re the number of
                resource elements (RE) in the frame
            noise_var(numpy.ndarray): M_rx x N_re array with the estimated noise variance at each RE
        """
        # remove null subcarriers
        resource_elements = frame_in_freq_domain[:, :, self._resource_element_mapping]
        channel_in_freq_domain = channel_in_freq_domain[:, :, :, self._resource_element_mapping]

        # remove reference symbols
        data_frame_indices = self.data_frame_indices[:, :, 0]
        resource_elements = resource_elements[:, data_frame_indices]
        channel_in_freq_domain = channel_in_freq_domain[:, :, data_frame_indices]

        # MIMO decode
        data_symbols, channel, noise_var = self._mimo.decode(resource_elements, channel_in_freq_domain, noise_var)

        # equalize
        if self.param.equalization == "MMSE":
            snr = (channel * np.conj(channel))**2 / noise_var
            equalizer = 1 / channel * (snr / (snr + 1.))
        else:  # ZF
            equalizer = 1 / channel

        noise_var = noise_var * np.abs(equalizer)**2
        data_symbols = data_symbols * equalizer

        return data_symbols, noise_var

    def _decode(self, frame, noise_var):
        """Decode the frequency-domain OFDM frame according to a given precoding method

        The precoding algorithm is defined in 'self.param.precoding'. Currently, only DFT-spread precoding is supported

        Args:
            frame(numpy.array): an M_rx x N_re array containing the OFDM resource elements, with M_rx the number of
                receive antennas and N_re the number of data resource elements in the frame
            noise_var(numpy.array): an M_rx x N_re array containing the noise variance

        Returns:
            frame(numpy.array): symbols after decoding
            noise_var(numpy.array): noise variance after decoding
        """
        if self.param.precoding == "DFT":
            frame_idx = 0
            for symbol in range(self._number_ofdm_symbols):
                data_indices = self.data_frame_indices[symbol, :, 0]
                if np.any(data_indices):
                    idx_end = frame_idx + self._data_resource_elements_per_symbol[symbol]
                    data_symbols = (frame[:, frame_idx: idx_end])
                    noise_var_data = noise_var[:, frame_idx: idx_end]

                    frame[:, frame_idx:idx_end] = np.fft.ifft(data_symbols, norm='ortho')
                    noise_var[:, frame_idx: idx_end] = np.broadcast_to(np.mean(noise_var_data, axis=1),
                                                                       (self._data_resource_elements_per_symbol[symbol],
                                                                        self.param.number_rx_antennas)).T

                frame_idx += self._data_resource_elements_per_symbol[symbol]

        return frame, noise_var

    def channel_estimation(self, rx_signal: np.ndarray,
                           timestamp_in_samples: int) -> np.ndarray:
        """Performs channel estimation

        This methods estimates the frequency response of the channel for all OFDM symbols in a frame. The estimation
        algorithm is defined in the parameter variable `self.param`.

        With ideal channel estimation, the channel state information is obtained directly from the channel.
        The CSI can be considered to be known only at the beginning/middle/end of the frame
        (estimation_type='IDEAL_PREAMBLE'/'IDEAL_MIDAMBLE'/ 'IDEAL_POSTAMBLE'), or at every OFDM symbol ('IDEAL').

        With reference-based estimation, the specified reference subcarriers are employed for channel estimation.

        Args:
            rx_signal(numpy.ndarray): frequency-domain samples of the received signal over the whole frame
            timestamp_in_samples(int): sample index inside the drop of the first sample in frame

        Returns:
            numpy.ndarray:
                channel estimate in the frequency domain. It is a R x T x K x N array, with N the FFT size and K the
                number of data OFDM symbols in the frame. R denotes the number of receive antennas and T of the transmit
                antennas.
        """
        initial_timestamp_in_samples = copy(timestamp_in_samples)

        ####
        # old determine timestamp of data symbols
        channel_sampling_timestamps = np.array([])
        for frame_element in self.param.frame_structure:
            if isinstance(frame_element, OfdmSymbolConfig):
                channel_sampling_timestamps = np.append(channel_sampling_timestamps, timestamp_in_samples)
                samples_in_element = frame_element.no_samples + frame_element.cyclic_prefix_samples
            else:
                samples_in_element = frame_element.no_samples
            timestamp_in_samples += samples_in_element * self.param.oversampling_factor
        channel_timestamps_old = channel_sampling_timestamps / self.param.sampling_rate
        number_of_symbols_old = channel_sampling_timestamps.size
        ####

        channel_timestamps = ((self.channel_sampling_timestamps * self.param.oversampling_factor
                              + initial_timestamp_in_samples) / self.param.sampling_rate)

        number_of_symbols = channel_timestamps.size

        channel_in_freq_domain: np.ndarray

        if self.param.channel_estimation == 'IDEAL':  # ideal channel estimation at each transmitted OFDM symbol
            channel_in_freq_domain = self.get_ideal_channel_estimation(channel_timestamps)
            channel_in_freq_domain = np.moveaxis(channel_in_freq_domain, 0, -1)

        elif self.param.channel_estimation in {'IDEAL_PREAMBLE', 'IDEAL_MIDAMBLE', 'IDEAL_POSTAMBLE'}:
            if self.param.channel_estimation == 'IDEAL_PREAMBLE':
                channel_timestamps = initial_timestamp_in_samples / self.param.sampling_rate
            elif self.param.channel_estimation == 'IDEAL_MIDAMBLE':
                channel_timestamps = ((initial_timestamp_in_samples + self.samples_in_frame/2)
                                      / self.param.sampling_rate)
            elif self.param.channel_estimation == 'IDEAL_POSTAMBLE':
                channel_timestamps = ((initial_timestamp_in_samples + self.samples_in_frame)
                                      / self.param.sampling_rate)

            channel_in_freq_domain = np.tile(self.get_ideal_channel_estimation(np.array([channel_timestamps])),
                                             number_of_symbols)
            channel_in_freq_domain = np.moveaxis(channel_in_freq_domain, 0, -1)

        elif self.param.channel_estimation in {"LS", "LEAST_SQUARE"}:
            # self.param.channel_estimation == "REFERENCE_SIGNAL":
            channel_in_freq_domain = self.reference_based_channel_estimation(rx_signal)
            channel_in_freq_domain = np.repeat(channel_in_freq_domain[:, :, np.newaxis, :], number_of_symbols, axis=2)
        else:
            raise ValueError('invalid channel estimation type')

        return channel_in_freq_domain

    def get_ideal_channel_estimation(
            self, channel_timestamp: np.array) -> np.ndarray:
        """returns ideal channel estimation

        This method extracts the frequency-domain response from a known channel impulse response. The channel is the one
        from `self.channel`.

        Args:
            channel_timestamp(np.array): timestamp (in seconds) at which the channel impulse response should be
                measured

        Returns:
            np.ndarray:
                channel in freqency domain in shape `FFT_SIZE x #rx_antennas x #tx_antennas x #timestamps
        """
        channel_in_freq_domain_MIMO = np.zeros(
            (self.param.fft_size * self.param.oversampling_factor,
             self._channel.number_rx_antennas,
             self._channel.number_tx_antennas,
             channel_timestamp.size),
            dtype=complex
        )
        cir = self._channel.get_impulse_response(channel_timestamp)
        cir = np.swapaxes(cir, 0, 3)

        for rx_antenna_idx in range(self._channel.number_rx_antennas):
            for tx_antenna_idx in range(self._channel.number_tx_antennas):
                channel_in_freq_domain_MIMO[:, rx_antenna_idx, tx_antenna_idx, :] = (
                    np.fft.fft(
                        cir[:, rx_antenna_idx, tx_antenna_idx, :],
                        n=self.param.fft_size * self.param.oversampling_factor,
                        axis=0
                    )
                )

        if self.param.oversampling_factor > 1:
            channel_in_freq_domain_MIMO = np.delete(
                channel_in_freq_domain_MIMO,
                slice(int(self.param.fft_size / 2), -int(self.param.fft_size / 2)),
                axis=0
            )

        return channel_in_freq_domain_MIMO

    def reference_based_channel_estimation(self, rx_signal, frequency_bins=np.array([])):
        """returns channel estimation base don reference signals

        This method estimates the channel using reference symbols. Only LS method is curently implemented. The function
        will return only a single value for each subcarrier. If several reference symbols are available, then the
        estimate will be averaged over all OFDM symbols.

        Args:
            rx_signal(np.array): frequency domain received signal of size N_rx x N_symb x N_sc
            frequency bins (np.array): optional parameter, if estimates are desired at different frequencies from the
                subcarriers of the current modem.

        Returns:
            np.ndarray:
                channel in freqency domain in shape `FFT_SIZE x #rx_antennas x #tx_antennas x #timestamps
        """

        # adjust sizes of matrices, consider only occupied subcarriers
        reference_frame = np.moveaxis(self.reference_frame, -1, 0)
        rx_signal = rx_signal[:, :, self._resource_element_mapping]
        ref_freq_idx = np.any(reference_frame, axis=(0, 1))
        ref_idx = reference_frame != 0

        # LS channel estimation (averaged over time)
        channel_estimation_time_freq = np.zeros(rx_signal.shape, dtype=complex)
        channel_estimation_time_freq[ref_idx] = rx_signal[ref_idx] / reference_frame[ref_idx]
        channel_estimation = np.zeros((self.param.number_rx_antennas, self.param.number_tx_antennas,
                                       self.param.number_occupied_subcarriers), dtype=complex)
        channel_estimation[0, 0, ref_freq_idx] = (np.sum(channel_estimation_time_freq[:, :, ref_freq_idx], axis=1) /
                                                  np.sum(ref_idx[:, :, ref_freq_idx], axis=1))

        # extend matrix to all N_FFT subcarriers
        channel_estimation_freq = np.zeros((self.param.number_rx_antennas, self.param.number_tx_antennas,
                                            self.param.fft_size), dtype=complex)
        channel_estimation_freq[:, :, self._resource_element_mapping] = channel_estimation

        """
        if np.any(channel_estimation_freq[:, :, self._resource_element_mapping] == 0) or frequency_bins.size:
            # if channel_estimation is missing at any frequency or different frequencies
            # then interpolate
            ch_est_freqs = np.where(channel_estimation != 0)[1]
            ch_est_freqs[ch_est_freqs > self.param.fft_size / 2] = (ch_est_freqs[ch_est_freqs > self.param.fft_size / 2]
                                                                    - self.param.fft_size)
            ch_est_freqs = ch_est_freqs * self.param.subcarrier_spacing
            ch_est_freqs = np.fft.fftshift(ch_est_freqs)

            interp_function = interpolate.interp1d(ch_est_freqs, np.fft.fftshift(channel_estimation))

            channel_estimation = interp_function(frequency_bins)
        """

        # multiple antennas
        # check interpolation

        return channel_estimation_freq

    def get_bit_energy(self) -> float:
        """returns the theoretical (discrete) bit energy.

        Returns:
            float:
                raw bit energy. For the OFDM signal, the average bit energy of all data symbols, including
                the cyclic prefix overhead, is considered.
        """

        return self.param.oversampling_factor / \
            self._mapping.bits_per_symbol * self._cyclic_prefix_overhead

    def get_symbol_energy(self) -> float:
        """returns the theoretical symbol energy.

        Returns:
            float:
                raw symbol energy. For the OFDM signal, the average energy of a data resource element,
                including the cyclic prefix overhead, is considered.
        """

        return self.param.oversampling_factor * self._cyclic_prefix_overhead

    def get_power(self) -> float:
        return self.param.number_occupied_subcarriers / self.param.fft_size
