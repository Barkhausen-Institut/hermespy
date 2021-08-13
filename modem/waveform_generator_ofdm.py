from typing import List, Tuple
from copy import copy

import numpy as np
from scipy import signal

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
         
        self.reference_frame, self.data_frame_indices = self._get_frame_structure()

    def _get_frame_structure(self):
        """Creates the OFDM frame structure in time, frequency and space.

        This method interprets the OFDM parameters in 'self.param' that describe the OFDM frame and generates matrices
        with the allocation of all resource elements in a time/frequency/antenna grid

        Returns:
            ref_frame(numpy.ndarray): see description in class attributes
            data_frame_indices(numpy.ndarray): see description in class attributes
        """
        ref_frame = np.zeros((self._number_ofdm_symbols, self.param.number_occupied_subcarriers,
                              self.param.number_tx_antennas), dtype=complex)
        data_frame_indices = np.zeros((self._number_ofdm_symbols, self.param.number_occupied_subcarriers,
                                       self.param.number_tx_antennas), dtype=bool)

        ofdm_symbol_idx = 0
        for frame_element in self.param.frame_structure:
            if isinstance(frame_element, OfdmSymbolConfig):
                ref_idxs = self._get_subcarrier_indices(frame_element, ResourceType.REFERENCE)

                #######################################################
                # fill out resource elements with pilot symbols
                ref_symbols = np.tile(self.param.reference_symbols,
                                      int(np.ceil(ref_idxs.size / self.param.reference_symbols.size)))
                ref_symbols = ref_symbols[:ref_idxs.size]

                ref_frame[ofdm_symbol_idx, ref_idxs, 0] = ref_symbols

                data_idxs = self._get_subcarrier_indices(frame_element, ResourceType.DATA)
                data_frame_indices[ofdm_symbol_idx, data_idxs, :] = True

                ofdm_symbol_idx += 1

        return ref_frame, data_frame_indices

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
            frame(numpy.array): a 2D array containing the raw OFDM symbols in time domain. It is of size M_tx x N_s,
                with M_tx the number of transmit antennas and N_s = N_symb x N_fft.

        Returns:
            output_signal(numpy.array): an M_tx x N_samp array containing the time-domain OFDM frame.
        """
        output_signal: np.ndarray = np.zeros((self.param.number_tx_antennas, self._samples_in_frame_no_oversampling),
                                             dtype=complex)
        sample_index = 0
        ofdm_symbol_idx = 0
        for frame_element in self.param.frame_structure:
            if isinstance(frame_element, GuardInterval):
                sample_index += frame_element.no_samples
            else:
                ofdm_symbol = frame[ofdm_symbol_idx, :, :].T

                # add cyclic prefix
                cyclic_prefix_length = frame_element.cyclic_prefix_samples
                output_signal[:, sample_index:sample_index + cyclic_prefix_length] = \
                    ofdm_symbol[:, -cyclic_prefix_length:]
                sample_index += cyclic_prefix_length

                # add daza symbol
                output_signal[:, sample_index:sample_index + self.param.fft_size] = ofdm_symbol
                sample_index += self.param.fft_size

                ofdm_symbol_idx += 1

        return output_signal

    def _precode(self, frame):
        """Precode the frequemcy-domain OFDM frame

        The precoding algorithm is defined in 'self.param.precoding'. Currently, only DFT-spread precoding is supported

        Args:
            frame(numpy.array): a 3D array(N_symb x K_sc x M_tx) containing the OFDM resource elements

        Returns:
            frame(numpy.array): the precoded frame
        """

        if self.param.precoding == "DFT":
            for symbol_idx in range(self._number_ofdm_symbols):
                for antenna_idx in range(self.param.number_streams):
                    data_symbol_idx = self.data_frame_indices[symbol_idx, :, antenna_idx]
                    data = frame[symbol_idx, data_symbol_idx, antenna_idx]
                    if data.size:
                        data = np.fft.fft(data, norm="ortho")
                    frame[symbol_idx, data_symbol_idx, antenna_idx] = data

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
            channel_estimation = self.channel_estimation(
                rx_signal, timestamp_in_samples)

            rx_signal = rx_signal[:, self.samples_in_frame:]
            if self.param.oversampling_factor > 1:
                frame_signal = signal.decimate(
                    frame_signal, self.param.oversampling_factor)

            for frame_element_def in self.param.frame_structure:
                frame_element_samples = frame_element_def.no_samples
                if isinstance(frame_element_def, OfdmSymbolConfig):
                    # get channel estimation for this symbol
                    channel_in_freq_domain = channel_estimation[:, :, :, 0]

                    if channel_in_freq_domain.size:
                        channel_estimation = channel_estimation[:, :, :, 1:]

                    frame_element_samples += frame_element_def.cyclic_prefix_samples

                    bits_in_ofdm_symbol = self.get_bits_from_ofdm_symbol(
                        frame_element_def,
                        frame_signal[:, frame_element_def.cyclic_prefix_samples:],
                        channel_in_freq_domain,
                        noise_var
                    )
                    bits = np.append(bits, bits_in_ofdm_symbol)

                frame_signal = frame_signal[:,
                                            frame_element_samples:]
                timestamp_in_samples += frame_element_samples * \
                    self.param.oversampling_factor
        return list([bits]), rx_signal

    def get_bits_from_ofdm_symbol(
            self,
            ofdm_symbol_config: OfdmSymbolConfig,
            frame_signal: np.ndarray,
            channel_in_freq_domain: np.ndarray,
            noise_var: float) -> np.ndarray:
        """Detects the bits that are contained in an ofdm symbol in time domain.

        Args:
            ofdm_symbol_config(OfdmSymbolConfig): Configuration of current ofdm symbol.
            frame_signal(numpy.ndarray): array with the received signal in the whole remaining frame
            channel_in_freq_domain(np.ndarray):
                channel frequency response estimation. It should be a np.ndarray of shape
                fft_size x #rx_antennas x #tx_antennas
            noise_var(float): noise variance.

        Returns:
            Vector containing the detected bits for this particular symbol.
        """
        ofdm_symbol_resources: np.ndarray = frame_signal[:, :self.param.fft_size]
        symbols, noise_var = self.demodulate_ofdm_symbol(
            ofdm_symbol_config,
            ofdm_symbol_resources,
            channel_in_freq_domain,
            noise_var
        )

        symbols = symbols.flatten('F')
        bits_in_ofdm_symbol = self._mapping.detect_bits(symbols)

        return bits_in_ofdm_symbol

    def demodulate_ofdm_symbol(self, ofdm_symbol_config: OfdmSymbolConfig, ofdm_symbol_resources: np.ndarray,
                               channel_in_freq_domain: np.ndarray, noise_var: float) -> np.ndarray:
        """Demodulates  a single OFDM symbol

        This method performs the FFT of the time-domain signal and equalizes it with knowledge of the channel frequency
        response.

        Args:
            ofdm_symbol_config(OfdmSymbolConfig): Config of ofdm symbol to demodulate.
            ofdm_symbol_resources(numpy.ndarray):
                contains information the received OFDM symbol in time domain of shape
                #rx_antennas x #fft_size
            channel_in_freq_domain(numpy.ndarray):
                channel estimate in the frequency domain of shape
                fft_size x #rx_antennas x #tx_antennas
            noise_var(float): noise variance.

        Returns:
            (numpy.ndarray, numpy.ndarray)
                'data_symbols(numpy.ndarray)': estimate of the frequency-domain symbols at the data subcarriers
                'noise_var(numpy.ndarray)': noise variance of demodulated symbols

        """
        channel_in_freq_domain = np.moveaxis(channel_in_freq_domain, 0, -1)

        ofdm_symbol_resources_f = np.fft.fft(ofdm_symbol_resources, norm='ortho')
        data_symbols = ofdm_symbol_resources_f[:, self._resource_element_mapping]
        channel_in_freq_domain = channel_in_freq_domain[:, :, self._resource_element_mapping]

        data_symbols = self.discard_reference_symbols(ofdm_symbol_config, data_symbols)

        channel_in_freq_domain_reduced = np.zeros(
            (
                self.param.number_rx_antennas,
                self.param.number_tx_antennas,
                data_symbols.shape[1],
            ),
            dtype=complex
        )

        for rx_antenna_idx in range(self.param.number_rx_antennas):
            channel_in_freq_domain_reduced[rx_antenna_idx, :, :] = self.discard_reference_symbols(
                ofdm_symbol_config,
                channel_in_freq_domain[rx_antenna_idx, :, :],
            )

        channel_in_freq_domain = channel_in_freq_domain_reduced
        data_symbols, channel_in_freq_domain, noise_var = \
            self._mimo.decode(data_symbols, channel_in_freq_domain, noise_var)

        if self.param.equalization == "MMSE":
            SNR = (channel_in_freq_domain *
                   np.conj(channel_in_freq_domain))**2 / noise_var
            equalizer = 1 / channel_in_freq_domain * (SNR / (SNR + 1))
        else:
            # ZF equalization considering perfect channel state information
            # (SISO only)
            equalizer = 1 / channel_in_freq_domain

        noise_var = noise_var * np.abs(equalizer) ** 2
        data_symbols = data_symbols * equalizer

        if self.param.precoding == "DFT" and data_symbols.size:
            data_symbols = np.fft.ifft(data_symbols, norm='ortho')

            dftmtx = np.fft.fft(np.eye(data_symbols.size), norm='ortho')
            noise_var = dftmtx @ np.diag(noise_var.flatten()) @ dftmtx.T.conj()

        return data_symbols, noise_var

    def discard_reference_symbols(self,
                                  ofdm_symbol_config: OfdmSymbolConfig,
                                  ofdm_symbol_resources: np.ndarray) -> np.ndarray:
        """Discards symbols except data symbols in ofdm symbol.

        Args:
            ofdm_symbol_config(OfdmSymbolConfig): Config that defines the resource element mapping.
            ofdm_symbol_resources(np.ndarray): subcarriers of the frame.

        Returns:
            np.ndarray: Data subcarriers
        """

        data_resources_indices: np.array = np.array([], dtype=int)
        symbol_idx = 0
        for pattern in ofdm_symbol_config.resource_types:
            for pattern_el_idx in range(pattern.number):
                for res in pattern.MultipleRes:
                    if res.ResourceType == ResourceType.DATA:
                        data_resources_indices = np.append(
                            data_resources_indices,
                            np.arange(symbol_idx, symbol_idx + res.number))
                    symbol_idx += res.number
        return ofdm_symbol_resources[:, data_resources_indices]

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
            rx_signal(numpy.ndarray): time-domain samples of the received signal over the whole frame
            timestamp_in_samples(int): sample index inside the drop of the first sample in frame

        Returns:
            numpy.ndarray:
                channel estimate in the frequency domain. It is a N x R x T x K array, with N
                the FFT size and K the number of data OFDM symbols in the frame.
                R denotes the number of receiving antennas and T of the transmitting
                antennas.
        """

        # determine timestamp of data symbols
        channel_sampling_timestamps = np.array([])
        for frame_element in self.param.frame_structure:
            if isinstance(frame_element, OfdmSymbolConfig):
                channel_sampling_timestamps = np.append(channel_sampling_timestamps, timestamp_in_samples)
                samples_in_element = frame_element.no_samples + frame_element.cyclic_prefix_samples
            else:
                samples_in_element = frame_element.no_samples

            timestamp_in_samples += samples_in_element * self.param.oversampling_factor

        channel_timestamps = channel_sampling_timestamps / self.param.sampling_rate

        number_of_symbols = channel_sampling_timestamps.size
        channel_in_freq_domain = np.zeros((self.param.fft_size, self._channel.number_rx_antennas,
                                           self._channel.number_tx_antennas, number_of_symbols), dtype=complex)

        if self.param.channel_estimation == 'IDEAL':  # ideal channel estimation at each transmitted OFDM symbol
            channel_in_freq_domain = self.get_ideal_channel_estimation(channel_timestamps)

        elif self.param.channel_estimation in {'IDEAL_PREAMBLE', 'IDEAL_MIDAMBLE', 'IDEAL_POSTAMBLE'}:
            if self.param.channel_estimation == 'IDEAL_PREAMBLE':
                channel_timestamp = channel_timestamps[0]
            elif self.param.channel_estimation == 'IDEAL_MIDAMBLE':
                channel_timestamp = (channel_timestamps[0] + channel_timestamps[-1]) / 2
            elif self.param.channel_estimation == 'IDEAL_POSTAMBLE':
                channel_timestamp = np.array(channel_timestamps[-1])

            channel_in_freq_domain = np.tile( self.get_ideal_channel_estimation(channel_timestamp), number_of_symbols)

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
                channel in freqence domain in shape `FFT_SIZE x #rx_antennas x #tx_antennas
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

    def reference_based_channel_estimation(self, rx_signal, channel_timestamp: np.array):
        pass

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
