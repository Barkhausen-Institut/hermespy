# -*- coding: utf-8 -*-
"""
=================================
Filtered Single Carrier Waveforms
=================================

"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, Set

import matplotlib.pyplot as plt
import numpy as np

from hermespy.core import ChannelStateInformation, Executable, FloatingError, Serializable, Signal
from .waveform_generator import ConfigurablePilotWaveform, WaveformGenerator, Synchronization, \
    ChannelEstimation, ChannelEqualization, PilotSymbolSequence
from hermespy.modem.tools.psk_qam_mapping import PskQamMapping
from .symbols import Symbols
from .waveform_correlation_synchronization import CorrelationSynchronization

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class FilteredSingleCarrierWaveform(ConfigurablePilotWaveform):
    """This method provides a class for a generic PSK/QAM modem.

    The modem has the following characteristics:
    - root-raised cosine filter with arbitrary roll-off factor
    - arbitrary constellation, as defined in modem.tools.psk_qam_mapping:PskQamMapping

    This implementation has currently the following limitations:
    - hard output only (no LLR)
    - no reference signal
    - ideal channel estimation
    - equalization of ISI with FMCW in AWGN channel only
    - no equalization (only amplitude and phase of first propagation path is compensated)
    """

    __symbol_rate: float
    __num_preamble_symbols: int
    __num_data_symbols: int
    __num_postamble_symbols: int
    __guard_interval: float
    __mapping: PskQamMapping
    __pilot_rate: int
    _data_symbol_idx: Optional[np.ndarray]
    _pulse_correlation_matrix: Optional[np.ndarray]

    def __init__(self,
                 symbol_rate: float,
                 num_preamble_symbols: int,
                 num_data_symbols: int,
                 num_postamble_symbols: int = 0,
                 pilot_rate: int = 0,
                 guard_interval: float = 0.,
                 pilot_symbol_sequence: Optional[PilotSymbolSequence] = None,
                 repeat_pilot_symbol_sequence: bool = True,
                 **kwargs: Any) -> None:
        """Waveform Generator PSK-QAM initialization.

        Args:

            symbol_rate (float):
                Rate at which symbols are being generated in Hz.

            num_preamble_symbols (int):
                Number of preamble symbols within a single communication frame.

            num_data_symbols (int):
                Number of data symbols within a single communication frame.

            num_postamble_symbols (int, optional):
                Number of postamble symbols within a single communication frame.

            guard_interval (float, optional):
                Guard interval between communication frames in seconds.
                Zero by default.

            pilot_rate (int, optional):
                Pilot symbol rate.
                Zero by default, i.e. no pilot symbols.

            pilot_symbol_sequence (Optional[PilotSymbolSequence], optional):
                The configured pilot symbol sequence.
                Uniform by default.

            repeat_pilot_symbol_sequence (bool, optional):
                Allow the repetition of pilot symbol sequences.
                Enabled by default.

            kwargs (Any):
                Waveform generator base class initialization parameters.
        """

        self.symbol_rate = symbol_rate
        self.num_preamble_symbols = num_preamble_symbols
        self.num_data_symbols = num_data_symbols
        self.num_postamble_symbols = num_postamble_symbols
        self.pilot_rate = pilot_rate
        self.guard_interval = guard_interval

        #self._data_symbol_idx = None
        #self._symbol_idx = None
        #self._pulse_correlation_matrix = None

        # Initialize base classes
        ConfigurablePilotWaveform.__init__(self, symbol_sequence=pilot_symbol_sequence, repeat_symbol_sequence=repeat_pilot_symbol_sequence)
        WaveformGenerator.__init__(self, **kwargs)

    @abstractmethod
    def _transmit_filter(self) -> np.ndarray:
        """Pulse shaping filter applied to data symbols during transmission.

        Returns:

            The shaping filter impulse response.
        """
        ...

    @abstractmethod
    def _receive_filter(self) -> np.ndarray:
        """Pulse shaping filter applied to signal streams during reception.


        Returns:

            The shaping filter impulse response.
        """
        ...

    @property
    @abstractmethod
    def _filter_delay(self) -> int:
        """Cumulative delay introduced during transmit and receive filtering.

        Returns:
            Delay in samples.
        """
        ...

    @property
    def symbol_rate(self) -> float:
        """Repetition rate of symbols.


        Returns:
            Symbol rate in Hz.

        Raises:

            ValueError: For rates smaller or equal to zero.
        """

        return self.__symbol_rate

    @symbol_rate.setter
    def symbol_rate(self, value: float) -> None:

        if value <= 0.0:
            raise ValueError("Symbol rate must be greater than zero")

        self.__symbol_rate = value
        
    @property
    def num_preamble_symbols(self) -> int:
        """Number of preamble symbols.
        
        Transmitted at the beginning of communication frames.
        
        Returns: The number of symbols.
        
        Raises:
        
            ValueError: If the number of symbols is smaller than zero.
        """
        
        return self.__num_preamble_symbols
    
    @num_preamble_symbols.setter
    def num_preamble_symbols(self, value: int) -> None:
        
        if value < 0:
            raise ValueError("Nummber of preamble symbols must be greater or equal to zero")
        
        self.__num_preamble_symbols = value
        
    @property
    def num_postamble_symbols(self) -> int:
        """Number of postamble symbols.
        
        Transmitted at the end of communication frames.
        
        Returns: The number of symbols.
        
        Raises:
        
            ValueError: If the number of symbols is smaller than zero.
        """
        
        return self.__num_postamble_symbols
    
    @num_postamble_symbols.setter
    def num_postamble_symbols(self, value: int) -> None:
        
        if value < 0:
            raise ValueError("Nummber of postamble symbols must be greater or equal to zero")
        
        self.__num_postamble_symbols = value
        
    @WaveformGenerator.modulation_order.setter
    def modulation_order(self, value: int) -> None:

        self.__mapping = PskQamMapping(value, soft_output=False)
        WaveformGenerator.modulation_order.fset(self, value)
        
    @property
    def pilot_signal(self) -> Signal:
        
        pilot_symbols = np.zeros(1 + (self.num_preamble_symbols - 1) * self.oversampling_factor, dtype=complex)
        pilot_symbols[::self.oversampling_factor] = self.pilot_symbols(self.num_preamble_symbols)

        return Signal(np.convolve(pilot_symbols, self._transmit_filter()), sampling_rate=self.sampling_rate)

    def map(self, data_bits: np.ndarray) -> Symbols:
        return Symbols(self.__mapping.get_symbols(data_bits))

    def unmap(self, data_symbols: Symbols) -> np.ndarray:
        return self.__mapping.detect_bits(data_symbols.raw)

    def modulate(self, data_symbols: Symbols) -> Signal:

        frame = np.zeros(1 + (self._num_frame_symbols - 1) * self.oversampling_factor, dtype=complex)

        # Query the pilot symbols
        pilot_symbols = self.pilot_symbols(self.num_preamble_symbols + self._num_pilot_symbols + self.num_postamble_symbols)

        # Set preamble symbols
        num_preamble_samples = self.oversampling_factor * self.num_preamble_symbols
        frame[:num_preamble_samples:self.oversampling_factor] = pilot_symbols[:self.num_preamble_symbols]

        # Set payload symbols
        # num_payload_samples = self.oversampling_factor * self._num_payload_symbols
        payload_start = num_preamble_samples
        payload_stop = payload_start + self._num_payload_symbols * self.oversampling_factor
        payload_slice = frame[payload_start:payload_stop:self.oversampling_factor]
        payload_slice[self._data_symbol_indices] = data_symbols.raw
        payload_slice[self._pilot_symbol_indices] = pilot_symbols[self.num_preamble_symbols:]

        # Set postamble symbols
        num_postamble_samples = self.oversampling_factor * self.num_postamble_symbols
        postamble_start_idx = payload_stop
        postamble_stop_idx = postamble_start_idx + num_postamble_samples
        frame[postamble_start_idx:postamble_stop_idx:self.oversampling_factor] = pilot_symbols[self.num_preamble_symbols + self._num_pilot_symbols::]

        # Generate waveforms by treating the frame as a comb and convolving with the impulse response
        output_signal = np.convolve(frame, self._transmit_filter())
        return Signal(output_signal, self.sampling_rate)

    def demodulate(self,
                   baseband_signal: np.ndarray,
                   channel_state: ChannelStateInformation,
                   noise_variance: float = 0.) -> Tuple[Symbols, ChannelStateInformation, np.ndarray]:

        # Query filters
        filter_delay = self._filter_delay

        # Filter the signal and csi
        filtered_signal = np.convolve(baseband_signal, self._receive_filter())
        filter_states = np.zeros((channel_state.state.shape[0], channel_state.state.shape[1], filter_delay, channel_state.state.shape[3]), dtype=complex)
        filter_states[:, :, :, 0] = 1.
        channel_state.state = np.append(filter_states, channel_state.state, axis=2)

        # Extract frame symbols
        preamble_start_idx = filter_delay
        preamble_stop_idx = preamble_start_idx + self.oversampling_factor * self.num_preamble_symbols
        # preamble = filtered_signal[preamble_start_idx:preamble_stop_idx:self.oversampling_factor]
        
        payload_symbols = filtered_signal[preamble_stop_idx:preamble_stop_idx + self.oversampling_factor * self._num_payload_symbols:self.oversampling_factor]
        pilot_symbols = payload_symbols[self._pilot_symbol_indices] if self._num_pilot_symbols > 0 else np.empty(0, dtype=complex)
        data_symbols = payload_symbols[self._data_symbol_indices] if self._num_payload_symbols >0 else np.empty(0, dtype=complex)
        channel_state.state = channel_state.state[:, :, preamble_stop_idx:self.oversampling_factor * self._num_payload_symbols:self.oversampling_factor, :]

        snr = self.power / noise_variance if noise_variance > 0. else float('inf')

        # Apply equalization routines
        equalized_data_symbols = self._equalize(data_symbols, channel_state, noise_variance)

        return Symbols(data_symbols), channel_state, np.repeat(noise_variance, len(data_symbols))

        # Estimate the channel
        estimated_csi = self.channel_estimation.estimate_channel(signal, channel_state)

        # Re-build signal model
        signal = Signal(filtered_signal, sampling_rate=self.sampling_rate)


        # Equalize the signal
        snr = self.power / noise_variance if noise_variance > 0. else float('inf')
        equalized_signal = self.channel_equalization.equalize_channel(signal, estimated_csi, snr)

        # Extract data symbols
        num_data_samples = self.oversampling_factor * self.__num_data_symbols
        data_start_idx = preamble_stop_idx
        data_stop_idx = data_start_idx + num_data_samples
        data = equalized_signal.samples[0, data_start_idx:data_stop_idx:self.oversampling_factor]
        data_state = estimated_csi[:, :, data_start_idx:data_stop_idx:self.oversampling_factor, :]

        # Extract postamble symbols
        # num_postamble_samples = self.oversampling_factor * self.num_postamble_symbols
        # postamble_start_idx = data_stop_idx
        # postamble_stop_idx = postamble_start_idx + num_postamble_samples
        # postamble = filtered_signal[postamble_start_idx:postamble_stop_idx:self.oversampling_factor]

        noise = np.repeat(noise_variance, len(data))

    def _equalize(self,
                  data_symbols: np.ndarray,
                  csi: ChannelStateInformation,
                  noise: np.ndarray) -> np.ndarray:

        return data_symbols

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

        if self._pulse_correlation_matrix:
            snr_factor = 0  # ZF
            h_matrix = self._pulse_correlation_matrix
            h_matrix_hermitian = h_matrix.conjugate().T

            if self.equalization == FilteredSingleCarrierWaveform.Equalization.MMSE:
                snr_factor = noise_var * h_matrix

            isi_equalizer = np.matmul(h_matrix_hermitian,
                                      np.linalg.inv(np.matmul(h_matrix_hermitian, h_matrix) + snr_factor))

            equalized_symbols = np.matmul(isi_equalizer, data_symbols[:, np.newaxis]).flatten()
        else:
            equalized_symbols = data_symbols

        # compensate channel phase and amplitude
        equalized_symbols = equalized_symbols / channel

        return equalized_symbols

    @property
    def num_pilot_samples(self) -> int:
        """Number of samples within the pilot section of a frame.

        Returns:
            int: Number of pilot samples.
        """

        if self.num_preamble_symbols < 1 and self.num_postamble_symbols < 1:
            return 0

        if self.__pilot_rate == 0:
            return 0

        return int(round(self.sampling_rate / self.pilot_rate))

    @property
    def num_guard_samples(self) -> int:
        """Number of samples within the guarding section of a frame.

        Returns:
            int: Number of samples.
        """

        return int(round(self.guard_interval * self.sampling_rate))

    @property
    def guard_interval(self) -> float:
        """Frame guard interval.

        Returns:
            float: Interval in seconds.
        """

        return self.__guard_interval

    @guard_interval.setter
    def guard_interval(self, interval: float) -> None:
        """Modify frame guard interval.

        Args:
            interval (float): Interval in seconds.

        Raises:
            ValueError: If `interval` is smaller than zero.
        """

        if interval < 0.0:
            raise ValueError("Guard interval must be greater or equal to zero")

        self.__guard_interval = interval
        
    @property
    def pilot_rate(self) -> int:
        """Repetition rate of pilot symbols within the frame.

        A pilot rate of zero indicates no pilot symbols within the data frame.

        Returns:
            Rate in number of symbols

        Raises:
            ValueError: If the pilot rate is smaller than zero.
        """

        return self.__pilot_rate

    @pilot_rate.setter
    def pilot_rate(self, value: int) -> None:
        """Modify frame pilot symbol rate.

        Args:
            rate (float): Rate in seconds.

        Raises:
            ValueError: If `rate` is smaller than zero.
        """

        if value < 0:
            raise ValueError("Pilot symbol rate must be greater or equal to zero")

        self.__pilot_rate = int(value)

    @property
    def _num_pilot_symbols(self) -> int:

        if self.pilot_rate <= 0:
            return 0

        return int(self.num_data_symbols / self.pilot_rate)

    @property
    def _num_payload_symbols(self) -> int:

        num_symbols = self.num_data_symbols + self._num_pilot_symbols
        return num_symbols

    @property
    def _pilot_symbol_indices(self) -> np.ndarray:
        """Indices of pilot symbols within the ful communication frame.

        Returns:
            Numpy array containing pilot symbol indices.
        """

        if self.pilot_rate <= 0:
            return np.empty(0, dtype=int)

        pilot_indices = np.arange(self._num_pilot_symbols) * (1 + self.pilot_rate)
        return pilot_indices

    @property
    def _data_symbol_indices(self) -> np.ndarray:
        """Indices of data symbols within the full communication frame.

        Returns:
            Nump array containging data symbol indices.
        """

        data_indices = np.arange(self._num_payload_symbols)
        
        payload_indices = self._pilot_symbol_indices
        if len(payload_indices) > 0:
            data_indices = np.delete(data_indices, self._pilot_symbol_indices)
            
        return data_indices

    @property
    def _num_frame_symbols(self) -> int:
        """Overall number of symbols per frame.

        Includes preamble, postamble, data symbols and interleaved pilot symbols

        Returns:
            Number of symbols.
        """

        return self.num_preamble_symbols + self._num_payload_symbols + self.num_postamble_symbols

    @property
    def num_data_symbols(self) -> int:
        """Number of data symbols per frame.

        Returns:
            int: Number of data symbols.
        """

        return self.__num_data_symbols

    @num_data_symbols.setter
    def num_data_symbols(self, num: int) -> None:
        """Modify number of data symbols per frame.

        Args:
            num (int): Number of data symbols.

        Raises:
            ValueError: If `num` is smaller than zero.
        """

        if num < 0:
            raise ValueError("Number of data symbols must be greater or equal to zero")

        self.__num_data_symbols = num

    @property
    def samples_in_frame(self) -> int:

        return (self._num_frame_symbols - 1) * self.oversampling_factor + self._transmit_filter().shape[0]

    @property
    def bits_per_frame(self) -> int:
        return self.__num_data_symbols * int(np.log2(self.modulation_order))

    @property
    def samples_overhead_in_frame(self) -> int:
        """Number of samples appended to frame due to filtering impulse responses.

        Returns:
            int: Number of samples.
        """

        return self.tx_filter.delay_in_samples

    def _set_pulse_correlation_matrix(self):
        """ Creates a matrix with autocorrelation among pulses at different instants
        """

        if self._pulse_correlation_matrix is None:

            if self.tx_filter.filter_type == ShapingFilter.FilterType.FMCW and \
                    self.equalization != FilteredSingleCarrierWaveform.Equalization.NONE:
                ######################################################################################
                # calculate the correlation matrix between data symbols sampled at different instants
                ######################################################################################

                # generate an NxN matrix with the time differences between the sampling instants of the N symbols
                # i.e., time_delay_matrix(i,j) = T_i - T_j, with T_i the sampling instant of the i-th symbol
                time_delay_matrix = np.zeros((self.num_data_symbols, self.num_data_symbols))
                for row in range(self.num_data_symbols):
                    time_delay_matrix[row, :] = np.arange(row, row - self.num_data_symbols, -1)
                time_delay_matrix = time_delay_matrix / self.symbol_rate

                # the correlation between two symbols r_i and r_j
                # is obtained as a known function of the difference between their sampling instants
                non_zero_idx = np.nonzero(time_delay_matrix)
                isi_matrix = np.ones((self.num_data_symbols, self.num_data_symbols))
                isi_matrix[non_zero_idx] = (np.sin(np.pi * self.chirp_bandwidth * time_delay_matrix[non_zero_idx] *
                                                   (1 - np.abs(time_delay_matrix[non_zero_idx]
                                                               / self.chirp_duration))) /
                                            (np.pi * self.chirp_bandwidth * time_delay_matrix[non_zero_idx]))

                time_idx = np.nonzero(np.abs(time_delay_matrix) > self.chirp_duration)
                isi_matrix[time_idx] = 0

                self._pulse_correlation_matrix = isi_matrix
            else:
                self._pulse_correlation_matrix = np.array([])

    @property
    def symbols_per_frame(self) -> int:
        return self.__num_data_symbols

    @property
    def bit_energy(self) -> float:
        return 1 / self.bits_per_symbol

    @property
    def symbol_energy(self) -> float:
        return 1.0

    @property
    def power(self) -> float:
        return 1 / self.oversampling_factor

    @property
    def sampling_rate(self) -> float:
        return self.symbol_rate * self.oversampling_factor

    def plot_filter_correlation(self) -> plt.Figure:
        """Plot the convolution between transmit and receive filter shapes.

        Returns:
            Handle to the generated matplotlib figure.
        """

        with Executable.style_context():

            tx_filter = self._transmit_filter()
            rx_filter = self._receive_filter()

            autocorrelation = np.convolve(tx_filter, rx_filter)

            fig, axes = plt.subplots()
            fig.suptitle('Pulse Autocorrelation')

            axes.plot(np.abs(autocorrelation))

        return fig

    def plot_filter(self) -> plt.Figure:
        """Plot the transmit filter shape.

        Returns:
            Handle to the generated matplotlib figure.
        """

        with Executable.style_context():

            tx_filter = self._transmit_filter()

            fig, axes = plt.subplots()
            fig.suptitle('Pulse Shape')

            axes.plot(tx_filter.real)
            axes.plot(tx_filter.imag)

        return fig


class SingleCarrierSynchronization(Synchronization[FilteredSingleCarrierWaveform]):
    """Synchronization for chirp-based frequency shift keying communication waveforms."""

    def __init__(self,
                 waveform_generator: Optional[FilteredSingleCarrierWaveform] = None,
                 *args: Any) -> None:
        """
        Args:

            waveform_generator (WaveformGenerator, optional):
                The waveform generator this synchronization routine is attached to.
        """

        Synchronization.__init__(self, waveform_generator)

class SingleCarrierCorrelationSynchronization(CorrelationSynchronization[FilteredSingleCarrierWaveform]):
    """Correlation-based clock-synchronization for PSK-QAM waveforms."""
    
    yaml_tag = u'SC-Correlation'


class SingleCarrierChannelEstimation(ChannelEstimation[FilteredSingleCarrierWaveform], ABC):
    """Channel estimation for Psk Qam waveforms."""
    
    def __init__(self,
                 waveform_generator: Optional[FilteredSingleCarrierWaveform] = None,
                 *args: Any) -> None:
        """
        Args:

            waveform_generator (WaveformGenerator, optional):
                The waveform generator this synchronization routine is attached to.
        """

        ChannelEstimation.__init__(self, waveform_generator)


class SingleCarrierLeastSquaresChannelEstimation(Serializable, SingleCarrierChannelEstimation):
    """Least-Squares channel estimation for Psk Qam waveforms."""

    yaml_tag = u'SC-LS'
    """YAML serialization tag"""
    
    def __init__(self,
                 waveform_generator: Optional[FilteredSingleCarrierWaveform] = None,
                 *args: Any) -> None:
        """
        Args:

            waveform_generator (WaveformGenerator, optional):
                The waveform generator this synchronization routine is attached to.
        """

        SingleCarrierChannelEstimation.__init__(self, waveform_generator)

    def estimate_channel(self,
                         signal: Signal,
                         csi: Optional[ChannelStateInformation] = None) -> ChannelStateInformation:

        if self.waveform_generator is None:
            raise FloatingError("Error trying to fetch the pilot section of a floating channel estimator")

        # Extract preamble symbols
        filter_delay = self.waveform_generator._filter_delay

        # Extract preamble symbols
        symbol_distance = self.waveform_generator.oversampling_factor
        num_preamble_samples = symbol_distance * self.waveform_generator.num_preamble_symbols
        preamble_start_idx = filter_delay
        preamble_stop_idx = preamble_start_idx + num_preamble_samples

        num_preamble_symbols = self.waveform_generator.num_preamble_symbols
        preamble_symbols = signal.samples[0, preamble_start_idx:preamble_stop_idx:symbol_distance]

        # Reference preamble
        reference = self.waveform_generator.pilot_symbols(num_preamble_symbols)

        # Compute channel weight.
        channel_weight = np.mean(preamble_symbols / reference)

        # Re-construct csi
        csi = ChannelStateInformation.Ideal(signal.num_samples, signal.num_streams)
        csi.state *= channel_weight

        return csi


class SingleCarrierChannelEqualization(ChannelEqualization[FilteredSingleCarrierWaveform], ABC):
    """Channel estimation for Psk Qam waveforms."""

    def __init__(self,
                 waveform_generator: Optional[FilteredSingleCarrierWaveform] = None) -> None:
        """
        Args:

            waveform_generator (WaveformGenerator, optional):
                The waveform generator this equalization routine is attached to.
        """

        ChannelEqualization.__init__(self, waveform_generator)


class SingleCarrierZeroForcingChannelEqualization(Serializable, SingleCarrierChannelEqualization, ABC):
    """Zero-Forcing Channel estimation for Psk Qam waveforms."""
    
    yaml_tag = u'SC-ZF'
    """YAML serialization tag"""

    def __init__(self,
                 waveform_generator: Optional[FilteredSingleCarrierWaveform] = None) -> None:
        """
        Args:

            waveform_generator (WaveformGenerator, optional):
                The waveform generator this equalization routine is attached to.
        """

        SingleCarrierChannelEqualization.__init__(self, waveform_generator)

    def equalize_channel(self,
                         signal: Signal,
                         csi: ChannelStateInformation,
                         snr: float = float('inf')) -> Signal:

        signal = signal.copy()
        signal.samples /= csi.state[0, 0, :signal.num_samples, 0]

        return signal


class SingleCarrierMinimumMeanSquareChannelEqualization(Serializable, SingleCarrierChannelEqualization, ABC):
    """Minimum-Mean-Square Channel estimation for Psk Qam waveforms."""
    
    yaml_tag = u'SC-MMSE'
    """YAML serialization tag"""

    def __init__(self,
                 waveform_generator: Optional[FilteredSingleCarrierWaveform] = None) -> None:
        """
        Args:

            waveform_generator (WaveformGenerator, optional):
                The waveform generator this equalization routine is attached to.
        """

        SingleCarrierChannelEqualization.__init__(self, waveform_generator)

    def equalize_channel(self,
                         signal: Signal,
                         csi: ChannelStateInformation,
                         snr: float = float('inf')) -> Signal:

        signal = signal.copy()
        signal.samples /= (csi.state[0, 0, :signal.num_samples, 0] + 1 / snr)
        
        return signal


class RolledOffSingleCarrierWaveform(FilteredSingleCarrierWaveform):
    """Base class for single carrier waveforms applying linear filters longer than a single symbol duration."""

    __relative_bandwidth: float     # Pulse bandwidth relative to the configured symbol rate
    __roll_off: float               # Filter pulse roll off factor
    __filter_length: int            # Filter length in modulation symbols
    
    @staticmethod
    def _arg_signature() -> Set[str]:
        
        return {'symbol_rate', 'num_preamble_symbols', 'num_data_symbols'}

    def __init__(self,
                 relative_bandwidth: float = 1.,
                 roll_off: float = 0.,
                 filter_length: int = 16,
                 *args, **kwargs) -> None:
        """
        Args:

            relative_bandwidth (float, optional):
                Bandwidth relative to the configured symbol rate.
                One by default, meaning the pulse bandwidth is equal to the symbol rate in Hz,
                assuming zero `roll_off`.

            roll_off (float, optional):
                Filter pulse shape roll off factor between zero and one.
                Zero by default, meaning no inter-symbol interference at the sampling instances.

            filter_length (float, optional):
                Filter length in modulation symbols.
                16 by default.
        """

        self.relative_bandwidth = relative_bandwidth
        self.roll_off = roll_off
        self.filter_length = filter_length

        FilteredSingleCarrierWaveform.__init__(self, *args, **kwargs)

    @property
    def filter_length(self) -> int:
        """Filter length in modulation symbols.

        Configures how far the shaping filter stretches in terms of the number of
        modulation symbols it overlaps with.

        Returns:
            Filter length in number of modulation symbols.

        Raises:
            ValueError: For filter lengths smaller than one.
        """

        return self.__filter_length

    @filter_length.setter
    def filter_length(self, value: int) -> None:

        if value < 1:
            raise ValueError("Filter length must be greater than zero")

        self.__filter_length = value

    @property
    def relative_bandwidth(self) -> float:
        """Bandwidth relative to the configured symbol rate.

        Raises:
            ValueError: On values smaller or equal to zero.
        """

        return self.__relative_bandwidth

    @relative_bandwidth.setter
    def relative_bandwidth(self, value: float) -> None:
        
        if value <= 0.:
            raise ValueError("Relative pulse bandwidth must be greater than zero")

        self.__relative_bandwidth = value
    
    @property
    def roll_off(self) -> float:
        """Filter pulse shape roll off factor.

        Raises:
            ValueError: On values smaller than zero or larger than one.
        """

        return self.__roll_off

    @roll_off.setter
    def roll_off(self, value: float) -> None:
        
        if value < 0. or value > 1.:
            raise ValueError("Filter pulse shape roll off factor value must be between zero and one")

        self.__roll_off = value

    @property
    def bandwidth(self) -> float:

        return self.symbol_rate * self.relative_bandwidth * (1 + self.roll_off)

    @abstractmethod
    def _base_filter(self) -> np.ndarray:
        """Generate the base filter impulse response.

        Returns:
            The base filter impulse response as a numpy array.
        """
        ...
    
    def _transmit_filter(self) -> np.ndarray:

        return self._base_filter()

    def _receive_filter(self) -> np.ndarray:

        return self._base_filter()

    @property
    def _filter_delay(self) -> int:

        return 2 * int(.5 * self.filter_length) * self.oversampling_factor


class RootRaisedCosineWaveform(RolledOffSingleCarrierWaveform, Serializable):
    """Root-Raised-Cosine filtered single carrier mdulation.
    
    
    .. plot::

       import matplotlib.pyplot as plt
   
       from hermespy.modem import RaisedCosineWaveform
   
   
       waveform = RaisedCosineWaveform(oversampling_factor=16)
       waveform.plot_filter()
       plt.show()

    .. plot::

       import matplotlib.pyplot as plt
   
       from hermespy.modem import RaisedCosineWaveform
   
   
       waveform = RaisedCosineWaveform(oversampling_factor=16)
       waveform.plot_filter_correlation()
       plt.show()
    """

    yaml_tag = u'SC-RootRaisedCosine'
    """YAML serialization tag"""

    def __init__(self, *args, **kwargs) -> None:

        RolledOffSingleCarrierWaveform.__init__(self, *args, **kwargs)

    def _base_filter(self) -> np.ndarray:

        impulse_response = np.zeros(self.oversampling_factor * self.filter_length)

        # Generate timestamps
        time = np.linspace(-int(.5 * self.filter_length), int(.5 * self.filter_length), self.filter_length*self.oversampling_factor, endpoint=(self.filter_length % 2 == 1)) * self.relative_bandwidth

        # Build filter response
        idx_0_by_0 = (time == 0)  # indices with division of zero by zero

        if self.roll_off != 0:
            # indices with division by zero
            idx_x_by_0 = (abs(time) == 1 / (4 * self.roll_off))
        else:
            idx_x_by_0 = np.zeros_like(time, dtype=bool)
        idx = (~idx_0_by_0) & (~idx_x_by_0)

        impulse_response[idx] = ((np.sin(np.pi * time[idx] * (1 - self.roll_off)) + 4 * self.roll_off * time[idx] * np.cos(np.pi * time[idx] * (1 + self.roll_off))) / (np.pi * time[idx] * (1 - (4 * self.roll_off * time[idx])**2)))
        if np.any(idx_x_by_0):
            impulse_response[idx_x_by_0] = self.roll_off / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * self.roll_off)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * self.roll_off)))
        impulse_response[idx_0_by_0] = 1 + self.roll_off * (4 / np.pi - 1)

        return impulse_response / np.linalg.norm(impulse_response)


class RaisedCosineWaveform(RolledOffSingleCarrierWaveform, Serializable):
    """Root-Raised-Cosine filtered single carrier mdulation.
    
    .. plot::

       import matplotlib.pyplot as plt
   
       from hermespy.modem import RootRaisedCosineWaveform
   
   
       waveform = RootRaisedCosineWaveform(oversampling_factor=16)
       waveform.plot_filter()
       plt.show()

    .. plot::

       import matplotlib.pyplot as plt
   
       from hermespy.modem import RootRaisedCosineWaveform
   
   
       waveform = RootRaisedCosineWaveform(oversampling_factor=16)
       waveform.plot_filter_correlation()
       plt.show()
    """

    yaml_tag = u'SC-RaisedCosine'
    """YAML serialization tag"""

    def __init__(self, *args, **kwargs) -> None:

        RolledOffSingleCarrierWaveform.__init__(self, *args, **kwargs)

    def _base_filter(self) -> np.ndarray:

        impulse_response = np.zeros(self.oversampling_factor * self.filter_length)

        # Generate timestamps
        time = np.linspace(-int(.5 * self.filter_length), int(.5 * self.filter_length), self.filter_length*self.oversampling_factor, endpoint=(self.filter_length % 2 == 1)) * self.relative_bandwidth

        # Build filter response
        if self.roll_off != 0:
            # indices with division of zero by zero
            idx_0_by_0 = (abs(time) == 1 / (2 * self.roll_off))
        else:
            idx_0_by_0 = np.zeros_like(time, dtype=bool)
        idx = ~idx_0_by_0
        impulse_response[idx] = (np.sinc(time[idx]) * np.cos(np.pi * self.roll_off * time[idx])
                                    / (1 - (2 * self.roll_off * time[idx]) ** 2))
        if np.any(idx_0_by_0):
            impulse_response[idx_0_by_0] = np.pi / \
                4 * np.sinc(1 / (2 * self.roll_off))

        return impulse_response / np.linalg.norm(impulse_response)


class RectangularWaveform(Serializable, FilteredSingleCarrierWaveform):
    """Rectangular filtered single carrier modulation.
    
    .. plot::

       import matplotlib.pyplot as plt
   
       from hermespy.modem import RectangularWaveform
   
   
       waveform = RectangularWaveform(oversampling_factor=16)
       waveform.plot_filter()
       plt.show()

    .. plot::

       import matplotlib.pyplot as plt
   
       from hermespy.modem import RectangularWaveform
   
   
       waveform = RectangularWaveform(oversampling_factor=16)
       waveform.plot_filter_correlation()
       plt.show()
    """

    yaml_tag = u'SC-Rectangular'
    """YAML serialization tag"""
    
    __relative_bandwidth: float
    
    @staticmethod
    def _arg_signature() -> Set[str]:
        
        return {'symbol_rate', 'num_preamble_symbols', 'num_data_symbols'}

    def __init__(self,
                 relative_bandwidth: float = 1.,
                 *args, **kwargs) -> None:

        self.relative_bandwidth = relative_bandwidth

        FilteredSingleCarrierWaveform.__init__(self, *args, **kwargs)

    @property
    def relative_bandwidth(self) -> float:
        """Bandwidth relative to the configured symbol rate.

        Raises:
            ValueError: On values smaller or equal to zero.
        """

        return self.__relative_bandwidth

    @relative_bandwidth.setter
    def relative_bandwidth(self, value: float) -> None:
        
        if value <= 0.:
            raise ValueError("Relative pulse bandwidth must be greater than zero")

        self.__relative_bandwidth = value

    @property
    def bandwidth(self) -> float:

        return self.symbol_rate * self.relative_bandwidth

    def _transmit_filter(self) -> np.ndarray:

        pulse_width = int(self.oversampling_factor / self.relative_bandwidth)
        return np.ones(pulse_width, dtype=complex) / np.sqrt(pulse_width)

    def _receive_filter(self) -> np.ndarray:

        return self._transmit_filter()

    @property
    def _filter_delay(self) -> int:

        return int(self.oversampling_factor / self.relative_bandwidth) - 1

    
class FMCWWaveform(Serializable, FilteredSingleCarrierWaveform):
    """Frequency Modulated Continuous Waveform Filter Modulation Scheme.
    
    .. plot::

       import matplotlib.pyplot as plt
   
       from hermespy.modem import FMCWWaveform
   
   
       waveform = FMCWWaveform(oversampling_factor=16, bandwidth=1e6)
       waveform.plot_filter()
       plt.show()

    .. plot::

       import matplotlib.pyplot as plt
   
       from hermespy.modem import FMCWWaveform
   
   
       waveform = FMCWWaveform(oversampling_factor=16, bandwidth=1e6)
       waveform.plot_filter_correlation()
       plt.show()
    """

    yaml_tag = u'SC-FMCW'
    """YAML serialization tag"""

    __bandwidth: float          # Chirp bandwidth in Hz
    __chirp_duration: float     # Chirp duration in seconds
    
    @staticmethod
    def _arg_signature() -> Set[str]:
        
        return {'symbol_rate', 'num_preamble_symbols', 'num_data_symbols'}

    def __init__(self,
                 bandwidth: float,
                 chirp_duration: float = 0.,
                 *args, **kwargs) -> None:
        """
        Args:

            bandwidth (float):
                The chirp bandwidth in Hz.
                
            chirp_duration (float, optional):
                Duration of each FMCW chirp in seconds.
                By default, the inverse symbol rate is assumed.
        """

        self.bandwidth = bandwidth
        self.chirp_duration = chirp_duration

        FilteredSingleCarrierWaveform.__init__(self, *args, **kwargs)
    
    @property
    def chirp_duration(self) -> float:
        """FMCW Chirp duration.
        
        A duration of zero will result in the inverse symbol rate as chirp duration.

        Returns:
        
            Chirp duration in seconds.
        
        Raises:
        
            ValueError: If the duration is smaller than zero.
        """
        
        return self.__chirp_duration
    
    @chirp_duration.setter
    def chirp_duration(self, value: float) -> None:
        
        if value < 0.:
            raise ValueError("Chirp duration must be greater or equal to zero")
        
        self.__chirp_duration = value
        
    @property
    def __true_chirp_duration(self) -> float:
        """Chirp duration for internal calculations.
        
        Returns:
        
            The inverse symbol rate or the specified chirp duration.
        """
        
        if self.__chirp_duration <= 0.:
            return 1 / self.symbol_rate
        
        return self.chirp_duration

    @property
    def bandwidth(self) -> float:

        return self.__bandwidth

    @bandwidth.setter
    def bandwidth(self, value: float) -> None:

        if value <= 0.:
            raise ValueError("Chirp bandwidth must be greater than zero")

        self.__bandwidth = value

    @property
    def chirp_slope(self) -> float:
        """Chirp slope.

        The slope is equal to the chirp bandwidth divided by its duration.

        Returns:
        
            Slope in Hz/s.
        """

        return self.bandwidth / self.__true_chirp_duration

    def _transmit_filter(self) -> np.ndarray:

        time = np.linspace(0, 1 / self.symbol_rate, self.oversampling_factor)
        impulse_response = np.exp(1j * np.pi * (self.bandwidth * time + self.chirp_slope * time ** 2))
        impulse_response[time > self.__true_chirp_duration] = 0.     # Cut off the chirp appropriately
        
        return impulse_response / np.sqrt(self.oversampling_factor)

    def _receive_filter(self) -> np.ndarray:

        return np.flip(self._transmit_filter().conj())

    @property
    def _filter_delay(self) -> int:

        return self.oversampling_factor - 1
