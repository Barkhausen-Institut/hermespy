# -*- coding: utf-8 -*-
"""Waveform Generation for Phase-Shift-Keying Quadrature Amplitude Modulation."""

from __future__ import annotations
from typing import Any, Tuple, Optional, Type, Union
from enum import Enum

import numpy as np
from ruamel.yaml import MappingNode, SafeRepresenter, SafeConstructor

from hermespy.channel import ChannelStateInformation
from hermespy.modem.waveform_generator import WaveformGenerator
from hermespy.modem.tools.shaping_filter import ShapingFilter
from hermespy.modem.tools.psk_qam_mapping import PskQamMapping
from hermespy.signal import Signal

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class WaveformGeneratorPskQam(WaveformGenerator):
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

    class Equalization(Enum):
        """Equalization method applied at receiver."""

        NONE = 0
        ZF = 1      # Zero Forcing
        MMSE = 2    # Minimum Mean Squared Error

    # YAML tag
    yaml_tag = WaveformGenerator.yaml_tag + "PskQam"

    tx_filter: ShapingFilter
    rx_filter: ShapingFilter
    __chirp_duration: float
    __chirp_bandwidth: float
    __pulse_width: float  # ToDO: Check where pulse-width has to be used for initialization
    __equalization: Equalization
    __num_preamble_symbols: int
    __num_data_symbols: int
    __num_postamble_symbols: int
    __pilot_symbol_rate: float
    __guard_interval: float
    __mapping: PskQamMapping
    complex_modulation: bool
    _data_symbol_idx: Optional[np.ndarray]
    _symbol_idx: Optional[np.ndarray]
    _pulse_correlation_matrix: Optional[np.ndarray]
    __symbol_rate: float

    def __init__(self,
                 symbol_rate: float = 100e6,
                 tx_filter: Optional[ShapingFilter] = None,
                 rx_filter: Optional[ShapingFilter] = None,
                 chirp_duration: float = 1e-6,
                 chirp_bandwidth: float = 100e6,
                 equalization: Union[str, WaveformGeneratorPskQam.Equalization] = 'NONE',
                 num_preamble_symbols: int = 2,
                 num_data_symbols: int = 100,
                 num_postamble_symbols: int = 0,
                 pilot_rate: float = 1e6,
                 guard_interval: float = 1e-6,
                 complex_modulation: bool = True,
                 **kwargs: Any) -> None:
        """Waveform Generator PSK-QAM initialization.

        Args:

            symbol_rate (float, optional):
                Rate at which symbols are being generated.

            tx_filter (ShapingFilter, optional):
                The shaping filter applied during signal generation.

            rx_filter (ShapingFilter, optional):
                The shaping filter applied during signal reception.

            chirp_duration (float, optional):
                Duration of a single chirp in seconds.

            chirp_bandwidth (float, optional):
                Bandwidth of a single chirp in Hz.

            num_preamble_symbols (int, optional):
                Number of preamble symbols within a single communication frame.

            num_data_symbols (int, optional):
                Number of data symbols within a single communication frame.

            num_postamble_symbols (int, optional):
                Number of postamble symbols within a single communication frame.

            pilot_rate (int, optional):
                Pilot symbol rate.

            kwargs (Any):
                Waveform generator base class initialization parameters.
        """

        # Initialize base class
        WaveformGenerator.__init__(self, **kwargs)

        if tx_filter is None:
            self.tx_filter = ShapingFilter(ShapingFilter.FilterType.NONE, self.oversampling_factor)

        else:
            self.tx_filter = tx_filter
            
        if rx_filter is None:
            self.rx_filter = ShapingFilter(ShapingFilter.FilterType.NONE, self.oversampling_factor)

        else:
            self.rx_filter = tx_filter

        self.symbol_rate = symbol_rate
        self.chirp_duration = chirp_duration
        self.chirp_bandwidth = chirp_bandwidth
        self.num_preamble_symbols = num_preamble_symbols
        self.num_data_symbols = num_data_symbols
        self.num_postamble_symbols = num_postamble_symbols
        self.pilot_rate = pilot_rate
        self.guard_interval = guard_interval
        self.complex_modulation = complex_modulation
        self.equalization = equalization

        self.__mapping = PskQamMapping(self.modulation_order, is_complex=self.complex_modulation, soft_output=False)
        self._data_symbol_idx = None
        self._symbol_idx = None
        self._pulse_correlation_matrix = None

    @property
    def chirp_duration(self) -> float:
        """Access the chirp duration.

        Returns:
            float:
                Chirp duration in seconds.
        """

        return self.__chirp_duration

    @chirp_duration.setter
    def chirp_duration(self, duration: float) -> None:
        """Modify the chirp duration.

        Args:
            duration (float):
                The new duration in seconds.

        Raises:
            ValueError:
                If the duration is less or equal to zero.
        """

        if duration < 0.0:
            raise ValueError("Chirp duration must be greater than zero")

        self.__chirp_duration = duration

    @property
    def symbol_rate(self) -> float:
        """Rate of symbols.

        Inverse of the chirp duration.

        Returns:
            float: Symbol rate in Hz.
        """

        return self.__symbol_rate

    @symbol_rate.setter
    def symbol_rate(self, rate: float) -> None:
        """Modify rate of symbols.
        
        Args:
            rate: New symbol rate in Hz.
            
        Raises:
            ValueError: If symbol rate is smaller than zero.
        """

        if rate < 0.0:
            raise ValueError("Symbol rate must be greater or equal to zero")

        self.__symbol_rate = rate

    @property
    def chirp_bandwidth(self) -> float:
        """Access the chirp bandwidth.

        Returns:
            float:
                The chirp bandwidth in Hz.
        """

        return self.__chirp_bandwidth

    @chirp_bandwidth.setter
    def chirp_bandwidth(self, bandwidth: float) -> None:
        """Modify the chirp bandwidth.

        Args:
            bandwidth (float):
                The new bandwidth in Hz.

        Raises:
            ValueError:
                If the bandwidth is les sor equal to zero.
        """

        if bandwidth < 0.0:
            raise ValueError("Chirp bandwidth must be greater than zero")

        self.__chirp_bandwidth = bandwidth

    def map(self, data_bits: np.ndarray) -> np.ndarray:
        return self.__mapping.get_symbols(data_bits)

    def unmap(self, data_symbols: np.ndarray) -> np.ndarray:
        return self.__mapping.detect_bits(data_symbols)

    def modulate(self, data_symbols: np.ndarray) -> Signal:

        self._set_sampling_indices()
        self._set_pulse_correlation_matrix()

        frame = np.zeros(self.samples_in_frame, dtype=complex)
        frame[self._symbol_idx[:self.num_preamble_symbols]] = 1
        start_index_data = self.num_preamble_symbols
        end_index_data = self.num_preamble_symbols + self.num_data_symbols
        frame[self._symbol_idx[start_index_data: end_index_data]] = data_symbols

        output_signal = self.tx_filter.filter(frame)
        return Signal(output_signal, self.sampling_rate, self.modem.carrier_frequency)

    def demodulate(self,
                   baseband_signal: np.ndarray,
                   channel_state: ChannelStateInformation,
                   noise_variance: float) -> Tuple[np.ndarray, ChannelStateInformation, np.ndarray]:

        self._set_sampling_indices()
        self._set_pulse_correlation_matrix()

        # Filter the signal
        filtered_signal = self.rx_filter.filter(baseband_signal)
        #filtered_channel_state = self.rx_filter.filter(channel_state.to_impulse_response().state)

        # Equalize the signal
        symbol_indices = self._data_symbol_idx + self.rx_filter.delay_in_samples + self.tx_filter.delay_in_samples
        channel_state = channel_state[:, :, symbol_indices, :]
        equalized_symbols = self._equalizer(filtered_signal[symbol_indices],
                                            channel_state.state[0, 0, :, 0],
                                            noise_variance)
        noise = np.repeat(noise_variance, len(equalized_symbols))

        return equalized_symbols, channel_state, noise

    @property
    def bandwidth(self) -> float:

        # The bandwidth is assumed to be identical to the QAM chirp bandwidth
        return self.__chirp_bandwidth

    @property
    def equalization(self) -> WaveformGeneratorPskQam.Equalization:
        """Equalization method.

        Returns:
            WaveformGeneratorPskQam.Equalization: Equalization method.
        """

        return self.__equalization

    @equalization.setter
    def equalization(self, method: Union[str, WaveformGeneratorPskQam.Equalization]) -> None:
        """Modify equalization method.

        Args:
            method: Equalization method.
        """

        if isinstance(method, str):
            self.__equalization = WaveformGeneratorPskQam.Equalization[method]

        else:
            self.__equalization = method

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

            if self.equalization == WaveformGeneratorPskQam.Equalization.MMSE:
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
    def pilot_rate(self) -> float:
        """Frame pilot symbol rate.

        Returns:
            float: Rate in seconds.
        """

        return self.__pilot_rate

    @pilot_rate.setter
    def pilot_rate(self, rate: float) -> None:
        """Modify frame pilot symbol rate.

        Args:
            rate (float): Rate in seconds.

        Raises:
            ValueError: If `rate` is smaller than zero.
        """

        if rate < 0.0:
            raise ValueError("Pilot symbol rate must be greater or equal to zero")

        self.__pilot_rate = rate

    @property
    def samples_in_frame(self) -> int:

        return int((self.num_preamble_symbols + self.num_postamble_symbols) *
                   self.num_pilot_samples +
                   self.num_guard_samples +
                   self.oversampling_factor * self.num_data_symbols)

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
    def bits_per_frame(self) -> int:
        return self.__num_data_symbols * int(np.log2(self.modulation_order))

    @property
    def samples_overhead_in_frame(self) -> int:
        """Number of samples appended to frame due to filtering impulse responses.

        Returns:
            int: Number of samples.
        """

        return self.rx_filter.delay_in_samples

    def _set_sampling_indices(self) -> None:
        """ Determines the sampling instants for pilots and data at a given frame
        """

        if self._data_symbol_idx is None:
            # create a vector with the position of every pilot and data symbol in a
            # frame
            preamble_symbol_idx = np.arange(
                self.num_preamble_symbols) * self.num_pilot_samples
            start_idx = self.num_preamble_symbols * self.num_pilot_samples
            self._data_symbol_idx = start_idx + \
                np.arange(self.num_data_symbols) * \
                self.oversampling_factor
            start_idx += self.num_data_symbols * self.oversampling_factor
            postamble_symbol_idx = start_idx + \
                np.arange(self.num_postamble_symbols) * \
                self.num_pilot_samples
            self._symbol_idx = np.concatenate(
                (preamble_symbol_idx, self._data_symbol_idx, postamble_symbol_idx))

            self._data_symbol_idx += int(.5 * self.oversampling_factor)
            self._symbol_idx += int(.5 * self.oversampling_factor)

    def _set_pulse_correlation_matrix(self):
        """ Creates a matrix with autocorrelation among pulses at different instants
        """

        if self._pulse_correlation_matrix is None:

            if self.tx_filter.filter_type == ShapingFilter.FilterType.FMCW and \
                    self.equalization != WaveformGeneratorPskQam.Equalization.NONE:
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

    @classmethod
    def to_yaml(cls: Type[WaveformGeneratorPskQam],
                representer: SafeRepresenter,
                node: WaveformGeneratorPskQam) -> MappingNode:
        """Serialize an `WaveformGeneratorPskQam` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (WaveformGeneratorPskQam):
                The `WaveformGeneratorPskQam` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        state = {
            "oversampling_factor": node.oversampling_factor,
            "modulation_order": node.modulation_order,
            "tx_filter": node.tx_filter,
            "rx_filter": node.rx_filter,
            "chirp_duration": node.chirp_duration,
            "chirp_bandwidth": node.chirp_bandwidth,
            "num_preamble_symbols": node.num_preamble_symbols,
            "num_data_symbols": node.num_data_symbols,
            "num_postamble_symbols": node.num_postamble_symbols,
            "pilot_rate": node.pilot_rate,
            "guard_interval": node.guard_interval,
            "equalization": node.equalization
        }
        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[WaveformGeneratorPskQam],
                  constructor: SafeConstructor,
                  node: MappingNode) -> WaveformGeneratorPskQam:
        """Recall a new `WaveformGeneratorPskQam` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `WaveformGeneratorPskQam` serialization.

        Returns:
            WaveformGeneratorPskQam:
                Newly created `WaveformGeneratorPskQam` instance.
        """
        state = constructor.construct_mapping(node)
        shaping_filter = state.pop('filter', None)

        generator = cls(**state)

        if shaping_filter is not None:

            # TODO: Patch-through for sampling rate
            samples_per_symbol = generator.oversampling_factor  # int(1e3 / generator.symbol_rate)
            shaping_filter = ShapingFilter(**shaping_filter, samples_per_symbol=samples_per_symbol)

            if shaping_filter.filter_type == ShapingFilter.FilterType.FMCW:
                bandwidth_factor = generator.chirp_bandwidth / generator.symbol_rate
            else:
                bandwidth_factor = 1.

            tx_filter = ShapingFilter(shaping_filter.filter_type,
                                      samples_per_symbol,
                                      is_matched=False,
                                      length_in_symbols=shaping_filter.length_in_symbols,
                                      roll_off=shaping_filter.roll_off,
                                      bandwidth_factor=bandwidth_factor)
            # TODO: Check if chirp bandwidth is identical to full BW)

            if shaping_filter.filter_type == ShapingFilter.FilterType.RAISED_COSINE:
                # for raised cosine, receive filter is a low-pass filter with
                # bandwidth Rs(1+roll-off)/2
                rx_filter = ShapingFilter(ShapingFilter.FilterType.RAISED_COSINE,
                                          samples_per_symbol,
                                          length_in_symbols=shaping_filter.length_in_symbols,
                                          roll_off=0,
                                          bandwidth_factor=1. + shaping_filter.roll_off)
            else:
                # for all other filter types, receive filter is a matched filter
                rx_filter = ShapingFilter(shaping_filter.filter_type,
                                          samples_per_symbol,
                                          is_matched=True,
                                          length_in_symbols=shaping_filter.length_in_symbols,
                                          roll_off=shaping_filter.roll_off,
                                          bandwidth_factor=bandwidth_factor)

            generator.tx_filter = tx_filter
            generator.rx_filter = rx_filter

        return generator
