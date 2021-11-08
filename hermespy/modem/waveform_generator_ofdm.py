# -*- coding: utf-8 -*-
"""HermesPy Orthogonal Frequency Division Multiplexing Waveform Generation."""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple, Optional, Type, Union
from copy import copy
from scipy import signal
from enum import Enum
from abc import abstractmethod
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode, ScalarNode
from scipy.constants import pi
from functools import lru_cache
import numpy as np

from modem import WaveformGenerator
from modem.tools import PskQamMapping

if TYPE_CHECKING:
    from modem import Modem

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "André Noll Barreto"
__email__ = "andre.nollbarreto@barkhauseninstitut.org"
__status__ = "Prototype"


class ElementType(Enum):
    """Type of resource element."""

    REFERENCE = 0
    DATA = 1
    NULL = 2


class ChannelEstimation(Enum):
    """Applied channel estimation algorithm after reception."""

    IDEAL = 0
    IDEAL_PREAMBLE = 1
    IDEAL_MIDAMBLE = 2
    IDEAL_POSTAMBLE = 3
    REFERENCE_SIGNAL = 4


class FrameElement:

    type: ElementType
    repetitions: int = 1

    def __init__(self,
                 type: Union[str, ElementType],
                 repetitions: int = 1) -> None:

        if type is None:
            self.type = ElementType.NULL

        elif isinstance(type, str):
            self.type = ElementType[type]

        else:
            self.type = type

        self.repetitions = repetitions


class FrameResource:
    """Configures one sub-section of an OFDM symbol section in time AND frequency."""

    __repetitions: int
    __cp_ratio: float
    elements: List[FrameElement]

    def __init__(self,
                 repetitions: int = 1,
                 cp_ratio: float = 0.0,
                 elements: Optional[List[FrameElement]] = None) -> None:

        self.repetitions = repetitions
        self.cp_ratio = cp_ratio
        self.elements = elements if elements is not None else []

    @property
    def repetitions(self) -> int:
        """Number of block repetitions along the frequency axis.

        Returns:
            int: Number of repetitions.
        """

        return self.__repetitions

    @repetitions.setter
    def repetitions(self, reps: int) -> None:
        """Modify the number of repetitions.

        Args:
            reps (int): Number of repetitions.

        Raises:
            ValueError: If `reps` is smaller than one.
        """

        if reps < 1:
            raise ValueError("Number of frame resource repetitions must be greater or equal to one")

        self.__repetitions = reps

    @property
    def cp_ratio(self) -> float:
        """Ratio between full block length and cyclic prefix.

        Returns:
            float: The ratio between zero and one.
        """

        return self.__cp_ratio

    @cp_ratio.setter
    def cp_ratio(self, ratio: float) -> None:
        """Modify the ratio between full block element length and cyclic prefix.

        Args:
            ratio (float): New ratio between zero and one.

        Raises:
            ValueError: If ratio is less than zero or larger than one.
        """

        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("Cyclic prefix ratio must be between zero and one")

        self.__cp_ratio = ratio

    @property
    def num_subcarriers(self) -> int:
        """Number of occupied subcarriers.

        Returns:
            int: Number of occupied subcarriers.
        """

        num: int = 0
        for element in self.elements:
            num += element.repetitions

        return self.__repetitions * num

    @property
    def num_symbols(self) -> int:
        """Number of data symbols this resource can modulate.

        Return:
            Number of modulated symbols.
        """

        num: int = 0
        for element in self.elements:
            if element.type == ElementType.DATA:
                num += element.repetitions

        return self.__repetitions * num

    @property
    def num_references(self) -> int:
        """Number of references symbols this resource can modulate.

        Return:
            Number of modulated symbols.
        """

        num: int = 0
        for element in self.elements:
            if element.type == ElementType.REFERENCE:
                num += element.repetitions

        return self.__repetitions * num

    @property
    def resource_mask(self) -> np.ndarray:

        # Initialize the base mask as all false
        mask = np.ndarray((len(ElementType), self.num_subcarriers), dtype=bool) * False

        element_count = 0
        for element in self.elements:

            mask[element.type.value, element_count:element_count+element.repetitions] = True
            element_count += element.repetitions

        # Repeat the subcarrier masks according to the configured number of repetitions.
        mask = mask[:, :element_count].repeat(self.__repetitions, axis=1)

        return mask


class FrameSection:
    """OFDM Frame configuration time axis."""

    frame: Optional[WaveformGeneratorOfdm]
    num_repetitions: int

    def __init__(self,
                 num_repetitions: int = 1,
                 frame: Optional[WaveformGeneratorOfdm] = None) -> None:

        self.frame = frame
        self.num_repetitions = num_repetitions

    @property
    def num_symbols(self) -> int:
        """Number of data symbols this section can modulate.

        Returns:
            int: The number of symbols
        """

        return 0

    @property
    def num_references(self) -> int:
        """Number of data symbols this section can modulate.

        Returns:
            int: The number of symbols
        """

        return 0

    @property
    def num_words(self) -> int:
        """Number of OFDM symbols, i.e. words of subcarrierr symbols this section can modulate.

        Returns:
            int: The number of words.
        """

        return 0

    @property
    def num_subcarriers(self) -> int:
        """Number of subcarriers this section requires.

        Returns:
            int: The number of subcarriers.
        """

        return 0

    @property
    def resource_mask(self) -> np.ndarray:
        return np.empty((len(ElementType), 0, 0), dtype=bool)

    @property
    @abstractmethod
    def num_samples(self) -> int:
        """Number of samples within this OFDM time-section.

        Returns:
            int: Number of samples
        """
        ...

    @property
    @abstractmethod
    def duration(self) -> float:
        """Duration of this frame element in time domain.

        Returns:
            float: Duration in seconds.
        """
        ...

    @abstractmethod
    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        """Modulate this section into a complex base-band signal.

        Args:
            symbols (np.ndarray):
                The complex data symbols encoded in this OFDM section.

        Returns:
            np.ndarray: The modulated signal vector.
        """
        ...

    @abstractmethod
    def demodulate(self,
                   baseband_signal: np.ndarray,
                   ideal_channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Demodulate a time section of a complex OFDM base-band signal into data symbols.

        Args:
            baseband_signal (np.ndarray): Vector of complex-valued base-band samples.
            ideal_channel (np.ndarray): The ideal channel impulse response.

        Returns:
            (np.ndarray, np.ndarray):
                Section symbol grid and channel response grid.
        """
        ...


class FrameSymbolSection(FrameSection):

    yaml_tag: str = u'Symbol'
    pattern: List[int]

    def __init__(self,
                 num_repetitions: int = 1,
                 pattern: Optional[List[int]] = None,
                 frame: Optional[WaveformGeneratorOfdm] = None) -> None:

        FrameSection.__init__(self, num_repetitions=num_repetitions, frame=frame)
        self.pattern = pattern if pattern is not None else []
        self.frame = frame

    @property
    def num_symbols(self) -> int:

        num = 0
        for resource_idx in self.pattern:

            resource = self.frame.resources[resource_idx]
            num += resource.num_symbols

        return self.num_repetitions * num

    @property
    def num_references(self) -> int:

        num = 0
        for resource_idx in self.pattern:

            resource = self.frame.resources[resource_idx]
            num += resource.num_references

        return self.num_repetitions * num

    @property
    def num_words(self) -> int:
        return self.num_repetitions * len(self.pattern)

    @property
    def num_subcarriers(self) -> int:

        # ToDo: Resources with different numbers of subcarriers are currently not supported
        num = 0
        if len(self.pattern) > 0:
            num = self.frame.resources[self.pattern[0]].num_subcarriers

        return num

    @property
    def num_timeslots(self) -> int:
        return len(self.pattern) * self.num_repetitions

    @property
    def duration(self) -> float:

        duration = self.num_timeslots / self.frame.subcarrier_spacing
        return duration

    def modulate(self, symbols: np.ndarray) -> np.ndarray:

        # Collect resource masks
        mask = self.resource_mask

        # Fill up the time-frequency grid exploiting the mask
        grid = np.empty(mask.shape[1:], complex)

        # Reference fields all currently carry the complex symbol 1+j0
        # ToDo: Implement reference symbol configurations
        grid.T[mask[ElementType.REFERENCE.value, ::].T] = 1.

        # Data fields carry the supplied data symbols
        grid.T[mask[ElementType.DATA.value, ::].T] = symbols

        # NULL fields are just that... zero ToDo: Check with André if this is correct
        grid.T[mask[ElementType.NULL.value, ::].T] = 0.

        # By convention, the length of each time slot is the inverse of the sub-carrier spacing
        num_samples_per_slot = self.frame.modem.scenario.sampling_rate / self.frame.subcarrier_spacing

        num_slot_samples = int(round(self.frame.modem.scenario.sampling_rate / self.frame.subcarrier_spacing))

        idft_matrix = self.frame.inverse_fourier_weights(num_slot_samples,
                                                        grid.shape[0],
                                                        self.frame.subcarrier_spacing,
                                                        self.frame.dc_suppression)

        resource_signals = idft_matrix @ grid

        # Add the cyclic prefix to each time slot while simultaneously flatten the resource signals into time domain
        signals = []
        for resource_idx, resource_samples in enumerate(resource_signals.T):

            pattern_idx = resource_idx % len(self.pattern)
            cp_ratio = self.frame.resources[self.pattern[pattern_idx]].cp_ratio

            num_prefix_samples = int(round(num_samples_per_slot * cp_ratio))
            signals.append(np.append(resource_samples[-num_prefix_samples:], resource_samples))


        return np.concatenate(signals, axis=0)

    def demodulate(self,
                   baseband_signal: np.ndarray,
                   ideal_channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        rel_samples_per_slot = self.frame.modem.scenario.sampling_rate / self.frame.subcarrier_spacing
        samples_per_slot = int(round(rel_samples_per_slot))

        # Samples without prefixes in the time-time grid, essentially sections of the demodulating stft
        slot_samples = np.empty((samples_per_slot, self.num_timeslots), dtype=complex)
        slot_channels = np.empty((samples_per_slot, self.num_timeslots), dtype=complex)

        # Remove the cyclic prefixes before transformation into time-domain
        sample_index = 0
        for slot_idx in range(len(self.pattern) * self.num_repetitions):

            pattern_idx = slot_idx % len(self.pattern)
            resource = self.frame.resources[self.pattern[pattern_idx]]

            num_prefix_samples = int(round(rel_samples_per_slot * resource.cp_ratio))

            sample_index += num_prefix_samples
            slot_samples[:, slot_idx] = baseband_signal[sample_index:sample_index+samples_per_slot]
            slot_channels[:, slot_idx] = ideal_channel[sample_index:sample_index + samples_per_slot]

            sample_index += samples_per_slot

        num_slot_samples = int(round(self.frame.modem.scenario.sampling_rate / self.frame.subcarrier_spacing))

        # Transform grid back to data symbols
        dft_matrix = self.frame.fourier_weights(num_slot_samples,
                                                self.num_subcarriers,
                                                self.frame.subcarrier_spacing,
                                                self.frame.dc_suppression)

        ofdm_grid = dft_matrix @ slot_samples
        channel_grid = dft_matrix @ slot_channels

        return ofdm_grid, channel_grid

    @property
    def resource_mask(self) -> np.ndarray:

        # Initialize the base mask as all false
        num_subcarriers = self.num_subcarriers
        mask = np.ndarray((len(ElementType), num_subcarriers, self.num_timeslots), dtype=bool) * False

        for resource_section, resource_idx in enumerate(self.pattern):

            resource = self.frame.resources[resource_idx]
            resource_mask = resource.resource_mask

            repeated_mask = np.repeat(resource_mask[..., np.newaxis], self.num_repetitions, axis=2)
            mask[:, :resource_mask.shape[1], resource_section::len(self.pattern)] = repeated_mask

        return mask

    @property
    def num_samples(self) -> int:
        
        num = 0

        num_samples_per_slot = self.frame.modem.scenario.sampling_rate / self.frame.subcarrier_spacing

        # Add up the additional samples from cyclic prefixes
        for resource_idx in self.pattern:
            num += int(round(num_samples_per_slot * self.frame.resources[resource_idx].cp_ratio))

        # Add up the base samples from each timeslot
        num += int(round(num_samples_per_slot)) * len(self.pattern)
        return num * self.num_repetitions

    @classmethod
    def from_yaml(cls: Type[FrameSymbolSection],
                  constructor: SafeConstructor,
                  node: Union[ScalarNode, MappingNode]) -> FrameSymbolSection:

        if isinstance(node, ScalarNode):
            return cls()

        return cls(**constructor.construct_mapping(node))


class FrameGuardSection(FrameSection):

    yaml_tag: str = u'Guard'
    __duration: float

    def __init__(self,
                 duration: float,
                 num_repetitions: int = 1) -> None:

        FrameSection.__init__(self, num_repetitions=num_repetitions)
        self.duration = duration

    @property
    def duration(self) -> float:
        return self.__duration

    @duration.setter
    def duration(self, secs: float) -> None:
        """Modify guard section duration.

        Args:
            secs (float): New duration in seconds.

        Raises:
            ValueError: If secs is smaller than zero.
        """

        if secs < 0.0:
            raise ValueError("Guard section duration must be greater or equal to zero")

        self.__duration = secs

    @property
    def num_samples(self) -> int:

        num = int(round(self.num_repetitions * self.__duration * self.frame.modem.scenario.sampling_rate))
        return num

    def modulate(self, symbols: np.ndarray) -> np.ndarray:

        if len(symbols) > 0:
            raise ValueError("Guard sections may not hold modulation symbols")

        return np.zeros(self.num_samples, dtype=complex)

    def demodulate(self,
                   baseband_signal: np.ndarray,
                   ideal_channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        # Guard sections naturally don't encode anything
        return np.empty(0, dtype=complex), np.empty(0, dtype=complex)

    @classmethod
    def from_yaml(cls: Type[FrameGuardSection],
                  constructor: SafeConstructor,
                  node: Union[ScalarNode, MappingNode]) -> FrameGuardSection:

        if isinstance(node, ScalarNode):
            return cls()

        return cls(**constructor.construct_mapping(node))


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

    yaml_tag: str = WaveformGenerator.yaml_tag + u'OFDM'

#    pilot_subcarriers: List[np.ndarray]
#    pilot_symbols: List[np.ndarray]
#    reference_symbols: List[np.ndarray]
#    __fft_size: int
    __subcarrier_spacing: float
    dc_suppression: bool
    resources: List[FrameResource]
    structure: List[FrameSection]

    def __init__(self,
#                 pilot_subcarriers: Optional[List[np.ndarray]] = None,
#                 pilot_symbols: Optional[List[np.ndarray]] = None,
#                 reference_symbols: Optional[List[np.ndarray]] = None,
#                 fft_size: int = 1,
                 channel_estimation: Union[str, ChannelEstimation] = ChannelEstimation.IDEAL,
                 subcarrier_spacing: float = 0.,
                 dc_suppression: bool = True,
                 resources: Optional[List[FrameResource]] = None,
                 structure: Optional[List[FrameSection]] = None,
                 modem: Modem = None,
                 oversampling_factor: int = 1,
                 modulation_order: int = 64) -> None:
        """Orthogonal Frequency Division Multiplexing Waveform Generator initialization.

        Args:

            subcarrier_spacing (float, optional):
                Spacing between individual subcarriers in Hz. ToDo: Check.

            dc_suppression (bool, optional):
                Suppress the direct current component during waveform generation.

        """

        # Init base class
        WaveformGenerator.__init__(self, modem=modem, oversampling_factor=oversampling_factor,
                                   modulation_order=modulation_order)

        # Parameter initialization
#        self.pilot_subcarriers = pilot_subcarriers if pilot_subcarriers is not None else []
#        self.pilot_symbols = pilot_symbols if pilot_symbols is not None else []
#        self.reference_symbols = reference_symbols if reference_symbols is not None else []
#        self.fft_size = fft_size
        if isinstance(channel_estimation, str):
            channel_estimation = ChannelEstimation[channel_estimation]
        self.channel_estimation_algorithm = channel_estimation
        self.subcarrier_spacing = subcarrier_spacing
        self.dc_suppression = dc_suppression
        self.resources = [] if resources is None else resources
        self.structure = [] if structure is None else structure

        # Initial parameter checks
        # TODO

        self._samples_in_frame_no_oversampling = 0
        self._mapping = PskQamMapping(self.modulation_order)

#        self._resource_element_mapping: np.array = self._calculate_resource_element_mapping()
#        self._samples_in_frame_no_oversampling, self._cyclic_prefix_overhead = (
#            self._calculate_samples_in_frame()
#        )

#        self.reference_frame = np.zeros((self.symbols_per_frame, self.num_occupied_subcarriers), dtype=complex)
#        self.data_frame_indices = np.zeros((self.symbols_per_frame, self.num_occupied_subcarriers), dtype=bool)
        self.guard_time_indices = np.array([], dtype=int)
        self.prefix_time_indices = np.array([], dtype=int)
        self.data_time_indices = np.array([], dtype=int)
        self.channel_sampling_timestamps = np.array([], dtype=int)

        # derived variables for precoding
        self._data_resource_elements_per_symbol = np.array([])
        # self._generate_frame_structure()

    def add_section(self, section: FrameSection) -> None:

        self.structure.append(section)
        section.frame = self

    @property
    def subcarrier_spacing(self) -> float:
        """Subcarrier spacing between frames.

        Returns:
            float: Spacing in Hz.
        """

        return self.__subcarrier_spacing

    @subcarrier_spacing.setter
    def subcarrier_spacing(self, spacing: float) -> None:
        """Modify the subcarrier spacing between frames.

        Args:
            spacing (float): New spacing in Hz.

        Raises:
            ValueError: If `spacing` is smaller or equal to zero.
        """

        if spacing <= 0.0:
            raise ValueError("Subcarrier spacing must be greater than zero")

        self.__subcarrier_spacing = spacing

    @property
    def symbols_per_frame(self) -> int:

        num_symbols = 0
        for section in self.structure:
            num_symbols += section.num_symbols

        return num_symbols

    @property
    def words_per_frame(self) -> int:

        num_words = 0
        for section in self.structure:
            num_words += section.num_words

        return num_words

    @property
    def references_per_frame(self) -> int:

        num_symbols = 0
        for section in self.structure:
            num_symbols += section.num_references

        return num_symbols

    @property
    def frame_duration(self) -> float:

        duration = 0.

        for section in self.structure:
            duration += section.duration

        return duration

    ###################################
    # property definitions
    @property
    def samples_in_frame(self) -> int:
        """int: Returns read-only samples_in_frame"""

        num = 0
        for section in self.structure:
            num += section.num_samples

        return num

    @property
    def num_subcarriers(self) -> int:

        num = 0
        for section in self.structure:
            num = max(num, section.num_subcarriers)

        return num

    # property definitions END
    #############################################

    def map(self, data_bits: np.ndarray) -> np.ndarray:
        return self._mapping.get_symbols(data_bits)

    def unmap(self, data_symbols: np.ndarray) -> np.ndarray:

        detected_bits = self._mapping.detect_bits(data_symbols).astype(int)
        return detected_bits

    def modulate(self, data_symbols: np.ndarray, timestamps: np.ndarray) -> np.ndarray:

        # The number of samples in time domain the frame should contain, given the current sample frequency
        num_samples = int(round(self.frame_duration * self.modem.scenario.sampling_rate))
        output_signal = np.empty(0, dtype=complex)

        sent_data_symbols = 0
        for section in self.structure:

            # Number of data symbols encoded in this time-section of the OFDM frame
            section_num_data_symbols = section.num_symbols

            # Select the data symbols to be sent over this time-section of the OFDM frame
            section_data_symbols = data_symbols[sent_data_symbols:sent_data_symbols+section_num_data_symbols]
            sent_data_symbols += section_num_data_symbols

            # Modulate the signal
            section_signal = section.modulate(section_data_symbols)
            output_signal = np.append(output_signal, section_signal)

        return output_signal

    def demodulate(self,
                   signal: np.ndarray,
                   impulse_response: np.ndarray,
                   noise_variance: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Recover OFDM grid
        symbol_grid = np.empty((self.num_subcarriers, self.words_per_frame), dtype=complex)
        impulse_grid = np.empty((self.num_subcarriers, self.words_per_frame), dtype=complex)
        resource_mask = np.empty((len(ElementType), self.num_subcarriers, self.words_per_frame), dtype=bool)

        sample_index = 0
        word_index = 0
        for section in self.structure:

            num_samples = section.num_samples
            num_words = section.num_words

            if num_words < 1:

                sample_index += num_samples
                continue

            signal_section = signal[sample_index:sample_index+num_samples]
            impulse_response_section = impulse_response[sample_index:sample_index+num_samples]

            section_symbol_grid, section_response_grid = section.demodulate(signal_section, impulse_response_section)
            section_mask = section.resource_mask

            symbol_grid[:, word_index:word_index+num_words] = section_symbol_grid
            impulse_grid[:, word_index:word_index+num_words] = section_response_grid
            resource_mask[:, :, word_index:word_index+num_words] = section_mask

            sample_index += num_samples
            word_index += num_words

        # Estimate the channel given the recovered OFDM resources
        channel_estimation = self.__channel_estimation(symbol_grid, impulse_grid, resource_mask)

        # Recover the data symbols, as well as the respective channel weights from the resource grids
        data_symbols = symbol_grid.T[resource_mask[ElementType.DATA.value, ::].T]
        channel_weights = channel_estimation.T[resource_mask[ElementType.DATA.value, ::].T]
        noise_variances = np.repeat(noise_variance, self.symbols_per_frame)

        return data_symbols, channel_weights, noise_variances

    @property
    def bandwidth(self) -> float:

        # OFDM bandwidth currently is identical to the number of subcarriers times the subcarrier spacing
        b = self.num_subcarriers * self.subcarrier_spacing
        return b

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
        #elif self.param.precoding == "GFDM":
            # 1. get data symbols from frame (nix mit gfdm zu tun)
            # 2. definiere pulse shape, wie? li muss ahmad fragen
            # 3. create window (gfdm_func.g2Wtx), z.B. Wtx_FD = gfdm_func.g2Wtx(g, K, M, "FD")
            # 4. data_symbols = data_symbols.reshape(K, M), len(D) MUSS gleich K*M, sonst zero padding oder so
            # 5. data_symbols = GFDM_Mod(D, window von 3, K_set, TD/FD)
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
        #elif self.param.precoding == "GFDM":
            # 1. create window für rx, Wtx2Wrx(Window transmitter, "ZF", 1, 1)
            # 2. data_symbols = GFDM_Demod(data_symbols, window von 1, K_set, TD/FD)
            # (3. data_symbols = data_sy)
            # 3. data_symbols = von KxM auf K*Mx1... wie's halt sein muss

        return frame, noise_var

    def __channel_estimation(self, symbol_grid: np.ndarray, channel_grid: np.ndarray, resource_mask: np.ndarray) -> np.ndarray:
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

        # Ideally, the channel is estimated perfectly at each received symbol slot
        if self.channel_estimation_algorithm == ChannelEstimation.IDEAL:
            return channel_grid

        # An ideal preamble estimates the channel at the first symbol position
        if self.channel_estimation_algorithm == ChannelEstimation.IDEAL_PREAMBLE:
            return channel_grid[:, 0].repeat(channel_grid.shape[1], axis=1)

        if self.channel_estimation_algorithm == ChannelEstimation.IDEAL_MIDAMBLE:
            return channel_grid[int(round(channel_grid.shape[1]))].repeat(channel_grid.shape[1], axis=1)

        # An ideal postamble estimates the channel at the last symbol position
        if self.channel_estimation_algorithm == ChannelEstimation.IDEAL_POSTAMBLE:
            return channel_grid[:, -1].repeat(channel_grid.shape[1], axis=1)

        if self.channel_estimation_algorithm == ChannelEstimation.REFERENCE_SIGNAL:
            return self.__reference_based_channel_estimation(rx_signal)

        #if self.param.stream_responses in {"LS", "LEAST_SQUARE"}:
        #    return self.__reference_based_channel_estimation(rx_signal)
        #    channel_in_freq_domain = np.repeat(channel_in_freq_domain[:, :, np.newaxis, :], number_of_symbols, axis=2)
        #else:
        #    raise ValueError('invalid channel estimation type')
        #
        # return channel_in_freq_domain

    def __ideal_channel_estimation(self) -> np.ndarray:
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
            # if stream_responses is missing at any frequency or different frequencies
            # then interpolate
            ch_est_freqs = np.where(stream_responses != 0)[1]
            ch_est_freqs[ch_est_freqs > self.param.fft_size / 2] = (ch_est_freqs[ch_est_freqs > self.param.fft_size / 2]
                                                                    - self.param.fft_size)
            ch_est_freqs = ch_est_freqs * self.param.subcarrier_spacing
            ch_est_freqs = np.fft.fftshift(ch_est_freqs)

            interp_function = interpolate.interp1d(ch_est_freqs, np.fft.fftshift(stream_responses))

            stream_responses = interp_function(frequency_bins)
        """

        # multiple antennas
        # check interpolation

        return channel_estimation_freq

    @property
    def bits_per_frame(self) -> int:
        return self.symbols_per_frame * self._mapping.bits_per_symbol

    @property
    def bit_energy(self) -> float:

        return 1 / self._mapping.bits_per_symbol  # ToDo: Re-implement
        # return self.oversampling_factor / self._mapping.bits_per_symbol * self._cyclic_prefix_overhead

    @property
    def symbol_energy(self) -> float:

        return 1 / self._mapping.bits_per_symbol  # ToDo: Re-implement
        # return self.oversampling_factor * self._cyclic_prefix_overhead

    @property
    def power(self) -> float:

        return 1  # ToDo: Re-implement
#        return self.num_occupied_subcarriers / self.fft_size

    @classmethod
    def to_yaml(cls: Type[WaveformGeneratorOfdm],
                representer: SafeRepresenter,
                node: WaveformGeneratorOfdm) -> MappingNode:
        """Serialize an `WaveformGenerator` object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (WaveformGeneratorOfdm):
                The `WaveformGeneratorOfdm` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node
        """

        state = {
            "guard_interval": node.guard_interval,
        }

        mapping = representer.represent_mapping(cls.yaml_tag, state)
        mapping.value.extend(WaveformGenerator.to_yaml(representer, node).value)

        return mapping

    @classmethod
    def from_yaml(cls: Type[WaveformGeneratorOfdm], constructor: SafeConstructor, node: MappingNode)\
            -> WaveformGeneratorOfdm:
        """Recall a new `WaveformGeneratorOfdm` instance from YAML.

        Args:
            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `WaveformGeneratorOfdm` serialization.

        Returns:
            WaveformGeneratorOfdm:
                Newly created `WaveformGeneratorOfdm` instance.
        """

        state = constructor.construct_mapping(node, deep=True)
        state = {k.lower(): v for k, v in state.items()}

        structure: List[FrameSection] = state.pop('structure', None)
        resources = state.pop('resources', None)

        # Handle resource list to object conversion
        if resources is not None:
            for resource_idx, resource in enumerate(resources):

                element_objects = []
                elements = resource.pop('elements', [])
                for element_args in elements:
                    element_objects.append(FrameElement(**element_args))
                resource['elements'] = element_objects

                resources[resource_idx] = FrameResource(**resource)

        state['resources'] = resources

        # Create actual frame object from state dictionary
        ofdm = cls(**state)

        if structure is not None:
            for section in structure:
                ofdm.add_section(section)

        return ofdm

    @lru_cache(maxsize=5)
    def fourier_weights(self, num_timestamps, num_subcarriers, subcarrier_spacing, dc_suppression) -> np.ndarray:

        slot_timestamps = np.arange(num_timestamps) / self.modem.scenario.sampling_rate

        if dc_suppression:
            discrete_frequencies = 2 * pi * (1 + np.arange(num_subcarriers)) * subcarrier_spacing

        else:
            discrete_frequencies = 2 * pi * np.arange(num_subcarriers) * subcarrier_spacing

        return np.exp(-1j * np.outer(discrete_frequencies, slot_timestamps)) / np.sqrt(num_timestamps)

    @lru_cache(maxsize=5)
    def inverse_fourier_weights(self, num_timestamps, num_subcarriers, subcarrier_spacing, dc_suppression) -> np.ndarray:

        slot_timestamps = np.arange(num_timestamps) / self.modem.scenario.sampling_rate

        if dc_suppression:
            discrete_frequencies = 2 * pi * (1 + np.arange(num_subcarriers)) * subcarrier_spacing

        else:
            discrete_frequencies = 2 * pi * np.arange(num_subcarriers) * subcarrier_spacing

        return np.exp(1j * np.outer(slot_timestamps, discrete_frequencies)) / np.sqrt(num_timestamps)
