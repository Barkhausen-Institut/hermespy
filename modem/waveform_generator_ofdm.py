# -*- coding: utf-8 -*-
"""HermesPy Orthogonal Frequency Division Multiplexing Waveform Generation."""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple, Optional, Type, Union, Any
from copy import copy
from scipy import signal
from functools import lru_cache
from dataclasses import dataclass, field
from abc import ABC
from enum import Enum
from collections import namedtuple
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode, ScalarNode
import numpy as np

from modem import WaveformGenerator
from modem.tools import PskQamMapping

if TYPE_CHECKING:
    from modem import Modem

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "André Noll Barreto"
__email__ = "andre.nollbarreto@barkhauseninstitut.org"
__status__ = "Prototype"


class ElementType(Enum):

    REFERENCE = 0
    DATA = 1
    NULL = 2


class FrameElement:

    type: ElementType
    repetitions: int = 1

    def __init__(self,
                 type: Union[str, ElementType],
                 repetitions: int = 1) -> None:

        self.type = ElementType[type] if isinstance(type, str) else type
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
        """Modify the ratio betwen full block element length and cyclic prefix.

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


class FrameSection:
    """OFDM Frame configuration time axis."""

    frame: Optional[Frame]
    num_repetitions: int

    def __init__(self,
                 num_repetitions: int = 1,
                 frame: Optional[Frame] = None) -> None:

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
    def num_subcarriers(self) -> int:
        """Number of subcarriers this section requires.

        Returns:
            int: The number of subcarriers.
        """

        return 0


class FrameSymbolSection(FrameSection):

    yaml_tag: str = u'Symbol'
    __pattern: List[int]

    def __init__(self,
                 num_repetitions: int = 1,
                 pattern: Optional[List[int]] = None,
                 frame: Optional[Frame] = None) -> None:

        FrameSection.__init__(self, num_repetitions=num_repetitions, frame=frame)
        self.__pattern = pattern if pattern is not None else []
        self.frame = frame

    @property
    def num_symbols(self) -> int:

        num = 0
        for resource_idx in self.__pattern:

            resource = self.frame.resources[resource_idx]
            num += resource.num_symbols

        return self.num_repetitions * num

    @property
    def num_subcarriers(self) -> int:

        # ToDo: Resources with different numbers of subcarriers are currently not supported
        num = 0
        if len(self.__pattern) > 0:
            num = self.frame.resources[self.__pattern[0]].num_subcarriers

        return num


    @classmethod
    def from_yaml(cls: Type[FrameSymbolSection],
                  constructor: SafeConstructor,
                  node: Union[ScalarNode, MappingNode]) -> FrameSymbolSection:

        if isinstance(node, ScalarNode):
            return cls()

        return cls(**constructor.construct_mapping(node))


class FrameGuardSection(FrameSection):

    yaml_tag: str = u'Guard'
    __length: float

    def __init__(self,
                 length: float,
                 num_repetitions: int = 1) -> None:

        FrameSection.__init__(self, num_repetitions=num_repetitions)
        self.length = length

    @property
    def length(self) -> float:
        """Guard section length.

        Returns:
            float: Length in seconds.
        """

        return self.__length

    @length.setter
    def length(self, secs: float) -> None:
        """Modify guard section length-

        Args:
            secs (float): New length in seconds.

        Raises:
            ValueError: If secs is smaller than zero.
        """

        if secs < 0.0:
            raise ValueError("Guard section length must be greater or equal to zero")

        self.__length = secs

    @classmethod
    def from_yaml(cls: Type[FrameGuardSection],
                  constructor: SafeConstructor,
                  node: Union[ScalarNode, MappingNode]) -> FrameGuardSection:

        if isinstance(node, ScalarNode):
            return cls()

        return cls(**constructor.construct_mapping(node))


class Frame:

    yaml_tag = u'Frame'
    resources: List[FrameResource]
    structure: List[FrameSection]
    __subcarrier_spacing: float

    def __init__(self,
                 subcarrier_spacing: float,
                 resources: Optional[List[FrameResource]] = None,
                 structure: Optional[List[FrameSection]] = None) -> None:

        self.resources = resources if resources is not None else []
        self.structure = structure if structure is not None else []

    @property
    def num_symbols(self) -> int:
        """Number of symbol slots within this frame configuration.

        Return:
            int: Number of symbol slots.
        """

        num: int = 0
        for section in self.structure:
            num += section.num_symbols

        return num

    def add_section(self, section: FrameSection) -> None:

        self.structure.append(section)
        section.frame = self

    @classmethod
    def from_yaml(cls: Type[Frame], constructor: SafeConstructor, node: MappingNode) -> Frame:

        state: dict[str, Any] = constructor.construct_mapping(node, deep=True)
        state = {k.lower(): v for k, v in state.items()}

        structure: List[FrameSection] = state.pop('structure', None)

        # Handle resource list to object conversion
        resources = state.pop('resources', None)
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
        frame = cls(**state)

        if structure is not None:
            for section in structure:
                frame.add_section(section)

        return frame


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

    def demodulate(self, signal: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        pass

    yaml_tag: str = WaveformGenerator.yaml_tag + u'OFDM'

    __frame: Frame
    pilot_subcarriers: List[np.ndarray]
    pilot_symbols: List[np.ndarray]
    reference_symbols: List[np.ndarray]
    __guard_interval: float
    __fft_size: int
    __num_occupied_subcarriers: int
    __subcarrier_spacing: float
    dc_suppression: bool

    def __init__(self,
                 frame: Optional[Frame] = None,
                 pilot_subcarriers: Optional[List[np.ndarray]] = None,
                 pilot_symbols: Optional[List[np.ndarray]] = None,
                 reference_symbols: Optional[List[np.ndarray]] = None,
                 guard_interval: float = 0.0,
                 fft_size: int = 1,
                 num_occupied_subcarriers: int = 1,
                 subcarrier_spacing: float = 0.,
                 dc_suppression: bool = True,
                 modem: Modem = None,
                 oversampling_factor: int = 1,
                 modulation_order: int = 64) -> None:
        """Orthogonal Frequency Division Multiplexing Waveform Generator initialization.

        Args:

            frame_structure (List[FrameElement], optional):
                Structure configuration of the generated frame.

            guard_interval (float, optional):
                Spacing between individual frame transmission in seconds.

            fft_size (int, optional):
                Number of frequency bins in the Fast Fourier Transform.

            num_occupied_subcarriers (int, optional):
                Number of subcarriers occupied within the frame structure??
                TODO: Check.

            subcarrier_spacing (float, optional):
                Spacing between individual subcarriers in Hz. ToDo: Check.

            dc_suppression (bool, optional):
                Suppress the direct current component during waveform generation.

        """

        # Init base class
        WaveformGenerator.__init__(self, modem=modem, oversampling_factor=oversampling_factor,
                                   modulation_order=modulation_order)

        # Parameter initialization
        self.__frame = frame if frame is not None else Frame()
        self.pilot_subcarriers = pilot_subcarriers if pilot_subcarriers is not None else []
        self.pilot_symbols = pilot_symbols if pilot_symbols is not None else []
        self.reference_symbols = reference_symbols if reference_symbols is not None else []
        self.guard_interval = guard_interval
        self.fft_size = fft_size
        self.num_occupied_subcarriers = num_occupied_subcarriers
        self.subcarrier_spacing = subcarrier_spacing
        self.dc_suppression = dc_suppression

        # Initial parameter checks
        # TODO

        if frame is not None:
            _ = frame.num_symbols

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
        
    @property
    def guard_interval(self) -> float:
        """Guard interval between frames.

        Returns:
            float: Interval in seconds.
        """

        return self.__guard_interval

    @guard_interval.setter
    def guard_interval(self, interval: float) -> None:
        """Modify the guard interval between frames.

        Args:
            interval (float): New interval in seconds.

        Raises:
            ValueError: If `interval` is smaller than zero.
        """

        if interval < 0.0:
            raise ValueError("Guard interval must be greater or equal to zero")

        self.__guard_interval = interval

    @property
    def fft_size(self) -> int:
        """Number of frequency bins in the Fast Fourier Transform.

        Returns:
            int: Number of frequency bins
        """

        return self.__fft_size

    @fft_size.setter
    def fft_size(self, size: int) -> None:
        """Modify the number of frequency bins in the Fast Fourier Transform.

        Args:
            size (int): New number of frequency bins.

        Raises:
            ValueError: If `size` is smaller than one.
        """

        if size < 1:
            raise ValueError("Number of frequency bins must be greater or equal to one")

        self.__fft_size = size

    @property
    def num_occupied_subcarriers(self) -> int:
        """Number of occupied subcarrier bands within the bandwidth.

        Returns:
            int: Number of occupied subcarriers.
        """

        return self.__num_occupied_subcarriers

    @num_occupied_subcarriers.setter
    def num_occupied_subcarriers(self, num: int) -> None:
        """Modify the number of occupied subcarrier bands within the bandwidth.

        Args:
            num (int): New number of occupied subcarriers.

        Raises:
            ValueError: If `num` is smaller than zero.
        """

        if num < 0:
            raise ValueError("Number of occupied subcarriers must be greater or equal to zeros")

        self.__num_occupied_subcarriers = num
        
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
            ValueError: If `spacing` is smaller than zero.
        """

        if spacing < 0.0:
            raise ValueError("Subcarrier spacing must be greater or equal to zero")

        self.__subcarrier_spacing = spacing

    @property
    def symbols_per_frame(self) -> int:
        return self.__frame.num_symbols

    @property
    def frame_duration(self) -> float:
        pass

    def _generate_frame_structure(self):
        """Creates the OFDM frame structure in time, frequency and space.

        This method interprets the OFDM parameters in 'self.param' that describe the OFDM frame and generates matrices
        with the allocation of all resource elements in a time/frequency/antenna grid

        """
        ofdm_symbol_idx = 0
        sample_idx = 0
        self.channel_sampling_timestamps = np.array([], dtype=int)

        for frame_element in self.__frame.structure:

            if isinstance(frame_element, FrameGuardSection):

                num_samples = self.guard_interval * self.subcarrier_spacing * self.fft_size

                self.guard_time_indices = np.append(self.guard_time_indices,
                                                    np.arange(sample_idx, sample_idx + num_samples))
                sample_idx += num_samples

            elif isinstance(frame_element, FrameSymbolSection):

                ref_idxs = self._get_subcarrier_indices(frame_element, ResourceType.REFERENCE)
                self.channel_sampling_timestamps = np.append(self.channel_sampling_timestamps, sample_idx)

                # fill out resource elements with pilot symbols
                ref_symbols = np.tile(self.reference_symbols,
                                      int(np.ceil(ref_idxs.size / len(self.reference_symbols))))
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

        #if self.param.precoding != "NONE":
        #    # check if all symbols have the same number of data REs
        #    self._data_resource_elements_per_symbol = np.sum(self.data_frame_indices[:, :, 0], axis=1)

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

        for frame_element in self.__frame.structure:

            if isinstance(frame_element, FrameGuardSection):
                samples_in_frame_no_oversampling += frame_element.no_samples
            else:
                samples_in_frame_no_oversampling += frame_element.cyclic_prefix_samples
                number_cyclic_prefix_samples += frame_element.cyclic_prefix_samples

                samples_in_frame_no_oversampling += frame_element.no_samples
                number_of_data_samples += frame_element.no_samples

        cyclic_prefix_overhead = 0.0
        if number_of_data_samples > 0.0:
            cyclic_prefix_overhead = (number_of_data_samples + number_cyclic_prefix_samples) / number_of_data_samples

        return samples_in_frame_no_oversampling, cyclic_prefix_overhead

    def _calculate_resource_element_mapping(self) -> np.array:
        initial_index = self.fft_size - \
            int(np.ceil(self.num_occupied_subcarriers / 2))
        resource_element_mapping: np.array = np.arange(
            initial_index, self.fft_size)
        final_index = int(np.floor(self.num_occupied_subcarriers / 2))
        resource_element_mapping = np.append(
            resource_element_mapping, np.arange(
                self.dc_suppression, final_index + self.dc_suppression))
        return resource_element_mapping

    ###################################
    # property definitions
    @property
    def samples_in_frame(self) -> int:
        """int: Returns read-only samples_in_frame"""
        return self._samples_in_frame_no_oversampling * self.oversampling_factor

    @property
    def bits_in_frame(self) -> int:
        """int: Returns read-only bits_in_frame"""
        return self.symbols_per_frame * self._mapping.bits_per_symbol

    @property
    def cyclic_prefix_overhead(self) -> float:
        """int: Returns read-only cyclic_prefix_overhead"""
        return self._cyclic_prefix_overhead

    # property definitions END
    #############################################

    def map(self, data_bits: np.ndarray) -> np.ndarray:
        return self._mapping.get_symbols(data_bits)

    def unmap(self, data_symbols: np.ndarray) -> np.ndarray:
        return self._mapping.detect_bits(data_symbols)

    def modulate(self, data_symbols: np.ndarray, timestamps: np.ndarray) -> np.ndarray:

        f#ull_frame = copy(self.reference_frame)
        #full_frame[np.where(self.data_frame_indices)] = data_symbols

        frame_in_freq_domain = np.zeros((self.symbols_per_frame, self.fft_size), dtype=complex)
        frame_in_freq_domain[:, self._resource_element_mapping] = full_frame

        frame_in_time_domain = np.fft.ifft(frame_in_freq_domain, norm='ortho', axis=1)
        frame_in_time_domain = self._add_guard_intervals(frame_in_time_domain)

        output_signal = np.zeros(self._samples_in_frame_no_oversampling, dtype=complex)

        data_symbols = np.reshape(frame, (self.symbols_per_frame * self.fft_size,
                                          self.param.number_tx_antennas))
        data_symbols = data_symbols.transpose()
        output_signal[:, self.data_time_indices] = data_symbols
        output_signal[:, self.prefix_time_indices] = output_signal[:, self.prefix_time_indices + self.param.fft_size]

        return output_signal

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

    @property
    def bits_per_frame(self) -> int:
        return self.symbols_per_frame * np.log2(self.modulation_order)

    @property
    def bit_energy(self) -> float:
        return self.oversampling_factor / self._mapping.bits_per_symbol * self._cyclic_prefix_overhead

    @property
    def symbol_energy(self) -> float:
        return self.oversampling_factor * self._cyclic_prefix_overhead

    @property
    def power(self) -> float:
        return self.num_occupied_subcarriers / self.fft_size

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

        state = constructor.construct_mapping(node)
        state = {k.lower(): v for k, v in state.items()}

        return cls(**state)
