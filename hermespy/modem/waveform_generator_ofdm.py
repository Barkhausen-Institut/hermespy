# -*- coding: utf-8 -*-
"""HermesPy Orthogonal Frequency Division Multiplexing Waveform Generation."""

from __future__ import annotations
from typing import List, Tuple, Optional, Type, Union, Any
from enum import Enum
from abc import abstractmethod

import numpy as np
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode, ScalarNode
from scipy.fft import fft, ifft
from scipy.interpolate import griddata

from hermespy.core.factory import Serializable
from hermespy.core.channel_state_information import ChannelStateInformation, ChannelStateDimension
from hermespy.core.signal_model import Signal
from hermespy.modem import WaveformGenerator
from hermespy.modem.tools import PskQamMapping

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
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
    REFERENCE = 4


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
    def mask(self) -> np.ndarray:
        """Boolean mask selecting a specific type of element from the OFDM grid.

        Returns:
            np.ndarray:
                Mask of dimension `num_element_types`x`num_subcarriers`.
        """

        # Initialize the base mask as all false
        mask = np.ndarray((len(ElementType), self.num_subcarriers), dtype=bool) * False

        element_count = 0
        for element in self.elements:

            mask[element.type.value, element_count:element_count+element.repetitions] = True
            element_count += element.repetitions

        # Repeat the subcarrier masks according to the configured number of repetitions.
        mask = np.tile(mask[:, :element_count], (1, self.__repetitions))
        return mask


class FrameSection:
    """OFDM Frame configuration time axis."""

    frame: Optional[WaveformGeneratorOfdm]
    __num_repetitions: int

    def __init__(self,
                 num_repetitions: int = 1,
                 frame: Optional[WaveformGeneratorOfdm] = None) -> None:

        self.frame = frame
        self.num_repetitions = num_repetitions

    @property
    def num_repetitions(self) -> int:
        """Number of section repetitions in the time-domain of an OFDM grid.

        Returns:
            int: The number of repetitions.
        """

        return self.__num_repetitions

    @num_repetitions.setter
    def num_repetitions(self, value: int) -> None:
        """Number of section repetitions in the time-domain of an OFDM grid.

        Args:
            value (int): The number of repetitions.

        Raises:
            ValueError: If `value` is smaller than one.
        """

        if value < 1:
            raise ValueError("OFDM frame number of repetitions must be greater or equal to one")

        self.__num_repetitions = value

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
        """Number of OFDM symbols, i.e. words of subcarrier symbols this section can modulate.

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
                   signal: np.ndarray,
                   channel_state: ChannelStateInformation) -> Tuple[np.ndarray, ChannelStateInformation]:
        """Demodulate a time section of a complex OFDM base-band signal into data symbols.

        Args:
            signal (np.ndarray): Vector of complex-valued base-band samples.
            channel_state (ChannelStateInformation): Channel state.

        Returns:
            (np.ndarray, channel_state):
                Section symbol grid and channel response grid.
        """
        ...


class FrameSymbolSection(FrameSection, Serializable):

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

        num = 0

        for resource_idx in set(self.pattern):
            num = max(num, self.frame.resources[resource_idx].num_subcarriers)

        return num

    def modulate(self, symbols: np.ndarray) -> np.ndarray:

        # Collect resource masks
        mask = self.resource_mask

        # Fill up the time-frequency grid exploiting the mask
        grid = np.zeros((self.frame.num_subcarriers, self.num_words), complex)

        # Reference fields all currently carry the complex symbol 1+j0
        # ToDo: Implement reference symbol configurations
        grid[mask[ElementType.REFERENCE.value]] = 1. + 0j

        # Data fields carry the supplied data symbols
        grid.T[mask[ElementType.DATA.value].T] = symbols

        # NULL fields are just that... zero
        grid[mask[ElementType.NULL.value]] = 0j

        # By convention, the length of each time slot is the inverse of the sub-carrier spacing
        num_slot_samples = self.frame.num_subcarriers * self.frame.oversampling_factor
        resource_signals = ifft(grid, n=num_slot_samples, axis=0, norm='ortho')

        # Add the cyclic prefix to each time slot while simultaneously flatten the resource signals into time domain
        signals = []
        for resource_idx, resource_samples in enumerate(resource_signals.T):

            pattern_idx = resource_idx % len(self.pattern)
            cp_ratio = self.frame.resources[self.pattern[pattern_idx]].cp_ratio

            num_prefix_samples = int(num_slot_samples * cp_ratio)

            if num_prefix_samples > 0:
                signals.append(resource_samples[-num_prefix_samples:])

            signals.append(resource_samples)

        signal_samples = np.concatenate(signals, axis=0)
        return signal_samples

    def demodulate(self,
                   signal: np.ndarray,
                   channel_state: ChannelStateInformation) -> Tuple[np.ndarray, ChannelStateInformation]:

        samples_per_slot = self.frame.num_subcarriers * self.frame.oversampling_factor

        # Remove the cyclic prefixes before transformation into time-domain
        sample_index = 0
        sample_indices = np.empty(0, dtype=int)
        channel_sample_indices = np.empty(0, dtype=int)
        num_slots = len(self.pattern) * self.num_repetitions

        for slot_idx in range(num_slots):

            pattern_idx = slot_idx % len(self.pattern)
            resource = self.frame.resources[self.pattern[pattern_idx]]

            num_prefix_samples = int(samples_per_slot * resource.cp_ratio)
            sample_index += num_prefix_samples

            sample_indices = np.append(sample_indices, np.arange(sample_index, sample_index + samples_per_slot))
            channel_sample_indices = np.append(channel_sample_indices, np.array(sample_index))

            sample_index += samples_per_slot

        slot_samples = signal[sample_indices].reshape((samples_per_slot, num_slots), order='F')
        slot_channel_state = channel_state[:, :, channel_sample_indices, :]\
            .to_frequency_selectivity(num_bins=self.frame.num_subcarriers)

        # Transform grid back to data symbols
        ofdm_grid = fft(slot_samples, n=samples_per_slot, axis=0, norm='ortho')[:self.frame.num_subcarriers, :]
        return ofdm_grid, slot_channel_state

    @property
    def resource_mask(self) -> np.ndarray:

        # Initialize the base mask as all false
        num_subcarriers = self.frame.num_subcarriers
        mask = np.zeros((len(ElementType), num_subcarriers, len(self.pattern)), dtype=bool)

        for word_idx, resource_idx in enumerate(self.pattern):

            resource = self.frame.resources[resource_idx]
            mask[:, :resource.num_subcarriers, word_idx] = resource.mask

        return np.tile(mask, (1, 1, self.num_repetitions))

    @property
    def num_samples(self) -> int:

        num_samples_per_slot = self.frame.num_subcarriers * self.frame.oversampling_factor
        num = len(self.pattern) * num_samples_per_slot

        # Add up the additional samples from cyclic prefixes
        for resource_idx in self.pattern:
            num += int(num_samples_per_slot * self.frame.resources[resource_idx].cp_ratio)

        # Add up the base samples from each timeslot
        return num * self.num_repetitions

    @classmethod
    def from_yaml(cls: Type[FrameSymbolSection],
                  constructor: SafeConstructor,
                  node: Union[ScalarNode, MappingNode]) -> FrameSymbolSection:

        if isinstance(node, ScalarNode):
            return cls()

        return cls(**constructor.construct_mapping(node))

    @classmethod
    def to_yaml(cls: Type[FrameSymbolSection], representer: SafeRepresenter, node: FrameSymbolSection) -> MappingNode:

        state = {
            'num_repetitions': node.num_repetitions,
            'pattern': node.pattern,
        }

        return representer.represent_mapping(node.yaml_tag, state)


class FrameGuardSection(FrameSection, Serializable):

    yaml_tag: str = u'Guard'
    __duration: float

    def __init__(self,
                 duration: float,
                 num_repetitions: int = 1,
                 frame: Optional[WaveformGeneratorOfdm] = None) -> None:

        FrameSection.__init__(self, num_repetitions=num_repetitions, frame=frame)
        self.duration = duration

    @property
    def duration(self) -> float:
        """Guard section duration in seconds.

        Returns:
            float: Duration in seconds.
        """

        return self.__duration

    @duration.setter
    def duration(self, value: float) -> None:
        """Guard section duration in seconds.

        Args:
            value (float): New duration.

        Raises:
            ValueError: If `value` is smaller than zero.
        """

        if value < 0.0:
            raise ValueError("Guard section duration must be greater or equal to zero")

        self.__duration = value

    @property
    def num_samples(self) -> int:

        return int(self.num_repetitions * self.__duration * self.frame.sampling_rate)

    def modulate(self, symbols: np.ndarray) -> np.ndarray:

        if len(symbols) > 0:
            raise ValueError("Guard sections may not hold modulation symbols")

        return np.zeros(self.num_samples, dtype=complex)

    def demodulate(self,
                   baseband_signal: np.ndarray,
                   channel_state: ChannelStateInformation) -> Tuple[np.ndarray, ChannelStateInformation]:

        # Guard sections naturally don't encode anything
        return np.empty((self.frame.num_subcarriers, 0), dtype=complex), ChannelStateInformation.Ideal(0)

    @classmethod
    def from_yaml(cls: Type[FrameGuardSection],
                  constructor: SafeConstructor,
                  node: MappingNode) -> FrameGuardSection:

        return cls(**constructor.construct_mapping(node))

    @classmethod
    def to_yaml(cls: Type[FrameGuardSection], representer: SafeRepresenter, node: FrameGuardSection) -> MappingNode:

        state = {
            'num_repetitions': node.num_repetitions,
            'duration': node.duration,
        }

        return representer.represent_mapping(cls.yaml_tag, state)


class WaveformGeneratorOfdm(WaveformGenerator, Serializable):
    """Generic Orthogonal-Frequency-Division-Multiplexing with a flexible frame configuration.

    The following features are supported:
        - The modem can transmit or receive custom-defined frames.
          Frames may contain UL/DL data symbols, null carriers, pilot subcarriers,
          reference signals and guard intervals.
        - SC-FDMA can also be implemented with a precoder.
        - Subcarriers can be modulated with BPSK/QPSK/16-/64-/256-QAM.
        - Cyclic prefixes for interference-free channel estimation and equalization are supported.

    This implementation has currently the following limitations:
        - All subcarriers use the same modulation scheme


    Attributes:

        __channel_estimation_algorithm (ChannelEstimation):
            Method deployed to simulate OFDM channel estimation.

        __num_subcarriers (int:
            Maximum number of subcarriers.
            Also the size of the FFT deployed during modulation,
            i.e. the difference between the configured number of subcarriers and the maximum number
            will be zero-padded.

        dc_suppression (bool):
            Suppress the direct current component during waveform generation.

        resources (List[FrameResource]):
            Frequency-domain resource section configurations.

        structure (List[FrameSection]):
            Time-domain frame configuration.
    """

    yaml_tag: str = u'OFDM'

    __channel_estimation_algorithm: ChannelEstimation
    __subcarrier_spacing: float
    __num_subcarriers: int
    dc_suppression: bool
    resources: List[FrameResource]
    structure: List[FrameSection]

    def __init__(self,
                 channel_estimation: Union[str, ChannelEstimation] = ChannelEstimation.IDEAL,
                 subcarrier_spacing: float = 1e3,
                 num_subcarriers: int = 1200,
                 dc_suppression: bool = True,
                 resources: Optional[List[FrameResource]] = None,
                 structure: Optional[List[FrameSection]] = None,
                 **kwargs: Any) -> None:
        """Orthogonal-Frequency-Division-Multiplexing Waveform Generator initialization.

        Args:

            channel_estimation (Union[str, ChannelEstimation], optional):
                Method deployed to simulate OFDM channel estimation.

            subcarrier_spacing (float, optional):
                Spacing between individual subcarriers in Hz.

            num_subcarriers (int, optional):
                Maximum number of subcarriers.
                Also the size of the FFT deployed during modulation,
                i.e. the difference between the configured number of subcarriers and the maximum number
                will be zero-padded.

            dc_suppression (bool, optional):
                Suppress the direct current component during waveform generation.

            resources (List[FrameResource], optional):
                Frequency-domain resource section configurations.

            structure (List[FrameSection], optional):
                Time-domain frame configuration.

            kwargs (Any):
                Waveform generator base class initialization parameters.
        """

        # Init base class
        WaveformGenerator.__init__(self, **kwargs)

        self.channel_estimation_algorithm = channel_estimation
        self.subcarrier_spacing = subcarrier_spacing
        self.num_subcarriers = num_subcarriers
        self.dc_suppression = dc_suppression
        self.resources = [] if resources is None else resources

        self.structure = []
        if structure is not None:
            for section in structure:
                self.add_section(section)

        self._mapping = PskQamMapping(self.modulation_order)

    def add_resource(self, resource: FrameResource) -> None:
        """Add a OFDM frequency resource to the waveform.

        Args:
            resource (FrameResource):
                The resource description to be added.
        """

        self.resources.append(resource)

    def add_section(self, section: FrameSection) -> None:
        """Add a frame section to the OFDM structure.

        Args:

            section (FrameSection):
                The section to be added.
        """

        self.structure.append(section)
        section.frame = self

    @property
    def channel_estimation_algorithm(self) -> ChannelEstimation:
        """Used channel estimation algorithm.

        Return:
            ChannelEstimation:
                The channel estimation algorithm.
        """

        return self.__channel_estimation_algorithm

    @channel_estimation_algorithm.setter
    def channel_estimation_algorithm(self, value: Union[str, ChannelEstimation]) -> None:
        """Modify the used channel estimation algorithm.

        Args:
            value (Union[str, ChannelEstimation]):
                New channel estimation algorithm.
        """

        if isinstance(value, str):
            self.__channel_estimation_algorithm = ChannelEstimation[value]

        else:
            self.__channel_estimation_algorithm = value

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

        """"duration = 0.

        for section in self.structure:
            duration += section.duration

        return duration"""
        return self.samples_in_frame / self.sampling_rate

    @property
    def samples_in_frame(self) -> int:
        """int: Returns read-only samples_in_frame"""

        num = 0
        for section in self.structure:
            num += section.num_samples

        return num

    def map(self, data_bits: np.ndarray) -> np.ndarray:
        return self._mapping.get_symbols(data_bits)

    def unmap(self, data_symbols: np.ndarray) -> np.ndarray:

        detected_bits = self._mapping.detect_bits(data_symbols).astype(int)
        return detected_bits

    def modulate(self, data_symbols: np.ndarray) -> Signal:

        # The number of samples in time domain the frame should contain, given the current sample frequency
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

        signal_model = Signal(output_signal, self.sampling_rate, carrier_frequency=self.modem.carrier_frequency)
        return signal_model

    def demodulate(self,
                   signal: np.ndarray,
                   channel_state: ChannelStateInformation,
                   noise_variance: float = 0.0) -> Tuple[np.ndarray, ChannelStateInformation, np.ndarray]:

        # Recover OFDM grid
        symbol_grid = np.empty((self.num_subcarriers, self.words_per_frame), dtype=complex)
        resource_mask = np.zeros((len(ElementType), self.num_subcarriers, self.words_per_frame), dtype=bool)
        section_channel_states: List[ChannelStateInformation] = []

        sample_index = 0
        word_index = 0
        for section in self.structure:

            num_samples = section.num_samples
            num_words = section.num_words

            if num_words < 1:

                sample_index += num_samples
                continue

            time_indices = np.arange(sample_index, sample_index+num_samples)
            signal_section = signal[time_indices]
            channel_state_section = channel_state[:, :, time_indices, :]

            section_symbol_grid, section_channel_state = section.demodulate(signal_section, channel_state_section)
            section_mask = section.resource_mask

            symbol_grid[:, word_index:word_index+num_words] = section_symbol_grid
            section_channel_states.append(section_channel_state)
            resource_mask[:, :, word_index:word_index+num_words] = section_mask

            sample_index += num_samples
            word_index += num_words

        ideal_channel_state = ChannelStateInformation.concatenate(section_channel_states,
                                                                  dimension=ChannelStateDimension.SAMPLES)

        # Estimate the channel given the recovered OFDM resources and convert it back to linear transformation matrices
        # Since we handle frequency bins here, the CSI transformations are diagonal over the last two dimensions
        channel_state_estimation = self.__channel_estimation(symbol_grid, ideal_channel_state, resource_mask)

        # Recover the data symbols, as well as the respective channel weights from the resource grids
        data_mask = resource_mask[ElementType.DATA.value]
        channel_state_estimation = channel_state_estimation[:, :, data_mask.flatten(), :]
        data_symbols = symbol_grid.T[data_mask.T]
        noise_variances = np.repeat(noise_variance, self.symbols_per_frame)

        return data_symbols, channel_state_estimation, noise_variances

    @property
    def bandwidth(self) -> float:

        # OFDM bandwidth currently is identical to the number of subcarriers times the subcarrier spacing
        b = self.num_subcarriers * self.subcarrier_spacing
        return b

    def __channel_estimation(self,
                             symbol_grid: np.ndarray,
                             channel_state: ChannelStateInformation,
                             resource_mask: np.ndarray) -> ChannelStateInformation:
        """Performs channel estimation over the OFDM grid.

        This methods estimates the frequency response of the channel for all OFDM symbols in a frame. The estimation
        algorithm is defined in the parameter variable `self.param`.

        With ideal channel estimation, the channel state information is obtained directly from the ideal channel state.
        The CSI can be considered to be known only at the beginning/middle/end of the frame
        (estimation_type='IDEAL_PREAMBLE'/'IDEAL_MIDAMBLE'/ 'IDEAL_POSTAMBLE'), or at every OFDM symbol ('IDEAL').

        With reference-based estimation, the specified reference subcarriers are employed for channel estimation.

        Args:

            symbol_grid (numpy.ndarray):
                Frequency-domain samples of the received signal over the whole frame.

            channel_state (ChannelStateInformation):
                Perfect channel state from which to simulate the channel state estimation.

            resource_mask (np.ndarray):
                Boolean mask for OFDM resource allocation.
                Required to distinguish between data, reference and null symbols within `symbol_grid`.

        Returns:
            ChannelStateInformation:
                The channel state estimate resulting from the selected method.
        """

        # Ideally, the channel is estimated perfectly at each received symbol slot
        if self.channel_estimation_algorithm == ChannelEstimation.IDEAL:
            return channel_state

        # Number of modulation symbols per ofdm word
        num_symbols = symbol_grid.shape[0]
        num_words = symbol_grid.shape[1]

        # An ideal pre-amble estimates the channel at the first sample position
        if self.channel_estimation_algorithm == ChannelEstimation.IDEAL_PREAMBLE:

            estimate = channel_state.state[:, :, :num_symbols, :]
            channel_state.state = np.tile(estimate, (1, 1, num_words, 1))
            return channel_state

        # An ideal mid-amble estimates the channel at the central symbol position
        if self.channel_estimation_algorithm == ChannelEstimation.IDEAL_MIDAMBLE:

            word_idx = int(.5 * num_words)
            estimate = channel_state.state[:, :, word_idx*num_symbols:(1+word_idx)*num_symbols, :]
            channel_state.state = np.tile(estimate, (1, 1, num_words, 1))
            return channel_state

        # An ideal post-amble estimates the channel at the last sample position
        if self.channel_estimation_algorithm == ChannelEstimation.IDEAL_POSTAMBLE:

            estimate = channel_state.state[:, :, -num_symbols:, :]
            channel_state.state = np.tile(estimate, (1, 1, num_words, 1))
            return channel_state

        if self.channel_estimation_algorithm == ChannelEstimation.REFERENCE:
            return self.reference_based_channel_estimation(symbol_grid, resource_mask)

        raise RuntimeError("Unknown OFDM channel estimation routine requested")

    @staticmethod
    def reference_based_channel_estimation(symbol_grid: np.ndarray,
                                           resource_mask: np.ndarray) -> ChannelStateInformation:
        """Perform a reference-symbol based channel estimation over the OFDM frame grid.

        This method estimates the channel using reference symbols. Only LS method is currently implemented. The function
        will return only a single value for each subcarrier. If several reference symbols are available, then the
        estimate will be averaged over all OFDM symbols.

        Args:

            symbol_grid (numpy.ndarray):
                Frequency-domain samples of the received signal over the whole frame.

            resource_mask (np.ndarray):
                Boolean mask for OFDM resource allocation.
                Required to distinguish between data, reference and null symbols within `symbol_grid`.

        Returns:

            ChannelStateInformation:
                The channel state estimate.
        """

        propagated_reference_symbols = symbol_grid.T[resource_mask[ElementType.REFERENCE.value, ::].T]
        reference_symbols = np.ones(len(propagated_reference_symbols), dtype=complex)
        reference_channel_estimation = propagated_reference_symbols / reference_symbols

        channel_estimation = np.zeros(symbol_grid.shape, dtype=complex)
        channel_estimation.T[resource_mask[ElementType.REFERENCE.value, ::].T] = reference_channel_estimation

        interpolation_stems = np.where(resource_mask[ElementType.REFERENCE.value, ::])
        holes = np.where(np.invert(resource_mask[ElementType.REFERENCE.value, ::]))

        # ToDo: Check with group what to do about missing values outside the convex hull
        interpolated_holes = griddata(interpolation_stems, reference_channel_estimation, holes, method='nearest')
        channel_estimation[holes] = interpolated_holes
        return channel_estimation[..., np.newaxis]   # Append an additional axis for multiple transmit antennas

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

    @property
    def num_subcarriers(self) -> int:
        """Maximum number of subcarriers.

        Sometimes also referred to as FFT-size.

        Returns:
            int: Number of subcarriers.
        """

        return self.__num_subcarriers

    @num_subcarriers.setter
    def num_subcarriers(self, value: int) -> None:
        """Modify the maximum number of subcarriers.

        Args:
            value (int): New maximum number of subcarriers.

        Raises:
            ValueError: If `value` is smaller than one.
        """

        if value < 1:
            raise ValueError("Number of subcarriers must be greater or equal to one")

        self.__num_subcarriers = value

    @property
    def sampling_rate(self) -> float:
        return self.oversampling_factor * self.subcarrier_spacing * self.__num_subcarriers

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
            'channel_estimation': node.__channel_estimation_algorithm.value,
            'subcarrier_spacing': node.__subcarrier_spacing,
            'num_subcarriers': node.__num_subcarriers,
            'dc_suppression': node.dc_suppression
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
