# -*- coding: utf-8 -*-
"""
==========================================
Orthogonal Frequency Division Multiplexing
==========================================
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from math import ceil
from typing import List, Tuple, Optional, Type, Union, Any, Set

import numpy as np
from ruamel.yaml import SafeConstructor, SafeRepresenter, MappingNode, Node
from scipy.fft import fft, fftshift, ifft, ifftshift
from scipy.interpolate import griddata
from scipy.signal import find_peaks

from hermespy.core import ChannelStateFormat, ChannelStateInformation, Serializable, SerializableEnum, Signal
from .symbols import StatedSymbols, Symbols
from .waveform import ChannelEqualization, ChannelEstimation, IdealChannelEstimation, ConfigurablePilotWaveform, Synchronization, WaveformGenerator, ZeroForcingChannelEqualization, MappedPilotSymbolSequence
from .waveform_correlation_synchronization import CorrelationSynchronization
from .tools import PskQamMapping

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["André Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ElementType(SerializableEnum):
    """Type of resource element."""

    REFERENCE = 0
    """Reference element within the time-frequency OFDM grid"""

    DATA = 1
    """Data element within the time-frequency OFDM grid"""

    NULL = 2
    """Empty element within the time-frequency OFDM grid"""


class PrefixType(SerializableEnum):
    """Type of prefix applied to the OFDM resource"""

    CYCLIC = 0
    """Cyclic prefix repeating the resource waveform in time-domain"""

    ZEROPAD = 1
    """Prefix zero-padding the prefix in time-domain"""

    NONE = 2
    """No prefix applied"""


class FrameElement(Serializable):
    yaml_tag = "FrameElement"
    serialized_attributes = {"type", "repetitions"}

    type: ElementType
    repetitions: int = 1

    def __init__(self, type: str | ElementType, repetitions: int = 1) -> None:
        self.type = ElementType[type] if isinstance(type, str) else type
        self.repetitions = repetitions


class FrameResource(Serializable):
    """Configures one sub-section of an OFDM symbol section in time AND frequency."""

    yaml_tag = "OFDM-Resource"
    serialized_attributes = {"prefix_type", "elements"}

    __repetitions: int
    __prefix_ratio: float

    prefix_type: PrefixType
    """Prefix type of the frame resource"""

    elements: List[FrameElement]
    """Individual resource elements"""

    def __init__(self, repetitions: int = 1, prefix_type: Union[PrefixType, str] = PrefixType.CYCLIC, prefix_ratio: float = 0.0, elements: Optional[List[FrameElement]] = None) -> None:
        self.repetitions = repetitions
        self.prefix_ratio = prefix_ratio
        self.prefix_type = PrefixType[prefix_type] if isinstance(prefix_type, str) else prefix_type
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
    def prefix_ratio(self) -> float:
        """Ratio between full block length and prefix length.

        Returns:
            float: The ratio between zero and one.

        Raises:
            ValueError: If ratio is less than zero or larger than one.
        """

        return self.__prefix_ratio

    @prefix_ratio.setter
    def prefix_ratio(self, ratio: float) -> None:
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("Cyclic prefix ratio must be between zero and one")

        self.__prefix_ratio = ratio

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
            mask[element.type.value, element_count : element_count + element.repetitions] = True
            element_count += element.repetitions

        # Repeat the subcarrier masks according to the configured number of repetitions.
        mask = np.tile(mask[:, :element_count], (1, self.__repetitions))
        return mask


class FrameSection:
    """OFDM Frame configuration time axis."""

    __frame: Optional[OFDMWaveform]
    __num_repetitions: int

    def __init__(self, num_repetitions: int = 1, frame: Optional[OFDMWaveform] = None) -> None:
        self.frame = frame
        self.num_repetitions = num_repetitions

    @property
    def frame(self) -> Optional[OFDMWaveform]:
        """OFDM frame this section belongs to.

        Returns:
            Handle to the OFDM frame.
            `None` if this section is considered floating.
        """

        return self.__frame

    @frame.setter
    def frame(self, value: Optional[OFDMWaveform]) -> None:
        self.__frame = value

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
        ...  # pragma: no cover

    def place_symbols(self, data_symbols: np.ndarray, reference_symbols: np.ndarray) -> np.ndarray:
        # Collect resource masks
        mask = self.resource_mask[:, : self.num_subcarriers, :]

        grid = np.zeros((self.num_subcarriers, self.num_words), dtype=complex)
        grid[mask[ElementType.REFERENCE.value, ::]] = reference_symbols
        grid[mask[ElementType.DATA.value, ::]] = data_symbols

        return grid

    def pick_symbols(self, grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Collect resource masks
        mask = self.resource_mask[:, : self.num_subcarriers, :]

        # Select correct subgrid
        subgrid = grid[: self.num_subcarriers, :]

        # Pick symbols
        reference_symbols = subgrid[mask[ElementType.REFERENCE.value]]
        data_symbols = subgrid[mask[ElementType.DATA.value]]

        return data_symbols, reference_symbols

    @abstractmethod
    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        """Modulate this section into a complex base-band signal.

        Args:

            symbols (np.ndarray):
                The palced complex symbols encoded in this OFDM section.
                This includes both reference and data symbols to be transmitted.

        Returns:
            np.ndarray: The modulated signal vector.
        """
        ...  # pragma: no cover

    @abstractmethod
    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """Demodulate a time section of a complex OFDM base-band signal into data symbols.

        Args:
            signal (np.ndarray): Vector of complex-valued base-band samples.

        Returns: Sequence of demodulated data and reference symbols.
        """
        ...  # pragma: no cover


class FrameSymbolSection(FrameSection, Serializable):
    yaml_tag: str = "Symbol"
    serialized_attributes = {"pattern"}

    pattern: List[int]

    def __init__(self, num_repetitions: int = 1, pattern: Optional[List[int]] = None, frame: Optional[OFDMWaveform] = None) -> None:
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

    def _add_prefix(self, resource_signals: np.ndarray) -> np.ndarray:
        """Add prefixes to time-domain resource signals.

        Args:

            resource_signals (np.ndarray):
                Numpy array of individual resource signals.

        Returns:
            Concatenated source signals with appended prefixes.
        """

        # Compute the number of required samples per resource
        padded_num_subcarriers = self.frame.num_subcarriers * self.frame.oversampling_factor

        signals = []
        for resource_idx, resource_samples in enumerate(resource_signals.T):
            # Infer pattern index
            pattern_idx = resource_idx % len(self.pattern)

            # Extract prefix parameters from configuration
            prefix_ratio = self.frame.resources[self.pattern[pattern_idx]].prefix_ratio
            prefix_type = self.frame.resources[self.pattern[pattern_idx]].prefix_type

            num_prefix_samples = int(padded_num_subcarriers * prefix_ratio)

            # Only add a prefix if required
            if num_prefix_samples > 0 and prefix_type != PrefixType.NONE:
                # Cyclic prefix
                if prefix_type == PrefixType.CYCLIC:
                    signals.append(resource_samples[-num_prefix_samples:])

                # Zero padding
                elif prefix_type == PrefixType.ZEROPAD:
                    signals.append(np.zeros(num_prefix_samples, dtype=complex))

                # Raise exception for unsupproted prefix types
                else:
                    raise RuntimeError("Unsupported prefix type configured")

            # Append base resource waveform after prefix
            signals.append(resource_samples)

        # The result is a concatenation in time domain of all prefixed resource signals
        signal_samples = np.concatenate(signals, axis=0)
        return signal_samples

    def _remove_prefix(self, signal_samples: np.ndarray) -> np.ndarray:
        """Remove prefixes and split signal into resource signals.

        Args:

            signal_samples(np.ndarray):
                Numpy vector of signal samples representing a single frame section.

        Returns: Two-dimensional numpy array representing signal samples of individual sections.
        """

        # Compute the number of required samples per resource
        padded_num_subcarriers = self.frame.num_subcarriers * self.frame.oversampling_factor

        sample_index = 0
        num_resources = len(self.pattern) * self.num_repetitions
        resource_samples = np.empty((padded_num_subcarriers, num_resources), dtype=complex)

        for resource_idx in range(num_resources):
            # Infer pattern index
            pattern_idx = resource_idx % len(self.pattern)

            # Extract prefix parameters from configuration
            resource = self.frame.resources[self.pattern[pattern_idx]]
            prefix_ratio = resource.prefix_ratio
            prefix_type = resource.prefix_type

            num_prefix_samples = int(padded_num_subcarriers * prefix_ratio)

            # Only add a prefix if required
            if num_prefix_samples > 0 and prefix_type != PrefixType.NONE:
                # Advance the sample index by the prefix length, essentially skipping the prefix
                sample_index += num_prefix_samples

            # Sort resource samples into their respective matrix sections
            resource_samples[:, resource_idx] = signal_samples[sample_index : sample_index + padded_num_subcarriers]

            # Advance sample index by resource length
            sample_index += padded_num_subcarriers

        return resource_samples

    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        # Generate the resource grid of the oversampled OFDM frame
        padded_num_subcarriers = self.frame.num_subcarriers * self.frame.oversampling_factor
        grid = np.zeros((padded_num_subcarriers, self.num_words), dtype=complex)

        # Select the subgrid onto which to project this symbol section's resource configuration
        subgrid_start_idx = int(0.5 * (padded_num_subcarriers - self.num_subcarriers))
        grid[subgrid_start_idx : subgrid_start_idx + self.num_subcarriers, :] = symbols.T

        # Shift in order to suppress the dc component
        # Note that for configurations without any oversampling the DC component will not be suppressed
        if self.frame.dc_suppression:
            dc_index = int(0.5 * padded_num_subcarriers)
            grid[dc_index:, :] = np.roll(grid[dc_index:, :], 1, axis=0)

        # By convention, the length of each time slot is the inverse of the sub-carrier spacing
        resource_signals = ifft(ifftshift(grid, axes=0), axis=0, norm="ortho")

        # Add prefixes and concatenate resources
        signal_samples = self._add_prefix(resource_signals)

        return signal_samples

    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        padded_num_subcarriers = self.frame.num_subcarriers * self.frame.oversampling_factor

        # Remove the cyclic prefixes before transformation into time-domain
        resource_signals = self._remove_prefix(signal)

        # Transform grid back to data symbols
        grid = fftshift(fft(resource_signals, axis=0, norm="ortho"), axes=0)

        # Account for the DC suppression
        if self.frame.dc_suppression:
            dc_index = int(0.5 * padded_num_subcarriers)
            grid[dc_index:, :] = np.roll(grid[dc_index:, :], -1, axis=0)

        # Extract the subgrid relevant to this section
        subgrid_start_idx = int(0.5 * (padded_num_subcarriers - self.num_subcarriers))
        subgrid = grid[subgrid_start_idx : subgrid_start_idx + self.num_subcarriers, :]

        return subgrid.T

    @property
    def resource_mask(self) -> np.ndarray:
        # Initialize the base mask as all false
        mask = np.zeros((len(ElementType), self.num_subcarriers, len(self.pattern)), dtype=bool)

        for word_idx, resource_idx in enumerate(self.pattern):
            resource = self.frame.resources[resource_idx]
            mask[:, : resource.num_subcarriers, word_idx] = resource.mask

        return np.tile(mask, (1, 1, self.num_repetitions))

    @property
    def num_samples(self) -> int:
        num_samples_per_slot = self.frame.num_subcarriers * self.frame.oversampling_factor
        num = len(self.pattern) * num_samples_per_slot

        # Add up the additional samples from cyclic prefixes
        for resource_idx in self.pattern:
            num += int(num_samples_per_slot * self.frame.resources[resource_idx].prefix_ratio)

        # Add up the base samples from each timeslot
        return num * self.num_repetitions


class FrameGuardSection(FrameSection, Serializable):
    yaml_tag = "Guard"
    __duration: float

    def __init__(self, duration: float, num_repetitions: int = 1, frame: Optional[OFDMWaveform] = None) -> None:
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
        if len(symbols) != 0:
            raise ValueError("Guard sections may not hold modulation symbols")

        return np.zeros(self.num_samples, dtype=complex)

    def demodulate(self, baseband_signal: np.ndarray) -> np.ndarray:
        # Guard sections naturally don't encode anything
        return np.empty(0, dtype=complex)


class OFDMWaveform(ConfigurablePilotWaveform, Serializable):
    """Generic Orthogonal-Frequency-Division-Multiplexing with a flexible frame configuration.

    The internally applied FFT size is :meth:`OFDMWaveform.num_subcarriers` times :meth:`WaveformGenerator.oversampling_factor`.

    The following features are supported:
        - The modem can transmit or receive custom-defined frames.
          Frames may contain UL/DL data symbols, null carriers, pilot subcarriers,
          reference signals and guard intervals.
        - SC-FDMA can also be implemented with a precoder.
        - Subcarriers can be modulated with BPSK/QPSK/16-/64-/256-QAM.
        - Cyclic prefixes for interference-free channel estimation and equalization are supported.

    This implementation has currently the following limitations:
        - All subcarriers use the same modulation scheme
    """

    yaml_tag: str = "OFDM"

    __subcarrier_spacing: float
    __num_subcarriers: int
    __pilot_section: PilotSection | None
    dc_suppression: bool
    __resources: List[FrameResource]
    __structure: List[FrameSection]

    @staticmethod
    def _arg_signature() -> Set[str]:
        return {"modulation_order"}

    def __init__(self, subcarrier_spacing: float = 1e3, num_subcarriers: int = 1024, dc_suppression: bool = True, resources: Optional[List[FrameResource]] = None, structure: Optional[List[FrameSection]] = None, **kwargs: Any) -> None:
        """
        Args:

            subcarrier_spacing (float, optional):
                Spacing between individual subcarriers in Hz.
                :math:`1~\\mathrm{kHz}` by default.

            num_subcarriers (int, optional):
                Maximum number of assignable subcarriers.
                Unassigned subcarriers will be assumed to be zero.
                :math:`1024` by default.

            dc_suppression (bool, optional):
                Suppress the direct current component during waveform generation.
                Enabled by default.

            resources (List[FrameResource], optional):
                Frequency-domain resource section configurations.

            structure (List[FrameSection], optional):
                Time-domain frame configuration.

            **kwargs (Any):
                Waveform generator base class initialization parameters.
                Refer to :class:`WaveformGenerator` for details.
        """

        self.subcarrier_spacing = subcarrier_spacing
        self.num_subcarriers = num_subcarriers
        self.dc_suppression = dc_suppression
        self.__resources = [] if resources is None else resources
        self.channel_equalization = OFDMChannelEqualization(self)
        self.channel_estimation = ChannelEstimation[OFDMWaveform]()
        self.__pilot_section = None

        self.__structure = []
        if structure is not None:
            for section in structure:
                self.add_section(section)

        # Initialize  base class
        ConfigurablePilotWaveform.__init__(self, **kwargs)
        self.pilot_symbol_sequence = MappedPilotSymbolSequence(self._mapping)

    @property
    def resources(self) -> List[FrameResource]:
        """OFDM grid resources.

        Returns: List of resources.
        """

        return self.__resources

    @property
    def structure(self) -> List[FrameSection]:
        """OFDM frame configuration in time domain.

        Returns: List of frame elements.
        """

        return self.__structure

    @WaveformGenerator.modulation_order.setter  # type: ignore
    def modulation_order(self, value: int) -> None:
        WaveformGenerator.modulation_order.fset(self, value)  # type: ignore
        self._mapping = PskQamMapping(value)

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

        self.__structure.append(section)
        section.frame = self

    @property
    def pilot_section(self) -> PilotSection | None:
        """Static pilot section transmitted at the beginning of each OFDM frame.

        Required for time-domain synchronization and equalization of carrier frequency offsets.

        Returns:
            FrameSection of the pilot symbols, `None` if no pilot is configured.
        """

        return self.__pilot_section

    @pilot_section.setter
    def pilot_section(self, value: PilotSection | None) -> None:
        if value is None:
            self.__pilot_section = None
            return

        self.__pilot_section = value

        if value.frame is not self:
            value.frame = self

    @property
    def pilot_signal(self) -> Signal:
        if self.pilot_section:
            return Signal(self.pilot_section.modulate(), sampling_rate=self.sampling_rate)

        else:
            return Signal.empty(self.sampling_rate)

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
        """Number of symbols per OFDM frame."""

        num_symbols = 0
        for section in self.structure:
            num_symbols += section.num_symbols + section.num_references

        return num_symbols

    @property
    def words_per_frame(self) -> int:
        """Number of words per OFDM frame."""

        num_words = 0
        for section in self.structure:
            num_words += section.num_words

        return num_words

    @property
    def references_per_frame(self) -> int:
        """Number of reference symbols per OFDM frame."""

        num_symbols = 0
        for section in self.structure:
            num_symbols += section.num_references

        return num_symbols

    @property
    def frame_duration(self) -> float:
        return self.samples_in_frame / self.sampling_rate

    @property
    def samples_in_frame(self) -> int:
        num = 0
        for section in self.structure:
            num += section.num_samples

        if self.pilot_signal:
            num += self.pilot_signal.num_samples

        return num

    def map(self, data_bits: np.ndarray) -> Symbols:
        if len(data_bits) != self.bits_per_frame:
            raise ValueError("Incorrect number of information bits provided for mapping")

        # Map data bits to data symbols
        data_symbols = self._mapping.get_symbols(data_bits)

        # Query reference symbols
        reference_symbols = self.pilot_symbols(self.references_per_frame)

        # Generate the symbol sequence for a full OFDM frame
        symbols = Symbols()

        data_idx = 0
        reference_idx = 0
        for section in self.structure:
            appended_symbols = np.zeros((1, section.num_words, self.num_subcarriers), dtype=complex)

            num_data_symbols = section.num_symbols
            num_reference_symbols = section.num_references

            data = data_symbols[data_idx : data_idx + num_data_symbols]
            reference = reference_symbols[reference_idx : reference_idx + num_reference_symbols]

            appended_symbols[0, :, : section.num_subcarriers] = section.place_symbols(data, reference).T
            symbols.append_symbols(Symbols(appended_symbols))

            data_idx += num_data_symbols
            reference_idx += num_reference_symbols

        return symbols

    def unmap(self, symbols: Symbols) -> np.ndarray:
        raw_symbols = symbols.raw[0, :, :].T
        data_symbols = Symbols()
        block_idx = 0
        for section in self.structure:
            section_data_symbols, _ = section.pick_symbols(raw_symbols[:, block_idx : block_idx + section.num_words])

            data_symbols.append_symbols(section_data_symbols)
            block_idx += section.num_words

        detected_bits = self._mapping.detect_bits(data_symbols.raw.flatten()).astype(int)
        return detected_bits

    def modulate(self, symbols: Symbols) -> Signal:
        # Start the frame with a pilot section, if configured
        if self.pilot_section:
            output_signal = self.pilot_section.modulate()

        else:
            output_signal = np.empty(0, dtype=complex)

        # Abort here if no symbols were provided, returning only a pilot section
        if symbols.num_blocks < 1:
            return Signal(output_signal, self.sampling_rate)

        # Convert symbols
        symbol_blocks = symbols.raw[0, :, :]

        block_idx = 0
        for section in self.structure:
            # Modulate the signal
            section_signal = section.modulate(symbol_blocks[block_idx : block_idx + section.num_words, : section.num_subcarriers])
            output_signal = np.append(output_signal, section_signal)

            block_idx += section.num_words

        signal_model = Signal(output_signal, self.sampling_rate)
        return signal_model

    def demodulate(self, signal: np.ndarray) -> Symbols:
        sample_index = 0

        # If the frame contains a pilot section, skip the respective samples
        if self.pilot_section:
            sample_index += self.pilot_section.num_samples

        symbols = Symbols()
        for section in self.structure:
            appended_symbols = np.zeros((1, section.num_words, self.num_subcarriers), dtype=complex)

            num_samples = section.num_samples

            if section.num_symbols < 1:
                sample_index += num_samples
                continue

            signal_section = signal[sample_index : sample_index + num_samples]

            appended_symbols[0, :, : section.num_subcarriers] = section.demodulate(signal_section)
            symbols.append_symbols(Symbols(appended_symbols))

            sample_index += num_samples

        return symbols

    @property
    def _resource_mask(self) -> np.ndarray:
        """Resource mask of the full OFDM frame.

        Returns: The resource mask.
        """

        resource_mask = np.zeros((len(ElementType), self.num_subcarriers, self.words_per_frame), dtype=bool)

        word_idx = 0
        for section in self.structure:
            num_words = section.num_words
            resource_mask[:, : section.num_subcarriers, word_idx : word_idx + num_words] = section.resource_mask

            word_idx += num_words

        return resource_mask

    @property
    def bandwidth(self) -> float:
        # OFDM bandwidth currently is identical to the number of subcarriers times the subcarrier spacing
        b = self.num_subcarriers * self.subcarrier_spacing
        return b

    @property
    def bits_per_frame(self) -> int:
        num_data_symbols = 0
        for section in self.structure:
            num_data_symbols += section.num_symbols

        return num_data_symbols * self._mapping.bits_per_symbol

    @property
    def bit_energy(self) -> float:
        return 1 / self._mapping.bits_per_symbol  # ToDo: Check validity

    @property
    def symbol_energy(self) -> float:
        return 1  # ToDo: Check validity

    @property
    def power(self) -> float:
        return 1 / self.oversampling_factor

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


class PilotSection(FrameSection, Serializable):
    """Pilot symbol section within an OFDM frame."""

    yaml_tag = "OFDM-Pilot"
    """YAML serialization tag"""

    __pilot_elements: Optional[Symbols]
    __cached_num_subcarriers: int
    __cached_oversampling_factor: int
    __cached_pilot: Optional[np.ndarray]

    def __init__(self, pilot_elements: Optional[Symbols] = None, frame: Optional[OFDMWaveform] = None) -> None:
        """
        Args:

            pilot_elements (Optional[Symbols], optional):
                Symbols with which the subcarriers within the pilot will be modulated.
                By default, a pseudo-random sequence from the frame mapping will be generated.

            frame (Optional[WaveformGeneratorOfdm], optional):
                The frame configuration this pilot section belongs to.
        """

        self.__pilot_elements = pilot_elements
        self.__cached_num_subcarriers = -1
        self.__cached_oversampling_factor = -1
        self.__cached_pilot = None

        FrameSection.__init__(self, num_repetitions=1, frame=frame)

    @property
    def num_samples(self) -> int:
        return self.frame.num_subcarriers * self.frame.oversampling_factor

    @property
    def pilot_elements(self) -> Optional[Symbols]:
        """Symbols with which the subcarriers within the pilot will be modulated.

        Returns:

            A stream of symbols. `None`, if no subsymbols where specified.

        Raises:

            ValueError: If the configured symbols contains multiple streams.
        """

        return self.__pilot_elements

    @pilot_elements.setter
    def pilot_elements(self, value: Optional[Symbols]) -> None:
        if value is None:
            self.__pilot_elements = None
            return

        if value.num_streams != 1:
            raise ValueError("Subsymbol pilot configuration may only contain a single stream")

        if value.num_symbols < 1:
            raise ValueError("Subsymbol pilot configuration must contain at least one symbol")

        # Reset the cached pilot, since the subsymbols have changed
        self.__cached_pilot = None

        self.__pilot_elements = value

    def _pilot_sequence(self, num_symbols: int = None) -> Symbols:
        """Generate a new sequence of pilot elements.

        Args:

            num_symbols (int, optional):
                The required number of symbols.
                By default, a symbol for each subcarrier is generated.

        Returns:

            A sequence of symbols.
        """

        num_symbols = self.frame.num_subcarriers if num_symbols is None else num_symbols

        # Generate a pseudo-random symbol stream if no subsymbols are specified
        if self.__pilot_elements is None:
            rng = np.random.default_rng(50)
            num_bits = num_symbols * self.frame._mapping.bits_per_symbol
            subsymbols = self.frame._mapping.get_symbols(rng.integers(0, 2, num_bits))[None, None, :]

        else:
            num_repetitions = int(ceil(num_symbols / self.__pilot_elements.num_symbols))
            subsymbols = np.tile(self.__pilot_elements.raw, (1, 1, num_repetitions))

        return Symbols(subsymbols[:, :, :num_symbols])

    def modulate(self, _: Any | None = None) -> np.ndarray:
        # Return the cached pilot signal if available and the relevant frame parameters haven't changed
        if self.__cached_pilot is not None and self.__cached_num_subcarriers == self.frame.num_subcarriers and self.__cached_oversampling_factor == self.frame.oversampling_factor:
            return self.__cached_pilot

        pilot = self._pilot()

        # Cache the pilot
        self.__cached_pilot = pilot
        self.__cached_num_subcarriers = self.frame.num_subcarriers
        self.__cached_oversampling_factor = self.frame.oversampling_factor

        return pilot

    def demodulate(self, _: np.ndarray) -> np.ndarray:
        return np.empty(0, dtype=complex)

    def _pilot(self) -> np.ndarray:
        """Generate the samples for a pilot section in time domain.

        Returns:

            Complex base-band pilot section samples.
        """

        # Generate the resource grid of the oversampled OFDM frame
        padded_num_subcarriers = self.frame.num_subcarriers * self.frame.oversampling_factor
        grid = np.zeros(padded_num_subcarriers, dtype=complex)

        # Select the subgrid onto which to project this symbol section's resource configuration
        subgrid_start_idx = int(0.5 * (padded_num_subcarriers - self.frame.num_subcarriers))

        # Set grid symbols
        grid[subgrid_start_idx : subgrid_start_idx + self.frame.num_subcarriers] = self._pilot_sequence().raw.flatten()

        # Shift in order to suppress the dc component
        # Note that for configurations without any oversampling the DC component will not be suppressed
        if self.frame.dc_suppression:
            dc_index = int(0.5 * padded_num_subcarriers)
            grid[dc_index:] = np.roll(grid[dc_index:], 1)

        # By convention, the length of each time slot is the inverse of the sub-carrier spacing
        pilot = ifft(ifftshift(grid), norm="ortho")

        return pilot

    @classmethod
    def to_yaml(cls: Type[PilotSection], representer: SafeRepresenter, node: PilotSection) -> MappingNode:
        """Serialize a serializable object to YAML.

        Args:

            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (PilotSection):
                The channel instance to be serialized.

        Returns: The serialized YAML node.
        """

        additional_fields = {}

        if node.pilot_elements:
            additional_fields["pilot_elements"] = node.pilot_elements.raw

        return node._mapping_serialization_wrapper(representer, blacklist={"pilot_elements"}, additional_fields=additional_fields)

    @classmethod
    def from_yaml(cls: Type[PilotSection], constructor: SafeConstructor, node: Node) -> PilotSection:
        """Recall a new serializable class instance from YAML.

        Args:

            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `PilotSection` serialization.

        Returns: The de-serialized object.
        """

        state: dict = constructor.construct_mapping(node, deep=True)
        pilot_elements = state.pop("pilot_elements", None)

        if pilot_elements is not None:
            pilot_elements = Symbols(pilot_elements)
            state["pilot_elements"] = pilot_elements

        return cls.InitializationWrapper(state)


class SchmidlCoxPilotSection(PilotSection):
    """Pilot Symbol Section of the Schmidl Cox Algorithm.

    Refer to :footcite:t:`1997:schmidl` for a detailed description.
    """

    yaml_tag = "SchmidlCoxPilot"
    """YAML serialization tag"""

    @property
    def num_samples(self) -> int:
        return self.frame.num_subcarriers * self.frame.oversampling_factor

    def _pilot(self) -> np.ndarray:
        samples_per_symbol = self.frame.num_subcarriers * self.frame.oversampling_factor
        pilot_sequence = self._pilot_sequence(ceil(0.5 * self.frame.num_subcarriers)).raw.flatten()

        pilot_frequencies = np.zeros(samples_per_symbol, dtype=complex)

        subgrid_start_idx = ceil(0.5 * (samples_per_symbol - self.frame.num_subcarriers))
        pilot_frequencies[subgrid_start_idx : subgrid_start_idx + self.frame.num_subcarriers : 2] = pilot_sequence

        pilot_samples = ifft(ifftshift(pilot_frequencies), norm="ortho", n=samples_per_symbol)
        return pilot_samples

    def demodulate(self, _: np.ndarray) -> np.ndarray:
        return np.empty(0, dtype=complex)


class OFDMSynchronization(Synchronization[OFDMWaveform]):
    """Synchronization Routine for OFDM Waveforms."""

    ...  # pragma: no cover


class OFDMCorrelationSynchronization(CorrelationSynchronization[OFDMWaveform]):
    """Correlation-Based Pilot Detection and Synchronization for OFDM Waveforms."""

    yaml_tag = "OFDM-PilotCorrelation"


class SchmidlCoxSynchronization(OFDMSynchronization):
    """Schmidl-Cox Algorithm for OFDM Waveform Time Synchronization and Carrier Frequency Offset Equzalization.

    Applying the synchronization routine requires the respective waveform to have a :class:`.SchmidlCoxPilotSection` pilot
    symbol section configured.

    Refer to :footcite:t:`1997:schmidl` for a detailed description.
    """

    yaml_tag = "SchmidlCox"
    """YAML serialization tag"""

    def synchronize(self, signal: np.ndarray) -> List[int]:
        symbol_length = self.waveform_generator.oversampling_factor * self.waveform_generator.num_subcarriers

        # Abort if the supplied signal is shorter than one symbol length
        if signal.shape[-1] < symbol_length:
            return []

        half_symbol_length = int(0.5 * symbol_length)

        num_delay_candidates = 2 + signal.shape[-1] - symbol_length
        delay_powers = np.empty(num_delay_candidates, dtype=float)
        delay_powers[0] = 0.0  # In order to be able to detect a peak on the first sample
        for d in range(0, num_delay_candidates - 1):
            delay_powers[1 + d] = np.sum(abs(np.sum(signal[:, d : d + half_symbol_length].conj() * signal[:, d + half_symbol_length : d + 2 * half_symbol_length], axis=1)))

        num_samples = self.waveform_generator.samples_in_frame
        min_height = 0.75 * np.max(delay_powers)
        peaks, _ = find_peaks(delay_powers, distance=int(0.9 * num_samples), height=min_height)
        frame_indices = peaks - 1  # Correct for the first delay bin being prepended

        return frame_indices


class ReferencePosition(SerializableEnum):
    """Applied channel estimation algorithm after reception."""

    IDEAL = 0
    IDEAL_PREAMBLE = 1
    IDEAL_MIDAMBLE = 2
    IDEAL_POSTAMBLE = 3


class OFDMIdealChannelEstimation(IdealChannelEstimation[OFDMWaveform], Serializable):
    """Ideal channel state estimation for OFDM waveforms."""

    yaml_tag = "OFDM-Ideal"
    serialized_attributes = {"reference_position"}

    reference_position: ReferencePosition
    """Assumed position of the reference symbol within the frame."""

    def __init__(self, reference_position: ReferencePosition = ReferencePosition.IDEAL, *args, **kwargs) -> None:
        """
        Args:

            reference_position (ReferencPosition, optional):
                Assumed location of the reference symbols within the ofdm frame.
        """

        self.reference_position = reference_position
        IdealChannelEstimation.__init__(self, *args, **kwargs)  # type: ignore

    def estimate_channel(self, symbols: Symbols) -> Tuple[StatedSymbols, ChannelStateInformation]:
        csi = self._csi().to_frequency_selectivity(self.waveform_generator.num_subcarriers)
        return StatedSymbols(symbols.raw, csi.state[:, :, : symbols.num_blocks, :]), csi


class OFDMLeastSquaresChannelEstimation(ChannelEstimation[OFDMWaveform], Serializable):
    """Least-Squares channel estimation for OFDM waveforms."""

    yaml_tag = "OFDM-LS"
    """YAML serializtion tag"""

    def estimate_channel(self, symbols: Symbols) -> Tuple[StatedSymbols, ChannelStateInformation]:
        if symbols.num_streams != 1:
            raise NotImplementedError("Least-Squares channel estimation is only implemented for SISO links")

        resource_mask = self.waveform_generator._resource_mask

        propagated_references = symbols.raw[0, ::].T[resource_mask[ElementType.REFERENCE.value, ::]]
        reference_symbols = self.waveform_generator.pilot_symbols(len(propagated_references))
        reference_channel_estimation = propagated_references / reference_symbols

        channel_estimation = np.zeros(((1, 1, symbols.num_symbols, symbols.num_blocks)), dtype=complex)
        channel_estimation[0, 0, resource_mask[ElementType.REFERENCE.value, ::]] = reference_channel_estimation
        channel_estimation = channel_estimation.transpose((0, 1, 3, 2))

        interpolation_stems = np.where(resource_mask[ElementType.REFERENCE.value, ::])
        holes = np.where(np.invert(resource_mask[ElementType.REFERENCE.value, ::]))

        # ToDo: Check with group what to do about missing values outside the convex hull
        interpolated_holes = griddata(interpolation_stems, reference_channel_estimation, holes, method="nearest")
        channel_estimation[0, 0, holes[1], holes[0]] = interpolated_holes
        return StatedSymbols(symbols.raw, channel_estimation), ChannelStateInformation(ChannelStateFormat.FREQUENCY_SELECTIVITY, channel_estimation)


class OFDMChannelEqualization(ChannelEqualization[OFDMWaveform], ABC):
    """Channel estimation for OFDM waveforms."""

    yaml_tag = "OFDM-NoEqualization"

    def __init__(self, waveform_generator: Optional[OFDMWaveform] = None) -> None:
        """
        Args:

            waveform_generator (WaveformGenerator, optional):
                The waveform generator this equalization routine is attached to.
        """

        ChannelEqualization.__init__(self, waveform_generator)


class OFDMZeroForcingChannelEqualization(ZeroForcingChannelEqualization[OFDMWaveform]):
    """Zero-Forcing channel equalization for OFDM waveforms."""

    yaml_tag = "OFDM-ZF"
    """YAML serialization tag"""
