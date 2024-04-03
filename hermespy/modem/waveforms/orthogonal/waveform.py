# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, List, Sequence, Type, TypeVar, Union
import matplotlib.pyplot as plt

import numpy as np
from ruamel.yaml import MappingNode, Node, SafeConstructor, SafeRepresenter

from hermespy.core import Serializable, SerializableEnum, Signal, VisualizableAttribute
from hermespy.core.visualize import ImageVisualization, VAT
from ...symbols import Symbols, StatedSymbols
from ...tools import PskQamMapping
from ...waveform import (
    CommunicationWaveform,
    ConfigurablePilotWaveform,
    PilotSymbolSequence,
    MappedPilotSymbolSequence,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ElementType(SerializableEnum):
    """Type of resource element."""

    REFERENCE = 0
    """Reference element within the resource grid"""

    DATA = 1
    """Data element within the resource grid"""

    NULL = 2
    """Empty element within the resource grid"""


class PrefixType(SerializableEnum):
    """Type of prefix applied to the grid resource in time-domain."""

    CYCLIC = 0
    """Cyclic prefix repeating the resource waveform in time-domain"""

    ZEROPAD = 1
    """Prefix zero-padding the prefix in time-domain"""

    NONE = 2
    """No prefix applied"""


class GridElement(Serializable):
    yaml_tag = "Element"
    serialized_attributes = {"type", "repetitions"}

    type: ElementType
    repetitions: int = 1

    def __init__(self, type: str | ElementType, repetitions: int = 1) -> None:
        self.type = ElementType[type] if isinstance(type, str) else type
        self.repetitions = repetitions


class ReferencePosition(SerializableEnum):
    """Applied channel estimation algorithm after reception."""

    IDEAL = 0
    IDEAL_PREAMBLE = 1
    IDEAL_MIDAMBLE = 2
    IDEAL_POSTAMBLE = 3


class GridResource(Serializable):
    """Configures one sub-section of a resource grid in both dimensions."""

    yaml_tag = "Resource"
    serialized_attributes = {"prefix_type", "elements"}

    __repetitions: int
    __prefix_ratio: float

    prefix_type: PrefixType
    """Prefix type of the frame resource"""

    elements: List[GridElement]
    """Individual resource elements"""

    def __init__(
        self,
        repetitions: int = 1,
        prefix_type: Union[PrefixType, str] = PrefixType.CYCLIC,
        prefix_ratio: float = 0.0,
        elements: List[GridElement] | None = None,
    ) -> None:
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
            raise ValueError(f"Cyclic prefix ratio must be between zero and one, not {ratio}")

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


OWT = TypeVar("OWT", bound="OrthogonalWaveform")
"""Type variable for orthogonal waveform types."""


class GridSection(Generic[OWT], ABC):
    """Description of a part of a grid's time domain."""

    __wave: OWT | None
    __num_repetitions: int
    __sample_offset: int

    def __init__(
        self, num_repetitions: int = 1, sample_offset: int = 0, wave: OWT | None = None
    ) -> None:
        """
        Args:

            num_repetitions (int, optional): Number of times this section is repeated in time-domain.
            sample_offset (int, optional): Offset in samples to the start of the section.
            wave (OWT, optional): Waveform this section is associated with. Defaults to None.
        """

        # Initialize class attributes
        self.wave = wave
        self.sample_offset = sample_offset
        self.num_repetitions = num_repetitions

    @property
    def wave(self) -> OWT | None:
        """Waveform this section is associated with."""

        return self.__wave

    @wave.setter
    def wave(self, value: OWT | None) -> None:
        self.__wave = value

    @property
    def sample_offset(self) -> int:
        """Offset in samples to the start of the section.

        This can be used to explot cyclic prefixes and suffixes in order to be more robust
        against timing offsets.
        """

        return self.__sample_offset

    @sample_offset.setter
    def sample_offset(self, value: int) -> None:
        self.__sample_offset = value

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
        """Place this section's symbols into the resource grid.

        Args:

            data_symbols (np.ndarray): Data symbols to be placed. Numpy vector of size `num_symbols`.
            reference_symbols (np.ndarray): Reference symbols to be placed. Numpy vector of size `num_references`.

        Returns: Two dimensional numpy array of size `num_words`x`num_subcarriers`.
        """

        # Collect resource masks
        mask = self.resource_mask

        grid = np.zeros((self.num_words, self.num_subcarriers), dtype=np.complex_)
        grid[mask[ElementType.REFERENCE.value, ::]] = reference_symbols
        grid[mask[ElementType.DATA.value, ::]] = data_symbols

        return grid

    def pick_symbols(self, grid: np.ndarray) -> np.ndarray:
        """Pick this section's data symbols from the resource grid.

        Args:

            grid (np.ndarray): Resource grid. Two dimensional numpy array of size `num_words`x`num_subcarriers`.

        Returns: Data symbols. Numpy vector of size `num_symbols`.
        """

        # Collect resource masks
        mask = self.resource_mask

        # Select correct subgrid
        subgrid_selector = tuple(slice(None) for _ in range(grid.ndim - 1)) + (
            slice(0, self.num_subcarriers, 1),
        )
        subgrid = grid[subgrid_selector]

        # Pick symbols
        # reference_symbols = subgrid[mask[ElementType.REFERENCE.value]]
        selector = tuple(slice(None) for _ in range(subgrid.ndim - 2)) + (
            mask[ElementType.DATA.value],
        )
        picked_symbols = subgrid[selector]
        return picked_symbols

    @abstractmethod
    def place_samples(self, signal: np.ndarray) -> np.ndarray:
        """Place this section's samples into the time-domain signal.

        Args:

            signal (np.ndarray): Time-domain signal to be placed. Numpy vector of size `num_samples`.

        Returns: Time-domain signal with the section's samples placed.
        """
        ...  # pragma: no cover

    @abstractmethod
    def pick_samples(self, signal: np.ndarray) -> np.ndarray:
        """Pick this section's samples from the time-domain signal.

        Args:

            signal (np.ndarray): Time-domain signal to be picked from. Numpy vector of size `num_samples`.

        Returns: Time-domain signal with the section's samples picked.
        """
        ...  # pragma: no cover


class SymbolSection(GridSection["OrthogonalWaveform"], Serializable):
    yaml_tag: str = "Symbol"
    serialized_attributes = {"pattern"}

    pattern: List[int]

    def __init__(
        self,
        num_repetitions: int = 1,
        pattern: List[int] | None = None,
        sample_offset: int = 0,
        wave: OrthogonalWaveform | None = None,
    ) -> None:
        """
        Args:
            num_repetitions (int, optional): Number of times this section is repeated in time-domain.
            pattern (List[int], optional): Resource pattern within this symbol section.
            sample_offset (int, optional): Offset in samples to the start of the section.
            frame (OrthogonalWaveform | None, optional): _description_. Defaults to None.
        """

        # Initialize bae class
        GridSection.__init__(self, num_repetitions, sample_offset, wave)

        # Initialize class attributes
        self.pattern = pattern if pattern is not None else []

    @property
    def num_symbols(self) -> int:
        num = 0
        for resource_idx in self.pattern:
            resource = self.wave.grid_resources[resource_idx]
            num += resource.num_symbols

        return self.num_repetitions * num

    @property
    def num_references(self) -> int:
        num = 0
        for resource_idx in self.pattern:
            resource = self.wave.grid_resources[resource_idx]
            num += resource.num_references

        return self.num_repetitions * num

    @property
    def num_words(self) -> int:
        return self.num_repetitions * len(self.pattern)

    @property
    def num_subcarriers(self) -> int:
        num = 0

        for resource_idx in set(self.pattern):
            num = max(num, self.wave.grid_resources[resource_idx].num_subcarriers)

        return num

    @property
    def _padded_num_subcarriers(self) -> int:
        """Number of subcarriers required to represent this section in time-domain."""

        return self.wave.num_subcarriers * self.wave.oversampling_factor

    def place_samples(self, samples: np.ndarray) -> np.ndarray:
        placed_samples = np.empty(self.num_samples, dtype=np.complex_)
        sample_idx = 0
        resource_idx: int
        resource_samples: np.ndarray
        for resource_idx, resource_samples in enumerate(samples):
            # Infer pattern index
            pattern_idx = resource_idx % len(self.pattern)

            # Extract prefix parameters from configuration
            prefix_ratio = self.wave.grid_resources[self.pattern[pattern_idx]].prefix_ratio
            prefix_type = self.wave.grid_resources[self.pattern[pattern_idx]].prefix_type

            num_prefix_samples = int(self._padded_num_subcarriers * prefix_ratio)

            # Only add a prefix if required
            if num_prefix_samples > 0 and prefix_type != PrefixType.NONE:
                # Cyclic prefix
                if prefix_type == PrefixType.CYCLIC:
                    placed_samples[sample_idx : sample_idx + num_prefix_samples] = resource_samples[
                        -num_prefix_samples:
                    ]

                # Zero padding
                elif prefix_type == PrefixType.ZEROPAD:
                    placed_samples[sample_idx : sample_idx + num_prefix_samples] = np.zeros(
                        num_prefix_samples, dtype=np.complex_
                    )

                # Raise exception for unsupproted prefix types
                else:
                    raise RuntimeError("Unsupported prefix type configured")

                # Advance the sample index by the prefix length
                sample_idx += num_prefix_samples

            # Append base resource waveform after prefix
            placed_samples[sample_idx : sample_idx + resource_samples.size] = resource_samples
            sample_idx += resource_samples.size

        return placed_samples

    def pick_samples(self, samples: np.ndarray) -> np.ndarray:
        sample_index = 0
        num_symbols = len(self.pattern) * self.num_repetitions
        resource_samples = np.empty(
            (*samples.shape[:-1], num_symbols, self._padded_num_subcarriers), dtype=complex
        )
        prefix_slice = [slice(None)] * (resource_samples.ndim - 2)

        for resource_idx in range(num_symbols):
            # Infer pattern index
            pattern_idx = resource_idx % len(self.pattern)

            # Extract prefix parameters from configuration
            resource = self.wave.grid_resources[self.pattern[pattern_idx]]
            prefix_ratio = resource.prefix_ratio
            prefix_type = resource.prefix_type

            num_prefix_samples = int(self._padded_num_subcarriers * prefix_ratio)

            # Only add a prefix if required
            if num_prefix_samples > 0 and prefix_type != PrefixType.NONE:
                # Advance the sample index by the prefix length, essentially skipping the prefix
                sample_index += num_prefix_samples

            # Sort resource samples into their respective matrix sections
            resource_slicing = (*prefix_slice, resource_idx, slice(None))
            signal_slicing = (
                *prefix_slice,
                slice(
                    sample_index - self.sample_offset,
                    sample_index + self._padded_num_subcarriers - self.sample_offset,
                ),
            )
            resource_samples[resource_slicing] = samples[signal_slicing]

            # Advance sample index by resource length
            sample_index += self._padded_num_subcarriers

        return resource_samples

    @property
    def resource_mask(self) -> np.ndarray:
        # Initialize the base mask as all false
        mask = np.zeros((len(ElementType), len(self.pattern), self.num_subcarriers), dtype=bool)

        for word_idx, resource_idx in enumerate(self.pattern):
            resource = self.wave.grid_resources[resource_idx]
            mask[:, word_idx, : resource.num_subcarriers] = resource.mask

        return np.tile(mask, (1, self.num_repetitions, 1))

    @property
    def num_samples(self) -> int:
        num_samples_per_slot = self.wave.num_subcarriers * self.wave.oversampling_factor
        num = len(self.pattern) * num_samples_per_slot

        # Add up the additional samples from cyclic prefixes
        for resource_idx in self.pattern:
            num += int(num_samples_per_slot * self.wave.grid_resources[resource_idx].prefix_ratio)

        # Add up the base samples from each timeslot
        return num * self.num_repetitions


class GuardSection(GridSection["OrthogonalWaveform"], Serializable):
    yaml_tag = "Guard"
    __duration: float

    def __init__(
        self, duration: float, num_repetitions: int = 1, frame: OrthogonalWaveform | None = None
    ) -> None:
        GridSection.__init__(self, num_repetitions=num_repetitions, wave=frame)
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
        return int(self.num_repetitions * self.__duration * self.wave.sampling_rate)

    def place_samples(self, signal: np.ndarray) -> np.ndarray:
        return np.zeros(self.num_samples, dtype=np.complex_)

    def pick_samples(self, signal: np.ndarray) -> np.ndarray:
        return np.empty(
            (0, self.wave.num_subcarriers * self.wave.oversampling_factor), dtype=np.complex_
        )


class PilotSection(Generic[OWT], GridSection[OWT], Serializable):
    """Pilot symbol section within an resource grid."""

    yaml_tag = "Pilot"
    """YAML serialization tag"""

    __pilot_elements: Symbols | None
    __cached_num_subcarriers: int
    __cached_oversampling_factor: int
    __cached_pilot: np.ndarray | None

    def __init__(self, pilot_elements: Symbols | None = None, wave: OWT | None = None) -> None:
        """
        Args:

            pilot_elements (Symbols, optional):
                Symbols with which the subcarriers within the pilot will be modulated.
                By default, a pseudo-random sequence from the frame mapping will be generated.

            wave (OWT, optional):
                The waveform configuration this pilot section is associated with.
        """

        # Initialize base class
        GridSection.__init__(self, 1, 0, wave=wave)

        # Initialize class attributes
        self.__pilot_elements = pilot_elements
        self.__cached_num_subcarriers = -1
        self.__cached_oversampling_factor = -1
        self.__cached_pilot = None

    @GridSection.num_repetitions.setter  # type: ignore
    def num_repetitions(self, value: int) -> None:
        if value != 1:
            raise ValueError("Pilot sections may not be repeated")

        GridSection.num_repetitions.fset(self, value)  # type: ignore

    @GridSection.sample_offset.setter  # type: ignore
    def sample_offset(self, value: int) -> None:
        if value != 0:
            raise ValueError("Pilot sections may not have a sample offset")

        GridSection.sample_offset.fset(self, value)  # type: ignore

    @property
    def num_samples(self) -> int:
        return self.wave.num_subcarriers * self.wave.oversampling_factor

    @property
    def num_symbols(self) -> int:
        return 0

    @property
    def num_words(self) -> int:
        return 1

    @property
    def num_subcarriers(self) -> int:
        return self.wave.num_subcarriers if self.wave else 0

    @property
    def num_references(self) -> int:
        if self.__pilot_elements or self.wave is None:
            return 0
        return self.wave.num_subcarriers

    @property
    def resource_mask(self) -> np.ndarray:
        mask = np.zeros(
            (len(ElementType), 1, self.wave.num_subcarriers if self.wave else 0), dtype=bool
        )
        mask[ElementType.REFERENCE.value, 0, ::] = True
        return mask

    @property
    def pilot_elements(self) -> Symbols | None:
        """Symbols with which the orthogonal subcarriers within the pilot will be modulated.

        Returns:
            A stream of symbols.
            `None`, if no pilot symbols were specified.

        Raises:

            ValueError: If the configured symbols contains multiple streams.
        """

        return self.__pilot_elements

    @pilot_elements.setter
    def pilot_elements(self, value: Symbols | None) -> None:
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

    def _pilot_sequence(self, num_symbols: int = None) -> np.ndarray:
        """Generate a new sequence of pilot elements.

        Args:

            num_symbols (int, optional):
                The required number of symbols.
                By default, a symbol for each subcarrier is generated.

        Returns:

            A sequence of symbols.
        """

        num_symbols = self.wave.num_subcarriers if num_symbols is None else num_symbols

        # Generate a pseudo-random symbol stream if no subsymbols are specified
        if self.__pilot_elements is None:
            rng = np.random.default_rng(50)
            num_bits = num_symbols * self.wave.mapping.bits_per_symbol
            subsymbols = self.wave.mapping.get_symbols(rng.integers(0, 2, num_bits))

        else:
            num_repetitions = int(np.ceil(num_symbols / self.__pilot_elements.num_symbols))
            subsymbols = np.tile(self.__pilot_elements.raw.flat, (num_repetitions))

        return subsymbols[:num_symbols]

    def place_symbols(self, data_symbols: np.ndarray, reference_symbols: np.ndarray) -> np.ndarray:
        reference_symbols = self._pilot_sequence(self.wave.num_subcarriers)
        return GridSection.place_symbols(self, data_symbols, reference_symbols)

    def place_samples(self, signal: np.ndarray) -> np.ndarray:
        # Just a stub, since the pilot section does not consider any prefixing
        return signal

    def pick_samples(self, signal: np.ndarray) -> np.ndarray:
        # Just a stub, since the pilot section does not consider any prefixing
        return signal

    def generate(self) -> np.ndarray:
        if self.wave is None:
            raise RuntimeError("Pilot section must be associated with a waveform")

        """Generate the pilot section in time domain."""
        # Return the cached pilot signal if available and the relevant frame parameters haven't changed
        if (
            self.__cached_pilot is not None
            and self.__cached_num_subcarriers == self.wave.num_subcarriers
            and self.__cached_oversampling_factor == self.wave.oversampling_factor
        ):
            return self.__cached_pilot

        pilot_symbols = self._pilot_sequence(self.wave.num_subcarriers)
        pilot = self.wave._forward_transformation(pilot_symbols[np.newaxis, :])

        # Cache the pilot
        self.__cached_pilot = pilot
        self.__cached_num_subcarriers = self.wave.num_subcarriers
        self.__cached_oversampling_factor = self.wave.oversampling_factor

        return pilot

    @classmethod
    def to_yaml(
        cls: Type[PilotSection], representer: SafeRepresenter, node: PilotSection
    ) -> MappingNode:
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

        return node._mapping_serialization_wrapper(
            representer, blacklist={"pilot_elements"}, additional_fields=additional_fields
        )

    @classmethod
    def from_yaml(
        cls: Type[PilotSection], constructor: SafeConstructor, node: Node
    ) -> PilotSection:
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


class GridVisualization(VisualizableAttribute[ImageVisualization]):
    """Plot the grid structure of an orthogonal waveform."""

    def __init__(self, wave: OrthogonalWaveform) -> None:
        """
        Args:

            wave (OrthogonalWaveform): Waveform this plot is associated with.
        """

        # Initialize base class
        super().__init__()

        # Initialize class attributes
        self.__wave = wave

    @property
    def title(self) -> str:
        return "Resource Grid"

    def __generate_image(self) -> np.ndarray:
        mask = self.__wave.resource_mask

        grid = np.zeros(mask.shape[1:], dtype=np.int_)
        grid[mask[ElementType.NULL.value]] = 1
        grid[mask[ElementType.REFERENCE.value]] = 2
        grid[mask[ElementType.DATA.value]] = 3

        return grid.T

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> ImageVisualization:
        ax: plt.Axes = axes.flat[0]
        image = ax.imshow(self.__generate_image(), cmap="viridis", aspect="auto")
        ax.set_ylabel("Resource")
        ax.set_xlabel("Time")

        return ImageVisualization(figure, axes, image)

    def _update_visualization(self, visualization: ImageVisualization, **kwargs) -> None:
        visualization.image.set_data(self.__generate_image())


class OrthogonalWaveform(ConfigurablePilotWaveform, ABC):
    """Base class for wavforms with orthogonal subcarrier modulation."""

    __mapping: PskQamMapping
    __num_subcarriers: int
    __grid_resources: Sequence[GridResource]
    __grid_structure: Sequence[GridSection]
    __grid_visualization: GridVisualization

    def __init__(
        self,
        num_subcarriers: int,
        grid_resources: Sequence[GridResource],
        grid_structure: Sequence[GridSection],
        pilot_section: PilotSection | None = None,
        pilot_sequence: PilotSymbolSequence | None = None,
        repeat_pilot_sequence: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:

            num_subcarriers (int): Number of available orthogonal subcarriers per symbol.
            grid_resources (Sequence[GridResource]): Grid resources available for modulation.
            grid_structure (Sequence[GridSection]): Grid structure of the time-domain.
            pilot_section (PilotSection, optional): Pilot section transmitted at the beginning of each frame.
            pilot_sequence (PilotSymbolSequence, optional): Sequence of pilot / reference symbols.
            repeat_pilot_sequence (bool, optional): Repeat the pilot sequence if it is shorter than the frame.
        """

        # Initialize base class
        ConfigurablePilotWaveform.__init__(self, pilot_sequence, repeat_pilot_sequence, **kwargs)

        # Initialize the class attributes
        self.num_subcarriers = num_subcarriers
        self.__grid_resources = grid_resources
        self.__grid_structure = grid_structure
        self.__mapping = PskQamMapping(self.modulation_order)
        self.__grid_visualization = GridVisualization(self)
        self.pilot_section = pilot_section

        if not pilot_sequence:
            self.pilot_symbol_sequence = MappedPilotSymbolSequence(self.mapping)

        for section in self.__grid_structure:
            section.wave = self

    @abstractmethod
    def _forward_transformation(self, symbol_grid: np.ndarray) -> np.ndarray:
        """Forward transformation of the orthogonal symbol grid into the time-domain.

        Args:

            symbol_grid (np.ndarray): The grid of modulated symbols to be transformed.

        Returns: The time-domain signal grid.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _backward_transformation(self, signal_grid: np.ndarray) -> np.ndarray:
        """Backward transformation of the time-domain signal grid into the orthogonal symbol grid.

        Args:

            signal_grid (np.ndarray): The time-domain signal grid to be transformed.

        Returns: The grid of modulated symbols.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _correct_sample_offset(self, symbol_subgrid: np.ndarray, sample_offset: int) -> np.ndarray:
        """Correct the sample offset of a symbol subgrid.

        Args:

            symbol_subgrid (np.ndarray): The symbol subgrid to be corrected.
            sample_offset (int): The sample offset to be corrected.

        Returns: The corrected symbol subgrid.
        """
        ...  # pragma: no cover

    @property
    def plot_grid(self) -> GridVisualization:
        """Visualize the resource grid."""

        return self.__grid_visualization

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

        if value.wave is not self:
            value.wave = self

    @property
    def pilot_signal(self) -> Signal:
        if self.pilot_section:
            return Signal(self.pilot_section.generate(), sampling_rate=self.sampling_rate)

        else:
            return Signal.empty(self.sampling_rate)

    @CommunicationWaveform.modulation_order.setter  # type: ignore
    def modulation_order(self, value: int) -> None:
        CommunicationWaveform.modulation_order.fset(self, value)  # type: ignore
        self.__mapping = PskQamMapping(value)

    @property
    def bit_energy(self) -> float:
        return 1 / self.mapping.bits_per_symbol  # ToDo: Check validity

    @property
    def symbol_energy(self) -> float:
        return 1  # ToDo: Check validity

    @property
    def symbol_duration(self) -> float:
        return 1 / self.bandwidth

    @property
    def power(self) -> float:
        return 1.0

    @property
    def mapping(self) -> PskQamMapping:
        """Constellation mapping used to translate bit sequences into complex symbols."""

        return self.__mapping

    @property
    def num_subcarriers(self) -> int:
        """Number of available orthogonal subcarriers.

        Raises:

            ValueError: If smaller than one.
        """

        return self.__num_subcarriers

    @num_subcarriers.setter
    def num_subcarriers(self, value: int) -> None:
        if value < 1:
            raise ValueError("Number of subcarriers must be greater or equal to one")

        self.__num_subcarriers = value

    @property
    def grid_resources(self) -> Sequence[GridResource]:
        """Available resources within the time-subcarrier grid."""

        return self.__grid_resources

    @property
    def grid_structure(self) -> Sequence[GridSection]:
        """Structure of the time-subcarrier grid."""

        return self.__grid_structure

    @property
    def symbols_per_frame(self) -> int:
        """Number of modulated symbols within each transmitted communication frame.

        This includes both data and reference symbols.
        """

        num_symbols = 0
        for section in self.grid_structure:
            num_symbols += section.num_symbols + section.num_references

        return num_symbols

    @property
    def num_data_symbols(self) -> int:
        num_symbols = 0
        for section in self.grid_structure:
            num_symbols += section.num_symbols

        return num_symbols

    @property
    def words_per_frame(self) -> int:
        """Number of words per communication frame."""

        num_words = 0
        for section in self.grid_structure:
            num_words += section.num_words

        return num_words

    @property
    def references_per_frame(self) -> int:
        """Number of reference symbols per communication frame."""

        num_symbols = 0
        for section in self.grid_structure:
            num_symbols += section.num_references

        return num_symbols

    @property
    def samples_per_frame(self) -> int:
        num = 0
        for section in self.grid_structure:
            num += section.num_samples

        if self.pilot_section:
            num += self.pilot_section.num_samples

        return num

    @property
    def resource_mask(self) -> np.ndarray:
        """Boolean resource mask of the full OFDM frame."""

        resource_mask = np.zeros(
            (len(ElementType), self.words_per_frame, self.num_subcarriers), dtype=bool
        )

        word_idx = 0
        for section in self.grid_structure:
            num_words = section.num_words
            resource_mask[
                :, word_idx : word_idx + num_words, : section.num_subcarriers
            ] = section.resource_mask

            word_idx += num_words

        return resource_mask

    def map(self, data_bits: np.ndarray) -> Symbols:
        return Symbols(self.mapping.get_symbols(data_bits)[None, :, None])

    def unmap(self, symbols: Symbols) -> np.ndarray:
        return self.mapping.detect_bits(symbols.raw.flatten()).astype(int)

    def place(self, symbols: Symbols) -> Symbols:
        # Prepare symbols to be placed
        data_symbols = symbols.raw.flatten()
        reference_symbols = self.pilot_symbols(self.references_per_frame)

        # Make sure the number of provided symbols matches the number of symbols in the frame
        if data_symbols.size != self.num_data_symbols:
            raise ValueError(
                f"Number of provided data symbols does not match the number of symbols in the frame ({data_symbols.size} != {self.num_data_symbols})"
            )

        # Generate the symbol sequence for a full OFDM frame
        num_words = 0
        for section in self.grid_structure:
            num_words += section.num_words
        placed_symbols = np.zeros((1, num_words, self.num_subcarriers), dtype=np.complex_)

        data_idx = 0
        reference_idx = 0
        word_idx = 0
        for section in self.grid_structure:
            num_data_symbols = section.num_symbols
            num_reference_symbols = section.num_references
            num_words = section.num_words

            data = data_symbols[data_idx : data_idx + num_data_symbols]
            reference = reference_symbols[reference_idx : reference_idx + num_reference_symbols]

            placed_symbols[
                0, word_idx : word_idx + num_words, : section.num_subcarriers
            ] = section.place_symbols(data, reference)

            data_idx += num_data_symbols
            reference_idx += num_reference_symbols
            word_idx += num_words

        return Symbols(placed_symbols)

    def pick(self, placed_symbols: StatedSymbols) -> StatedSymbols:
        raw_symbols = placed_symbols.raw
        raw_states = placed_symbols.dense_states()
        raw_picked_symbols = np.empty(
            (placed_symbols.num_streams, self.num_data_symbols, 1), dtype=np.complex_
        )
        raw_picked_states = np.empty(
            (
                placed_symbols.num_streams,
                placed_symbols.num_transmit_streams,
                self.num_data_symbols,
                1,
            ),
            dtype=np.complex_,
        )

        block_idx = 0
        symbol_idx = 0
        for section in self.grid_structure:
            raw_picked_symbols[
                :, symbol_idx : symbol_idx + section.num_symbols, 0
            ] = section.pick_symbols(raw_symbols[:, block_idx : block_idx + section.num_words, :])
            raw_picked_states[
                :, :, symbol_idx : symbol_idx + section.num_symbols, 0
            ] = section.pick_symbols(raw_states[:, :, block_idx : block_idx + section.num_words, :])

            block_idx += section.num_words
            symbol_idx += section.num_symbols

        return StatedSymbols(raw_picked_symbols, raw_picked_states)

    def modulate(self, symbols: Symbols) -> np.ndarray:
        frame_samples = np.empty(self.samples_per_frame, dtype=np.complex_)
        sample_idx = 0

        # Start the frame with a pilot section, if configured
        if self.pilot_section:
            frame_samples[: self.pilot_section.num_samples] = self.pilot_section.generate()
            sample_idx += self.pilot_section.num_samples

        # Transform the symbols into the time-domain
        symbol_grid = symbols.raw[0, :, :]
        signal_grid = self._forward_transformation(symbol_grid)

        # Place the time-domain samples of each section into their respective frame section
        # This includes the application of prefixes
        block_idx = 0
        for section in self.grid_structure:
            # Modulate the signal
            frame_samples[sample_idx : sample_idx + section.num_samples] = section.place_samples(
                signal_grid[block_idx : block_idx + section.num_words, :]
            )
            block_idx += section.num_words
            sample_idx += section.num_samples

        return frame_samples

    def demodulate(self, signal: np.ndarray) -> Symbols:
        sample_idx = 0

        # If the frame contains a pilot section, skip the respective samples
        if self.pilot_section:
            sample_idx += self.pilot_section.num_samples

        signal_grid = np.empty(
            (self.words_per_frame, self.num_subcarriers * self.oversampling_factor),
            dtype=np.complex_,
        )

        # Pick the time-domain samples of each section from the frame
        block_idx = 0
        for section in self.grid_structure:
            signal_grid[block_idx : block_idx + section.num_words, :] = section.pick_samples(
                signal[sample_idx : sample_idx + section.num_samples]
            )
            block_idx += section.num_words
            sample_idx += section.num_samples

        # Transform the time-domain samples into the orthogonal symbol grid
        symbol_grid = self._backward_transformation(signal_grid)

        # Correct the effect of prefix offsets on symbols within the individual sections
        block_idx = 0
        for section in self.grid_structure:
            if section.sample_offset != 0:
                symbol_grid[
                    block_idx : block_idx + section.num_words, :
                ] = self._correct_sample_offset(
                    symbol_grid[block_idx : block_idx + section.num_words, :], section.sample_offset
                )

        return Symbols(symbol_grid[np.newaxis, :, :])
