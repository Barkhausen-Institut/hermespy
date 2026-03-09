# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Type
from typing_extensions import override

import numpy as np

from hermespy.core import SerializationProcess, DeserializationProcess
from hermespy.modem.waveform import Synchronization
from .orthogonal import (
    ElementType,
    OFDMWaveform,
    SymbolSection,
    PrefixType,
    GridElement,
    GridResource,
    PilotSection,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2026, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


# Constants for the 5G NR waveform mockup
IEEE_5GNR_RB_NUM_SUBCARRIERS = 12  # Referred to as N^RB_SC in the standard
IEEE_5GNR_CYCLIC_PREFIX_RATIO = 0.07  # Ratio of cyclic prefix duration to useful symbol duration, typical for 5G NR (normal CP)
IEEE_5GNR_NUM_SYMBOLS_PER_SLOT = 14
IEEE_5GNR_NUM_SUBFRAMES_PER_FRAME = 10
IEEE_5GNR_MIN_NUM_RBS = 24  # Minimum number of resource blocks for a single slot in 5G NR


def nr_subcarrier_spacing(numerology: int) -> float:
    """Calculate the subcarrier spacing for a given 5G NR numerology index.

    Args:
        numerology (int): The 5G NR numerology index (0 to 6).

    Returns:
        float: The subcarrier spacing in Hz.
    """
    if numerology < 0 or numerology > 6:
        raise ValueError("Numerology index must be between 0 and 6")

    return 15e3 * (2 ** numerology)


def nr_bandwidth(numerology: int, num_resource_blocks: int = IEEE_5GNR_MIN_NUM_RBS) -> float:
    """Calculate the bandwidth for a given 5G NR numerology index and number of resource blocks.

    Args:
        numerology (int): The 5G NR numerology index (0 to 6).
        num_resource_blocks (int): The number of resource blocks.

    Returns:
        float: The bandwidth in Hz.
    """

    return nr_subcarrier_spacing(numerology) * IEEE_5GNR_RB_NUM_SUBCARRIERS * num_resource_blocks


class PSS(PilotSection):
    """5G NR primary synchronization signal (PSS) section."""

    @override
    def _pilot_sequence(self, num_symbols: int | None = None) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        # Section 7.4.2.2.1 of 3GPP TS 138.211 V18.8.0 (2026-02)

        # Physical layer identity within cell identity group, can be set to any value in { 0 ... 2 }
        # Denoted by N_ID^(2) in 3GPP TS 138.211 V18.8.0 (2026-02)
        n_id_2 = 0

        num_symbols = 127

        # Denoted by x(m) / x(i) in section 7.4.2.2.1 of 3GPP TS 138.211 V18.8.0 (2026-02)
        x = np.empty(num_symbols, dtype=np.int64)
        x[0:7] = [1, 1, 1, 0, 1, 1, 0]
        for i in range(7, num_symbols):
            x[i] = (x[i - 3] + x[i - 7]) % 2

        # Denoted by m in section 7.4.2.2.1 of 3GPP TS 138.211 V18.8.0 (2026-02)
        sequence_cell_selector = (np.arange(num_symbols) + 43 * n_id_2) % 127

        # Denoted by d_PSS(n) in section 7.4.2.2.1 of 3GPP TS 138.211 V18.8.0 (2026-02)
        symbol_sequence = 1 - 2 * x[sequence_cell_selector]

        return symbol_sequence.astype(np.complex128)


class SSS(PilotSection):
    """5G NR secondary synchronization signal (SSS) section."""

    @override
    def _pilot_sequence(self, num_symbols: int | None = None) -> np.ndarray[tuple[int, ...], np.dtype[np.complex128]]:
        # Section 7.4.2.3.1 of 3GPP TS 138.211 V18.8.0 (2026-02)

        # Cell identity group index, can be set to any value in { 0 ... 2 }
        # Denoted by N_ID^(1) in 3GPP TS 138.211 V18.8.0 (2026-02)
        n_id_1 = 0

        # Physical layer identity within cell identity group, can be set to any value in { 0 ... 2 }
        # Denoted by N_ID^(2) in 3GPP TS 138.211 V18.8.0 (2026-02)
        n_id_2 = 0

        m_0 = 15 * np.floor(n_id_1 / 112) + 5 * n_id_2
        m_1 = n_id_1 % 112

        num_symbols = 127

        # Denoted by x_0(i) in section 7.4.2.3.1 of 3GPP TS 138.211 V18.8.0 (2026-02)
        x_0 = np.empty(num_symbols, dtype=np.int64)
        x_0[0:7] = [0, 0, 0, 0, 0, 0, 1]

        # Denoted by x_1(i) in section 7.4.2.3.1 of 3GPP TS 138.211 V18.8.0 (2026-02)
        x_1 = np.empty(num_symbols, dtype=np.int64)
        x_1[0:7] = [0, 0, 0, 0, 0, 0, 1]

        for i in range(7, num_symbols):
            x_0[i] = (x_0[i - 3] + x_0[i - 7]) % 2
            x_1[i] = (x_1[i - 6] + x_1[i - 7]) % 2

        n = np.arange(num_symbols)
        pilot_symbol_sequence = (1 - 2 * x_0 * ((n + m_0) % 127)) * (1 - 2 * x_1 * ((n + m_1) % 127))

        # Map bits to modulation symbols
        return pilot_symbol_sequence.view(np.complex128)


class NRSlot(OFDMWaveform):
    """Mock of a 5G NR slot.

    Note that only the rough frame structure is implemented, reference symbols and synchronization patterns are not placed according to the actual 5G NR standard.
    This is intended for testing and demonstration purposes only.
    """

    @staticmethod
    def _init_pss_resource(num_subcarriers: int) -> GridResource:
        """Generate a grid represenatation of the primary synchronization signal,
        that is transmitted in the first slot of each frame and occupies 127 subcarriers around the DC subcarrier.
        """

        return GridResource(
            repetitions=1,
            prefix_type=PrefixType.CYCLIC,
            prefix_ratio=IEEE_5GNR_CYCLIC_PREFIX_RATIO,
            elements=[
                GridElement(ElementType.NULL, repetitions=int(.5 * num_subcarriers) - 64),
                GridElement(ElementType.REFERENCE, repetitions=127),
                GridElement(ElementType.NULL, repetitions=int(.5 * num_subcarriers) - 63),
            ],
        )

    @staticmethod
    def _init_sss_resource(num_subcarriers: int) -> GridResource:
        """Generate a grid represenatation of the secondary synchronization signal,
        that is transmitted in the first slot of each frame and occupies 127 subcarriers around the DC subcarrier.
        """

        return GridResource(
            repetitions=1,
            prefix_type=PrefixType.CYCLIC,
            prefix_ratio=IEEE_5GNR_CYCLIC_PREFIX_RATIO,
            elements=[
                GridElement(ElementType.NULL, repetitions=int(.5 * num_subcarriers) - 64),
                GridElement(ElementType.REFERENCE, repetitions=127),
                GridElement(ElementType.NULL, repetitions=int(.5 * num_subcarriers) - 63),
            ],
        )

    @staticmethod
    def _init_pbch_resource(num_subcarriers: int, num_resource_blocks: int) -> GridResource:
        """Generate a grid represenatation of the physical broadcast channel,
        that is transmitted in the first slot of each frame and occupies 240 subcarriers around the DC subcarrier.
        """

        return GridResource(
            repetitions=num_resource_blocks,
            prefix_type=PrefixType.CYCLIC,
            prefix_ratio=IEEE_5GNR_CYCLIC_PREFIX_RATIO,
            elements=[
                GridElement(ElementType.REFERENCE, repetitions=1),  # PBCH DMRS
                GridElement(ElementType.DATA, repetitions=3),       # PBCH data
                GridElement(ElementType.REFERENCE, repetitions=1),  # PBCH DMRS
                GridElement(ElementType.DATA, repetitions=3),       # PBCH data
                GridElement(ElementType.REFERENCE, repetitions=1),  # PBCH DMRS
                GridElement(ElementType.DATA, repetitions=3),       # PBCH data
            ],
        )

    def __init__(self, num_resource_blocks: int = IEEE_5GNR_MIN_NUM_RBS) -> None:
        """
        Args:

            num_resource_blocks:
                Number of resource blocks within a single slot.
                Must be at least 24 to meet the minimum slot bandwidth requirements of 5G NR.
                The maximum number depends on the overall bandwidth available for the given frequency range.
        """

        num_subcarriers = num_resource_blocks * IEEE_5GNR_RB_NUM_SUBCARRIERS

        pss_resource = self._init_pss_resource(num_subcarriers)
        sss_resource = self._init_sss_resource(num_subcarriers)
        pbch_resource = self._init_pbch_resource(num_subcarriers, num_resource_blocks)

        # Build a frame structure
        grid_resources = [pss_resource, sss_resource, pbch_resource]
        grid_structure = [
            SymbolSection(1, [0]),
            SymbolSection(1, [2]),
            SymbolSection(1, [1]),
            SymbolSection(IEEE_5GNR_NUM_SYMBOLS_PER_SLOT - 3, [2]),
        ]

        OFDMWaveform.__init__(
            self,
            grid_resources,
            grid_structure,
            num_subcarriers=num_subcarriers,
            dc_suppression=False,
        )

    @property
    def num_resource_blocks(self) -> int:
        """Maximum number of resource blocks within a single OFDM slot."""

        return self.num_subcarriers // IEEE_5GNR_RB_NUM_SUBCARRIERS

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.num_resource_blocks, "num_resource_blocks")

    @override
    @classmethod
    def Deserialize(cls: Type[NRSlot], process: DeserializationProcess) -> NRSlot:
        return cls(
            num_resource_blocks=process.deserialize_integer("num_resource_blocks", IEEE_5GNR_MIN_NUM_RBS)
        )


class NRSubframe(NRSlot):
    """Mock of a 5G NR subframe.

    Note that only the rough frame structure is implemented, reference symbols and synchronization patterns are not placed according to the actual 5G NR standard.
    This is intended for testing and demonstration purposes only.
    """

    __numerology: int

    @staticmethod
    def num_subframe_slots(numerology: int) -> int:
        """Calculate the number of slots in a 5G NR subframe for a given numerology.

        Args:
            numerology (int): The 5G NR numerology index (0 to 6).

        Returns:
            int: The number of slots in a subframe.
        """

        return 2 ** numerology

    def __init__(self, numerology: int, num_resource_blocks: int = IEEE_5GNR_MIN_NUM_RBS) -> None:
        """
        Args:

            numerology:
                The 5G NR numerology index (0 to 6) defining the subcarrier spacing and slot duration.

            num_resource_blocks:
                Number of resource blocks within a single slot.
                Must be at least 24 to meet the minimum slot bandwidth requirements of 5G NR.
                The maximum number depends on the overall bandwidth available for the given frequency range.
        """

        self.__numerology = numerology
        num_subcarriers = num_resource_blocks * IEEE_5GNR_RB_NUM_SUBCARRIERS
        num_symbols = 2**numerology * IEEE_5GNR_NUM_SYMBOLS_PER_SLOT

        pss_resource = self._init_pss_resource(num_subcarriers)
        sss_resource = self._init_sss_resource(num_subcarriers)
        pbch_resource = self._init_pbch_resource(num_subcarriers, num_resource_blocks)

        # Build a frame structure
        grid_resources = [pss_resource, sss_resource, pbch_resource]
        grid_structure = [
            SymbolSection(1, [0]),
            SymbolSection(1, [2]),
            SymbolSection(1, [1]),
            SymbolSection(num_symbols - 3, [2]),
        ]

        OFDMWaveform.__init__(
            self,
            grid_resources,
            grid_structure,
            num_subcarriers=num_subcarriers,
            dc_suppression=False,
        )

    @property
    def numerology(self) -> int:
        """5G NR numerology index.

        Zero to six.
        """

        return self.__numerology

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_integer(self.numerology, "numerology")
        process.serialize_integer(self.num_resource_blocks, "num_resource_blocks")

    @override
    @classmethod
    def Deserialize(cls: Type[NRSubframe], process: DeserializationProcess) -> NRSubframe:
        return cls(
            numerology=process.deserialize_integer("numerology", 0),
            num_resource_blocks=process.deserialize_integer("num_resource_blocks", IEEE_5GNR_MIN_NUM_RBS)
        )


class NRFrame(NRSubframe):
    """Mock of a 5G NR frame.

    Note that only the rough frame structure is implemented, reference symbols and synchronization patterns are not placed according to the actual 5G NR standard.
    This is intended for testing and demonstration purposes only.
    """

    def __init__(self, numerology: int, num_resource_blocks: int = IEEE_5GNR_MIN_NUM_RBS) -> None:

        # Init subframe
        NRSubframe.__init__(self, numerology, num_resource_blocks)

        # Repeat the subframe structure 10x to build a frame
        self.grid_structure = IEEE_5GNR_NUM_SUBFRAMES_PER_FRAME * list(self.grid_structure)


class NRSynchronization(Synchronization[NRSubframe]):
    """Synchronization routine detecting the delay and frequency offset of a 5G NR subframe based on the PSS and SSS patterns."""

    @override
    def synchronize(self, signal: np.ndarray[tuple[int, ...], np.dtype[np.complex128]], bandwidth: float, oversampling_factor: int) -> list[int]:
        ...
