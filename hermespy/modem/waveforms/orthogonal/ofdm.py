# -*- coding: utf-8 -*-

from __future__ import annotations
from math import ceil
from typing import List, Any, Set, Sequence

import numpy as np
from scipy.fft import fft, fftfreq, fftshift, ifft, ifftshift
from scipy.signal import find_peaks

from hermespy.core import Serializable
from ...waveform import Synchronization
from ...waveform_correlation_synchronization import CorrelationSynchronization
from .waveform import (
    GridResource,
    GridSection,
    OrthogonalWaveform,
    PilotSection,
    PilotSymbolSequence,
)

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["André Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class OFDMWaveform(OrthogonalWaveform, Serializable):
    """Generic Orthogonal Frequency Division Multiplexing waveform description."""

    yaml_tag: str = "OFDM"

    __subcarrier_spacing: float
    dc_suppression: bool

    @staticmethod
    def _arg_signature() -> Set[str]:
        return {"modulation_order"}  # pragma: no cover

    def __init__(
        self,
        grid_resources: Sequence[GridResource],
        grid_structure: Sequence[GridSection],
        num_subcarriers: int = 1024,
        subcarrier_spacing: float = 1e3,
        dc_suppression: bool = True,
        pilot_section: PilotSection | None = None,
        pilot_sequence: PilotSymbolSequence | None = None,
        repeat_pilot_sequence: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Args:

            grid_resources (Sequence[GridResource]):
                Frequency-domain resource section configurations.

            grid_structure (Sequence[GridSection]):
                Time-domain frame configuration.

            num_subcarriers (int, optional):
                Maximum number of assignable subcarriers.
                Unassigned subcarriers will be assumed to be zero.
                :math:`1024` by default.

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

            pilot_section (PilotSection, optional):
                Pilot section preceding the frame's payload.
                If not specified, no dedicated pilot section will be generated.

            pilot_sequence (PilotSymbolSequence, optional):
                Sequence of symbols used for the pilot section and reference symbols
                within the frame. If not specified, pseudo-random sequences will be generated
                from the set of data symbols.

            **kwargs (Any):
                Waveform generator base class initialization parameters.
                Refer to :class:`CommunicationWaveform` for details.
        """

        # Initialize the base class
        OrthogonalWaveform.__init__(
            self,
            num_subcarriers,
            grid_resources,
            grid_structure,
            pilot_section,
            pilot_sequence,
            repeat_pilot_sequence,
            **kwargs,
        )

        # Initialize the OFDM specific attributes
        self.subcarrier_spacing = subcarrier_spacing
        self.dc_suppression = dc_suppression

    def _forward_transformation(self, symbol_grid: np.ndarray) -> np.ndarray:
        # Normalize the frequency-domain data symbols for unit power transmission
        normalized_symbols = symbol_grid / np.sqrt(self.num_subcarriers)

        # Zero-pad the grid to account for oversampling
        padded_symbol_grid = np.zeros(
            (normalized_symbols.shape[0], self.oversampling_factor * self.num_subcarriers),
            dtype=np.complex128,
        )
        padding_start_idx = (
            self.oversampling_factor * self.num_subcarriers
        ) // 2 - self.num_subcarriers // 2
        padded_symbol_grid[:, padding_start_idx : self.num_subcarriers + padding_start_idx] = (
            normalized_symbols
        )

        # Shift in order to suppress the dc component
        # Note that for configurations without any oversampling the DC component will not be suppressed
        if self.dc_suppression:
            dc_index = int(0.5 * self.num_subcarriers * self.oversampling_factor)
            padded_symbol_grid[:, dc_index:] = np.roll(padded_symbol_grid[:, dc_index:], 1, axis=-1)

        # By convention, the length of each time slot is the inverse of the sub-carrier spacing
        sample_grid = ifft(
            ifftshift(padded_symbol_grid, axes=-1),
            self.num_subcarriers * self.oversampling_factor,
            axis=-1,
            norm="forward",
        )

        return sample_grid

    def _backward_transformation(
        self, sample_sections: np.ndarray, normalize: bool = True
    ) -> np.ndarray:
        # Transform the time-domain resource signals to frequency-domain data symbols
        symbol_grid = fft(
            sample_sections,
            n=self.num_subcarriers * self.oversampling_factor,
            axis=-1,
            norm="backward",
        )

        # Shift fft bins to the center
        symbol_grid = fftshift(symbol_grid, axes=-1)

        # Account for the DC suppression
        if self.dc_suppression:
            dc_index = int(0.5 * self.num_subcarriers * self.oversampling_factor)
            symbol_grid[..., dc_index:] = np.roll(symbol_grid[..., dc_index:], -1, axis=-1)

        # Remove the zero padding due to the oversampling from the symbol grid
        padding_start_idx = (
            self.oversampling_factor * self.num_subcarriers
        ) // 2 - self.num_subcarriers // 2
        original_symbol_grid = symbol_grid[
            ..., padding_start_idx : self.num_subcarriers + padding_start_idx
        ]

        # Normalize the frequency-domain data symbols for unit power reception
        if normalize:
            original_symbol_grid *= (self.num_subcarriers) ** -0.5 / self.oversampling_factor

        return original_symbol_grid

    def _correct_sample_offset(self, symbol_subgrid: np.ndarray, sample_offset: int) -> np.ndarray:
        frame_start_idx = (
            self.oversampling_factor * self.num_subcarriers
        ) // 2 - self.num_subcarriers // 2
        freqs = fftshift(fftfreq(self.oversampling_factor * self.num_subcarriers))[
            frame_start_idx : frame_start_idx + self.num_subcarriers
        ]

        if self.dc_suppression:
            dc_index = int(0.5 * symbol_subgrid.shape[1])
            freqs[dc_index:] = np.roll(freqs[dc_index:], -1)

        return symbol_subgrid * np.exp(2j * np.pi * freqs * sample_offset)

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
    def samples_per_frame(self) -> int:
        num = 0
        for section in self.grid_structure:
            num += section.num_samples

        if self.pilot_section:
            num += self.pilot_section.num_samples

        return num

    @property
    def bandwidth(self) -> float:
        # OFDM bandwidth currently is identical to the number of subcarriers times the subcarrier spacing
        b = self.num_subcarriers * self.subcarrier_spacing
        return b

    @property
    def sampling_rate(self) -> float:
        return self.oversampling_factor * self.subcarrier_spacing * self.num_subcarriers


class SchmidlCoxPilotSection(PilotSection[OFDMWaveform]):
    """Pilot Symbol Section of the Schmidl Cox Algorithm.

    Refer to :footcite:t:`1997:schmidl` for a detailed description.
    """

    yaml_tag = "SchmidlCoxPilot"
    """YAML serialization tag"""

    def _pilot_sequence(self, num_symbols: int = None) -> np.ndarray:
        # The schmidl-cox pilot sequence is zero-stuffed in frequency domain
        stuffed_pilot_sequence = np.zeros(self.wave.num_subcarriers, dtype=complex)
        stuffed_pilot_sequence[::2] = PilotSection._pilot_sequence(
            self, ceil(0.5 * self.wave.num_subcarriers)
        )

        if self.wave.dc_suppression:
            dc_index = int(0.5 * self.wave.num_subcarriers)
            stuffed_pilot_sequence[:dc_index] = np.roll(stuffed_pilot_sequence[:dc_index], 1)

        return stuffed_pilot_sequence


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
        symbol_length = self.waveform.oversampling_factor * self.waveform.num_subcarriers

        # Abort if the supplied signal is shorter than one symbol length
        if signal.shape[-1] < symbol_length:
            return []

        half_symbol_length = int(0.5 * symbol_length)

        num_delay_candidates = 2 + signal.shape[-1] - symbol_length
        delay_powers = np.empty(num_delay_candidates, dtype=float)
        delay_powers[0] = 0.0  # In order to be able to detect a peak on the first sample
        for d in range(0, num_delay_candidates - 1):
            delay_powers[1 + d] = np.sum(
                abs(
                    np.sum(
                        signal[:, d : d + half_symbol_length].conj()
                        * signal[:, d + half_symbol_length : d + 2 * half_symbol_length],
                        axis=1,
                    )
                )
            )

        num_samples = self.waveform.samples_per_frame
        min_height = 0.75 * np.max(delay_powers)
        peaks, _ = find_peaks(delay_powers, distance=int(0.9 * num_samples), height=min_height)
        frame_indices = peaks - 1  # Correct for the first delay bin being prepended

        return frame_indices
