# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Sequence

import numpy as np

from ...waveform import PilotSymbolSequence
from .waveform import GridResource, GridSection, OrthogonalWaveform, PilotSection

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class OCDMWaveform(OrthogonalWaveform):
    """Orthogonal Chirp Division Multiplexing waveform."""

    __bandwidth: float

    def __init__(
        self,
        bandwidth: float,
        num_subcarriers: int,
        grid_resources: Sequence[GridResource],
        grid_structure: Sequence[GridSection],
        pilot_section: PilotSection | None = None,
        pilot_sequence: PilotSymbolSequence | None = None,
        repeat_pilot_sequence: bool = True,
        **kwargs,
    ) -> None:
        # Initialize base class
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

        # Initialize class attributes
        self.bandwidth = bandwidth

    @property
    def __DFnT(self) -> np.ndarray:
        """Discrete Fresenl Transform matrix."""

        N = self.num_subcarriers
        correction = 0.0 if N % 2 == 0 else 0.5

        # Discrete Fresnel Transform
        transform = np.zeros((N, N * self.oversampling_factor), dtype=np.complex_)
        for m, n in np.ndindex(N, N * self.oversampling_factor):
            transform[m, n] = N**-0.5 * np.exp(
                1j * np.pi * ((m + correction - n / self.oversampling_factor) ** 2 / N - 0.25)
            )

        return transform

    @property
    def bandwidth(self) -> float:
        return self.__bandwidth

    @bandwidth.setter
    def bandwidth(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError("Bandwidth must be gerater than zero")

        self.__bandwidth = value

    @property
    def sampling_rate(self) -> float:
        return self.bandwidth * self.oversampling_factor

    def _forward_transformation(self, symbol_grid: np.ndarray) -> np.ndarray:
        return symbol_grid @ self.__DFnT

    def _backward_transformation(self, sample_sections: np.ndarray) -> np.ndarray:
        return (
            sample_sections
            @ self.__DFnT[:, : sample_sections.shape[-1]].T.conj()
            / self.oversampling_factor
        )

    def _correct_sample_offset(self, symbol_subgrid: np.ndarray, sample_offset: int) -> np.ndarray:
        # This is a stub for now
        return symbol_subgrid  # pragma: no cover
