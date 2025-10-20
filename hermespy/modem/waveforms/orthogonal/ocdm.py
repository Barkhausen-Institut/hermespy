# -*- coding: utf-8 -*-

from __future__ import annotations
from functools import cache
from typing_extensions import override

import numpy as np

from .waveform import OrthogonalWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class OCDMWaveform(OrthogonalWaveform):
    """Orthogonal Chirp Division Multiplexing waveform."""

    @cache
    @staticmethod
    def __DFnT(num_subcarriers: int, oversampling_factor: int) -> np.ndarray:
        """Discrete Fresenl Transform matrix.

        Args:
            num_subcarriers: Number of subcarriers in the waveform.
            oversampling_factor: Oversampling factor for the waveform.

        Returns:
            The Discrete Fresnel Transform matrix.
        """

        N = num_subcarriers
        correction = 0.0 if N % 2 == 0 else 0.5

        # Discrete Fresnel Transform
        transform = np.zeros((N, N * oversampling_factor), dtype=np.complex128)
        for m, n in np.ndindex(N, N * oversampling_factor):
            transform[m, n] = N**-0.5 * np.exp(
                1j * np.pi * ((m + correction - n / oversampling_factor) ** 2 / N - 0.25)
            )

        return transform

    @override
    def _forward_transformation(
        self, symbol_grid: np.ndarray, oversampling_factor: int
    ) -> np.ndarray:
        return symbol_grid @ OCDMWaveform.__DFnT(self.num_subcarriers, oversampling_factor)

    @override
    def _backward_transformation(
        self, sample_sections: np.ndarray, oversampling_factor: int
    ) -> np.ndarray:
        return (
            sample_sections
            @ OCDMWaveform.__DFnT(self.num_subcarriers, oversampling_factor)[
                :, : sample_sections.shape[-1]
            ].T.conj()
            / oversampling_factor
        )

    @override
    def _correct_sample_offset(
        self, symbol_subgrid: np.ndarray, sample_offset: int, oversampling_factor: int
    ) -> np.ndarray:
        # This is a stub for now
        return symbol_subgrid  # pragma: no cover
