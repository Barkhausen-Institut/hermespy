# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
from scipy.fft import fft, ifft

from .ofdm import OFDMWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class OTFSWaveform(OFDMWaveform):
    """Orthogonal Time Frequency Space (OTFS) waveform."""

    def _forward_transformation(self, symbol_grid: np.ndarray) -> np.ndarray:
        # Initial step: ISFFT
        delay_doppler_symbols = fft(ifft(symbol_grid, axis=-1, norm="ortho"), axis=-2, norm="ortho")

        # Second step: Heisenberg transform, i.e. the regular OFDM treatment
        sample_sections = OFDMWaveform._forward_transformation(self, delay_doppler_symbols)

        return sample_sections

    def _backward_transformation(
        self, sample_sections: np.ndarray, normalize: bool = True
    ) -> np.ndarray:
        # Initial step: Inverse Heisenberg transform, i.e. the regular OFDM treatment
        delay_doppler_symbols = OFDMWaveform._backward_transformation(
            self, sample_sections, normalize
        )

        # Second step: SFFT
        symbol_grid = ifft(fft(delay_doppler_symbols, axis=-1, norm="ortho"), axis=-2, norm="ortho")

        # Normalize the symbol grid
        if normalize:
            symbol_grid /= np.sqrt(np.prod(symbol_grid.shape[:-2]))

        return symbol_grid
