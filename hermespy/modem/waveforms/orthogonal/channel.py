# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC

import numpy as np
from scipy.interpolate import griddata

from hermespy.core import Serializable
from ...symbols import StatedSymbols, Symbols
from ...waveform import ChannelEstimation, ChannelEqualization, ZeroForcingChannelEqualization
from .waveform import ElementType, OrthogonalWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class OrthogonalLeastSquaresChannelEstimation(ChannelEstimation[OrthogonalWaveform], Serializable):
    """Least-Squares channel estimation for OFDM waveforms."""

    def estimate_channel(self, symbols: Symbols, delay: float = 0.0) -> StatedSymbols:
        if symbols.num_streams != 1:
            raise NotImplementedError(
                "Least-Squares channel estimation is only implemented for SISO links"
            )

        resource_mask = self.waveform.resource_mask

        propagated_references = symbols.raw[0, resource_mask[ElementType.REFERENCE.value, ::]]
        reference_symbols = self.waveform.pilot_symbols(len(propagated_references))
        reference_channel_estimation = propagated_references / reference_symbols

        channel_estimation = np.zeros(
            ((1, 1, symbols.num_blocks, symbols.num_symbols)), dtype=complex
        )
        channel_estimation[0, 0, resource_mask[ElementType.REFERENCE.value, ::]] = (
            reference_channel_estimation
        )

        # Interpolate over the holes, if there are any
        holes = np.where(np.invert(resource_mask[ElementType.REFERENCE.value, ::]))
        if holes[0].size != 0:
            interpolation_stems = np.where(resource_mask[ElementType.REFERENCE.value, ::])
            interpolated_holes = griddata(
                interpolation_stems, reference_channel_estimation, holes, method="linear"
            )
            channel_estimation[0, 0, holes[0], holes[1]] = interpolated_holes

        # Fill nan values with nearest neighbor
        nan_indices = np.where(np.isnan(channel_estimation))
        stem_indices = np.where(np.invert(np.isnan(channel_estimation)))
        channel_estimation[nan_indices] = griddata(
            stem_indices, channel_estimation[stem_indices], nan_indices, method="nearest"
        )

        return StatedSymbols(symbols.raw, channel_estimation)


class OrthogonalChannelEqualization(ChannelEqualization[OrthogonalWaveform], ABC):
    """Channel estimation for OFDM waveforms."""

    def __init__(self, waveform: OrthogonalWaveform | None = None) -> None:
        """
        Args:

            waveform:
                The waveform generator this equalization routine is attached to.
        """

        ChannelEqualization.__init__(self, waveform)


class OrthogonalZeroForcingChannelEqualization(ZeroForcingChannelEqualization[OrthogonalWaveform]):
    """Zero-Forcing channel equalization for OFDM waveforms."""

    ...
