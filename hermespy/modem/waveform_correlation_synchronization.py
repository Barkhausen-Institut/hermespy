# -*- coding: utf-8 -*-
"""
===========================================
Correlation-Based Waveform Synchronization
===========================================
"""

from __future__ import annotations
from typing import Any, Generic, List, Tuple, TypeVar

import numpy as np
from scipy.signal import correlate, find_peaks

from ..core.channel_state_information import ChannelStateInformation
from .waveform_generator import PilotWaveformGenerator, Synchronization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.6"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


PGT = TypeVar('PGT', bound=PilotWaveformGenerator)
"""Type of pilot-generating waveforms."""


class CorrelationSynchronization(Generic[PGT], Synchronization[PGT]):
    """Correlation-based clock synchronization for arbitrary communication waveforms."""

    __threshold: float          # Correlation threshold at which a pilot signal is detected
    __guard_ratio: float        # Guard ratio of frame duration

    def __init__(self,
                 threshold: float = 0.9,
                 guard_ratio: float = 0.8,
                 *args: Any,
                 **kwargs: Any) -> None:
        """
        Args:

            threshold (float, optional):
                Correlation threshold at which a pilot signal is detected.

            guard_ratio (float, optional):
                Guard ratio of frame duration.

            *args:
                Synchronization base class initialization parameters.
        """

        self.threshold = threshold
        self.guard_ratio = guard_ratio

        Synchronization.__init__(self, *args, **kwargs)

    @property
    def threshold(self) -> float:
        """Correlation threshold at which a pilot signal is detected.

        Returns:
            float: Threshold between zero and one.

        Raises:
            ValueError: If threshold is smaller than zero or greater than one.
        """

        return self.__threshold

    @threshold.setter
    def threshold(self, value: float):
        """Set correlation threshold at which a pilot signal is detected."""

        if value < 0. or value > 1.:
            raise ValueError("Synchronization threshold must be between zero and one.")

        self.__threshold = value

    @property
    def guard_ratio(self) -> float:
        """Correlation guard ratio at which a pilot signal is detected.

        After the detection of a pilot section, `guard_ratio` prevents the detection of another pilot in
        the following samples for a span relative to the configured frame duration.

        Returns:
            float: Guard Ratio between zero and one.

        Raises:
            ValueError: If guard ratio is smaller than zero or greater than one.
        """

        return self.__guard_ratio

    @guard_ratio.setter
    def guard_ratio(self, value: float):
        """Set correlation guard ratio at which a pilot signal is detected."""

        if value < 0. or value > 1.:
            raise ValueError("Synchronization guard ratio must be between zero and one.")

        self.__guard_ratio = value

    def synchronize(self,
                    signal: np.ndarray,
                    channel_state: ChannelStateInformation) -> List[Tuple[np.ndarray, ChannelStateInformation]]:

        # Query the pilot signal from the waveform generator
        pilot_sequence = self.waveform_generator.pilot.samples.flatten()

        # Raise a runtime error if pilot sequence is empty
        if len(pilot_sequence) < 1:
            raise RuntimeError("No pilot sequence configured, time-domain correlation synchronization impossible")

        correlation = correlate(signal, pilot_sequence, mode='full', method='fft')
        correlation /= (np.linalg.norm(pilot_sequence) ** 2)  # Normalize correlation

        # Determine the pilot sequence locations by performing a peak search over the correlation profile
        frame_length = self.waveform_generator.samples_in_frame
        pilot_indices, _ = find_peaks(abs(correlation), height=.9, distance=int(.8 * frame_length))
        pilot_indices -= len(pilot_sequence) - 1

        # Abort if no pilot section has been detected
        if len(pilot_indices) < 1:
            return []

        frames = []
        for pilot_index in pilot_indices:

            if pilot_index + frame_length <= int(1.05 * len(signal)):

                signal_frame = signal[pilot_index:pilot_index + frame_length]
                csi_frame = channel_state[:, :, pilot_index:pilot_index + frame_length, :]

                if len(signal_frame) < frame_length:

                    signal_frame = np.append(signal_frame, np.zeros(frame_length - len(signal_frame), dtype=complex))

                frames.append((signal_frame, csi_frame))

        return frames
