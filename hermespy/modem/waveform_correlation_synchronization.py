# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, Generic, List, TypeVar
from typing_extensions import override

import numpy as np
from scipy.signal import correlate, find_peaks

from hermespy.core import SerializationProcess, DeserializationProcess
from .waveform import PilotCommunicationWaveform, Synchronization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


PGT = TypeVar("PGT", bound=PilotCommunicationWaveform)
"""Type of pilot-generating waveforms."""


class CorrelationSynchronization(Generic[PGT], Synchronization[PGT]):
    """Correlation-based clock synchronization for arbitrary communication waveforms.

    The implemented algorithm is equivalent to :cite:p:`1976:knapp` without pre-filtering.
    """

    __DEFAULT_THRESHOLD: float = 0.9  # Default correlation threshold
    __DEFAULT_GUARD_RATIO: float = 0.8  # Default guard ratio
    __DEFAULT_PEAK_PROMINENCE: float = 0.2  # Default peak prominence

    __threshold: float  # Correlation threshold at which a pilot signal is detected
    __guard_ratio: float  # Guard ratio of frame duration
    __peak_prominence: float  # Minimum peak prominence for peak detection

    def __init__(
        self,
        threshold: float = __DEFAULT_THRESHOLD,
        guard_ratio: float = __DEFAULT_GUARD_RATIO,
        peak_prominence: float = __DEFAULT_PEAK_PROMINENCE,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Args:

            threshold:
                Correlation threshold at which a pilot signal is detected.

            guard_ratio:
                Guard ratio of frame duration.

            peak_prominence:
                Minimum peak prominence for peak detection in the interval (0, 1].
                :math:`0.2` is a good default value for most applications.

            *args:
                Synchronization base class initialization parameters.
        """

        self.threshold = threshold
        self.guard_ratio = guard_ratio
        self.__peak_prominence = peak_prominence

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

        if value < 0.0 or value > 1.0:
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

        if value < 0.0 or value > 1.0:
            raise ValueError("Synchronization guard ratio must be between zero and one.")

        self.__guard_ratio = value

    def synchronize(self, signal: np.ndarray) -> List[int]:
        # Expand the dimensionality for flat signal streams
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]

        # Query the pilot signal from the waveform generator
        pilot_sequence = self.waveform.pilot_signal.getitem().flatten()

        # Raise a runtime error if pilot sequence is empty
        if len(pilot_sequence) < 1:
            raise RuntimeError(
                "No pilot sequence configured, time-domain correlation synchronization impossible"
            )

        # Compute the correlation between each signal stream and the pilot sequence, sum up as a result
        correlation = np.zeros(len(pilot_sequence) + signal.shape[1] - 1, dtype=float)
        for stream in signal:
            correlation += abs(correlate(stream, pilot_sequence, mode="full", method="fft"))

        correlation /= correlation.max()  # Normalize correlation

        # Determine the pilot sequence locations by performing a peak search over the correlation profile
        frame_length = self.waveform.samples_per_frame
        pilot_indices, _ = find_peaks(
            abs(correlation), height=0.9, distance=int(0.8 * frame_length)
        )

        # Abort if no pilot section has been detected
        if len(pilot_indices) < 1:
            return []

        # Correct pilot indices by the convolution length
        pilot_length = len(pilot_sequence)
        pilot_indices -= pilot_length - 1

        # Correct infeasible pilot index choices
        pilot_indices = np.where(pilot_indices < 0, 0, pilot_indices)
        pilot_indices = np.where(
            pilot_indices > (signal.shape[1] - frame_length),
            abs(signal.shape[1] - frame_length),
            pilot_indices,
        )

        return pilot_indices.tolist()

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.threshold, "threshold")
        process.serialize_floating(self.guard_ratio, "guard_ratio")
        process.serialize_floating(self.__peak_prominence, "peak_prominence")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> CorrelationSynchronization:
        return cls(
            threshold=process.deserialize_floating("threshold", cls.__DEFAULT_THRESHOLD),
            guard_ratio=process.deserialize_floating("guard_ratio", cls.__DEFAULT_GUARD_RATIO),
            peak_prominence=process.deserialize_floating(
                "peak_prominence", cls.__DEFAULT_PEAK_PROMINENCE
            ),
        )
