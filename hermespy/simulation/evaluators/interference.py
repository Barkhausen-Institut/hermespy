# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Sequence
from typing_extensions import override

import numpy as np

from hermespy.core import (
    DeserializationProcess,
    ScalarEvaluator,
    ScalarEvaluationResult,
    GridDimension,
    Hook,
    PowerEvaluation,
    Serializable,
    SerializationProcess,
    ValueType,
)
from ..simulated_device import ProcessedSimulatedDeviceInput, SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SI(ScalarEvaluator, Serializable):
    """Evaluate a simulated device's self-interference power."""

    __device: SimulatedDevice
    _input_hook: Hook[ProcessedSimulatedDeviceInput]
    _input: ProcessedSimulatedDeviceInput | None

    def __init__(
        self,
        device: SimulatedDevice,
        confidence: float = 1.0,
        tolerance: float = 0.0,
        min_num_samples: int = 1024,
        plot_scale: str = "linear",
        tick_format: ValueType = ValueType.LIN,
        plot_surface: bool = True,
    ) -> None:
        """
        Args:
            device: The device to evaluate.
            confidence: Required confidence level for the given `tolerance` between zero and one.
            tolerance: Acceptable non-negative bound around the mean value of the estimated scalar performance indicator.
            min_num_samples: Minimum number of samples required to compute the confidence bound.
            plot_scale: Scale of the plot. Can be ``'linear'`` or ``'log'``.
            tick_format: Tick format of the plot.
            plot_surface: Enable surface plotting for two-dimensional grids. Enabled by default.
        """

        # Initialize the base class
        ScalarEvaluator.__init__(self, confidence, tolerance, min_num_samples, plot_scale, tick_format, plot_surface)

        # Register input hook
        self.__device = device
        self._input_hook = device.simulated_input_callbacks.add_callback(self.__input_callback)

    def __input_callback(self, input: ProcessedSimulatedDeviceInput) -> None:
        """Callback function notifying the evaluator of a new input."""

        self._input = input

    @property
    @override
    def abbreviation(self) -> str:
        return "SI"

    @property
    @override
    def title(self) -> str:
        return "Self-Interference"

    @override
    def evaluate(self) -> PowerEvaluation:
        if self._input is None:
            raise RuntimeError(
                "Self-interference evaluator could not fetch input. Has the device received data?"
            )

        return PowerEvaluation(self._input.leaking_signal.power)

    @override
    def generate_result(self, grid: Sequence[GridDimension], artifacts: np.ndarray) -> ScalarEvaluationResult:
        # Find the maximum number of receive ports over all artifacts
        max_ports = max(
            max(artifact.power.size for artifact in artifacts) for artifacts in artifacts.flat
        )

        mean_powers = np.zeros((*artifacts.shape, max_ports), dtype=np.float64)
        for grid_index, artifacts in np.ndenumerate(artifacts):
            for artifact in artifacts:
                mean_powers[grid_index] += artifact.power

            num_artifacts = len(artifacts)
            if num_artifacts > 0:
                mean_powers[grid_index] /= len(artifacts)

        return ScalarEvaluationResult(mean_powers, grid, self)

    def __del__(self) -> None:
        self._input_hook.remove()

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.__device, "device")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> SI:
        device = process.deserialize_object("device", SimulatedDevice)
        return cls(device)


class SSINR(SI):
    """Signal to self-interfernce plus noise power ratio evaluator."""

    @property
    @override
    def abbreviation(self) -> str:
        return "SSINR"

    @property
    @override
    def title(self) -> str:
        return "Signal to Self-Interference plus Noise Power Ratio"

    @override
    def evaluate(self) -> PowerEvaluation:
        if self._input is None:
            raise RuntimeError(
                "SSINR evaluator could not fetch input. Has the device received data?"
            )

        # Power of the noise realization per receive channel
        noise_power = self._input.noise_realization.power

        # Power of the self-interference
        si_power = self._input.leaking_signal.power

        # Power of the useful signal
        # ToDo: The definition of impinging signals is inconsistent
        signal_power = np.zeros(self._input.impinging_signals[0].num_streams, dtype=np.float64)
        for signal in self._input.impinging_signals:
            signal_power += signal.power

        # SSINR
        ssinr = signal_power / (si_power + noise_power)

        return PowerEvaluation(ssinr)
