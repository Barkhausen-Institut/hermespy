# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Sequence
from typing_extensions import override

import matplotlib.pyplot as plt
import numpy as np

from hermespy.beamforming import BeamFocus
from hermespy.core import (
    AntennaMode,
    DeserializationProcess,
    Evaluator,
    Evaluation,
    GridDimension,
    Hook,
    ScalarEvaluationResult,
    Serializable,
    SerializationProcess,
    ArtifactTemplate,
    PlotVisualization,
    VAT,
)
from ..simulated_device import ProcessedSimulatedDeviceInput, SimulatedDevice, SimulatedDeviceOutput

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SidelobeEvaluation(Evaluation[PlotVisualization]):
    """Evaluation of the side lobe power."""

    __powers: np.ndarray
    __focus_power: float

    def __init__(self, powers: np.ndarray, focus_power: float) -> None:
        """
        Args:
            powers: The directional powers of the codebook.
            focus_power: Power of the desired beamforming direction.
        """

        self.__powers = powers
        self.__focus_power = focus_power

    @override
    def artifact(self) -> ArtifactTemplate[float]:
        normalized_power = self.__focus_power / np.sum(self.__powers)
        return ArtifactTemplate(normalized_power)

    @override
    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> PlotVisualization:
        _ax = axes[0, 0]
        _ax.set_title("Sidelobe Level")
        _ax.set_xlabel("Codebook Index")
        _ax.set_ylabel("Sidelobe Power")

        lines = np.empty_like(axes, dtype=object)
        lines[0, 0] = _ax.semilogy(np.arange(len(self.__powers)), self.__powers, **kwargs)

        return PlotVisualization(figure, axes, lines)

    @override
    def _update_visualization(self, visualization, **kwargs):
        lines = visualization.lines[0, 0][0]
        lines.set_ydata(self.__powers)


class SidelobeEvaluator(Evaluator, Serializable):
    """Evaluate a simulated device's sidelobe power."""

    __device: SimulatedDevice
    __codebook: np.ndarray
    __desired_focus: BeamFocus
    __mode: AntennaMode
    __plot_surface: bool
    __input_hook: Hook[ProcessedSimulatedDeviceInput]
    __output_hook: Hook[SimulatedDeviceOutput]
    __input: ProcessedSimulatedDeviceInput | None
    __output: SimulatedDeviceOutput | None

    def __init__(
        self,
        device: SimulatedDevice,
        mode: AntennaMode,
        desired_focus: BeamFocus,
        codebook: np.ndarray | None = None,
        plot_surface: bool = False,
    ) -> None:
        """
        Args:

            device: The device to evaluate.
            mode:
                The duplex mode to evaluate.
                If :attr:`TX<hermespy.core.antennas.AntennaMode.TX>` then only the `device`'s tranmsit beamformer is evaluated,
                if :attr:`RX<hermespy.core.antennas.AntennaMode.RX>` then only the `device`'s receive beamformer is evaluated.
            desired_focus:
                The desired beamforming direction to evaluate against the codebook.
            codebook:
                Beamforming directions to evaluate against the desired beamforming direction.
                If :py:obj:`None`, the codebook is generated by sampling the device's positive z half-space with a resolution of twice the number of antennas.
            plot_surface: Visuale the final result as a surface plot if possible.
        """

        # Initialize the base class
        Evaluator.__init__(self)
        self.plot_scale = "log"
        self.__plot_surface = plot_surface

        # Register input hook
        self.__device = device
        self.__input_hook = device.simulated_input_callbacks.add_callback(self.__input_callback)
        self.__output_hook = device.simulated_output_callbacks.add_callback(self.__output_callback)

        # Initialize the codebook if not provided
        if codebook is None:

            num_antennas = (
                device.antennas.num_transmit_antennas
                if mode is AntennaMode.TX
                else device.antennas.num_receive_antennas
            )

            azimuth_candidates = np.linspace(0, 2 * np.pi, 2 * num_antennas, endpoint=False)
            zenith_candidates = np.linspace(0, np.pi, num_antennas, endpoint=True)[1:]

            self.__codebook = np.empty(
                (1 + azimuth_candidates.size * zenith_candidates.size, num_antennas), np.complex128
            )
            self.__codebook[0, :] = 1.0

            av, zv = np.meshgrid(azimuth_candidates, zenith_candidates)
            for i, (azimuth, zenith) in enumerate(zip(av.flat, zv.flat)):
                self.__codebook[1 + i, :] = device.antennas.spherical_phase_response(
                    device.carrier_frequency, azimuth, zenith, mode
                )

        else:
            self.__codebook = codebook

        # Initialize remaining attributes
        self.__mode = mode
        self.__desired_focus = desired_focus
        self.__input = None
        self.__output = None

    def __input_callback(self, input: ProcessedSimulatedDeviceInput) -> None:
        """ "Callback function notifying the evaluator of a new input."""

        self.__input = input

    def __output_callback(self, output: SimulatedDeviceOutput) -> None:
        """ "Callback function notifying the evaluator of a new output."""

        self.__output = output

    @property
    @override
    def abbreviation(self) -> str:
        return "SLL"

    @property
    @override
    def title(self) -> str:
        return "Sidelobe Level"

    @property
    def mode(self) -> AntennaMode:
        """The duplex mode to evaluate."""

        return self.__mode

    @mode.setter
    def mode(self, value: AntennaMode) -> None:
        self.__mode = value

    @property
    def desired_focus(self) -> BeamFocus:
        """The desired beamforming direction."""

        return self.__desired_focus

    @desired_focus.setter
    def desired_focus(self, value: BeamFocus) -> None:
        self.__desired_focus = value

    @override
    def evaluate(self) -> SidelobeEvaluation:
        samples: np.ndarray

        if self.mode is AntennaMode.TX:
            if self.__output is None:
                raise RuntimeError(
                    "Sidelobe evaluator could not fetch output. Has the device transmitted data?"
                )

            samples = self.__output.mixed_signal.getitem()

        elif self.mode is AntennaMode.RX:
            if self.__input is None:
                raise RuntimeError(
                    "Sidelobe evaluator could not fetch input. Has the device received data?"
                )

            samples = self.__input.baseband_signal.getitem()

        else:
            raise RuntimeError("Invalid duplex mode configuration in SidelobeEvaluator")

        # Get the device state
        # ToDo: There should be a callback interface to cache the device's runtime state
        # Alternatively, the device state could be added to the device input/output objects
        device_state = self.__device.state()
        focus_direction = self.desired_focus.spherical_angles(device_state)
        focus_beamformer = device_state.antennas.spherical_phase_response(
            device_state.carrier_frequency, focus_direction[0], focus_direction[1], self.mode
        )[np.newaxis, :]

        beamformed_power = (
            np.linalg.norm(
                np.concatenate([focus_beamformer, self.__codebook], axis=0) @ samples, axis=1
            )
            ** 2
            / np.linalg.norm(samples) ** 2
        )

        return SidelobeEvaluation(beamformed_power[1:], beamformed_power[0])

    @override
    def generate_result(
        self, grid: Sequence[GridDimension], artifacts: np.ndarray
    ) -> ScalarEvaluationResult:
        return ScalarEvaluationResult.From_Artifacts(grid, artifacts, self, self.__plot_surface)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.__device, "device")
        process.serialize_array(self.__codebook, "codebook")
        process.serialize_object(self.mode, "mode")
        process.serialize_object(self.desired_focus, "desired_focus")

    @classmethod
    @override
    def Deserialize(
        cls: type[SidelobeEvaluator], process: DeserializationProcess
    ) -> SidelobeEvaluator:
        return SidelobeEvaluator(
            process.deserialize_object("device", SimulatedDevice),
            process.deserialize_object("mode", AntennaMode),
            process.deserialize_object("desired_focus", BeamFocus),
            process.deserialize_array("codebook", np.complex128, None),
        )

    def __del__(self) -> None:
        self.__input_hook.remove()
        self.__output_hook.remove()
