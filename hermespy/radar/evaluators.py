# -*- coding: utf-8 -*-
"""
================
Radar Evaluation
================

This module introduces several evaluators for performance indicators in radar detection.
Refer to the :doc:`PyMonte</api/core.monte_carlo>` documentation for a detailed introduction to the concept of
:class:`Evaluators<hermespy.core.monte_carlo.Evaluator>`.

.. autoclasstree:: hermespy.modem.evaluators
   :alt: Communication Evaluator Class Tree
   :strict:
   :namespace: hermespy

The implemented :class:`RadarEvaluator<.RadarEvaluator>` all inherit from the identically named common
base which gets initialized by selecting one :class:`Modem<hermespy.modem.modem.Modem>`, whose performance will be
evaluated and one :class:`RadarChannel<hermespy.channel.radar_channel.RadarChannel>` instance, containing the ground
truth.
The currently considered performance indicators are

========================================= ================================ ============================================================
Evaluator                                 Artifact                         Performance Indicator
========================================= ================================ ============================================================
:class:`.DetectionProbEvaluator`          :class:`.DetectionProbArtifact`  Probability of detecting the target at the right bin
:class:`.ReceiverOperatingCharacteristic` :class:`.RocArtifact`            Probability of detection versus probability of false alarm
:class`.RootMeanSquareError`              :class:`.RootMeanSquareArtifact` Root mean square error of point detections
========================================= ================================ ============================================================

Configuring :class:`RadarEvaluators<.RadarEvaluator>` to evaluate the radar detection of
:class:`Modem<hermespy.modem.modem.Modem>` instances is rather straightforward:

.. code-block:: python

   # Create two separate modem instances
   modem = Modem()
   channel = RadarChannel()

   # Create a radar evaluation as an evaluation example
   radar_evaluator = DetectionProbEvaluator(modem, channel)

   # Extract evaluation
   radar_evaluation = radar_evaluator.evaluate()

   # Visualize evaluation
   radar_evaluation.plot()

"""

from __future__ import annotations
from abc import ABC
from collections.abc import Sequence
from itertools import product
from typing import Type
from typing_extensions import override

import matplotlib.pyplot as plt
import numpy as np
from rich import get_console
from rich.console import Console
from rich.progress import track
from scipy.stats import uniform
from scipy.integrate import simpson

from hermespy.core import (
    Executable,
    Hook,
    PlotVisualization,
    ScatterVisualization,
    Scenario,
    ScenarioMode,
    Serializable,
    SerializationBackend,
    VAT,
    Evaluator,
    Evaluation,
    EvaluationResult,
    EvaluationTemplate,
    GridDimension,
    ArtifactTemplate,
    Artifact,
    ScalarEvaluationResult,
)
from hermespy.radar import Radar, RadarReception
from hermespy.channel.radar.radar import RadarChannelBase, RadarChannelSample
from hermespy.radar.cube import RadarCube
from hermespy.radar.detection import RadarPointCloud
from hermespy.simulation import (
    SimulatedDevice,
    ProcessedSimulatedDeviceInput,
    SimulatedDeviceOutput,
)


__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarEvaluator(Evaluator, ABC):
    """Bastract base class for evaluating sensing performance.

    Inherits from the abstract :class:`Evaluator<hermespy.core.monte_carlo.Evaluator>` base class.
    Expects the abstract method :meth:`evaluate<hermespy.core.monte_carlo.Evaluator.evaluate>` as well as the abstract properties
    :attr:`abbreviation<hermespy.core.monte_carlo.Evaluator.abbreviation>` and :attr:`title<hermespy.core.monte_carlo.Evaluator.title>` to be implemented.

    There are currently three different :class:`RadarEvaluators<.RadarEvaluator>` implemented:

    .. toctree::

        evaluators.DetectionProbEvaluator
        evaluators.ReceiverOperatingCharacteristic
        evaluators.RootMeanSquareError
    """

    __receiving_radar: Radar  # Handle to the radar receiver
    __receiving_device: SimulatedDevice
    __transmitting_device: SimulatedDevice
    __radar_channel: RadarChannelBase
    __reception: RadarReception | None
    __receive_hook: Hook[RadarReception]
    __channel_sample: RadarChannelSample | None

    def __init__(
        self,
        receiving_radar: Radar,
        transmitting_device: SimulatedDevice,
        receiving_device: SimulatedDevice,
        radar_channel: RadarChannelBase,
    ) -> None:
        """
        Args:

            receiving_radar:
                Radar detector to be evaluated.

            transmitting_device:
                Device transmitting into the evaluated channel.

            receiving_device:
                Device receiving from the evaluated channel.
                The `receiving_radar` must be a receive DSP algorithm of this device.

            radar_channel:
                The radar channel containing the ground truth to be evaluated.

        Raises:

            ValueError: If the receiving radar is not an operator of the radar_channel receiver.
        """

        # Initialize base class
        Evaluator.__init__(self)

        # Check if the receiving radar is a dsp algorithm of the receiving device
        if receiving_radar not in receiving_device.receivers:
            raise ValueError(
                "Receiving radar not registered as DSP algorithm of the receiving device"
            )

        # Initialize class attributes
        self.__transmitting_device = transmitting_device
        self.__receiving_device = receiving_device
        self.__receiving_radar = receiving_radar
        self.__radar_channel = radar_channel

        # Register receive callback
        self.__reception = None
        self.__receive_hook = receiving_radar.add_receive_callback(self.__receive_callback)

        # Register sample callback
        self.__channel_sample = None
        radar_channel.add_sample_hook(
            self.__sample_callback, self.__transmitting_device, self.__receiving_device
        )

    def __receive_callback(self, reception: RadarReception) -> None:
        """Callback for receiving radar receptions.

        Args:

            reception:
                Received radar reception.
        """

        self.__reception = reception

    def __sample_callback(self, sample: RadarChannelSample) -> None:
        """Callback for sampling radar channel realizations.

        Args:

            sample:
                Sampled radar channel realization.
        """

        self.__channel_sample = sample

    def _fetch_reception(self) -> RadarReception:
        """Fetch the most recent radar reception.

        Returns: The most recent radar reception.

        Raises:

            RuntimeError: If no reception is available.
        """

        if self.__reception is None:
            raise RuntimeError(
                "No reception available. Has the radar's receive method been called?"
            )

        return self.__reception

    def _fetch_pcl(self) -> RadarPointCloud:
        """Fetch the most recent radar point cloud.

        Returns: The most recent radar point cloud.

        Raises:
            RunmtineError: If no reception is available or the reception does not contain a point cloud.
        """

        reception = self._fetch_reception()
        if reception.cloud is None:
            raise RuntimeError(
                "No point cloud available. Has the radar's receive method been called?"
            )

        return reception.cloud

    def _fetch_channel(self) -> RadarChannelSample:
        """Fetch the most recent radar channel sample.

        Returns: The most recent radar channel sample.

        Raises:

            RuntimeError: If no channel sample is available.
        """

        if self.__channel_sample is None:
            raise RuntimeError("No channel sample available. Has the radar channel been sampled?")

        return self.__channel_sample

    @property
    def receiving_device(self) -> SimulatedDevice:
        """Device receiving from the evaluated channel."""

        return self.__receiving_device

    @property
    def transmitting_device(self) -> SimulatedDevice:
        """Device transmitting into the evaluated channel."""

        return self.__transmitting_device

    @property
    def receiving_radar(self) -> Radar:
        """Radar detector, the output of which is to be evaluated."""

        return self.__receiving_radar

    @property
    def radar_channel(self) -> RadarChannelBase:
        """The considered radar channel."""

        return self.__radar_channel

    def generate_result(
        self, grid: Sequence[GridDimension], artifacts: np.ndarray
    ) -> EvaluationResult:
        return ScalarEvaluationResult(grid, artifacts, self)

    def __del__(self) -> None:
        self.__receive_hook.remove()


class DetectionProbArtifact(ArtifactTemplate[bool]):
    """Artifacto of the probability of detection for a radar detector.

    Represents a boolean indicator of whether a target was detected or not.
    Generated by the :class:`DetectionProbabilityEvaluation<.DetectionProbabilityEvaluation>`'s :meth:`artifact()<DetectionProbabilityEvaluation.artifact>` method.
    """

    ...  # pragma: no cover


class DetectionProbabilityEvaluation(EvaluationTemplate[bool, ScatterVisualization]):
    """Evaluation of the probability of detection for a radar detector.

    Represents a boolean indicator of whether a target was detected or not.
    Generated by the :class:`DetectionProbEvaluator<.DetectionProbEvaluator>`'s :meth:`evaluate()<DetectionProbEvaluator.evaluate>` method.
    """

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> ScatterVisualization:  # pragma: no cover
        raise NotImplementedError("Detection probability evaluation does not support visualization")

    def _update_visualization(
        self, visualization: ScatterVisualization, **kwargs
    ) -> None:  # pragma: no cover
        # ToDo: Implement updating the single-item scatter plot
        raise NotImplementedError("Detection probability evaluation does not support visualization")

    def artifact(self) -> DetectionProbArtifact:
        return DetectionProbArtifact(self.evaluation)


class DetectionProbEvaluator(Evaluator, Serializable):
    """Estimates the probability of detection for a given radar detector.

    Assumes a successful detection if the :class:`Radar's<hermespy.radar.radar.Radar>` :class:`RadarReception<hermespy.radar.radar.RadarReception>` contains a non-empty point cloud.
    This is the case if the configured :class:`RadarDetector<hermespy.radar.detection.RadarDetector>` made a positive decision
    for any bin within the processed :class:`RadarCube<hermespy.radar.cube.RadarCube>`.

    A minimal example within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`
    evaluating the probability of detection for a single radar target illuminated by an :class:`FMCW<hermespy.radar.fmcw.FMCW>` radar would be:

    .. literalinclude:: ../../scripts/examples/radar_evaluators_DetectionProbEvaluator.py
       :language: python
       :linenos:
       :lines: 03-27
    """

    __radar: Radar
    __cloud: RadarPointCloud | None
    __hook: Hook[RadarReception]

    def __init__(self, radar: Radar) -> None:
        """
        Args:

            radar:
                Radar detector to be evaluated.
        """

        # Initialize base class
        Evaluator.__init__(self)
        self.plot_scale = "log"  # Plot logarithmically by default

        # Initialize class attributes
        self.__cloud = None
        self.__radar = radar
        self.__hook = radar.add_receive_callback(self.__receive_callback)

    def __receive_callback(self, reception: RadarReception) -> None:
        """Callback for receiving radar receptions.

        Args:

            reception:
                Received radar reception.
        """

        self.__cloud = reception.cloud

    @property
    def radar(self) -> Radar:
        """Radar detector to be evaluated."""

        return self.__radar

    @property
    def abbreviation(self) -> str:
        return "PD"

    @property
    def title(self) -> str:
        return "Probability of Detection Evaluation"

    @staticmethod
    def _scalar_cdf(scalar: float) -> float:
        return uniform.cdf(scalar)

    def generate_result(
        self, grid: Sequence[GridDimension], artifacts: np.ndarray
    ) -> ScalarEvaluationResult:
        return ScalarEvaluationResult.From_Artifacts(grid, artifacts, self)

    def evaluate(self) -> DetectionProbabilityEvaluation:
        if self.__cloud is None:
            raise RuntimeError(
                "Detection evaluation requires a detector to be configured at the radar"
            )

        # Verify if a target is detected in any bin
        detection = self.__cloud.num_points > 0
        return DetectionProbabilityEvaluation(detection)

    def __del__(self) -> None:
        self.__hook.remove()


class RocArtifact(Artifact):
    """Artifact of receiver operating characteristics (ROC) evaluation"""

    __h0_value: float
    __h1_value: float

    def __init__(self, h0_value: float, h1_value: float) -> None:
        """
        Args:

            h0_value:
                Measured value for null-hypothesis (H0), i.e., noise only

            h1_value:
                Measured value for alternative hypothesis (H1)

        """

        self.__h0_value = h0_value
        self.__h1_value = h1_value

    def __str__(self) -> str:
        return f"({self.__h0_value:.4}, {self.__h1_value:.4})"

    @property
    def h0_value(self) -> float:
        return self.__h0_value

    @property
    def h1_value(self) -> float:
        return self.__h1_value

    def to_scalar(self) -> None:
        return None


class RocEvaluation(Evaluation[PlotVisualization]):
    """Evaluation of receiver operating characteristics (ROC)"""

    __cube_h0: RadarCube
    __cube_h1: RadarCube

    def __init__(self, cube_h0: RadarCube, cube_h1: RadarCube) -> None:
        """
        Args:
            cube_h0: H0 hypothesis radar cube.
            cube_h1: H1 hypothesis radar cube.
        """

        # Initialize base class
        super().__init__()

        # Initialize class attributes
        self.__cube_h0 = cube_h0
        self.__cube_h1 = cube_h1

    @property
    def cube_h0(self) -> RadarCube:
        """H0 hypothesis radar cube."""

        return self.__cube_h0

    @property
    def cube_h1(self) -> RadarCube:
        """H1 hypothesis radar cube."""

        return self.__cube_h1

    def artifact(self) -> RocArtifact:
        h0_value = self.__cube_h0.data.max()
        h1_value = self.__cube_h1.data.max()

        return RocArtifact(h0_value, h1_value)

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> PlotVisualization:
        lines = np.empty_like(axes, dtype=np.object_)
        h0_lines = axes[0, 0].plot(
            self.__cube_h0.range_bins, np.zeros_like(self.__cube_h0.range_bins)
        )
        h1_lines = axes[0, 0].plot(
            self.__cube_h1.range_bins, np.zeros_like(self.__cube_h1.range_bins)
        )
        lines[0, 0] = h0_lines + h1_lines

        return PlotVisualization(figure, axes, lines)

    def _update_visualization(self, visualization: PlotVisualization, **kwargs) -> None:
        _lines = visualization.lines[0, 0]
        _lines[0].set_ydata(self.__cube_h0.data)
        _lines[1].set_ydata(self.__cube_h1.data)


class RocEvaluationResult(EvaluationResult):
    """Final result of an receive operating characteristcs evaluation."""

    __detection_probabilities: np.ndarray
    __false_alarm_probabilities: np.ndarray

    def __init__(
        self,
        detection_probabilities: np.ndarray,
        false_alarm_probabilities: np.ndarray,
        grid: Sequence[GridDimension],
        evaluator: ReceiverOperatingCharacteristic | None = None,
    ) -> None:
        """
        Args:

            detection_probabilities:
                Detection probabilities for each grid point.

            false_alarm_probabilities:
                False alarm probabilities for each grid point.

            grid:
                Grid dimensions of the evaluation result.

            evaluator:
                Evaluator that generated the evaluation result.
        """

        # Initialize base class
        EvaluationResult.__init__(self, grid, evaluator)

        # Initialize class attributes
        self.__detection_probabilities = detection_probabilities
        self.__false_alarm_probabilities = false_alarm_probabilities

    @override
    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> PlotVisualization:
        ax: plt.Axes = axes.flat[0]

        # Configure axes labels
        ax.set_xlabel("False Alarm Probability")
        ax.set_ylabel("Detection Probability")

        # Configure axes limits
        ax.set_xlim(0.0, 1.1)
        ax.set_ylim(0.0, 1.1)

        line_list = []
        section_magnitudes = tuple(s.num_sample_points for s in self.grid)
        for section_indices in np.ndindex(section_magnitudes):
            # Generate the graph line label
            line_label = ""
            for i, v in enumerate(section_indices):
                line_label += f"{self.grid[i].title} = {self.grid[i].sample_points[v].title}, "
            line_label = line_label[:-2]
            line_list.extend(ax.plot([], [], label=line_label))

        # Only plot the legend for an existing sweep grid.
        if len(self.grid) > 0:
            with Executable.style_context():
                ax.legend()

        lines = np.empty_like(axes, dtype=np.object_)
        lines[0, 0] = line_list
        return PlotVisualization(figure, axes, lines)

    @override
    def _update_visualization(self, visualization: PlotVisualization, **kwargs) -> None:
        section_magnitudes = tuple(s.num_sample_points for s in self.grid)
        for section_indices, line in zip(np.ndindex(section_magnitudes), visualization.lines[0, 0]):
            # Select the graph line scalars
            x_axis = self.__false_alarm_probabilities[section_indices]
            y_axis = self.__detection_probabilities[section_indices]

            # Update the respective line
            line.set_data(x_axis, y_axis)

    @override
    def to_array(self) -> np.ndarray:
        return np.stack((self.__detection_probabilities, self.__false_alarm_probabilities), axis=-1)

    @override
    def to_str(self, grid_coordinates: Sequence[int]) -> str:
        false_alarm_probabilities = self.__false_alarm_probabilities[grid_coordinates]
        detection_probabilities = self.__detection_probabilities[grid_coordinates]

        integration = 2 * np.abs(simpson(detection_probabilities, false_alarm_probabilities)) - 1.0
        return f"{int(np.round(100*integration))}%"


class ReceiverOperatingCharacteristic(RadarEvaluator, Serializable):
    """Evaluate the receiver operating characteristics for a radar operator.

    The receiver operating characteristics (ROC) curve is a graphical plot that illustrates the performance of a detector
    by visualizing the probability of false alarm versus the probability of detection for a given parameterization.

    A minimal example within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`
    evaluating the probability of detection for a single radar target illuminated by an :class:`FMCW<hermespy.radar.fmcw.FMCW>` radar would be:

    .. literalinclude:: ../../scripts/examples/radar_evaluators_ReceiverOperatingCharacteristic.py
       :language: python
       :linenos:
       :lines: 03-23
    """

    _title = "Receiver Operating Characteristics"
    __num_thresholds: int
    __output_hook: Hook[SimulatedDeviceOutput]
    __input_hook: Hook[ProcessedSimulatedDeviceInput]
    __output: SimulatedDeviceOutput | None
    __input: ProcessedSimulatedDeviceInput | None

    def __init__(
        self,
        receiving_radar: Radar,
        transmitting_device: SimulatedDevice,
        receiving_device: SimulatedDevice,
        radar_channel: RadarChannelBase,
        num_thresholds=101,
    ) -> None:
        """
        Args:

            radar:
                Radar under test.

            radar_channel:
                Radar channel containing a desired target.

            num_thresholds:
                Number of different thresholds to be considered in ROC curve
        """

        # Initialize base class
        RadarEvaluator.__init__(
            self, receiving_radar, transmitting_device, receiving_device, radar_channel
        )

        # Initialize class attributes
        self.__num_thresholds = num_thresholds
        self.__input_hook = receiving_device.simulated_input_callbacks.add_callback(
            self.__input_callback
        )
        self.__output_hook = transmitting_device.simulated_output_callbacks.add_callback(
            self.__output_callback
        )
        self.__input = None
        self.__output = None

    def __input_callback(self, input: ProcessedSimulatedDeviceInput) -> None:
        """Callback notificying the evaluator of newly generated device inputs.

        Args:

            input:
                Newly generated device input.
        """

        self.__input = input

    def __output_callback(self, output: SimulatedDeviceOutput) -> None:
        """Callback notificying the evaluator of newly generated device outputs.

        Args:

            output:
                Newly generated device output.
        """

        self.__output = output

    @staticmethod
    def __evaluation_from_receptions(
        h0_reception: RadarReception, h1_reception: RadarReception
    ) -> RocEvaluation:
        """Subroutine to generate an evaluation given two hypothesis radar receptions.

        Args:

            h0_reception:
                Reception missing the target of interest.

            h1_reception:
                Reception containing the target of interest.

        Returns: An initialized :class:`RocEvaluation`.
        """

        # Retrieve radar cubes for both hypothesis
        radar_cube_h0 = h0_reception.cube
        radar_cube_h1 = h1_reception.cube

        # Return resulting evaluation
        return RocEvaluation(radar_cube_h0, radar_cube_h1)

    def evaluate(self) -> RocEvaluation:
        reception = self._fetch_reception()
        channel_sample = self._fetch_channel()

        if self.__output is None:
            raise RuntimeError("No device output available")

        if self.__input is None:
            raise RuntimeError("No device input available")

        # Collect required information from the simulation
        one_hypothesis_sample = channel_sample
        device_index = self.radar_channel.scenario.device_index(self.transmitting_device)
        operator_index = self.receiving_device.receivers.operator_index(self.receiving_radar)

        # Generate the null hypothesis detection radar cube by re-running the radar detection routine
        null_hypothesis_sample = one_hypothesis_sample.null_hypothesis()

        # Propagate again over the radar channel
        null_hypothesis_propagation = null_hypothesis_sample.propagate(self.__output)

        # Exchange the respective propagated signal
        impinging_signals = list(self.__input.impinging_signals).copy()
        impinging_signals[device_index] = null_hypothesis_propagation

        # Receive again
        devic_state = self.receiving_device.state(channel_sample.time)
        null_hypothesis_device_reception = self.receiving_device.process_from_realization(
            impinging_signals,
            self.__input,
            self.__output.trigger_realization,
            self.__input.leaking_signal,
            devic_state,
        )
        null_hypothesis_radar_reception = self.receiving_radar.receive(
            null_hypothesis_device_reception.operator_inputs[operator_index], devic_state, False
        )

        # Generate evaluation
        return self.__evaluation_from_receptions(null_hypothesis_radar_reception, reception)

    @property
    def abbreviation(self) -> str:
        return "ROC"  # pragma: no cover

    @property
    def title(self) -> str:
        return ReceiverOperatingCharacteristic._title  # pragma: no cover

    @staticmethod
    def GenerateResult(
        grid: Sequence[GridDimension],
        artifacts: np.ndarray,
        num_thresholds: int = 101,
        evaluator: ReceiverOperatingCharacteristic | None = None,
    ) -> RocEvaluationResult:
        """Generate a new receiver operating characteristics evaluation result.

        Args:

            grid:
                Grid dimensions of the evaluation result.

            artifacts:
                Artifacts of the evaluation result.

            num_thresholds:
                Number of different thresholds to be considered in ROC curve
                101 by default.

            evaluator:
                Evaluator that generated the evaluation result.

        Returns: The generated result.
        """

        # Prepare result containers
        if len(grid) > 0:
            dimensions = tuple(g.num_sample_points for g in grid)
        else:
            dimensions = (1,)
            artifacts = artifacts.reshape(dimensions)

        detection_probabilities = np.empty((*dimensions, num_thresholds), dtype=float)
        false_alarm_probabilities = np.empty((*dimensions, num_thresholds), dtype=float)

        # Convert artifacts to raw data array
        for grid_coordinates in np.ndindex(dimensions):
            artifact_line = artifacts[grid_coordinates]
            roc_data = np.array([[a.h0_value, a.h1_value] for a in artifact_line])

            for t, threshold in enumerate(
                np.linspace(roc_data.min(), roc_data.max(), num_thresholds, endpoint=True)
            ):
                threshold_coordinates = grid_coordinates + (t,)
                detection_probabilities[threshold_coordinates] = np.mean(
                    roc_data[:, 1] >= threshold
                )
                false_alarm_probabilities[threshold_coordinates] = np.mean(
                    roc_data[:, 0] >= threshold
                )

        return RocEvaluationResult(
            detection_probabilities, false_alarm_probabilities, grid, evaluator
        )

    def generate_result(
        self, grid: Sequence[GridDimension], artifacts: np.ndarray
    ) -> RocEvaluationResult:
        """Generate a new receiver operating characteristics evaluation result.

        Args:

            grid:
                Grid dimensions of the evaluation result.

            artifacts:
                Artifacts of the evaluation result.

        Returns: The generated result.
        """

        return self.GenerateResult(grid, artifacts, self.__num_thresholds, self)

    @staticmethod
    def FromScenarios(
        h0_scenario: Scenario,
        h1_scenario: Scenario,
        num_drops: int,
        h0_operator: Radar | None = None,
        h1_operator: Radar | None = None,
        num_thresholds: int = 101,
    ) -> RocEvaluationResult:
        """Compute an ROC evaluation result from two scenarios.

        Args:

            h0_scenario:
                Scenario of the null hypothesis.

            h1_scenario:
                Scenario of the alternative hypothesis.

            num_drops:
                Number of drops to be considered in the evaluation.
                The more drops, the smoother the estimated ROC curve will be.

            h0_operator:
                Radar operator of the null hypothesis.
                If not provided, the first radar operator of the null hypothesis scenario will be used.

            h1_operator:
                Radar operator of the alternative hypothesis.
                If not provided, the first radar operator of the alternative hypothesis scenario will be used.

            num_thresholds:
                Number of different thresholds to be considered in ROC curve

        Returns: The ROC evaluation result.

        Raises:

            ValueError:
                - If the number of drops is less than one
                - If the operators are not registered within the scenarios
                - If, for any reason, the operated devices cannot be determined
        """

        # There should be at least a single drop available
        if num_drops < 1:
            raise ValueError("At least one drop is required for the ROC evaluation")

        # Select operators if none were provided
        if h0_operator:
            if h0_operator not in h0_scenario.operators:
                raise ValueError(
                    "Null hypthesis radar not an operator within the null hypothesis scenario"
                )

        else:
            if h0_scenario.num_operators < 1:
                raise ValueError("Null hypothesis radar has no registered operators")

            inferred_h0_operator = None
            for operator in h0_scenario.operators:
                if isinstance(operator, Radar):
                    inferred_h0_operator = operator
                    break

            if inferred_h0_operator is None:
                raise ValueError("Could not infer radar operator from list of operators")

            h0_operator = inferred_h0_operator

        if h1_operator:
            if h1_operator not in h1_scenario.operators:
                raise ValueError(
                    "One hypthesis radar not an operator within the null hypothesis scenario"
                )

        else:
            if h1_scenario.num_operators < 1:
                raise ValueError("One hypothesis radar has no registered operators")

            inferred_h1_operator = None
            for operator in h1_scenario.operators:
                if isinstance(operator, Radar):
                    inferred_h1_operator = operator
                    break

            if inferred_h1_operator is None:
                raise ValueError("Could not infer radar operator from list of operators")

            h1_operator = inferred_h1_operator

        # Find the indices of the operators within the scenarios
        h0_device_index = -1
        h1_device_index = -1
        h0_operator_index = -1
        h1_operator_index = -1
        for d, device in enumerate(h0_scenario.devices):
            if h0_operator in device.receivers:
                h0_device_index = d
                h0_operator_index = device.receivers.index(h0_operator)
        for d, device in enumerate(h1_scenario.devices):
            if h1_operator in device.receivers:
                h1_device_index = d
                h1_operator_index = device.receivers.index(h1_operator)

        if (
            h0_device_index < 0
            or h1_device_index < 0
            or h0_operator_index < 0
            or h1_operator_index < 0
        ):
            raise ValueError(
                "Could not detect devices and operators within the provided scenarios."
            )

        # Collect artifacts
        artifacts = np.empty(1, dtype=object)
        artifacts[0] = []
        for _ in range(num_drops):
            h0_drop = h0_scenario.drop()
            h1_drop = h1_scenario.drop()
            h0_reception = h0_drop.device_receptions[h0_device_index].operator_receptions[
                h0_operator_index
            ]
            h1_reception = h1_drop.device_receptions[h1_device_index].operator_receptions[
                h1_operator_index
            ]

            evaluation = ReceiverOperatingCharacteristic.__evaluation_from_receptions(
                h0_reception, h1_reception
            )
            artifacts[0].append(evaluation.artifact())

        # Generate results
        grid: Sequence[GridDimension] = []
        result = ReceiverOperatingCharacteristic.GenerateResult(grid, artifacts, num_thresholds)
        return result

    @staticmethod
    def FromScenario(
        scenario: Scenario,
        operator: Radar,
        h0_campaign: str = "h0",
        h1_campaign: str = "h1",
        drop_offset: int = 0,
        num_drops: int = 0,
        backend: SerializationBackend = SerializationBackend.HDF,
        console: Console | None = None,
    ) -> RocEvaluationResult:
        """Extract an ROC evaluation from an existing scenario.

        Args:
            scenario: Scenario from which to extract the ROC from.
            operator: Radar operator to be evaluated.
            h0_campaign: Campaign identifier of the null hypothesis measurements.
            h1_campaign: Campaign identifier of the alternative hypothesis measurements.
            drop_offset: Index of the first drop to be replayed.
            num_drops: Number of drops to be replayed.
            backend: Serialization backend to be used for the evaluation.
            console: Rich console to be used for progress tracking.

        Raises:
            ValueError: If the scenario is not in replay mode.

        Returns:
            The ROC evaluation result.
        """

        _console = get_console() if console is None else console

        # Check if the scenario is in replay mode
        if scenario.mode != ScenarioMode.REPLAY:
            raise ValueError("Scenario is not in replay mode")

        null_receptions: list[RadarReception] = []
        alt_receptions: list[RadarReception] = []

        # Find the operator index within the scenario
        device_index = -1
        operator_index = -1
        for d, device in enumerate(scenario.devices):
            if operator in device.receivers:
                operator_index = device.receivers.index(operator)
                device_index = d

        if device_index < 0 or operator_index < 0:
            raise ValueError("Could not detect device and operator within the provided scenario.")

        # Collect receptions for both hypothesis
        for campaign, hypothesis, receptions in zip(
            [h0_campaign, h1_campaign],
            ["null hypothesis", "alt hypothesis"],
            [null_receptions, alt_receptions],
        ):
            num_available_drops = scenario.replay(
                campaign=campaign, drop_offset=drop_offset, backend=backend
            )
            num_replayed_drops = min(num_drops, num_available_drops)
            for _ in track(
                range(num_replayed_drops), description="Replaying " + hypothesis, console=_console
            ):
                drop = scenario.drop()
                receptions.append(
                    drop.device_receptions[device_index].operator_receptions[operator_index]
                )

        # Generate the evaluation result
        grid: Sequence[GridDimension] = []
        artifacts = np.empty(1, dtype=object)
        artifacts[0] = [
            ReceiverOperatingCharacteristic.__evaluation_from_receptions(h0, h1).artifact()
            for h0, h1 in zip(null_receptions, alt_receptions)
        ]
        return ReceiverOperatingCharacteristic.GenerateResult(grid, artifacts)

    @classmethod
    def FromFile(
        cls: Type[ReceiverOperatingCharacteristic],
        file: str,
        h0_campaign="h0_measurements",
        h1_campaign="h1_measurements",
        num_thresholds: int = 101,
        drop_offset: int = 0,
        num_drops: int | None = None,
        backend: SerializationBackend = SerializationBackend.HDF,
    ) -> RocEvaluationResult:
        """Compute an ROC evaluation result from a savefile.

        Args:

            file:
                Savefile containing the measurements.
                Either as file system location or h5py `File` handle.

            h0_campaign:
                Campaign identifier of the h0 hypothesis measurements.
                By default, `h0_measurements` is assumed.

            h1_campaign:
                Campaign identifier of the h1 hypothesis measurements.
                By default, `h1_measurements` is assumed.

            num_thresholds:
                Number of different thresholds to be considered in ROC curve
                By default, 101 is assumed.

            drop_offset:
                Index of the first drop to be replayed.

            num_drops:
                Number of drops to be replayed.
                If not provided, all available drops will be replayed.

            backend:
                Serialization backend to be used for the evaluation.
                By default, `HDF` is assumed.

        Returns: The ROC evaluation result.
        """

        # Load scenarios from the savefile
        h0_scenario, h0_num_drops = Scenario.Replay(file, h0_campaign, drop_offset, backend)
        h1_scenario, h1_num_drops = Scenario.Replay(file, h1_campaign, drop_offset, backend)

        # Ensure that the number of drops is the same for both scenarios
        _num_drops = min(h0_num_drops, h1_num_drops)
        if num_drops is not None:
            _num_drops = min(num_drops, _num_drops)

        # Resort to the from scenarios routine for computing the evaluation result
        result = cls.FromScenarios(
            h0_scenario, h1_scenario, _num_drops, num_thresholds=num_thresholds
        )

        # Close the scenarios properly
        h0_scenario.stop()
        h1_scenario.stop()

        return result

    def __del__(self) -> None:
        self.__input_hook.remove()
        self.__output_hook.remove()


class RootMeanSquareArtifact(Artifact):
    """Artifact of a root mean square evaluation"""

    __num_errors: int
    __cummulation: float

    def __init__(self, num_errors: int, cummulation: float) -> None:
        """
        Args:

            num_errors:
                Number of errros.

            cummulation:
                Sum of squared errors distances.
        """

        self.__num_errors = num_errors
        self.__cummulation = cummulation

    def to_scalar(self) -> float:
        return np.sqrt(self.cummulation / self.num_errors)

    def __str__(self) -> str:
        return f"{self.to_scalar():4.0f}"

    @property
    def num_errors(self) -> int:
        """Number of cummulated errors"""

        return self.__num_errors

    @property
    def cummulation(self) -> float:
        """Cummulated squared error"""

        return self.__cummulation


class RootMeanSquareEvaluation(Evaluation[ScatterVisualization]):
    """Result of a single root mean squre evaluation."""

    __pcl: RadarPointCloud
    __ground_truth: np.ndarray

    def __init__(self, pcl: RadarPointCloud, ground_truth: np.ndarray) -> None:
        self.__pcl = pcl
        self.__ground_truth = ground_truth

    def artifact(self) -> RootMeanSquareArtifact:
        num_errors = self.__pcl.num_points * self.__ground_truth.shape[0]
        cummulative_square_error = 0.0

        for point, truth in product(self.__pcl.points, self.__ground_truth):
            cummulative_square_error += float(np.linalg.norm(point.position - truth)) ** 2

        return RootMeanSquareArtifact(num_errors, cummulative_square_error)

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> ScatterVisualization:
        return self.__pcl._prepare_visualization(figure, axes, **kwargs)

    def _update_visualization(self, visualization: ScatterVisualization, **kwargs) -> None:
        self.__pcl._update_visualization(visualization, **kwargs)


class RootMeanSquareErrorResult(ScalarEvaluationResult):
    """Result of a root mean square error evaluation."""

    ...  # pragma: no cover


class RootMeanSquareError(RadarEvaluator):
    """Root mean square error of estimated points to ground truth.

    Root mean square error (RMSE) is a widely used metric for evaluating the performance of a radar detector.
    It estimates the average distance between the estimated and the ground truth position of a target.

    A minimal example within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`
    evaluating the probability of detection for a single radar target illuminated by an :class:`FMCW<hermespy.radar.fmcw.FMCW>` radar would be:

    .. literalinclude:: ../../scripts/examples/radar_evaluators_RootMeanSquareError.py
       :language: python
       :linenos:
       :lines: 03-29
    """

    def evaluate(self) -> Evaluation:
        point_cloud = self._fetch_pcl()
        channel_sample = self._fetch_channel()

        # Consolide the ground truth
        ground_truth = np.array(
            [p.ground_truth[0] for p in channel_sample.paths if p.ground_truth is not None]
        )
        return RootMeanSquareEvaluation(point_cloud, ground_truth)

    @property
    def title(self) -> str:
        return "Root Mean Square Error"

    @property
    def abbreviation(self) -> str:
        return "RMSE"

    def generate_result(
        self, grid: Sequence[GridDimension], artifacts: np.ndarray
    ) -> RootMeanSquareErrorResult:
        rmse_scalar_results = np.empty(artifacts.shape, dtype=float)
        for coordinates, section_artifacts in np.ndenumerate(artifacts):
            cummulative_errors = 0.0
            error_count = 0

            artifact: RootMeanSquareArtifact
            for artifact in section_artifacts:
                cummulative_errors += artifact.cummulation
                error_count += artifact.num_errors

            rmse = np.sqrt(cummulative_errors / error_count)
            rmse_scalar_results[coordinates] = rmse

        return RootMeanSquareErrorResult(grid, rmse_scalar_results, self)
