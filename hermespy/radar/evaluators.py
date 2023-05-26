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
from typing import Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from scipy.stats import uniform

from hermespy.core import ReplayScenario, Scenario, ScenarioMode, Serializable
from hermespy.core.monte_carlo import Evaluator, Evaluation, EvaluationResult, EvaluationTemplate, GridDimension, ArtifactTemplate, Artifact, ScalarEvaluationResult
from hermespy.radar import Radar, RadarReception
from hermespy.channel import RadarChannelBase
from hermespy.radar.cube import RadarCube
from hermespy.radar.detection import RadarPointCloud


__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarEvaluator(Evaluator, ABC):
    """Base class for evaluating sensing performance."""

    __receiving_radar: Radar  # Handle to the radar receiver
    __radar_channel: RadarChannelBase  # Handle to the radar channel

    def __init__(self, receiving_radar: Radar, radar_channel: RadarChannelBase | None = None) -> None:
        """
        Args:

            receiving_radar (Radar): nRadar under test.
            radar_channel (RadarChannelBase): Radar channel modeling a desired target.

        Raises:

            ValueError: If the receiving radar is not an operator of the radar_channel receiver.
        """

        if radar_channel is not None and receiving_radar not in radar_channel.receiver.receivers:
            raise ValueError("The radar operator is not a receiver within the radar channel receiving device")

        self.__receiving_radar = receiving_radar
        self.__radar_channel = radar_channel

        # Initialize base class
        Evaluator.__init__(self)

    @property
    def receiving_radar(self) -> Radar:
        """Radar detector with target present.

        Returns:
            Modem: Handle to the receiving radar, when target is present.
        """

        return self.__receiving_radar

    @property
    def radar_channel(self) -> RadarChannelBase:
        """The considered radar channel."""

        return self.__radar_channel

    def generate_result(self, grid: Sequence[GridDimension], artifacts: np.ndarray) -> EvaluationResult:
        return ScalarEvaluationResult(grid, artifacts, self)


class DetectionProbArtifact(ArtifactTemplate[bool]):
    """Artifact of a detection probability evaluation for a radar detector."""

    ...  # pragma: no cover


class DetectionProbabilityEvaluation(EvaluationTemplate[bool]):
    def artifact(self) -> DetectionProbArtifact:
        return DetectionProbArtifact(self.evaluation)


class DetectionProbEvaluator(RadarEvaluator, Serializable):
    """Evaluate detection probability at a radar detector, considering any bin, i.e., detection is considered if any
    bin in the radar cube is above the threshold"""

    yaml_tag = "DetectionProbEvaluator"
    """YAML serialization tag"""

    def __init__(self, receiving_radar: Radar) -> None:
        """
        Args:
            receiving_radar (Radar):
                Radar detector
        """

        RadarEvaluator.__init__(self, receiving_radar)
        self.plot_scale = "log"  # Plot logarithmically by default

    @property
    def abbreviation(self) -> str:
        return "PD"

    @property
    def title(self) -> str:
        return "Probability of Detection Evaluation"

    @staticmethod
    def _scalar_cdf(scalar: float) -> float:
        return uniform.cdf(scalar)

    def generate_result(self, grid: Sequence[GridDimension], artifacts: np.ndarray) -> ScalarEvaluationResult:
        return ScalarEvaluationResult(grid, artifacts, self)

    def evaluate(self) -> DetectionProbabilityEvaluation:
        # Retrieve transmitted and received bits
        cloud = self.receiving_radar.reception.cloud

        if cloud is None:
            raise RuntimeError("Detection evaluation requires a detector to be configured at the radar")

        # Verify if a target is detected in any bin
        detection = cloud.num_points > 0
        return DetectionProbabilityEvaluation(detection)


class RocArtifact(Artifact):
    """Artifact of receiver operating characteristics (ROC) evaluation"""

    __h0_value: float
    __h1_value: float

    def __init__(self, h0_value: float, h1_value: float) -> None:
        """
        Args:

            h0_value (float):
                Measured value for null-hypothesis (H0), i.e., noise only

            h1_value (float):
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


class RocEvaluation(Evaluation):
    """Evaluation of receiver operating characteristics (ROC)"""

    data_h0: np.ndarray
    data_h1: np.ndarray

    def __init__(self, cube_h0: RadarCube, cube_h1: RadarCube) -> None:
        self.data_h0 = cube_h0.data
        self.data_h1 = cube_h1.data

    def artifact(self) -> RocArtifact:
        h0_value = self.data_h0.max()
        h1_value = self.data_h1.max()

        return RocArtifact(h0_value, h1_value)


class RocEvaluationResult(EvaluationResult):
    """Final result of an receive operating characteristcs evaluation."""

    __grid: Sequence[GridDimension]
    __detection_probabilities: np.ndarray
    __false_alarm_probabilities: np.ndarray

    def __init__(self, grid: Sequence[GridDimension], detection_probabilities: np.ndarray, false_alarm_probabilities: np.ndarray, title: str = "Receiver Operating Characteristics") -> None:
        self.__grid = grid
        self.__detection_probabilities = detection_probabilities
        self.__false_alarm_probabilities = false_alarm_probabilities
        self.__title = title

    def _plot(self, axes: plt.Axes) -> None:
        # Configure axes labels
        axes.set_xlabel("False Alarm Probability")
        axes.set_ylabel("Detection Probability")

        # Configure axes limits
        axes.set_xlim(0.0, 2)
        axes.set_ylim(0.0, 2)

        section_magnitudes = tuple(s.num_sample_points for s in self.__grid)
        for section_indices in np.ndindex(section_magnitudes):
            # Generate the graph line label
            line_label = ""
            for i, v in enumerate(section_indices):
                line_label += f"{self.__grid[i].title} = {self.__grid[i].sample_points[v]}, "
            line_label = line_label[:-2]

            # Select the graph line scalars
            x_axis = self.__false_alarm_probabilities[section_indices]
            y_axis = self.__detection_probabilities[section_indices]

            # Plot the graph line

            axes.plot(x_axis, y_axis, label=line_label)

        # Only plot the legend for an existing sweep grid.
        if len(self.__grid) > 0:
            axes.legend()

    def to_array(self) -> np.ndarray:
        return np.stack((self.__detection_probabilities, self.__false_alarm_probabilities), axis=-1)


class ReceiverOperatingCharacteristic(RadarEvaluator, Serializable):
    """Evaluate the receiver operating characteristics for a radar operator."""

    yaml_tag = "ROC"
    """YAML serialization tag."""

    __num_thresholds: int

    def __init__(self, radar: Radar, radar_channel: RadarChannelBase | None = None, num_thresholds=101) -> None:
        """
        Args:

            radar (Radar):
                Radar under test.

            radar_channel (RadarChannelBase, optional):
                Radar channel containing a desired target.
                If the radar channel is not specified, the :meth:`.evaluate` routine will not be available.

            num_thresholds (int, optional)
                Number of different thresholds to be considered in ROC curve
        """

        # Make sure the channel belongs to a simulation scenario
        if radar_channel is not None:
            if radar_channel.transmitter is None or radar_channel.receiver is None:
                raise ValueError("ROC evaluator must be defined within a simulation scenario")

        RadarEvaluator.__init__(self, receiving_radar=radar, radar_channel=radar_channel)
        self.__num_thresholds = num_thresholds

    @staticmethod
    def __evaluation_from_receptions(h0_reception: RadarReception, h1_reception: RadarReception) -> RocEvaluation:
        """Subroutine to generate an evaluation given two hypothesis radar receptions.

        Args:

            h0_reception (RadarReception):
                Reception missing the target of interest.

            h1_reception (RadarReception):
                Reception containing the target of interest.

        Returns: An initialized :class:`RocEvaluation`.
        """

        # Retrieve radar cubes for both hypothesis
        radar_cube_h0 = h0_reception.cube
        radar_cube_h1 = h1_reception.cube

        # Return resulting evaluation
        return RocEvaluation(radar_cube_h0, radar_cube_h1)

    def evaluate(self) -> RocEvaluation:
        if self.radar_channel is None:
            raise RuntimeError("Radar channel must be specified in order to evaluate during rutime")

        # Collect required information from the simulation
        one_hypothesis_channel_realization = self.radar_channel.realization
        device_output = self.radar_channel.transmitter.output
        device_input = self.radar_channel.receiver.input
        device_index = self.radar_channel.scenario.device_index(self.radar_channel.transmitter)
        operator_index = self.receiving_radar.device.receivers.operator_index(self.receiving_radar)

        if one_hypothesis_channel_realization is None:
            raise RuntimeError("Channel has not been realized yet")

        if device_output is None or device_input is None:
            raise RuntimeError("Channel devices lack cached transmission / reception information")

        # Generate the null hypothesis detection radar cube by re-running the radar detection routine
        null_hypothesis_channel_realization = self.radar_channel.null_hypothesis(device_output.mixed_signal.num_samples, device_output.sampling_rate, one_hypothesis_channel_realization)

        # Propagate again over the radar channel
        null_hypothesis_propagation = self.radar_channel.Propagate(device_output.mixed_signal, null_hypothesis_channel_realization)

        # Exchange the respective propagated signal
        impinging_signals = list(device_input.impinging_signals).copy()
        impinging_signals[device_index] = null_hypothesis_propagation

        # Receive again
        null_hypothesis_device_reception = self.radar_channel.receiver.process_from_realization(impinging_signals, device_input, device_output.trigger_realization, device_input.leaking_signal, False, channel_state=null_hypothesis_channel_realization)
        null_hypothesis_radar_reception = self.receiving_radar.receive(null_hypothesis_device_reception.operator_inputs[operator_index][0], None, False)

        # Generate evaluation
        return self.__evaluation_from_receptions(null_hypothesis_radar_reception, self.receiving_radar.reception)

    @property
    def abbreviation(self) -> str:
        return "ROC"  # pragma no cover

    @property
    def title(self) -> str:
        return "Operating Characteristics"  # pragma no cover

    def generate_result(self, grid: Sequence[GridDimension], artifacts: np.ndarray) -> RocEvaluationResult:
        # Prepare result containers
        dimensions = tuple(g.num_sample_points for g in grid)
        detection_probabilities = np.empty((*dimensions, self.__num_thresholds), dtype=float)
        false_alarm_probabilities = np.empty((*dimensions, self.__num_thresholds), dtype=float)

        # Convert artifacts to raw data array
        for grid_coordinates in np.ndindex(dimensions):
            artifact_line = artifacts[grid_coordinates]
            roc_data = np.array([[a.h0_value, a.h1_value] for a in artifact_line])

            for t, threshold in enumerate(np.linspace(roc_data.min(), roc_data.max(), self.__num_thresholds, endpoint=True)):
                threshold_coordinates = grid_coordinates + (t,)
                detection_probabilities[threshold_coordinates] = np.mean(roc_data[:, 1] >= threshold)
                false_alarm_probabilities[threshold_coordinates] = np.mean(roc_data[:, 0] >= threshold)

        return RocEvaluationResult(grid, detection_probabilities, false_alarm_probabilities, self.title)

    @classmethod
    def From_Scenarios(cls: Type[ReceiverOperatingCharacteristic], h0_scenario: Scenario, h1_scenario: Scenario, h0_operator: Optional[Radar] = None, h1_operator: Optional[Radar] = None) -> RocEvaluationResult:
        # Assert that both scenarios are in replay mode
        if h0_scenario.mode != ScenarioMode.REPLAY:
            raise ValueError("Null hypothesis scenario is not in replay mode")

        if h1_scenario.mode != ScenarioMode.REPLAY:
            raise ValueError("One hypothesis scenario is not in replay mode")

        # Assert that both scenarios have at least a single drop recorded
        if h0_scenario.num_drops < 1:
            raise ValueError("Null hypothesis scenario has no recorded drops")

        if h1_scenario.num_drops < 1:
            raise ValueError("One hypothesis scenario has no recorded drops")

        # Select operators if none were provided
        if h0_operator:
            if h0_operator not in h0_scenario.operators:
                raise ValueError("Null hypthesis radar not an operator within the null hypothesis scenario")

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
                raise ValueError("One hypthesis radar not an operator within the null hypothesis scenario")

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

        # The overall number of considered drops is bounded by the H1 hypothesis
        num_drops = h1_scenario.num_drops
        artifacts = np.empty(1, dtype=object)
        artifacts[0] = []

        # Collect artifacts
        for _ in range(num_drops):
            _ = h0_scenario.drop()
            _ = h1_scenario.drop()

            evaluation = cls.__evaluation_from_receptions(h0_operator.reception, h1_operator.reception)
            artifacts[0].append(evaluation.artifact())

        # Generate results
        grid = [GridDimension(h1_scenario, "num_drops", [0.0])]
        result = cls(h1_operator).generate_result(grid, artifacts)
        return result

    @classmethod
    def From_HDF(cls: Type[ReceiverOperatingCharacteristic], file: Union[str, File], h0_campaign="h0_measurements", h1_campaign="h1_measurements") -> RocEvaluationResult:
        """Compute an ROC evaluation result from a savefile.

        Args:

            file (Union[str, File]):
                Savefile containing the measurements.
                Either as file system location or h5py `File` handle.

            h0_campaign (str, optional):
                Campaign identifier of the h0 hypothesis measurements.
                By default, `h0_measurements` is assumed.

            h1_campaign (str, optional):
                Campaign identifier of the h1 hypothesis measurements.
                By default, `h1_measurements` is assumed.

        Returns: The ROC evaluation result.
        """

        # Load scenarios with the respective campaigns from the specified savefile
        h0_scenario = ReplayScenario.Replay(file, h0_campaign)
        h1_scenario = ReplayScenario.Replay(file, h1_campaign)

        # Resort to the from scenarios routine for computing the evaluation result
        result = cls.From_Scenarios(h0_scenario=h0_scenario, h1_scenario=h1_scenario)

        # Close the scenarios properly
        h0_scenario.stop()
        h1_scenario.stop()

        return result


class RootMeanSquareArtifact(Artifact):
    """Artifact of a root mean square evaluation"""

    num_errors: int
    cummulation: float

    def __init__(self, num_errors: int, cummulation: float) -> None:
        """
        Args:

            num_errors (int):
                Number of errros.

            cummulation (float):
                Sum of squared errors distances.
        """

        self.num_errors = num_errors
        self.cummulation = cummulation

    def to_scalar(self) -> float:
        return np.sqrt(self.cummulation / self.num_errors)

    def __str__(self) -> str:
        return f"{self.to_scalar():4.0f}"


class RootMeanSquareEvaluation(Evaluation):
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


class RootMeanSquareErrorResult(ScalarEvaluationResult):
    """Result of a root mean square error evaluation."""

    ...  # pragma no cover


class RootMeanSquareError(RadarEvaluator):
    """Root mean square estimation error of point detections."""

    def evaluate(self) -> Evaluation:
        point_cloud = self.receiving_radar.reception.cloud
        ground_truth = self.radar_channel.realization.ground_truth

        return RootMeanSquareEvaluation(point_cloud, ground_truth)

    @property
    def title(self) -> str:
        return "Root Mean Square Error"

    @property
    def abbreviation(self) -> str:
        return "RMSE"

    def generate_result(self, grid: Sequence[GridDimension], artifacts: np.ndarray) -> RootMeanSquareErrorResult:
        rmse_section_artifacts = np.empty(artifacts.shape, dtype=float)
        for coordinates, section_artifacts in np.ndenumerate(artifacts):
            cummulative_errors = 0.0
            error_count = 0

            artifact: RootMeanSquareArtifact
            for artifact in section_artifacts:
                cummulative_errors += artifact.cummulation
                error_count += artifact.num_errors

            rmse = np.sqrt(cummulative_errors / error_count)
            rmse_section_artifacts[coordinates] = rmse

        return RootMeanSquareErrorResult.From_Artifacts(grid, rmse_section_artifacts, self)
