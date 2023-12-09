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
from typing import Type, Union
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
from h5py import File
from scipy.stats import uniform

from hermespy.core import ReplayScenario, Scenario, ScenarioMode, Serializable, VAT
from hermespy.core.monte_carlo import (
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
from hermespy.channel import RadarChannelBase
from hermespy.radar.cube import RadarCube
from hermespy.radar.detection import RadarPointCloud
from hermespy.simulation import SimulatedDevice


__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarEvaluator(Evaluator, ABC):
    """Bastract base class for evaluating sensing performance.

    Inherits from the abstract :class:`Evaluator<hermespy.core.monte_carlo.Evaluator>` base class.
    Expects the abstract method :meth:`evaluate` as well as the abstract properties
    :meth:`abbreviation<abbreviation>` and :meth:`title<title>` to be implemented.

    There are currently three different :class:`RadarEvaluators<.RadarEvaluator>` implemented:

    .. toctree::

        radar.evaluators.DetectionProbEvaluator
        radar.evaluators.ReceiverOperatingCharacteristic
        radar.evaluators.RootMeanSquareError
    """

    __receiving_radar: Radar  # Handle to the radar receiver
    __radar_channel: RadarChannelBase  # Handle to the radar channel
    __receiving_device: SimulatedDevice
    __transmitting_device: SimulatedDevice

    def __init__(self, receiving_radar: Radar, radar_channel: RadarChannelBase) -> None:
        """
        Args:

            receiving_radar (Radar): Radar under test.
            radar_channel (RadarChannelBase): Radar channel modeling a desired target.

        Raises:

            ValueError: If the receiving radar is not an operator of the radar_channel receiver.
        """

        if radar_channel.alpha_device is None or radar_channel.beta_device is None:
            raise ValueError("Radar channel must be configured within a simulation scenario")

        if receiving_radar.device is None:
            raise ValueError("Radar must be assigned a device within a simulation scenario")

        self.__receiving_radar = receiving_radar
        self.__radar_channel = radar_channel

        if receiving_radar.device is radar_channel.alpha_device:
            self.__receiving_device = radar_channel.alpha_device
            self.__transmitting_device = radar_channel.beta_device

        elif receiving_radar.device is radar_channel.beta_device:
            self.__receiving_device = radar_channel.beta_device
            self.__transmitting_device = radar_channel.alpha_device

        else:
            raise ValueError(
                "Recieving radar to be evaluated must be assigned to the radar channel"
            )

        # Initialize base class
        Evaluator.__init__(self)

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


class DetectionProbArtifact(ArtifactTemplate[bool]):
    """Artifacto of the probability of detection for a radar detector.

    Represents a boolean indicator of whether a target was detected or not.
    Generated by the :class:`DetectionProbabilityEvaluation<.DetectionProbabilityEvaluation>`'s :meth:`artifact()<DetectionProbabilityEvaluation.artifact>` method.
    """

    ...  # pragma: no cover


class DetectionProbabilityEvaluation(EvaluationTemplate[bool]):
    """Evaluation of the probability of detection for a radar detector.

    Represents a boolean indicator of whether a target was detected or not.
    Generated by the :class:`DetectionProbEvaluator<.DetectionProbEvaluator>`'s :meth:`evaluate()<DetectionProbEvaluator.evaluate>` method.
    """

    def artifact(self) -> DetectionProbArtifact:
        return DetectionProbArtifact(self.evaluation)


class DetectionProbEvaluator(Evaluator, Serializable):
    """Estimates the probability of detection for a given radar detector.

    Assumes a successful detection if the :class:`Radar's<hermespy.radar.radar.Radar>` :meth:`reception<hermespy.radar.radar.Radar.reception>` contains a non-empty point cloud.
    This is the case if the configured :class:`RadarDetector<hermespy.radar.detection.RadarDetector>` made a positive decision
    for any bin within the processed :class:`RadarCube<hermespy.radar.cube.RadarCube>`.

    A minimal example within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`
    evaluating the probability of detection for a single radar target illuminated by an :class:`FMCW<hermespy.radar.fmcw.FMCW>` radar would be:

    .. literalinclude:: ../scripts/examples/radar_evaluators_DetectionProbEvaluator.py
       :language: python
       :linenos:
       :lines: 03-27
    """

    yaml_tag = "DetectionProbEvaluator"

    __radar: Radar

    def __init__(self, radar: Radar) -> None:
        """
        Args:

            radar (Radar):
                Radar detector to be evaluated.
        """

        # Initialize base class
        Evaluator.__init__(self)

        # Initialize class attributes
        self.__radar = radar
        self.plot_scale = "log"  # Plot logarithmically by default

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
        # Retrieve transmitted and received bits
        cloud = self.radar.reception.cloud

        if cloud is None:
            raise RuntimeError(
                "Detection evaluation requires a detector to be configured at the radar"
            )

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

    def to_scalar(self) -> None:
        return None


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

    __detection_probabilities: np.ndarray
    __false_alarm_probabilities: np.ndarray

    def __init__(
        self,
        grid: Sequence[GridDimension],
        evaluator: ReceiverOperatingCharacteristic,
        detection_probabilities: np.ndarray,
        false_alarm_probabilities: np.ndarray,
    ) -> None:
        """
        Args:

            grid (Sequence[GridDimension]):
                Grid dimensions of the evaluation result.

            evaluator (ReceiverOperatingCharacteristic):
                Evaluator that generated the evaluation result.

            detection_probabilities (np.ndarray):
                Detection probabilities for each grid point.

            false_alarm_probabilities (np.ndarray):
                False alarm probabilities for each grid point.
        """

        # Initialize base class
        EvaluationResult.__init__(self, grid, evaluator)

        self.__detection_probabilities = detection_probabilities
        self.__false_alarm_probabilities = false_alarm_probabilities

    def _plot(self, axes: VAT) -> None:
        ax: plt.Axes = axes.flat[0]

        # Configure axes labels
        ax.set_xlabel("False Alarm Probability")
        ax.set_ylabel("Detection Probability")

        # Configure axes limits
        ax.set_xlim(0.0, 1.1)
        ax.set_ylim(0.0, 1.1)

        section_magnitudes = tuple(s.num_sample_points for s in self.grid)
        for section_indices in np.ndindex(section_magnitudes):
            # Generate the graph line label
            line_label = ""
            for i, v in enumerate(section_indices):
                line_label += f"{self.grid[i].title} = {self.grid[i].sample_points[v].title}, "
            line_label = line_label[:-2]

            # Select the graph line scalars
            x_axis = self.__false_alarm_probabilities[section_indices]
            y_axis = self.__detection_probabilities[section_indices]

            # Plot the graph line
            ax.plot(x_axis, y_axis, label=line_label)

        # Only plot the legend for an existing sweep grid.
        if len(self.grid) > 0:
            ax.legend()

    def to_array(self) -> np.ndarray:
        return np.stack((self.__detection_probabilities, self.__false_alarm_probabilities), axis=-1)


class ReceiverOperatingCharacteristic(RadarEvaluator, Serializable):
    """Evaluate the receiver operating characteristics for a radar operator.

    The receiver operating characteristics (ROC) curve is a graphical plot that illustrates the performance of a detector
    by visualizing the probability of false alarm versus the probability of detection for a given parameterization.

    A minimal example within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`
    evaluating the probability of detection for a single radar target illuminated by an :class:`FMCW<hermespy.radar.fmcw.FMCW>` radar would be:

    .. literalinclude:: ../scripts/examples/radar_evaluators_ReceiverOperatingCharacteristic.py
       :language: python
       :linenos:
       :lines: 03-23
    """

    yaml_tag = "ROC"

    _title = "Receiver Operating Characteristics"
    __num_thresholds: int

    def __init__(self, radar: Radar, radar_channel: RadarChannelBase, num_thresholds=101) -> None:
        """
        Args:

            radar (Radar):
                Radar under test.

            radar_channel (RadarChannelBase):
                Radar channel containing a desired target.

            num_thresholds (int, optional)
                Number of different thresholds to be considered in ROC curve
        """

        RadarEvaluator.__init__(self, receiving_radar=radar, radar_channel=radar_channel)
        self.__num_thresholds = num_thresholds

    @staticmethod
    def __evaluation_from_receptions(
        h0_reception: RadarReception, h1_reception: RadarReception
    ) -> RocEvaluation:
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
        # Collect required information from the simulation
        one_hypothesis_channel_realization = self.radar_channel.realization
        device_output = self.transmitting_device.output
        device_input = self.receiving_device.input
        device_index = self.radar_channel.scenario.device_index(self.transmitting_device)
        operator_index = self.receiving_device.receivers.operator_index(self.receiving_radar)

        # Check if the channel has been realized yet
        if one_hypothesis_channel_realization is None:
            raise RuntimeError("Channel has not been realized yet")

        # Check if the devices have been realized yet
        if device_output is None or device_input is None:
            raise RuntimeError("Channel devices lack cached transmission / reception information")

        # Generate the null hypothesis detection radar cube by re-running the radar detection routine
        null_hypothesis_channel_realization = one_hypothesis_channel_realization.null_hypothesis()

        # Propagate again over the radar channel
        null_hypothesis_propagation = null_hypothesis_channel_realization.propagate(device_output)

        # Exchange the respective propagated signal
        impinging_signals = list(device_input.impinging_signals).copy()
        impinging_signals[device_index] = null_hypothesis_propagation.signal

        # Receive again
        null_hypothesis_device_reception = self.receiving_device.process_from_realization(
            impinging_signals,
            device_input,
            device_output.trigger_realization,
            device_input.leaking_signal,
            False,
        )
        null_hypothesis_radar_reception = self.receiving_radar.receive(
            null_hypothesis_device_reception.operator_inputs[operator_index], False
        )

        # Generate evaluation
        return self.__evaluation_from_receptions(
            null_hypothesis_radar_reception, self.receiving_radar.reception
        )

    @property
    def abbreviation(self) -> str:
        return "ROC"  # pragma: no cover

    @property
    def title(self) -> str:
        return ReceiverOperatingCharacteristic._title  # pragma: no cover

    def generate_result(
        self, grid: Sequence[GridDimension], artifacts: np.ndarray
    ) -> RocEvaluationResult:
        """Generate a new receiver operating characteristics evaluation result.

        Args:

            grid (Sequence[GridDimension]):
                Grid dimensions of the evaluation result.

            artifacts (numpy.ndarray):
                Artifacts of the evaluation result.

        Returns: The generated result.
        """

        # Prepare result containers
        if len(grid) > 0:
            dimensions = tuple(g.num_sample_points for g in grid)
        else:
            dimensions = (1,)
            artifacts = artifacts.reshape(dimensions)

        detection_probabilities = np.empty((*dimensions, self.__num_thresholds), dtype=float)
        false_alarm_probabilities = np.empty((*dimensions, self.__num_thresholds), dtype=float)

        # Convert artifacts to raw data array
        for grid_coordinates in np.ndindex(dimensions):
            artifact_line = artifacts[grid_coordinates]
            roc_data = np.array([[a.h0_value, a.h1_value] for a in artifact_line])

            for t, threshold in enumerate(
                np.linspace(roc_data.min(), roc_data.max(), self.__num_thresholds, endpoint=True)
            ):
                threshold_coordinates = grid_coordinates + (t,)
                detection_probabilities[threshold_coordinates] = np.mean(
                    roc_data[:, 1] >= threshold
                )
                false_alarm_probabilities[threshold_coordinates] = np.mean(
                    roc_data[:, 0] >= threshold
                )

        return RocEvaluationResult(grid, self, detection_probabilities, false_alarm_probabilities)

    def from_scenarios(
        self,
        h0_scenario: Scenario,
        h1_scenario: Scenario,
        h0_operator: Radar | None = None,
        h1_operator: Radar | None = None,
    ) -> RocEvaluationResult:
        """Compute an ROC evaluation result from two scenarios.

        Args:

            h0_scenario (Scenario):
                Scenario of the null hypothesis.

            h1_scenario (Scenario):
                Scenario of the alternative hypothesis.

            h0_operator (Radar, optional):
                Radar operator of the null hypothesis.
                If not provided, the first radar operator of the null hypothesis scenario will be used.

            h1_operator (Radar, optional):
                Radar operator of the alternative hypothesis.
                If not provided, the first radar operator of the alternative hypothesis scenario will be used.

        Returns: The ROC evaluation result.
        """

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

        # The overall number of considered drops is bounded by the H1 hypothesis
        num_drops = h1_scenario.num_drops
        artifacts = np.empty(1, dtype=object)
        artifacts[0] = []

        # Collect artifacts
        for _ in range(num_drops):
            _ = h0_scenario.drop()
            _ = h1_scenario.drop()

            evaluation = self.__evaluation_from_receptions(
                h0_operator.reception, h1_operator.reception
            )
            artifacts[0].append(evaluation.artifact())

        # Generate results
        grid: Sequence[GridDimension] = []
        result = self.generate_result(grid, artifacts)
        return result

    @classmethod
    def From_Scenarios(
        cls: Type[ReceiverOperatingCharacteristic],
        h0_scenario: Scenario,
        h1_scenario: Scenario,
        h0_operator: Radar | None = None,
        h1_operator: Radar | None = None,
    ) -> RocEvaluationResult:
        """Compute an ROC evaluation result from two scenarios.

        Args:

            h0_scenario (Scenario):
                Scenario of the null hypothesis.

            h1_scenario (Scenario):
                Scenario of the alternative hypothesis.

            h0_operator (Radar, optional):
                Radar operator of the null hypothesis.
                If not provided, the first radar operator of the null hypothesis scenario will be used.

            h1_operator (Radar, optional):
                Radar operator of the alternative hypothesis.
                If not provided, the first radar operator of the alternative hypothesis scenario will be used.

        Returns: The ROC evaluation result.
        """

        # Generate a mock evaluator
        # ToDo: Resolve this workaround to avoid the need for a mock evaluator
        channel = Mock()
        channel.alpha_device = Mock()
        channel.beta_device = channel.alpha_device
        radar = Mock()
        radar.device = channel.alpha_device
        evaluator = ReceiverOperatingCharacteristic(radar, channel)
        return evaluator.from_scenarios(h0_scenario, h1_scenario, h0_operator, h1_operator)

    @classmethod
    def From_HDF(
        cls: Type[ReceiverOperatingCharacteristic],
        file: Union[str, File],
        h0_campaign="h0_measurements",
        h1_campaign="h1_measurements",
    ) -> RocEvaluationResult:
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

    __num_errors: int
    __cummulation: float

    def __init__(self, num_errors: int, cummulation: float) -> None:
        """
        Args:

            num_errors (int):
                Number of errros.

            cummulation (float):
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

    ...  # pragma: no cover


class RootMeanSquareError(RadarEvaluator):
    """Root mean square error of estimated points to ground truth.

    Root mean square error (RMSE) is a widely used metric for evaluating the performance of a radar detector.
    It estimates the average distance between the estimated and the ground truth position of a target.

    A minimal example within the context of a :class:`Simulation<hermespy.simulation.simulation.Simulation>`
    evaluating the probability of detection for a single radar target illuminated by an :class:`FMCW<hermespy.radar.fmcw.FMCW>` radar would be:

    .. literalinclude:: ../scripts/examples/radar_evaluators_RootMeanSquareError.py
       :language: python
       :linenos:
       :lines: 03-29
    """

    def evaluate(self) -> Evaluation:
        reception = self.receiving_radar.reception

        if reception is None:
            raise RuntimeError(
                "Root mean square evaluation requires its radar to have received a reception"
            )

        point_cloud = reception.cloud
        channel_realization = self.radar_channel.realization

        if point_cloud is None:
            raise RuntimeError(
                "Root mean square evaluation requires a detector to be configured at the radar"
            )

        if channel_realization is None:
            raise RuntimeError("Root mean square evaluation requires a realized radar channel")

        return RootMeanSquareEvaluation(point_cloud, channel_realization.ground_truth())

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
