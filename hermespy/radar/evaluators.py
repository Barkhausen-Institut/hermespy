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
from itertools import product
from re import A
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

from hermespy.core.executable import Executable
from hermespy.core.factory import Serializable
from hermespy.core.monte_carlo import Evaluator, Evaluation, EvaluationResult, EvaluationTemplate, GridDimension, \
    ArtifactTemplate, Artifact, ScalarEvaluationResult, ProcessedScalarEvaluationResult
from hermespy.radar import Radar
from hermespy.channel.radar_channel import RadarChannel
from hermespy.radar.cube import RadarCube
from hermespy.radar.detection import RadarPointCloud


__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "André Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RadarEvaluator(Evaluator, ABC):
    """Base class for evaluating communication processes between two modems."""

    __receiving_radar: Radar   # Handle to the radar receiver
    __radar_channel: RadarChannel   # Handle to the radar channel

    def __init__(self,
                 receiving_radar: Radar,
                 radar_channel: Optional[RadarChannel] = None
                 ) -> None:
        """
        Args:

            receiving_radar (Radar):
                Modem detecting radar in case of a target.

            radar_channel (RadarChannel):
                Radar channel containing a desired target.

        """

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
    def radar_channel(self) -> RadarChannel:
        """Radar channel

        Returns:
            RadarChannel: Handle to the radar channel
        """

        return self.__radar_channel

    def generate_result(self,
                        grid: List[GridDimension],
                        artifacts: np.ndarray) -> EvaluationResult:

        return ScalarEvaluationResult(grid, artifacts, self)


class DetectionProbArtifact(ArtifactTemplate[bool]):
    """Artifact of a detection probability evaluation for a radar detector."""

    def to_scalar(self) -> float:

        return float(self.artifact)
    
    
class DetectionProbabilityEvaluation(EvaluationTemplate[bool]):
    
    def artifact(self) -> DetectionProbArtifact:
        
        return DetectionProbArtifact(self.evaluation)


class DetectionProbEvaluator(RadarEvaluator, Serializable):
    """Evaluate detection probability at a radar detector, considering any bin, i.e., detection is considered if any
       bin in the radar cube is above the threshold"""

    yaml_tag = u'DetectionProbEvaluator'
    """YAML serialization tag"""

    def __init__(self,
                 receiving_radar: Radar) -> None:
        """
        Args:
            receiving_radar (Radar):
                Radar detector
        """

        RadarEvaluator.__init__(self, receiving_radar)
        self.plot_scale = 'log'  # Plot logarithmically by default

    @property
    def abbreviation(self) -> str:
        return "PD"

    @property
    def title(self) -> str:
        return "Probability of Detection Evaluation"

    @staticmethod
    def _scalar_cdf(scalar: float) -> float:
        return uniform.cdf(scalar)
    
    def generate_result(self,
                        grid: List[GridDimension],
                        artifacts: np.ndarray) -> ScalarEvaluationResult:
        
        return ScalarEvaluationResult(grid, artifacts, self)

    def evaluate(self) -> DetectionProbArtifact:

        # Retrieve transmitted and received bits
        cloud = self.receiving_radar.cloud
        
        if cloud is None:
            RuntimeError("Detection evaluation requires a detector to be configured at the radar")
            
        # Verify if a target is detected in any bin
        detection = cloud.num_points > 0
        return DetectionProbabilityEvaluation(detection)


class RocArtifact(Artifact):
    """Artifact of receiver operating characteristics (ROC) evaluation"""

    __h0_value: float
    __h1_value: float

    def __init__(self,
                 h0_value: float,
                 h1_value: float) -> None:
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

    def __init__(self,
                 cube_h0: RadarCube,
                 cube_h1: RadarCube) -> None:

        self.data_h0 = cube_h0.data
        self.data_h1 = cube_h1.data

    def artifact(self) -> RocArtifact:

        h0_value = self.data_h0.max()
        h1_value = self.data_h1.max()

        return RocArtifact(h0_value, h1_value)
    
    
class RocEvaluationResult(EvaluationResult):
    """Final result of an receive operating characteristcs evaluation."""
    
    __evaluator: ReceiverOperatingCharacteristic
    __grid: List[GridDimension]
    __detection_probabilities: np.ndarray
    __false_alarm_probabilities: np.ndarray
    
    def __init__(self,
                 evaluator: ReceiverOperatingCharacteristic,
                 grid: List[GridDimension],
                 detection_probabilities: np.ndarray,
                 false_alarm_probabilities: np.ndarray) -> None:
        
        self.__evaluator = evaluator
        self.__grid = grid
        self.__detection_probabilities = detection_probabilities
        self.__false_alarm_probabilities = false_alarm_probabilities
        
    def plot(self) -> plt.Figure:
        
        with Executable.style_context():

            figure = plt.figure()
            figure.suptitle(self.__evaluator.title)

            # Create single axes
            axes = figure.add_subplot()

            # Configure axes labels
            axes.set_xlabel("False Alarm Probability")
            axes.set_ylabel("Detection Probability")
            
            # Configure axes limits
            axes.set_xlim(0., 1.)
            axes.set_ylim(0., 1.)

            section_magnitudes = tuple(s.num_sample_points for s in self.__grid)
            for section_indices in np.ndindex(section_magnitudes):

                # Generate the graph line label
                line_label = ''
                for i, v in enumerate(section_indices):
                    line_label += f'{self.__grid[i].title} = {self.__grid[i].sample_points[v]}, '
                line_label = line_label[:-2]

                # Select the graph line scalars
                x_axis = self.__false_alarm_probabilities[section_indices]
                y_axis = self.__detection_probabilities[section_indices]

                # Plot the graph line
                
                axes.plot(x_axis, y_axis, label=line_label)

            # Only plot the legend for an existing sweep grid.
            if len(self.__grid) > 0:
                axes.legend()
            
            return figure
        
    def to_array(self) -> np.ndarray:
        
        return np.stack((self.__detection_probabilities, self.__false_alarm_probabilities), axis=-1)
        

class ReceiverOperatingCharacteristic(RadarEvaluator, Serializable):
    """Evaluate the receiver operating characteristics for a radar operator."""
    
    yaml_tag = u'ROC'
    """YAML serialization tag."""

    receiving_radar: Radar
    __receiving_radar_null_hypothesis: Radar    # handle to a radar receiver with only noise (H0)
    __num_thresholds: int

    def __init__(self,
                 receiving_radar: Radar,
                 receiving_radar_null_hypothesis: Radar,
                 radar_channel: RadarChannel = None,
                 num_thresholds=101) -> None:
        """
        Args:

            receiving_radar (Radar):
                Modem detecting radar in case of a target.

            receiving_radar_null_hypothesis(Radar):
                Radar receiver containing only noise.

            radar_channel (RadarChannel, Optional):
                Radar channel containing a desired target.
                If a radar channel is given, then the ROC is calculated for the bin that contains the target, or else
                a detection is performed if the output of any bin is above the threshold.

            num_thresholds (int, Optional)
                Number of different thresholds to be considered in ROC curve
        """
        
        RadarEvaluator.__init__(self, receiving_radar=receiving_radar,
                                radar_channel=radar_channel)
                        
        self.__receiving_radar_null_hypothesis = receiving_radar_null_hypothesis
        self.__num_thresholds = num_thresholds

    def evaluate(self) -> RocEvaluation:

        # Retrieve transmitted and received bits
        radar_cube_h0 = self.receiving_radar_null_hypothesis.cube
        radar_cube_h1 = self.receiving_radar.cube

        return RocEvaluation(radar_cube_h0, radar_cube_h1)

    @property
    def receiving_radar_null_hypothesis(self) -> Radar:
        """Radar detector with only noise

        Returns:
            Modem: Handle to the receiving modem, with only noise at receiver.
        """

        return self.__receiving_radar_null_hypothesis

    @property
    def abbreviation(self) -> str:
        return "ROC"
    
    @property
    def title(self) -> str:
        return "Operating Characteristics"

    def generate_result(self,
                        grid: List[GridDimension],
                        artifacts: np.ndarray) -> EvaluationResult:

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
                
        return RocEvaluationResult(self, grid, detection_probabilities, false_alarm_probabilities)


class RootMeanSquareArtifact(Artifact):
    """Artifact of a root mean square evaluation"""
    
    num_errors: int
    cummulation: float

    def __init__(self,
                 num_errors: int,
                 cummulation: float) -> None:
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
    
    def __init__(self,
                 pcl: RadarPointCloud,
                 ground_truth: np.ndarray) -> None:

        self.__pcl = pcl
        self.__ground_truth = ground_truth
        
    def artifact(self) -> RootMeanSquareArtifact:
        
        num_errors = self.__pcl.num_points * self.__ground_truth.shape[0]
        cummulative_square_error = 0.
        
        for point, truth in product(self.__pcl.points, self.__ground_truth):
            cummulative_square_error += np.linalg.norm(point.position - truth) ** 2
            
        return RootMeanSquareArtifact(num_errors, cummulative_square_error)


class RootMeanSquareErrorResult(ProcessedScalarEvaluationResult):
    """Result of a root mean square error evaluation."""
    ...  # pragma no cover


class RootMeanSquareError(RadarEvaluator):
    """Root mean square estimation error of point detections."""
    
    def evaluate(self) -> Evaluation:
        
        point_cloud = self.receiving_radar.cloud
        ground_truth = self.radar_channel.ground_truth
        
        return RootMeanSquareEvaluation(point_cloud, ground_truth)

    @property
    def title(self) -> str:
        return 'Root Mean Square Error'
    
    @property
    def abbreviation(self) -> str:
        return 'RMSE'

    def generate_result(self, grid: List[GridDimension], artifacts: np.ndarray) -> RootMeanSquareErrorResult:

        rmse_section_artifacts = np.empty(artifacts.shape, dtype=float)
        for coordinates, section_artifacts in np.ndenumerate(artifacts):
            
            cummulative_errors = 0.
            error_count = 0
            
            artifact: RootMeanSquareArtifact
            for artifact in section_artifacts:
                
                cummulative_errors += artifact.cummulation
                error_count += artifact.num_errors
                
            rmse = np.sqrt(cummulative_errors / error_count)
            rmse_section_artifacts[coordinates] = rmse
            
        return RootMeanSquareErrorResult(grid, rmse_section_artifacts, self)
