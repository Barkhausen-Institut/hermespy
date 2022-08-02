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

================================ ================================ ======================================================
Evaluator                        Artifact                         Performance Indicator
================================ ================================ ======================================================
:class:`.DetectionProbEvaluator` :class:`.DetectionProbArtifact`  Probability of detecting the target at the right bin
================================ ================================ ======================================================

Configuring :class:`RadarEvaluators<.RadarEvaluator>` to evaluate the radar detection of
:class:`Modem<hermespy.modem.modem.Modem>` instances is rather straightforward:

.. code-block:: python

   # Create two separate modem instances
   modem = Modem()
   channel = RadarChannel()

   # Create a radar evaluation as an evaluation example
   radar_evaluator = DetectionProbEvaluator(modem, channel)

   # Extract evaluation artifact
   radar_artifact = radar_evaluator.evaluate()

   # Visualize artifact
   radar_artifact.plot()

"""

from __future__ import annotations
from abc import ABC
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

from hermespy.core.executable import Executable
from hermespy.core.factory import Serializable
from hermespy.core.scenario import Scenario
from hermespy.core.monte_carlo import Evaluator, Evaluation, EvaluationResult, EvaluationTemplate, GridDimension, ArtifactTemplate, Artifact, ScalarEvaluationResult
from hermespy.radar import Radar
from hermespy.channel.radar_channel import RadarChannel
from hermespy.radar.cube import RadarCube


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
    __receiving_radar_null_hypothesis: Radar    # handle to a radar receiver with only noise (H0)

    def __init__(self,
                 receiving_radar: Radar,
                 radar_channel: Optional[RadarChannel] = None,
                 receiving_radar_null_hypothesis: Optional[Radar] = None,
                 ) -> None:
        """
        Args:

            receiving_radar (Radar):
                Modem detecting radar in case of a target.

            radar_channel (RadarChannel):
                Radar channel containing a desired target.

            receiving_radar_null_hypothesis(Optional, Radar):
                Radar receiver containing only noise.
        """

        self.__receiving_radar = receiving_radar
        self.__receiving_radar_null_hypothesis = receiving_radar_null_hypothesis
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
    def receiving_radar_null_hypothesis(self) -> Radar:
        """Radar detector with only noise

        Returns:
            Modem: Handle to the receiving modem, with only noise at receiver.
        """

        return self.__receiving_radar_null_hypothesis

    @property
    def radar_channel(self) -> RadarChannel:
        """Radar channel

        Returns:
            RadarChannel: Handle to the radar channel
        """

        return self.__radar_channel


class DetectionProbArtifact(ArtifactTemplate[bool]):
    """Artifact of a detection probability evaluation for a radar detector."""

    def to_scalar(self) -> float:

        return float(self.artifact)
    
    
class DetectionProbEvaluation(EvaluationTemplate[float]):
    
    def artifact(self) -> DetectionProbArtifact:
        
        return DetectionProbArtifact(self.evaluation)
    
    
class DetectionProbResult(ScalarEvaluationResult):
    ...


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
                        artifacts: np.ndarray) -> DetectionProbResult:
        
        return DetectionProbResult(grid, artifacts, self)

    def evaluate(self) -> DetectionProbEvaluation:

        # Retrieve transmitted and received bits
        cloud = self.receiving_radar.cloud
        
        if cloud is None:
            RuntimeError("Detection evaluation requires a detector to be configured at the radar")
            
        # Verify if a target is detected in any bin
        detection = cloud.num_points > 0
        return DetectionProbEvaluation(detection)
    
    
class DetectionArtifact(ArtifactTemplate[bool]):
    """Artifact generated from a single detection evaluation."""
    ...

class DetectionEvaluation(Evaluation):
    """Extraction of a detection probability evaluation for a radar detector."""
    
    detection: RadarCube
    null_hypothesis: RadarCube
    
    def __init__(self,
                 detection: RadarCube,
                 null_hypothesis: RadarCube) -> None:
        
        self.detection = detection
        self.null_hypothesis = null_hypothesis
        
    def to_scalar(self) -> float:

        return float('inf')
    
    def artifact(self) -> DetectionArtifact:
        
        # Verify if a target is detected in any bin
        detection_value_h1 = np.max(self.detection.data)
        detection_value_h0 = np.max(self.null_hypothesis.data)
        
        # Do something
        return DetectionArtifact(False)
    

class DetectionEvaluationResult(ScalarEvaluationResult):
    ...
    

class ReceiverOperatingCharacteristic(RadarEvaluator, Serializable):
    """Evaluate the receiver operator characteristics for a radar operator.

    """
    yaml_tag = u'ROC'
    """YAML serialization tag."""

    def __init__(self,
                 receiving_radar: Radar,
                 receiving_radar_null_hypothesis: Radar,
                 radar_channel: RadarChannel = None,
                 num_thresholds=100
                 ) -> None:
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
        
        Serializable.__init__(self)
        RadarEvaluator.__init__(self, receiving_radar=receiving_radar,
                                receiving_radar_null_hypothesis=receiving_radar_null_hypothesis,
                                radar_channel=radar_channel)
        
    def evaluate(self) -> DetectionEvaluation:

        # Retrieve transmitted and received bits
        radar_cube_h1 = self.receiving_radar.cube
        radar_cube_h0 = self.receiving_radar_null_hypothesis.cube
        
        return DetectionEvaluation(radar_cube_h1, radar_cube_h0)

    @property
    def abbreviation(self) -> str:
        return "ROC"
    
    @property
    def title(self) -> str:
        
        return "Operating Characteristics"

    def generate_result(self, grid: List[GridDimension], artifacts: np.ndarray) -> DetectionEvaluationResult:
        
        return DetectionEvaluationResult(grid, artifacts, self)
