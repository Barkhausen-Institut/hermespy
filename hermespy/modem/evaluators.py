# -*- coding: utf-8 -*-
"""
========================
Communication Evaluators
========================

This module introduces several evaluators for performance indicators in communication scenarios.
Refer to the :doc:`PyMonte</api/core.monte_carlo>` documentation for a detailed introduction to the concept of
:class:`Evaluators<hermespy.core.monte_carlo.Evaluator>`.

.. autoclasstree:: hermespy.modem.evaluators
   :alt: Communication Evaluator Class Tree
   :strict:
   :namespace: hermespy

The implemented :class:`CommunicationEvaluators<.CommunicationEvaluator>` all inherit from the identically named common
base which gets initialized by selecting the two :class:`Modem<hermespy.modem.modem.Modem>` instances whose communication
should be evaluated.
The currently considered performance indicators are

============================= ============================= ========================================================
Evaluator                     Artifact                      Performance Indicator
============================= ============================= ========================================================
:class:`.BitErrorEvaluator`   :class:`.BitErrorArtifact`    Errors comparing two bit streams
:class:`.BlockErrorEvaluator` :class:`.BlockErrorArtifact`  Errors comparing two bit streams divided into blocks
:class:`.FrameErrorEvaluator` :class:`.FrameErrorArtifact`  Errors comparing two bit streams divided into frames
:class:`.ThroughputEvaluator` :class:`.ThroughputArtifact`  Rate of correct frames multiplied by the frame bit rate
============================= ============================= ========================================================

Configuring :class:`CommunicationEvaluators<.CommunicationEvaluator>` to evaluate the communication process between two
:class:`Modem<hermespy.modem.modem.Modem>` instances is rather straightforward:

.. code-block:: python

   # Create two separate modem instances
   modem_alpha = Modem()
   modem_beta = Modem()

   # Create a bit error evaluation as a communication evaluation example
   communication_evaluator = BitErrorEvaluator(modem_alpha, modem_beta)

   # Extract evaluation artifact
   communication_artifact = communication_evaluator.evaluate()

   # Visualize artifact
   communication_artifact.plot()

"""

from __future__ import annotations
from abc import ABC
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import uniform

from ..core.executable import Executable
from ..core.factory import Serializable
from ..core.scenario import Scenario
from ..core.monte_carlo import Artifact, ArtifactTemplate, Evaluator, EvaluationResult, EvaluationTemplate, GridDimension, ScalarEvaluationResult
from .modem import Modem

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class CommunicationEvaluator(Evaluator, ABC):
    """Base class for evaluating communication processes between two modems."""

    __transmitting_modem: Modem     # Handle to the transmitting modem
    __receiving_modem: Modem        # Handle to the receiving modem

    def __init__(self,
                 transmitting_modem: Modem,
                 receiving_modem: Modem) -> None:
        """
        Args:

            transmitting_modem (Modem):
                Modem transmitting information.

            receiving_modem (Modem):
                Modem receiving information.
        """

        self.__transmitting_modem = transmitting_modem
        self.__receiving_modem = receiving_modem

        # Initialize base class
        Evaluator.__init__(self)

    @property
    def transmitting_modem(self) -> Modem:
        """Modem transmitting information.

        Returns:
            Modem: Handle to the transmitting modem.
        """

        return self.__transmitting_modem

    @property
    def receiving_modem(self) -> Modem:
        """Modem receiving information.

        Returns:
            Modem: Handle to the receiving modem.
        """

        return self.__receiving_modem
    
    def generate_result(self,
                        grid: List[GridDimension],
                        artifacts: np.ndarray) -> EvaluationResult:
        
        return ScalarEvaluationResult(grid, artifacts, self)


class BitErrorArtifact(ArtifactTemplate[float]):
    """Artifact of a bit error evaluation between two modems exchanging information."""

    def to_scalar(self) -> float:

        return self.artifact


class BitErrorEvaluation(EvaluationTemplate[np.ndarray]):
    """Bit error evaluation of a single communication process between modems."""

    def plot(self) -> List[plt.Figure]:

        with Executable.style_context():

            figure, axes = plt.subplots()
            figure.suptitle("Bit Error Evaluation")

            axes.stem(self.evaluation)
            axes.set_xlabel("Bit Index")
            axes.set_ylabel("Bit Error Indicator")

            return [figure]

    def artifact(self) -> BitErrorArtifact:

        ber = np.mean(self.evaluation)
        return BitErrorArtifact(ber)


class BitErrorEvaluator(CommunicationEvaluator, Serializable):
    """Evaluate bit errors between two modems exchanging information."""

    yaml_tag = u'BitErrorEvaluator'
    """YAML serialization tag"""

    def __init__(self,
                 transmitting_modem: Modem,
                 receiving_modem: Modem) -> None:
        """
        Args:

            transmitting_modem (Modem):
                Modem transmitting information.

            receiving_modem (Modem):
                Modem receiving information.
        """

        CommunicationEvaluator.__init__(self, transmitting_modem, receiving_modem)
        self.plot_scale = 'log'  # Plot logarithmically by default

    def evaluate(self) -> BitErrorEvaluation:

        # Retrieve transmitted and received bits
        transmitted_bits = self.transmitting_modem.transmission.bits
        received_bits = self.receiving_modem.reception.bits

        # Pad bit sequences (if required)
        num_bits = max(len(received_bits), len(transmitted_bits))
        padded_transmission = np.append(transmitted_bits, np.zeros(num_bits - len(transmitted_bits)))
        padded_reception = np.append(received_bits, np.zeros(num_bits - len(received_bits)))

        # Compute bit errors as the positions where both sequences differ.
        # Note that this requires the sequences to be in 0/1 format!
        bit_errors = np.abs(padded_transmission - padded_reception)

        return BitErrorEvaluation(bit_errors)

    @property
    def abbreviation(self) -> str:
        return "BER"

    @property
    def title(self) -> str:
        return "Bit Error Rate Evaluation"

    @staticmethod
    def _scalar_cdf(scalar: float) -> float:
        return uniform.cdf(scalar)


class BlockErrorArtifact(ArtifactTemplate[float]):
    """Artifact of a block error evaluation between two modems exchanging information."""

    def to_scalar(self) -> float:

        return self.artifact
    
    
class BlockErrorEvaluation(EvaluationTemplate[np.ndarray]):
    """Block error evaluation of a single communication process between modems."""

    def plot(self) -> List[plt.Figure]:

        with Executable.style_context():

            figure, axes = plt.subplots()
            figure.suptitle("Block Error Evaluation")

            axes.stem(self.evaluation)
            axes.set_xlabel("Block Index")
            axes.set_ylabel("Block Error Indicator")

            return [figure]

    def artifact(self) -> BitErrorArtifact:

        bler = np.mean(self.evaluation)
        return BlockErrorArtifact(bler)


class BlockErrorEvaluator(CommunicationEvaluator, Serializable):
    """Evaluate block errors between two modems exchanging information."""

    yaml_tag = u'BlockErrorEvaluator'
    """YAML serialization tag"""

    def __init__(self,
                 transmitting_modem: Modem,
                 receiving_modem: Modem) -> None:
        """
        Args:

            transmitting_modem (Modem):
                Modem transmitting information.

            receiving_modem (Modem):
                Modem receiving information.
        """

        CommunicationEvaluator.__init__(self, transmitting_modem, receiving_modem)
        self.plot_scale = 'log'  # Plot logarithmically by default

    def evaluate(self) -> BlockErrorEvaluation:

        # Retrieve transmitted and received bits
        transmitted_bits = self.transmitting_modem.transmission.bits
        received_bits = self.receiving_modem.reception.bits
        block_size = self.receiving_modem.encoder_manager.bit_block_size

        # Pad bit sequences (if required)
        received_bits = np.append(received_bits, np.zeros(received_bits.shape[0] % block_size))

        if transmitted_bits.shape[0] >= received_bits.shape[0]:

            transmitted_bits = transmitted_bits[:received_bits.shape[0]]

        else:

            transmitted_bits = np.append(transmitted_bits, -np.ones(received_bits.shape[0] - transmitted_bits.shape[0]))

        # Compute bit errors as the positions where both sequences differ.
        # Note that this requires the sequences to be in 0/1 format!
        bit_errors = np.abs(transmitted_bits - received_bits)
        block_errors = (bit_errors.reshape((-1, block_size)).sum(axis=1) > 0)

        return BlockErrorEvaluation(block_errors)

    @property
    def title(self) -> str:

        return "Block Error Rate"

    @property
    def abbreviation(self) -> str:

        return "BLER"

    @staticmethod
    def _scalar_cdf(scalar: float) -> float:
        return uniform.cdf(scalar)


class FrameErrorArtifact(ArtifactTemplate[float]):
    """Artifact of a frame error evaluation between two modems exchanging information."""

    def to_scalar(self) -> float:

        return self.artifact
    
    
class FrameErrorEvaluation(EvaluationTemplate[np.ndarray]):
    """Frame error evaluation of a single communication process between modems."""

    def plot(self) -> List[plt.Figure]:

        with Executable.style_context():

            figure, axes = plt.subplots()
            figure.suptitle("Frame Error Evaluation")

            axes.stem(self.evaluation)
            axes.set_xlabel("Frame Index")
            axes.set_ylabel("Frame Error Indicator")

            return [figure]

    def artifact(self) -> FrameErrorArtifact:

        bler = np.mean(self.evaluation)
        return FrameErrorArtifact(bler)


class FrameErrorEvaluator(CommunicationEvaluator, Serializable):
    """Evaluate frame errors between two modems exchanging information."""

    yaml_tag = u'FrameErrorEvaluator'
    """YAML serialization tag"""

    def __init__(self,
                 transmitting_modem: Modem,
                 receiving_modem: Modem) -> None:
        """
        Args:

            transmitting_modem (Modem):
                Modem transmitting information.

            receiving_modem (Modem):
                Modem receiving information.
        """

        CommunicationEvaluator.__init__(self, transmitting_modem, receiving_modem)
        self.plot_scale = 'log'  # Plot logarithmically by default

    def evaluate(self) -> FrameErrorEvaluation:

        # Retrieve transmitted and received bits
        transmitted_bits = self.transmitting_modem.transmission.bits
        received_bits = self.receiving_modem.reception.bits
        frame_size = self.receiving_modem.num_data_bits_per_frame
        
        if frame_size < 1:
            return FrameErrorArtifact(np.empty(0, dtype=np.unit8))

        # Pad bit sequences (if required)
        received_bits = np.append(received_bits, np.zeros(received_bits.shape[0] % frame_size))

        if transmitted_bits.shape[0] >= received_bits.shape[0]:

            transmitted_bits = transmitted_bits[:received_bits.shape[0]]

        else:

            transmitted_bits = np.append(transmitted_bits, -np.ones(received_bits.shape[0] - transmitted_bits.shape[0]))

        # Compute bit errors as the positions where both sequences differ.
        # Note that this requires the sequences to be in 0/1 format!
        bit_errors = np.abs(transmitted_bits - received_bits)
        frame_errors = (bit_errors.reshape((-1, frame_size)).sum(axis=1) > 0)

        return FrameErrorEvaluation(frame_errors)

    @property
    def title(self) -> str:

        return "Frame Error Rate"

    @property
    def abbreviation(self) -> str:

        return "FER"

    @staticmethod
    def _scalar_cdf(scalar: float) -> float:
        return uniform.cdf(scalar)
    
    
class ThroughputArtifact(ArtifactTemplate[float]):
    """Artifact of a throughput evaluation between two modems exchanging information."""

    def to_scalar(self) -> float:

        return self.artifact
    

class ThroughputEvaluation(EvaluationTemplate[np.ndarray]):
    """Throughput evaluation between two modems exchanging information."""

    __bits_per_frame: int           # Number of bits per communication frame
    __frame_duration: float         # Duration of a single communication frame in seconds

    def __init__(self,
                 bits_per_frame: int,
                 frame_duration: float,
                 frame_errors: np.ndarray) -> None:
        """
        Args:

            bits_per_frame (int):
                Number of bits per communication frame

            frame_duration (float):
                Duration of a single communication frame in seconds

            frame_errors (np.ndarray):
                Frame error indicators
        """
        
        EvaluationTemplate.__init__(self, frame_errors)

        self.__bits_per_frame = bits_per_frame
        self.__frame_duration = frame_duration

    def __str__(self) -> str:
        return f"{self.to_scalar():.3f}"

    def artifact(self) -> ThroughputArtifact:

        num_frames = len(self.evaluation)
        num_correct_frames = np.sum(np.invert(self.evaluation))
        throughput = num_correct_frames * self.__bits_per_frame / (num_frames * self.__frame_duration)

        return ThroughputArtifact(throughput)


class ThroughputEvaluator(FrameErrorEvaluator, Serializable):
    """Evaluate data throughput between two modems exchanging information."""

    yaml_tag = u'ThroughputEvaluator'
    """YAML serialization tag"""

    def __init__(self,
                 transmitting_modem: Modem,
                 receiving_modem: Modem) -> None:
        """
        Args:

            transmitting_modem (Modem):
                Modem transmitting information.

            receiving_modem (Modem):
                Modem receiving information.
        """

        FrameErrorEvaluator.__init__(self, transmitting_modem, receiving_modem)

    def evaluate(self) -> ThroughputEvaluation:

        # Get the frame errors
        frame_errors = FrameErrorEvaluator.evaluate(self).evaluation.flatten()

        # Transform frame errors to data throughput
        bits_per_frame = self.receiving_modem.num_data_bits_per_frame
        frame_duration = self.receiving_modem.frame_duration

        return ThroughputEvaluation(bits_per_frame, frame_duration, frame_errors)

    @property
    def title(self) -> str:

        return "Data Throughput"

    @property
    def abbreviation(self) -> str:

        return "DRX"
