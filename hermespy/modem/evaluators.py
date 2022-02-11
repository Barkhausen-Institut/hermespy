# -*- coding: utf-8 -*-
"""
========================
Communication Evaluators
========================
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
from ..core.monte_carlo import Evaluator, ArtifactTemplate, Artifact
from .modem import Modem

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class CommunicationEvaluator(Evaluator[Scenario], ABC):
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


class BitErrorArtifact(ArtifactTemplate[np.ndarray]):
    """Artifact of a block error evaluation between two modems exchanging information."""

    def to_scalar(self) -> float:

        return np.sum(self.artifact) / len(self.artifact)

    def plot(self) -> List[plt.Figure]:

        with Executable.style_context():

            figure, axes = plt.subplots()
            figure.suptitle("Bit Error Evaluation")

            axes.stem(self.artifact)
            axes.set_xlabel("Bit Index")
            axes.set_ylabel("Bit Error Indicator")

            return [figure]


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

    def evaluate(self, investigated_object: Optional[Scenario] = None) -> BitErrorArtifact:

        # Retrieve transmitted and received bits
        transmitted_bits = self.transmitting_modem.transmitted_bits
        received_bits = self.receiving_modem.received_bits

        # Pad bit sequences (if required)
        num_bits = max(len(received_bits), len(transmitted_bits))
        padded_transmission = np.append(transmitted_bits, np.zeros(num_bits - len(transmitted_bits)))
        padded_reception = np.append(received_bits, np.zeros(num_bits - len(received_bits)))

        # Compute bit errors as the positions where both sequences differ.
        # Note that this requires the sequences to be in 0/1 format!
        bit_errors = np.abs(padded_transmission - padded_reception)

        return BitErrorArtifact(bit_errors)

    @property
    def abbreviation(self) -> str:
        return "BER"

    @property
    def title(self) -> str:
        return "Bit Error Rate Evaluation"

    @staticmethod
    def _scalar_cdf(scalar: float) -> float:
        return uniform.cdf(scalar)


class BlockErrorArtifact(ArtifactTemplate[np.ndarray]):
    """Artifact of a block error evaluation between two modems exchanging information."""

    def to_scalar(self) -> float:

        return np.sum(self.artifact) / len(self.artifact)


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

    def evaluate(self, investigated_object: Scenario) -> BlockErrorArtifact:

        # Retrieve transmitted and received bits
        transmitted_bits = self.transmitting_modem.transmitted_bits
        received_bits = self.receiving_modem.received_bits
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

        return BlockErrorArtifact(block_errors)

    @property
    def title(self) -> str:

        return "Block Error Rate"

    @property
    def abbreviation(self) -> str:

        return "BLER"


class FrameErrorArtifact(ArtifactTemplate[np.ndarray]):
    """Artifact of a frame error evaluation between two modems exchanging information."""

    def to_scalar(self) -> float:

        return np.sum(self.artifact) / len(self.artifact)


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

    def evaluate(self, investigated_object: Scenario) -> FrameErrorArtifact:

        # Retrieve transmitted and received bits
        transmitted_bits = self.transmitting_modem.transmitted_bits
        received_bits = self.receiving_modem.received_bits
        frame_size = self.receiving_modem.num_data_bits_per_frame

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

        return FrameErrorArtifact(frame_errors)

    @property
    def title(self) -> str:

        return "Frame Error Rate"

    @property
    def abbreviation(self) -> str:

        return "FER"


class ThroughputArtifact(Artifact):
    """Artifact of a throughput evaluation between two modems exchanging information."""

    __bits_per_frame: int           # Number of bits per communication frame
    __frame_duration: float         # Duration of a single communication frame in seconds
    __frame_errors: np.ndarray      # Frame error indicators

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

        self.__bits_per_frame = bits_per_frame
        self.__frame_duration = frame_duration
        self.__frame_errors = frame_errors

    def __str__(self) -> str:
        return f"{self.to_scalar():.3f}"

    def to_scalar(self) -> float:

        num_frames = len(self.__frame_errors)
        num_correct_frames = np.sum(np.invert(self.__frame_errors))
        throughput = num_correct_frames * self.__bits_per_frame / (num_frames * self.__frame_duration)

        return throughput


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

    def evaluate(self, investigated_object: Scenario) -> ThroughputArtifact:

        # Get the frame errors
        frame_errors = FrameErrorEvaluator.evaluate(self, investigated_object).artifact.flatten()

        # Transform frame errors to data throughput
        bits_per_frame = self.receiving_modem.num_data_bits_per_frame
        frame_duration = self.receiving_modem.frame_duration

        return ThroughputArtifact(bits_per_frame, frame_duration, frame_errors)

    @property
    def title(self) -> str:

        return "Data Throughput"

    @property
    def abbreviation(self) -> str:

        return "DRX"
