# -*- coding: utf-8 -*-
"""Modem evaluators."""

from __future__ import annotations
from abc import ABC
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import SafeRepresenter, Node

from ..core.executable import Executable
from ..core.factory import Serializable
from ..core.scenario import Scenario
from ..core.monte_carlo import Evaluator, ArtifactTemplate
from .modem import Modem

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
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


class BlockErrorArtifact(ArtifactTemplate[np.ndarray]):
    """Artifact of a block error evaluation between two modems exchanging information."""

    def to_scalar(self) -> float:

        return np.sum(self.artifact) / len(self.artifact)


class BlockErrorEvaluator(CommunicationEvaluator):
    """Evaluate block errors between two modems exchanging information."""

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

    def __str__(self) -> str:

        return "BLER"

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


class FrameErrorEvaluator(CommunicationEvaluator):
    """Evaluate frame errors between two modems exchanging information."""

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

    def __str__(self) -> str:

        return "FER"

    @property
    def title(self) -> str:

        return "Frame Error Rate"

    @property
    def abbreviation(self) -> str:

        return "FER"
