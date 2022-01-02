# -*- coding: utf-8 -*-
"""Modem evaluators."""

from __future__ import annotations
from abc import ABC

import numpy as np

from hermespy.core.scenario import Scenario
from hermespy.core.monte_carlo import Evaluator, Artifact, ArtifactTemplate, MO
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

    def to_scalar(self) -> float:

        return np.sum(self.artifact) / len(self.artifact)


class BitErrorEvaluator(CommunicationEvaluator):
    """Evaluate bit errors between two modems exchanging information."""

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

    def evaluate(self, investigated_object: Scenario) -> BitErrorArtifact:

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

    def __str__(self) -> str:

        return "BER"
