# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import TYPE_CHECKING
from typing_extensions import override

import numpy as np

from hermespy.core import DeserializationProcess, SerializationProcess, Signal
from .coupling import Coupling

if TYPE_CHECKING:
    from hermespy.simulation import SimulatedDevice  # pragma: no cover

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ImpedanceCoupling(Coupling):
    """Imedance based mutual coupling model"""

    __transmit_correlation: np.ndarray | None
    __receive_correlation: np.ndarray | None
    __transmit_impedance: np.ndarray | None
    __receive_impedance: np.ndarray | None
    __matching_impedance: np.ndarray | None

    def __init__(
        self,
        device: SimulatedDevice | None = None,
        transmit_correlation: np.ndarray | None = None,
        receive_correlation: np.ndarray | None = None,
        transmit_impedance: np.ndarray | None = None,
        receive_impedance: np.ndarray | None = None,
        matching_impedance: np.ndarray | None = None,
    ) -> None:
        """
        Args:

            device (SimulatedDevice, optional)
                Device the model is configured to.

            transmit_correlation (numpy.ndarray, optional):
                Correlation matrix of the transmit antenna array.
                Defaults to the identity matrix.

            receive_correlation (numpy.ndarray, optional):
                Correlation matrix of the receive antenna array.
                Defaults to the identity matrix.

            transmit_impedance (numpy.ndarray, optional):
                Impedance matrix of the transmit antenna array.
                Defaults to the identity matrix.

            receive_impedance (numpy.ndarray, optional):
                Impedance matrix of the receive antenna array.
                Defaults to the identity matrix.

            matching_impedance (numpy.ndarray, optional):
                Impedance matrix of the matching network.
                Defaults to the identity matrix.
        """

        Coupling.__init__(self, device=device)

        self.transmit_correlation = transmit_correlation
        self.receive_correlation = receive_correlation
        self.transmit_impedance = transmit_impedance
        self.receive_impedance = receive_impedance
        self.matching_impedance = matching_impedance

    @property
    def transmit_correlation(self) -> np.ndarray | None:
        return self.__transmit_correlation

    @transmit_correlation.setter
    def transmit_correlation(self, value: np.ndarray | None) -> None:
        if value is None:
            self.__transmit_correlation = None
            return

        if value.ndim != 2:
            raise ValueError("Transmit correlation must be a two dimensional array")

        if value.shape[0] != value.shape[1]:
            raise ValueError("Transmit correlation must be square")

        self.__transmit_correlation = value

    @property
    def receive_correlation(self) -> np.ndarray:
        return self.__receive_correlation

    @receive_correlation.setter
    def receive_correlation(self, value: np.ndarray | None) -> None:
        if value is None:
            self.__receive_correlation = None
            return

        if value.ndim != 2:
            raise ValueError("Receive correlation must be a two dimensional array")

        if value.shape[0] != value.shape[1]:
            raise ValueError("Receive correlation must be square")

        self.__receive_correlation = value

    @property
    def transmit_impedance(self) -> np.ndarray | None:
        return self.__transmit_impedance

    @transmit_impedance.setter
    def transmit_impedance(self, value: np.ndarray | None) -> None:
        if value is None:
            self.__transmit_impedance = None
            return

        if value.ndim != 2:
            raise ValueError("Transmit impedance must be a two dimensional array")

        if value.shape[0] != value.shape[1]:
            raise ValueError("Transmit impedance must be square")

        self.__transmit_impedance = value

    @property
    def receive_impedance(self) -> np.ndarray | None:
        return self.__receive_impedance

    @receive_impedance.setter
    def receive_impedance(self, value: np.ndarray | None) -> None:
        if value is None:
            self.__receive_impedance = None
            return

        if value.ndim != 2:
            raise ValueError("Receive impedance must be a two dimensional array")

        if value.shape[0] != value.shape[1]:
            raise ValueError("Receive impedance must be square")

        self.__receive_impedance = value

    @property
    def matching_impedance(self) -> np.ndarray | None:
        return self.__matching_impedance

    @matching_impedance.setter
    def matching_impedance(self, value: np.ndarray | None) -> None:
        if value is None:
            self.__matching_impedance = None
            return

        if value.ndim != 2:
            raise ValueError("Matching impedances must be a two dimensional array")

        if value.shape[0] != value.shape[1]:
            raise ValueError("Matching impedances must be square")

        self.__matching_impedance = value

    def _transmit(self, signal: Signal) -> Signal:
        transmit_impedance = (
            np.eye(self.device.antennas.num_transmit_antennas)
            if self.transmit_impedance is None
            else self.transmit_impedance
        )
        transmit_correlation = (
            np.eye(self.device.antennas.num_transmit_antennas)
            if self.transmit_correlation is None
            else self.transmit_correlation
        )

        transmit_coupling = transmit_impedance.real**-0.5 @ transmit_correlation**0.5
        transmitted_samples = transmit_coupling @ signal.getitem()

        return signal.from_ndarray(transmitted_samples)

    def _receive(self, signal: Signal) -> Signal:
        receive_impedance = (
            np.eye(self.device.antennas.num_receive_antennas)
            if self.receive_impedance is None
            else self.receive_impedance
        )
        receive_correlation = (
            np.eye(self.device.antennas.num_receive_antennas)
            if self.receive_correlation is None
            else self.receive_correlation
        )
        matching_impedance = (
            np.eye(self.device.antennas.num_receive_antennas)
            if self.matching_impedance is None
            else self.matching_impedance
        )

        receive_coupling = (
            2
            * receive_impedance[0, 0].real
            * matching_impedance.real**0.5
            @ np.linalg.inv(matching_impedance + receive_correlation)
            @ receive_correlation**0.5
        )
        received_samples = receive_coupling @ signal.getitem()

        return signal.from_ndarray(received_samples)

    @override
    def serialize(self, serialization_process: SerializationProcess) -> None:
        if self.transmit_correlation is not None:
            serialization_process.serialize_array(self.transmit_correlation, "transmit_correlation")
        if self.receive_correlation is not None:
            serialization_process.serialize_array(self.receive_correlation, "receive_correlation")
        if self.transmit_impedance is not None:
            serialization_process.serialize_array(self.transmit_impedance, "transmit_impedance")
        if self.receive_impedance is not None:
            serialization_process.serialize_array(self.receive_impedance, "receive_impedance")
        if self.matching_impedance is not None:
            serialization_process.serialize_array(self.matching_impedance, "matching_impedance")

    @override
    @classmethod
    def Deserialize(cls, deserialization_process: DeserializationProcess) -> ImpedanceCoupling:
        return cls(
            transmit_correlation=deserialization_process.deserialize_array(
                "transmit_correlation", np.float64, None
            ),
            receive_correlation=deserialization_process.deserialize_array(
                "receive_correlation", np.float64, None
            ),
            transmit_impedance=deserialization_process.deserialize_array(
                "transmit_impedance", np.float64, None
            ),
            receive_impedance=deserialization_process.deserialize_array(
                "receive_impedance", np.float64, None
            ),
            matching_impedance=deserialization_process.deserialize_array(
                "matching_impedance", np.float64, None
            ),
        )
