# -*- coding: utf-8 -*-
"""
==================
Impedance Coupling
==================
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING

import numpy as np

from hermespy.core import Serializable, Signal
from .coupling import Coupling

if TYPE_CHECKING:
    from hermespy.simulation import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ImpedanceCoupling(Serializable, Coupling):
    """Ideal mutual coupling between two antenna arrays."""

    yaml_tag = "Impedance-Coupling"

    __transmit_correlation: Optional[np.ndarray]
    __receive_correlation: Optional[np.ndarray]
    __transmit_impedance: Optional[np.ndarray]
    __receive_impedance: Optional[np.ndarray]
    __matching_impedance: Optional[np.ndarray]

    def __init__(self, device: Optional[SimulatedDevice] = None, transmit_correlation: Optional[np.ndarray] = None, receive_correlation: Optional[np.ndarray] = None, transmit_impedance: Optional[np.ndarray] = None, receive_impedance: Optional[np.ndarray] = None, matching_impedance: Optional[np.ndarray] = None) -> None:
        """
        Args:

            device (SimulatedDevice, optional): Device the model is configured to.
        """

        Coupling.__init__(self, device=device)

        self.transmit_correlation = transmit_correlation
        self.receive_correlation = receive_correlation
        self.transmit_impedance = transmit_impedance
        self.receive_impedance = receive_impedance
        self.matching_impedance = matching_impedance

    @property
    def transmit_correlation(self) -> Optional[np.ndarray]:
        return self.__transmit_correlation

    @transmit_correlation.setter
    def transmit_correlation(self, value: Optional[np.ndarray]) -> None:
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
    def receive_correlation(self, value: Optional[np.ndarray]) -> None:
        if value is None:
            self.__receive_correlation = None
            return

        if value.ndim != 2:
            raise ValueError("Receive correlation must be a two dimensional array")

        if value.shape[0] != value.shape[1]:
            raise ValueError("Receive correlation must be square")

        self.__receive_correlation = value

    @property
    def transmit_impedance(self) -> Optional[np.ndarray]:
        return self.__transmit_impedance

    @transmit_impedance.setter
    def transmit_impedance(self, value: Optional[np.ndarray]) -> None:
        if value is None:
            self.__transmit_impedance = None
            return

        if value.ndim != 2:
            raise ValueError("Transmit impedance must be a two dimensional array")

        if value.shape[0] != value.shape[1]:
            raise ValueError("Transmit impedance must be square")

        self.__transmit_impedance = value

    @property
    def receive_impedance(self) -> Optional[np.ndarray]:
        return self.__receive_impedance

    @receive_impedance.setter
    def receive_impedance(self, value: Optional[np.ndarray]) -> None:
        if value is None:
            self.__receive_impedance = None
            return

        if value.ndim != 2:
            raise ValueError("Receive impedance must be a two dimensional array")

        if value.shape[0] != value.shape[1]:
            raise ValueError("Receive impedance must be square")

        self.__receive_impedance = value

    @property
    def matching_impedance(self) -> Optional[np.ndarray]:
        return self.__matching_impedance

    @matching_impedance.setter
    def matching_impedance(self, value: Optional[np.ndarray]) -> None:
        if value is None:
            self.matching_impedance = None
            return

        if value.ndim != 2:
            raise ValueError("Matching impedances must be a two dimensional array")

        if value.shape[0] != value.shape[1]:
            raise ValueError("Matching impedances must be square")

        self.__matching_impedance = value

    def _transmit(self, signal: Signal) -> Signal:
        transmit_impedance = np.eye(self.device.antennas.num_transmit_antennas) if self.transmit_impedance is None else self.transmit_impedance
        transmit_correlation = np.eye(self.device.antennas.num_transmit_antennas) if self.transmit_correlation is None else self.transmit_correlation

        transmit_coupling = transmit_impedance.real**-0.5 @ transmit_correlation**0.5
        transmitted_samples = transmit_coupling @ signal.samples

        return Signal(transmitted_samples, signal.sampling_rate, signal.carrier_frequency)

    def _receive(self, signal: Signal) -> Signal:
        receive_impedance = np.eye(self.device.antennas.num_receive_antennas) if self.receive_impedance is None else self.receive_impedance
        receive_correlation = np.eye(self.device.antennas.num_receive_antennas) if self.receive_correlation is None else self.receive_correlation
        matching_impedance = np.eye(self.device.antennas.num_receive_antennas) if self.matching_impedance is None else self.matching_impedance

        receive_coupling = 2 * receive_impedance[0, 0].real * matching_impedance.real**0.5 @ np.linalg.inv(matching_impedance + receive_correlation) @ receive_correlation**0.5
        received_samples = receive_coupling @ signal.samples

        return Signal(received_samples, signal.sampling_rate, signal.carrier_frequency)
