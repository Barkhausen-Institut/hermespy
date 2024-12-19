# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

from hermespy.core import Signal, FloatingError

if TYPE_CHECKING:
    from ..simulated_device import SimulatedDevice  # pragma: no cover

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Coupling(ABC):
    """Base class for mutual coupling model implementations."""

    __device: SimulatedDevice | None

    def __init__(self, device: SimulatedDevice | None = None) -> None:
        """
        Args:

            device (SimulatedDevice, optional): Device the model is configured to.
        """

        self.device = device

    @property
    def device(self) -> SimulatedDevice | None:
        """Device the model is configured to.

        Returns:
            Handle to the device.
            `None`, if the model is considered floating.
        """

        return self.__device

    @device.setter
    def device(self, value: SimulatedDevice | None) -> None:
        self.__device = value

    def transmit(self, signal: Signal) -> Signal:
        """Apply the mutual coupling model during signal transmission.

        Args:

            signal (Signal): The signal to be transmitted.

        Returns: The signal resulting from coupling modeling.

        Raises:

            FloatingError: If the device is not specified.
            ValueError: If the number of signal streams does not match the number of transmitting antennas.
        """

        if self.device is None:
            raise FloatingError("Error trying to simulate coupling of a floating model")

        if self.device.num_transmit_antennas != signal.num_streams:
            raise ValueError(
                f"Number of signal streams ({signal.num_streams}) does not match the number of transmitting antenna ports ({self.device.num_transmit_antennas})"
            )

        return self._transmit(signal)

    @abstractmethod
    def _transmit(self, signal: Signal) -> Signal:
        """Apply the mutual coupling model during signal transmission.

        Args:

            signal (Signal): The signal to be transmitted.

        Returns: The signal resulting from coupling modeling.
        """
        ...  # pragma: no cover

    def receive(self, signal: Signal) -> Signal:
        """Apply the mutual coupling model during signal reception.

        Args:

            signal (Signal): The signal to be received.

        Returns: The signal resulting from coupling modeling.

        Raises:

            FloatingError: If the device is not specified.
            ValueError: If the number of signal streams does not match the number of transmitting antennas.
        """

        if self.device is None:
            raise FloatingError("Error trying to simulate coupling of a floating model")

        if self.device.num_receive_antenna_ports != signal.num_streams:
            raise ValueError(
                f"Number of signal streams ({signal.num_streams}) does not match the number of receiving antenna ports ({self.device.num_receive_antenna_ports})"
            )

        return self._receive(signal)

    @abstractmethod
    def _receive(self, signal: Signal) -> Signal:
        """Apply the mutual coupling model during signal reception.

        Args:

            signal (Signal): The signal to be received.

        Returns: The signal resulting from coupling modeling.
        """
        ...  # pragma: no cover
