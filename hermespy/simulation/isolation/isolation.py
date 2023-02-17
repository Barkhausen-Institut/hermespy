# -*- coding: utf-8 -*-
"""
==========================
Antenna Isolation Modeling
==========================


.. toctree::
   :glob:

   simulation.isolation.perfect
   simulation.isolation.impedance
   simulation.isolation.specific
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from abc import ABC, abstractmethod

from hermespy.core import Signal, FloatingError

if TYPE_CHECKING:
    from ..simulated_device import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Isolation(ABC):
    """Base class for antenna isolation modeling."""

    __device: Optional[SimulatedDevice]

    def __init__(self, device: Optional[SimulatedDevice] = None) -> None:
        """
        Args:

            device (SimulatedDevice, optional): Device the model is configured to.
        """

        self.device = device

    @property
    def device(self) -> Optional[SimulatedDevice]:
        """Device the model is configured to.

        Returns:
            Handle to the device.
            `None`, if the model is considered floating.
        """

        return self.__device

    @device.setter
    def device(self, value: Optional[SimulatedDevice]) -> None:

        self.__device = value

    @abstractmethod
    def _leak(self, signal: Signal) -> Signal:
        """Compute leakage between RF transmit and receive chains.

        Args:

            signal (Signal): The signal transmitted over the respective antenna RF chains.


        Returns: The signal components leaking into receive chains.
        """
        ...  # pragma no cover

    def _assert_leaking_signal(self, signal: Signal | None) -> Signal:
        """Assert the validity of a leaking signal.

        Args:

            signal (Signal, None):
                The signal transmitted over the respective antenna RF chains.

        Raises:

            ValueError: If `signal` is invalid.

        Returns: The validated signal.
        """

        if self.device is None:
            raise FloatingError("Error trying to simulate leakage of a floating model")

        if signal is None:
            raise ValueError("Leaking signal not specified")

        if self.device.antennas.num_transmit_antennas != signal.num_streams:
            raise ValueError("Number of signal streams ({signal.num_streams}) does not match the number of transmitting antennas ({self.device.antennas.num_transmit_antennas})")

        return signal

    def leak(self, signal: Signal | None) -> Signal:
        """Compute leakage between RF transmit and receive chains.

        Args:

            signal (Signal, None):
                The signal transmitted over the respective antenna RF chains.


        Returns: The signal components leaking into receive chains.

        Raises:

            FloatingError: If the device is not specified.
            ValueError: If the number of signal streams does not match the number of transmitting antennas.
            ValueError: If `signal` is not specified but required.
        """

        _signal = self._assert_leaking_signal(signal)
        leaking_signal = self._leak(_signal)

        return leaking_signal
