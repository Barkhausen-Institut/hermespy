# -*- coding: utf-8 -*-
"""
================
Physical Devices
================
"""

from __future__ import annotations

import numpy as np
from abc import abstractmethod

from hermespy.core import Device
from hermespy.core.signal_model import Signal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.6"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalDevice(Device):
    """Base representing any device controlling real hardware."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:
            *args:
                Device base class initialization parameters.

            **kwargs:
                Device base class initialization parameters.
        """

        # Init base class
        Device.__init__(self, *args, **kwargs)

    @abstractmethod
    def trigger(self) -> None:
        """Trigger the device."""
        ...

    def receive(self, signal: Signal) -> None:
        """Receive a new signal at this physical device.

        Args:
            signal (Signal):
                Signal model to be received.
        """

        # Signal is now a baseband-signal
        signal.carrier_frequency = 0.

        for receiver in self.receivers:

            receiver.cache_reception(signal)

    @property
    def velocity(self) -> np.ndarray:

        raise NotImplementedError("The velocity of physical devices is undefined by default")
