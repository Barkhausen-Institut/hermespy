# -*- coding: utf-8 -*-
"""
=====================
Physical Device Dummy
=====================

Implements a physical device dummy for testing and demonstration purposes.
"""

from __future__ import annotations

from ..simulation import SimulatedDevice
from .physical_device import PhysicalDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalDeviceDummy(SimulatedDevice, PhysicalDevice):
    """Physical device dummy for testing and demonstration."""

    def trigger(self) -> None:

        # Compute signal to be transmitted
        signal = self.transmit()

        # Patch it directly to the receive chain
        PhysicalDevice.receive(self, signal[0])
