# -*- coding: utf-8 -*-
"""
=====================
Physical Device Dummy
=====================

Implements a physical device dummy for testing and demonstration purposes.
"""

from __future__ import annotations

from hermespy.core import DeviceReception, DeviceTransmission, Signal
from hermespy.simulation import SimulatedDevice
from .physical_device import PhysicalDevice
from .scenario import PhysicalScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalDeviceDummy(SimulatedDevice, PhysicalDevice):
    """Physical device dummy for testing and demonstration.

    The physical device dummy always receives back its most recent transmission.
    """

    __cached_signal: Signal

    def __init__(self, *args, **kwargs) -> None:

        SimulatedDevice.__init__(self, *args, **kwargs)
        PhysicalDevice.__init__(self)

        self.__cached_signal = Signal.empty(1.0, self.num_antennas)

    def _upload(self, signal: Signal) -> None:

        self.__cached_signal = signal

    def _download(self) -> Signal:

        return self.__cached_signal

    def transmit(self, clear_cache: bool = True) -> DeviceTransmission:

        return PhysicalDevice.transmit(self, clear_cache)

    def process_input(self, *args) -> Signal:

        return PhysicalDevice.process_input(self, *args)

    def receive(self, *args, **kwargs) -> DeviceReception:

        return PhysicalDevice.receive(self, *args, **kwargs)

    def trigger(self) -> None:

        # Triggering a dummy does nothing
        return

    @property
    def max_sampling_rate(self) -> float:

        return self.sampling_rate


class PhysicalScenarioDummy(PhysicalScenario[PhysicalDeviceDummy]):
    """Physical scenario for testing and demonstration."""

    def _trigger(self) -> None:
        return

    def new_device(self, *args, **kwargs) -> PhysicalDeviceDummy:

        device = PhysicalDeviceDummy(*args, **kwargs)
        self.add_device(device)

        return device
