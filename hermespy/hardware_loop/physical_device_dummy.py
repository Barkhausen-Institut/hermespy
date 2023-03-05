# -*- coding: utf-8 -*-
"""
=====================
Physical Device Dummy
=====================

Implements a physical device dummy for testing and demonstration purposes.
"""

from __future__ import annotations
from collections.abc import Sequence
from typing import Tuple

from hermespy.core import ChannelStateInformation, DeviceInput, DeviceReception, Signal, SNRType
from hermespy.simulation import ProcessedSimulatedDeviceInput, SimulatedDevice, SimulatedDeviceOutput, SimulatedDeviceTransmission
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

    def transmit(self, clear_cache: bool = True) -> SimulatedDeviceTransmission:
        physical_transmission = PhysicalDevice.transmit(self, clear_cache)
        simulated_transmission = SimulatedDeviceTransmission(physical_transmission.operator_transmissions, physical_transmission.mixed_signal, physical_transmission.sampling_rate, physical_transmission.num_antennas, physical_transmission.carrier_frequency)

        return simulated_transmission

    def process_input(self, impinging_signals: DeviceInput | Signal | Sequence[Signal] | Sequence[Tuple[Sequence[Signal], ChannelStateInformation | None]] | SimulatedDeviceOutput | None = None, cache: bool = True, snr: float = float("inf"), snr_type: SNRType = SNRType.PN0, leaking_signal: Signal | None = None, channel_state: ChannelStateInformation | None = None) -> ProcessedSimulatedDeviceInput:
        _impinging_signals = self.__cached_signal if impinging_signals is None else impinging_signals
        return SimulatedDevice.process_input(self, _impinging_signals, cache, snr, snr_type, leaking_signal, channel_state)

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
