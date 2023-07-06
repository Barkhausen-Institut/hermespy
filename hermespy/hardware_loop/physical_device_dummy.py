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

import numpy as np

from hermespy.core import AntennaArrayBase, ChannelStateInformation, DeviceInput, Serializable, Signal, SNRType
from hermespy.simulation import ProcessedSimulatedDeviceInput, SimulatedDevice, SimulatedDeviceOutput, SimulatedDeviceReception, SimulatedDeviceTransmission, TriggerRealization
from .physical_device import PhysicalDevice
from .scenario import PhysicalScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalDeviceDummy(SimulatedDevice, PhysicalDevice, Serializable):
    """Physical device dummy for testing and demonstration.

    The physical device dummy always receives back its most recent transmission.
    """

    yaml_tag = "PhysicalDeviceDummy"

    __receive_transmission: bool
    __uploaded_signal: Signal
    __downloaded_signal: Signal

    def __init__(self, *args, max_receive_delay: float = 0.0, antennas: AntennaArrayBase | None = None, noise_power: np.ndarray | None = None, receive_transmission: bool = True, **kwargs) -> None:
        # Initialize base classes
        SimulatedDevice.__init__(self, *args, **kwargs)
        PhysicalDevice.__init__(self, max_receive_delay=max_receive_delay, antennas=antennas, noise_power=noise_power)

        # Initialize internal state
        self.receive_transmission = receive_transmission
        self.__uploaded_signal = Signal.empty(1.0, self.num_antennas)
        self.__downloaded_signal = Signal.empty(1.0, self.num_antennas)

    @property
    def receive_transmission(self) -> bool:
        """Whether the device receives back its own transmission."""

        return self.__receive_transmission

    @receive_transmission.setter
    def receive_transmission(self, value: bool) -> None:
        self.__receive_transmission = value

    def _upload(self, signal: Signal) -> None:
        self.__uploaded_signal = signal

    def _download(self) -> Signal:
        return self.__downloaded_signal

    def transmit(self, cache: bool = True, trigger_realization: TriggerRealization | None = None) -> SimulatedDeviceTransmission:
        # Generate device transmission
        device_transmission = SimulatedDevice.transmit(self, cache, trigger_realization)

        # Upload mixed signal
        self._upload(device_transmission.mixed_signal)

        return device_transmission

    def process_input(
        self,
        impinging_signals: DeviceInput | Signal | Sequence[Signal] | Sequence[Tuple[Sequence[Signal], ChannelStateInformation | None]] | SimulatedDeviceOutput | None = None,
        cache: bool = True,
        trigger_realization: TriggerRealization | None = None,
        snr: float = float("inf"),
        snr_type: SNRType = SNRType.PN0,
        leaking_signal: Signal | None = None,
        channel_state: ChannelStateInformation | None = None,
    ) -> ProcessedSimulatedDeviceInput:
        _impinging_signals = self.__uploaded_signal if impinging_signals is None else impinging_signals
        return SimulatedDevice.process_input(self, _impinging_signals, cache, trigger_realization, snr, snr_type, leaking_signal, channel_state)

    def receive(self, impinging_signals: DeviceInput | Signal | Sequence[Signal] | None = None, *args, **kwargs) -> SimulatedDeviceReception:
        if impinging_signals is None:
            impinging_signals = self._download()

        return SimulatedDevice.receive(self, impinging_signals, *args, **kwargs)

    def trigger(self) -> None:
        if self.receive_transmission:
            self.__downloaded_signal = self.__uploaded_signal

        else:
            samples = np.zeros(self.__uploaded_signal.samples.shape)
            self.__downloaded_signal = Signal(samples, self.sampling_rate, self.carrier_frequency)

    def trigger_direct(self, signal: Signal, calibrate: bool = True) -> Signal:
        if self.receive_transmission:
            input = signal

        else:
            input = Signal(np.zeros((self.antennas.num_receive_antennas, signal.num_samples), dtype=np.complex_), self.sampling_rate, self.carrier_frequency)

        # Apply the simulation receive model
        processed_input = self.process_input(input, False, leaking_signal=signal)
        baseband_signal = processed_input.baseband_signal

        # Apply correction routines if calibrations are available
        corrected_signal = baseband_signal if not calibrate or self.leakage_calibration is None else self.leakage_calibration.remove_leakage(signal, baseband_signal, self.delay_calibration.delay)

        return corrected_signal

    @property
    def max_sampling_rate(self) -> float:
        return self.sampling_rate


class PhysicalScenarioDummy(PhysicalScenario[PhysicalDeviceDummy], Serializable):
    """Physical scenario for testing and demonstration."""

    yaml_tag = "PhysicalScenarioDummy"

    def _trigger(self) -> None:
        return

    def new_device(self, *args, **kwargs) -> PhysicalDeviceDummy:
        device = PhysicalDeviceDummy(*args, **kwargs)
        self.add_device(device)

        return device
