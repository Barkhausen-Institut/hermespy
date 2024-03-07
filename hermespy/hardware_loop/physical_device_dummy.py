# -*- coding: utf-8 -*-
"""
=====================
Physical Device Dummy
=====================

Implements a physical device dummy for testing and demonstration purposes.
"""

from __future__ import annotations
from collections.abc import Sequence

import numpy as np

from hermespy.core import DeviceInput, Serializable, Signal, SNRType
from hermespy.channel import ChannelPropagation
from hermespy.simulation import (
    ProcessedSimulatedDeviceInput,
    SimulatedAntennaArray,
    SimulatedDevice,
    SimulatedDeviceOutput,
    SimulatedDeviceReception,
    SimulatedDeviceTransmission,
    SimulationScenario,
    TriggerRealization,
)
from .physical_device import PhysicalDevice
from .scenario import PhysicalScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
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

    def __init__(
        self,
        max_receive_delay: float = 0.0,
        antennas: SimulatedAntennaArray | None = None,
        noise_power: np.ndarray | None = None,
        receive_transmission: bool = True,
        **kwargs,
    ) -> None:
        # Initialize base classes
        SimulatedDevice.__init__(self, antennas=antennas, **kwargs)
        PhysicalDevice.__init__(self, max_receive_delay=max_receive_delay, noise_power=noise_power)

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

    def transmit(
        self, cache: bool = True, trigger_realization: TriggerRealization | None = None
    ) -> SimulatedDeviceTransmission:
        # Generate device transmission
        device_transmission = SimulatedDevice.transmit(self, cache, trigger_realization)

        # Upload mixed signal
        self._upload(device_transmission.mixed_signal)

        return device_transmission

    def process_input(
        self,
        impinging_signals: (
            DeviceInput
            | Signal
            | Sequence[Signal]
            | ChannelPropagation
            | Sequence[ChannelPropagation]
            | SimulatedDeviceOutput
            | None
        ) = None,
        cache: bool = True,
        trigger_realization: TriggerRealization | None = None,
        snr: float = float("inf"),
        snr_type: SNRType = SNRType.PN0,
        leaking_signal: Signal | None = None,
    ) -> ProcessedSimulatedDeviceInput:
        _impinging_signals = (
            self.__uploaded_signal if impinging_signals is None else impinging_signals
        )
        return SimulatedDevice.process_input(
            self, _impinging_signals, cache, trigger_realization, snr, snr_type, leaking_signal
        )

    def receive(
        self,
        impinging_signals: (
            DeviceInput
            | Signal
            | Sequence[Signal]
            | ChannelPropagation
            | Sequence[ChannelPropagation]
            | SimulatedDeviceOutput
            | None
        ) = None,
        *args,
        **kwargs,
    ) -> SimulatedDeviceReception:
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
            input = Signal(
                np.zeros(
                    (self.antennas.num_receive_antennas, signal.num_samples), dtype=np.complex_
                ),
                self.sampling_rate,
                self.carrier_frequency,
            )

        # Apply the simulation receive model
        leaking_signal = self.isolation.leak(signal)
        processed_input = self.process_input(input, False, leaking_signal=leaking_signal)
        baseband_signal = processed_input.baseband_signal

        # Apply correction routines if calibrations are available
        corrected_signal = (
            baseband_signal
            if not calibrate or self.leakage_calibration is None
            else self.leakage_calibration.remove_leakage(
                signal, baseband_signal, self.delay_calibration.delay
            )
        )

        return corrected_signal

    @property
    def max_sampling_rate(self) -> float:
        return self.sampling_rate


class PhysicalScenarioDummy(
    SimulationScenario, PhysicalScenario[PhysicalDeviceDummy], Serializable
):
    """Physical scenario for testing and demonstration."""

    yaml_tag = "PhysicalScenarioDummy"

    def __init__(
        self, seed: int | None = None, devices: Sequence[PhysicalDeviceDummy] | None = None
    ) -> None:
        # Initialize base classes
        SimulationScenario.__init__(self, seed=seed, devices=devices)
        PhysicalScenario.__init__(self, seed=seed, devices=devices)

    def new_device(self, *args, **kwargs) -> PhysicalDeviceDummy:
        device = PhysicalDeviceDummy(*args, **kwargs)
        self.add_device(device)

        return device

    def add_device(self, device: SimulatedDevice | PhysicalDeviceDummy) -> None:
        # Adding a device resolves to the simulation scenario's add device method
        SimulationScenario.add_device(self, device)

    def receive_devices(
        self,
        impinging_signals: (
            Sequence[DeviceInput]
            | Sequence[Signal]
            | Sequence[Sequence[Signal]]
            | Sequence[Sequence[ChannelPropagation]]
            | None
        ) = None,
        cache: bool = True,
        trigger_realizations: Sequence[TriggerRealization] | None = None,
    ) -> Sequence[SimulatedDeviceReception]:
        if impinging_signals is None:
            physical_device_receptions = PhysicalScenario.receive_devices(self, None, cache)
            impinging_signals = [r.impinging_signals for r in physical_device_receptions]

        return SimulationScenario.receive_devices(
            self, impinging_signals, cache, trigger_realizations
        )

    def _trigger(self) -> None:
        # Triggering is equivalent to generating a new simulation drop
        SimulationScenario.drop(self)  # type: ignore
