# -*- coding: utf-8 -*-
"""
========================
Physical Device Scenario
========================
"""

from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from time import time
from typing import Generic, Optional, TypeVar

from hermespy.core import DeviceInput, DeviceReception, Scenario, Drop, Signal
from hermespy.simulation import SimulatedDeviceReception, SimulationScenario, TriggerRealization
from .physical_device import PDT

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalScenario(Generic[PDT], Scenario[PDT]):
    """Scenario of physical device bindings.

    Managing physical devices by a scenario enables synchronized triggering
    and shared random seed configuration.
    """

    def __init__(self, seed: Optional[int] = None, devices: Optional[Sequence[PDT]] = None) -> None:
        Scenario.__init__(self, seed, devices)

    @abstractmethod
    def _trigger(self) -> None:
        """Trigger synchronzed transmission and reception for all managed devices."""
        ...  # pragma no cover

    def receive_devices(
        self,
        impinging_signals: (
            Sequence[DeviceInput] | Sequence[Signal] | Sequence[Sequence[Signal]] | None
        ) = None,
        cache: bool = True,
    ) -> Sequence[DeviceReception]:
        """Receive over all scenario devices.

        Internally calls :meth:`Scenario.process_inputs` and :meth:`Scenario.receive_devices`.

        Args:

            impinging_signals (Sequence[DeviceInput | Signal | Sequence[Signal]] | None, optional):
                List of signals impinging onto the devices.
                If not specified, the device will download the signal samples from its binding.

            cache (bool, optional):
                Cache the operator inputs at the registered receive operators for further processing.
                Enabled by default.

        Returns: List of the processed device input information.

        Raises:

            ValueError: If the number of `impinging_signals` does not match the number of registered devices.
        """

        impinging_signals = (
            [None] * self.num_devices if impinging_signals is None else impinging_signals
        )

        # Generate inputs
        device_inputs = [d.process_input(i, cache) for d, i in zip(self.devices, impinging_signals)]  # type: ignore

        # Generate operator receptions
        receptions = self.receive_operators(device_inputs)

        # Generate device receptions
        return [
            DeviceReception.From_ProcessedDeviceInput(i, r)
            for i, r in zip(device_inputs, receptions)
        ]

    def _drop(self) -> Drop:
        # Generate device transmissions
        device_transmissions = self.transmit_devices()

        # Trigger the full scenario for phyiscal transmission and reception
        timestamp = time()
        self._trigger()

        # Generate device receptions
        device_receptions = self.receive_devices()

        return Drop(timestamp, device_transmissions, device_receptions)

    def add_device(self, device: PDT) -> None:
        Scenario.add_device(self, device)


PhysicalScenarioType = TypeVar("PhysicalScenarioType", bound=PhysicalScenario)
"""Type of physical scenario"""


class SimulatedPhysicalScenario(SimulationScenario, PhysicalScenario):
    """Simulated physical scenario for testing purposes."""

    def _trigger(self) -> None:
        # Triggering does nothing
        pass  # pragma: no cover

    def receive_devices(
        self,
        impinging_signals: (
            Sequence[DeviceInput] | Sequence[Signal] | Sequence[Sequence[Signal]] | None
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

        else:
            return SimulationScenario.receive_devices(
                self, impinging_signals, cache, trigger_realizations
            )
