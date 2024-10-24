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

from h5py import Group

from hermespy.core import DeviceInput, DeviceReception, Scenario, Signal, Drop
from .physical_device import PDT

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalScenario(Generic[PDT], Scenario[PDT, Drop]):
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

    @abstractmethod
    def _trigger_direct(
        self, transmissions: list[Signal], devices: list[PDT], calibrate: bool = True
    ) -> list[Signal]:
        """Trigger a synchronized transmission and reception for a selection of managed devices.

        Subroutine of :meth:`PhysicalScenario.trigger_direct<hermespy.hardware_loop.scenario.PhysicalScenario.trigger_direct>`.

        Args:

            transmissions: List of signals to be transmitted.
            devices: List of devices to be triggered.
            calibrate: Apply device calibrations during transmission and reception.

        Returns: List of signals received by all devices.
        """
        ...  # pragma: no cover

    def trigger_direct(
        self, transmissions: list[Signal], devices: list[PDT] | None = None, calibrate: bool = True
    ) -> list[Signal]:
        """Trigger a synchronized transmission and reception for all managed devices.

        Args:

            transmissions: List of signals to be transmitted.
            devices: List of devices to be triggered.
            calibrate: Apply device calibrations during transmission and reception.

        Returns: List of signals received by all devices.
        """

        _devices = self.devices if devices is None else devices

        if len(transmissions) != len(_devices):
            raise ValueError(
                f"The number of transmissions does not match the number of devices ({len(transmissions)} != {len(_devices)})"
            )

        return self._trigger_direct(transmissions, _devices, calibrate)

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

    def _recall_drop(self, group: Group) -> Drop:
        return Drop.from_HDF(group, self.devices)

    def add_device(self, device: PDT) -> None:
        Scenario.add_device(self, device)


PhysicalScenarioType = TypeVar("PhysicalScenarioType", bound=PhysicalScenario)
"""Type of physical scenario"""
