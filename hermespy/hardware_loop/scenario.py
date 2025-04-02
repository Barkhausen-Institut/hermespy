# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from time import time
from typing import Generic, TypeVar
from typing_extensions import override

from hermespy.core import DeviceInput, DeviceReception, DeviceState, Scenario, Signal, Drop
from .physical_device import PDT

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalScenario(Generic[PDT], Scenario[PDT, DeviceState, Drop]):
    """Scenario of physical device bindings.

    Managing physical devices by a scenario enables synchronized triggering
    and shared random seed configuration.
    """

    def __init__(self, seed: int | None = None, devices: Sequence[PDT] | None = None) -> None:
        Scenario.__init__(self, seed, devices)

    @classmethod
    @override
    def _drop_type(cls) -> type[Drop]:
        return Drop

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
        states: Sequence[DeviceState | None] | None = None,
        notify: bool = True,
    ) -> Sequence[DeviceReception]:
        """Receive over all scenario devices.

        Internally calls :meth:`Scenario.process_inputs<hermespy.core.scenario.Scenario.process_inputs>` and :meth:`Scenario.receive_devices<hermespy.core.scenario.Scenario.receive_devices>`.

        Args:

            impinging_signals:
                List of signals impinging onto the devices.
                If not specified, the device will download the signal samples from its binding.

            states:
                States of the transmitting devices.
                If not specified, the current device states will be queried by calling :meth:`Device.state<hermespy.core.device.Device.state>`.

            notify:
                Notify the receiving DSP layer's callbacks about the reception results.
                Enabled by default.

        Returns: List of the processed device input information.

        Raises:
            ValueError: If the number of `impinging_signals` does not match the number of registered devices.
        """

        _impinging_signals = (
            [None] * self.num_devices if impinging_signals is None else impinging_signals
        )
        _states = [None] * self.num_devices if states is None else states

        # Generate inputs
        device_inputs = [d.process_input(i, s) for d, s, i in zip(self.devices, _states, _impinging_signals)]  # type: ignore

        # Generate operator receptions
        receptions = self.receive_operators(device_inputs, _states, notify)

        # Generate device receptions
        return [
            DeviceReception.From_ProcessedDeviceInput(i, r)
            for i, r in zip(device_inputs, receptions)
        ]

    @override
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
