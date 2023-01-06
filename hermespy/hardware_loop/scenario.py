# -*- coding: utf-8 -*-
"""
========================
Physical Device Scenario
========================
"""

from abc import abstractmethod, ABC
from time import time
from typing import Generic, Iterable, List, Optional, TypeVar, Union

from hermespy.core import DeviceInput, DeviceReception, Scenario, Drop, Signal
from hermespy.simulation import SimulationScenario
from .physical_device import PhysicalDeviceType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalScenario(Scenario[PhysicalDeviceType], ABC, Generic[PhysicalDeviceType]):
    """Scenario of physical device bindings.

    Managing physical devices by a scenario enables synchronized triggering
    and shared random seed configuration.
    """

    @abstractmethod
    def _trigger(self) -> None:
        """Trigger synchronzed transmission and reception for all managed devices."""
        ...  # pragma no cover

    def receive_devices(self,
                        impinging_signals: Optional[List[Union[DeviceInput, Signal, Iterable[Signal]]]] = None,
                        cache: bool = True) -> List[DeviceReception]:
        """Receive over all scenario devices.

        Internally calls :meth:`Scenario.process_inputs` and :meth:`Scenario.receive_devices`.
        
        Args:

            impinging_signals (List[Union[DeviceInput, Signal, Iterable[Signal]]], optional):
                List of signals impinging onto the devices.
                If not specified, the device will download the signal samples from its binding.

            cache (bool, optional):
                Cache the operator inputs at the registered receive operators for further processing.
                Enabled by default.

        Returns: List of the processed device input information.

        Raises:

            ValueError: If the number of `impinging_signals` does not match the number of registered devices.
        """

        impinging_signals = [None] * self.num_devices if impinging_signals is None else impinging_signals

        # Generate inputs
        device_inputs = [d.process_input(i, cache) for d, i in zip(self.devices, impinging_signals)]

        # Generate operator receptions
        receptions = self.receive_operators(device_inputs)

        # Generate device receptions
        return [DeviceReception.From_ProcessedDeviceInput(i, r) for i, r in zip(device_inputs, receptions)]


    def _drop(self) -> Drop:

        # Generate device transmissions
        device_transmissions = self.transmit_devices()

        # Trigger the full scenario for phyiscal transmission and reception
        timestamp = time()
        self._trigger()

        # Generate device receptions
        device_receptions = self.receive_devices()

        return Drop(timestamp, device_transmissions, device_receptions)


PhysicalScenarioType = TypeVar("PhysicalScenarioType", bound=PhysicalScenario)
"""Type of physical scenario"""


class SimulatedPhysicalScenario(SimulationScenario, PhysicalScenario):
    """Simulated physical scenario for testing purposes."""

    def _trigger(self) -> None:
        
        # Triggering does nothing
        pass
