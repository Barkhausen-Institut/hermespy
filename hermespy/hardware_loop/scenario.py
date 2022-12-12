# -*- coding: utf-8 -*-
"""
========================
Physical Device Scenario
========================
"""

from abc import abstractmethod, ABC
from time import time
from typing import Generic, List, Optional, TypeVar

from hermespy.core import Scenario, DeviceReception, DeviceTransmission, Drop, Signal
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

    def receive_devices(self, receptions: Optional[List[Signal]] = None) -> List[Signal]:

        if receptions is None:

            receptions = [device.receive() for device in self.devices]
            return receptions

        else:
            return Scenario.receive_devices(self, receptions)

    def _drop(self) -> Drop:

        # Generate device transmissions
        operator_transmissions = self.transmit_operators()
        transmitted_device_signals = self.transmit_devices()
        device_transmissions = [DeviceTransmission(s, i) for s, i in zip(transmitted_device_signals, operator_transmissions)]

        # Trigger the full scenario for phyiscal transmission and reception
        timestamp = time()
        self._trigger()

        received_device_signals = self.receive_devices()
        operator_receptions = self.receive_operators()
        device_receptions = [DeviceReception(s, None, i) for s, i in zip(received_device_signals, operator_receptions)]

        return Drop(timestamp, device_transmissions, device_receptions)


PhysicalScenarioType = TypeVar("PhysicalScenarioType", bound=PhysicalScenario)
"""Type of physical scenario"""
