# -*- coding: utf-8 -*-
"""
========================
Physical Device Scenario
========================
"""

from abc import abstractmethod, ABC
from typing import Generic

from hermespy.core import Scenario
from .physical_device import PhysicalDevice, PhysicalDeviceType


class PhysicalScenario(Generic[PhysicalDeviceType], Scenario[PhysicalDeviceType], ABC):
    """Scenario of physical device bindings.

    Managing physical devices by a scenario enables synchronized triggering
    and shared random seed configuration.
    """

    @abstractmethod
    def trigger(self) -> None:
        """Trigger synchronzed transmission and reception for all managed devices."""
        ...  # pragma no cover
