# -*- coding: utf-8 -*-

from __future__ import annotations
from typing_extensions import override

from hermespy.core import Signal
from ..scenario import PhysicalScenario
from .device import AudioDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class AudioScenario(PhysicalScenario[AudioDevice]):
    """Scenario of phyical device bindings to sound cards."""

    def __init__(self, *args, **kwargs) -> None:
        PhysicalScenario.__init__(self, *args, **kwargs)

    @classmethod
    @override
    def _device_type(cls) -> type[AudioDevice]:
        return AudioDevice

    @override
    def _trigger(self) -> None:
        for device in self.devices:
            device.trigger()

    @override
    def _trigger_direct(
        self, transmissions: list[Signal], devices: list[AudioDevice], calibrate: bool = True
    ) -> list[Signal]:
        transmissions = [d.trigger_direct(t) for d, t in zip(devices, transmissions)]
        return transmissions
