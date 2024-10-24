# -*- coding: utf-8 -*-

from hermespy.core import Signal
from ..scenario import PhysicalScenario
from .device import AudioDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class AudioScenario(PhysicalScenario[AudioDevice]):
    """Scenario of phyical device bindings to sound cards."""

    yaml_tag = "AudioSystem"

    def __init__(self, *args, **kwargs) -> None:
        PhysicalScenario.__init__(self, *args, **kwargs)

    def new_device(self, *args, **kwargs) -> AudioDevice:
        """Create a new UHD device managed by the system.

        Args:

            *args, \**kwargs:
                Device initialization parameters.
                Refer to :class:`AudioDevice` for further details.

        Returns: A handle to the initialized device.
        """

        device = AudioDevice(*args, **kwargs)
        self.add_device(device)

        return device

    def _trigger(self) -> None:
        for device in self.devices:
            device.trigger()

    def _trigger_direct(
        self, transmissions: list[Signal], devices: list[AudioDevice], calibrate: bool = True
    ) -> list[Signal]:
        transmissions = [d.trigger_direct(t) for d, t in zip(devices, transmissions)]
        return transmissions
