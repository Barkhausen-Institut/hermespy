# -*- coding: utf-8 -*-
"""
=====================
Audio Device Scenario
=====================
"""

from ..scenario import PhysicalScenario
from .device import AudioDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
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

            *args, **kwargs:
                Device initialization parameters.
                Refer to :class:`AudioDevice` for further details.

        Returns: A handle to the initialized device.
        """

        device = AudioDevice(*args, **kwargs)
        self.add_device(device)

        return device

    def _trigger(self) -> None:
        # Trigger of the audio scenario is not implemented
        return
