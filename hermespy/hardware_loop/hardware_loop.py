# -*- coding: utf-8 -*-
"""HermesPy hardware loop configuration.
"""

from __future__ import annotations

from hermespy.core import Executable, Scenario
from .physical_device import PhysicalDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class HardwareLoop(Executable, Scenario[PhysicalDevice]):
    """Hermespy hardware loop configuration.
    """

    yaml_tag = u'Loop'

    def __init__(self) -> None:

        Executable.__init__(self)
        Scenario[PhysicalDevice].__init__(self)

    def run(self) -> None:

        # Trigger devices
        for device in self.devices:
            device.trigger()
