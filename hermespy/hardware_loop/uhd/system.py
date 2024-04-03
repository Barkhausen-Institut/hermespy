# -*- coding: utf-8 -*-
"""
==========
UHD System
==========
"""

from usrp_client import System as _UsrpSystem

from hermespy.core import Serializable
from .usrp import UsrpDevice
from ..scenario import PhysicalScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class UsrpSystem(Serializable, PhysicalScenario[UsrpDevice]):
    """Scenario of USRPs running the UHD server application."""

    yaml_tag = "UsrpSystem"
    """YAML serialization tag"""

    def __init__(self, *args, **kwargs) -> None:
        PhysicalScenario.__init__(self, *args, **kwargs)

        # Hacked USRP system (hidden)
        self.__system = _UsrpSystem()

    def new_device(self, ip: str, port: int = 5555, *args, **kwargs) -> UsrpDevice:
        """Create a new UHD device managed by the system.

        Args:

            Device initialization parameters.
            Refer to :class:.UsrpDevice for further details.

        Returns: A handle to the initialized device.
        """
        device = UsrpDevice(ip, port, *args, **kwargs)

        self.add_device(device)
        return device

    def add_device(self, device: UsrpDevice) -> None:
        """Register an existing UHD device to be managed by the system.

        Args:

            device (UsrpDevice):
                The device to be added.
        """

        self.__system.addUsrp(usrpName=str(self.num_devices), client=device._client)
        PhysicalScenario.add_device(self, device)

    def _trigger(self) -> None:
        self.__system.execute()
