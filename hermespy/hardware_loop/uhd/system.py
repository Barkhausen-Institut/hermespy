# -*- coding: utf-8 -*-

from typing import Type
from typing_extensions import override

from usrp_client import System as _UsrpSystem

from hermespy.core import Signal
from .usrp import UsrpDevice
from ..physical_device import PDT
from ..scenario import PhysicalScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class UsrpSystem(PhysicalScenario[UsrpDevice]):
    """Scenario of USRPs running the UHD server application."""

    def __init__(self, *args, **kwargs) -> None:
        PhysicalScenario.__init__(self, *args, **kwargs)

        # Hacked USRP system (hidden)
        self.__system = _UsrpSystem()

    @classmethod
    @override
    def _device_type(cls) -> Type[UsrpDevice]:
        return UsrpDevice

    def new_device(self, ip: str, port: int = 5555, *args, **kwargs) -> UsrpDevice:
        """Create a new UHD device managed by the system.

        Args:

            ip: IP address of the UHD server application.
            port: Port number of the UHD server application.
            args, kwargs:
                Device initialization parameters.
                Refer to :class:`UsrpDevice<hermespy.hardware_loop.uhd.usrp.UsrpDevice>` for further details.

        Returns: A handle to the initialized device.
        """
        device = UsrpDevice(ip, port, *args, **kwargs)

        self.add_device(device)
        return device

    def add_device(self, device: UsrpDevice) -> None:
        """Register an existing UHD device to be managed by the system.

        Args:

            device (UsrpDevice): The device to be added.
        """

        self.__system.addUsrp(usrpName=str(self.num_devices), client=device._client)
        PhysicalScenario.add_device(self, device)

    def _trigger(self) -> None:
        self.__system.execute()

    def _trigger_direct(
        self, transmissions: list[Signal], devices: list[PDT], calibrate: bool = True
    ) -> list[Signal]:
        # Upload transmissions to devices
        for transmission, device in zip(transmissions, devices):
            device._upload(transmission)

        # Trigger devices
        self._trigger()

        # Download receptions from devices
        receptions = [device._download() for device in devices]
        return receptions
