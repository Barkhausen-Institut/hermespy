# -*- coding: utf-8 -*-

from typing import Generic

from .device import DeviceType
from .executable import Executable
from .scenario import ScenarioType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Pipeline(Executable, Generic[ScenarioType, DeviceType]):
    """Base class of HermesPy simulation pipelines."""

    __scenario: ScenarioType  # The simulated scenario.
    __num_drops: int  # Number of data drops being generated

    def __init__(self, scenario: ScenarioType, num_drops: int = 1, *args, **kwargs) -> None:
        """
        Args:

            scenario: The simulated scenario.

            num_drops:
                Number of consecutive triggers occuring during :meth:`run<hermespy.core.executable.Executable.run>`,
                resulting in `num_drops` data drops being generated.
                One by default.
        """

        Executable.__init__(self, *args, **kwargs)

        self.__scenario = scenario
        self.num_drops = num_drops

    @property
    def scenario(self) -> ScenarioType:
        """The simulated scenario.

        Returns: Handle to the scenario.
        """

        return self.__scenario

    @property
    def num_drops(self) -> int:
        """Number of generated data drops.

        Each drop is generated from a dedicated system triggering.

        Returns: The number of drops.

        Raises:
            ValueError: For `num_drops` smaller than one.
        """

        return self.__num_drops

    @num_drops.setter
    def num_drops(self, value: int) -> None:
        if value < 1:
            raise ValueError("Number of drops must be greater than zero")

        self.__num_drops = value

    def add_device(self, device: DeviceType) -> None:
        """Add an exsting device to the scenario.

        Convenience function pointing to :meth:`hermespy.core.scenario.Scenario.add_device`.

        Args:
            device: New device to be added to the scenario.

        Raises:
            ValueError: If the device already exists.
            RuntimeError: If the scenario is not in default mode.
            RuntimeError: If the scenario does not allow for the creation or addition of new devices.
        """

        self.scenario.add_device(device)

    def new_device(self, *args, **kwargs) -> DeviceType:
        """Add a new device to the scenario.

        Convenience function pointing to :meth:`hermespy.core.scenario.Scenario.new_device`.

        Returns: Handle to the created device.

        Raises:
            RuntimeError: If the scenario is not in default mode.
            RuntimeError: If the scenario does not allow for the creation or addition of new devices.
        """

        return self.scenario.new_device(*args, **kwargs)

    def device_index(self, device: DeviceType) -> int:
        """Get the index of a device in the scenario.

        Convenience function pointing to :meth:`hermespy.core.scenario.Scenario.device_index`.

        Args:
            device: Device to be searched for.

        Returns: The index of the device.

        Raises:
            ValueError: If the device does not exist.
        """

        return self.scenario.device_index(device)
