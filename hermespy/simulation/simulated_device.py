# -*- coding: utf-8 -*-
"""
=================
Simulated Devices
=================
"""

from __future__ import annotations
from abc import abstractmethod
from typing import Optional

import numpy as np
from scipy.constants import speed_of_light

from hermespy.core import Device, FloatingError
from .scenario import Scenario
from .rf_chain import RfChain

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SimulatedDevice(Device):
    """Representation of a device simulating hardware.

    Simulated devices are required to attach to a scenario in order to simulate proper channel propagation.
    """

    __slots__ = ['__scenario', '__rf_chain']

    __scenario: Optional[Scenario]          # Scenario this device is attached to

    def __init__(self,
                 scenario: Optional[Scenario] = None,
                 num_antennas: Optional[int] = None,
                 rf_chain: Optional[RfChain] = None,
                 *args,
                 **kwargs) -> None:
        """
        Args:

            scenario (Scenario, optional):
                Scenario this device is attached to.
                By default, the device is considered floating.

            num_antennas (int, optional):
                Number of antennas.
                The information is used to initialize the simulated device as a Uniform Linear Array with
                half-wavelength antenna spacing.

            rf_chain (RfChain, optional):
                Model of the device's radio frequency amplification chain.

            *args:
                Device base class initialization parameters.

            **kwargs:
                Device base class initialization parameters.
        """

        # Init base class
        Device.__init__(self, *args, **kwargs)

        self.scenario = scenario

        # If num_antennas is configured initialize the modem as a Uniform Linear Array
        # with half wavelength element spacing
        if num_antennas is not None:

            if not np.array_equal(self.topology, np.zeros((1, 3))):
                raise ValueError("The num_antennas and topology parameters are mutually exclusive")

            # For a carrier frequency of 0.0 we will initialize all antennas at the same position.
            half_wavelength = 0.0
            if self.__carrier_frequency > 0.0:
                half_wavelength = .5 * speed_of_light / self.__carrier_frequency

            self.topology = half_wavelength * np.outer(np.arange(num_antennas), np.array([1., 0., 0.]))

    @property
    def scenario(self) -> Scenario:
        """Scenario this device is attached to.

        Returns:
            Scenario:
                Handle to the scenario this device is attached to.

        Raises:
            FloatingError: If the device is currently floating.
            RuntimeError: Trying to overwrite the scenario of an already attached device.
        """

        if self.__scenario is None:
            raise FloatingError("Error trying to access the scenario of a floating modem")

        return self.__scenario

    @scenario.setter
    def scenario(self, scenario: Scenario) -> None:
        """Set the scenario this device is attached to. """

        if hasattr(self, '_SimulatedDevice__scenario') and self.__scenario is not None:
            raise RuntimeError("Error trying to modify the scenario of an already attached modem")

        self.__scenario = scenario

    @property
    def attached(self) -> bool:
        """Attachment state of this device.

        Returns:
            bool: `True` if the device is currently attached, `False` otherwise.
        """

        return self.__scenario is not None
