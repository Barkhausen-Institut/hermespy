# -*- coding: utf-8 -*-
"""
================
Physical Devices
================
"""

from __future__ import annotations

from abc import abstractmethod

from hermespy.core import Device

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PhysicalDevice(Device):
    """Base representing any device controlling real hardware."""

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:
            *args:
                Device base class initialization parameters.

            **kwargs:
                Device base class initialization parameters.
        """

        # Init base class
        Device.__init__(self, *args, **kwargs)

    @abstractmethod
    def trigger(self) -> None:
        """Trigger the device."""
        ...
