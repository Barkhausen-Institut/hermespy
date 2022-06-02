# -*- coding: utf-8 -*-
"""Test HermesPy physical device module."""

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.hardware_loop.physical_device import PhysicalDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestPhysicalDevice(TestCase):
    """Test the physical device base class."""

    def setUp(self) -> None:

        self.device = PhysicalDevice()
