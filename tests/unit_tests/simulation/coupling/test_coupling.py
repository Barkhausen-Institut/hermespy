# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from hermespy.core import FloatingError, Signal
from hermespy.simulation import Coupling, SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockCoupling(Coupling):
    """Implementation of the abstract coupling base class for testing only"""

    def _transmit(self, signal: Signal) -> Signal:
        return signal

    def _receive(self, signal: Signal) -> Signal:
        return signal


class TestCoupling(TestCase):
    """Test coupling model"""

    def setUp(self) -> None:
        self.device = SimulatedDevice()
        self.coupling = MockCoupling(device=self.device)

    def test_device_setget(self) -> None:
        """Device property getter should return setter argument"""

        expected_device = SimulatedDevice()
        self.coupling.device = expected_device

        self.assertEqual(self.coupling.device, expected_device)

    def test_transmit_validation(self) -> None:
        """Transmit method should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.coupling.transmit(Signal(np.zeros((2, 1)), 1.0, 0.0))

        self.coupling.device = None
        with self.assertRaises(FloatingError):
            self.coupling.transmit(Mock())

    def test_receive_validation(self) -> None:
        """Receive method should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.coupling.receive(Signal(np.zeros((2, 1)), 1.0, 0.0))

        self.coupling.device = None
        with self.assertRaises(FloatingError):
            self.coupling.receive(Mock())
