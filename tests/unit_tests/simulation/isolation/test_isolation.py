# -*- coding: utf-8 -*-

from unittest import TestCase

from numpy.random import default_rng
from numpy.testing import assert_array_equal

from hermespy.core import FloatingError, Signal
from hermespy.simulation import Isolation, SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class IsolationDummy(Isolation):
    """Isolation implementation for test purposes"""

    def _leak(self, signal: Signal) -> Signal:
        return signal


class TestIsolation(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.device = SimulatedDevice()
        self.isolation = IsolationDummy(device=self.device)

    def test_assert_leak(self) -> None:
        """Leak routine should raise ValueErrors on invalid configurations"""

        with self.assertRaises(ValueError):
            _ = self.isolation.leak(None)

        with self.assertRaises(ValueError):
            _ = self.isolation.leak(Signal.Empty(1.0, self.device.antennas.num_receive_antennas + 1, carrier_frequency=0.0))

        with self.assertRaises(FloatingError):
            isolation = IsolationDummy()
            _ = isolation.leak(Signal.Empty(1.0, self.device.antennas.num_receive_antennas, carrier_frequency=0.0))

    def test_leak(self) -> None:
        """Leak routine should be properly called"""

        expected_signal = Signal.Create(self.rng.normal(size=10) + 1j * self.rng.normal(size=10), self.device.antennas.num_receive_antennas, carrier_frequency=0.0)
        leaked_signal = self.isolation.leak(expected_signal)

        assert_array_equal(expected_signal, leaked_signal)
