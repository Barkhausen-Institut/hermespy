# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import patch, PropertyMock

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_equal

from hermespy.core import Signal
from hermespy.simulation import SimulatedDevice, SimulatedUniformArray, SimulatedIdealAntenna, SpecificIsolation
from hermespy.tools import db2lin, lin2db

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSpecificIsolation(TestCase):
    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.device = SimulatedDevice()
        self.isolation = SpecificIsolation(device=self.device)

    def test_isolation_validation(self) -> None:
        """Isolation property setter should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            self.isolation.isolation = np.array([[[1.234]]])

        self.device.antennas = SimulatedUniformArray(SimulatedIdealAntenna, 1.0, (2, 1, 1))

        with self.assertRaises(ValueError):
            self.isolation.isolation = 4.56

    def test_isolation_setget(self) -> None:
        """Isolation property getter should return setter argument interpretation"""

        isolation = np.array([[1.234]])
        self.isolation.isolation = isolation

        assert_array_equal(isolation, self.isolation.isolation)

        scalar_isolation = 4.56
        self.isolation.isolation = scalar_isolation

        assert_array_equal(np.array([[scalar_isolation]]), self.isolation.isolation)

    def test_leaking_power(self) -> None:
        """The correct amount of power should be leaked"""

        num_samples = 100
        expected_isolation = 20
        self.isolation.isolation = db2lin(expected_isolation)

        signal = Signal(self.rng.normal(size=(self.device.num_antennas, num_samples)) + 1j * self.rng.normal(size=(self.device.num_antennas, num_samples)), 1.0)
        leaking_signal = self.isolation.leak(signal)

        realised_isolation = lin2db(signal.power) - lin2db(leaking_signal.power)
        self.assertAlmostEqual(expected_isolation, realised_isolation)

    def test_leak_validation(self) -> None:
        """Leak subroutine should raise RuntimeErrors on invalid internal states"""

        signal = Signal.empty(1, 1, 0)

        self.isolation._SpecificIsolation__leakage_factors = None
        with self.assertRaises(RuntimeError):
            _ = self.isolation._leak(signal)

        self.isolation._SpecificIsolation__leakage_factors = np.ones((2, 1))
        with self.assertRaises(RuntimeError):
            _ = self.isolation._leak(signal)

        self.isolation._SpecificIsolation__leakage_factors = np.ones((1, 2))
        with self.assertRaises(RuntimeError):
            _ = self.isolation._leak(signal)
