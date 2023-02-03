# -*- coding: utf-8 -*-

from unittest import TestCase

from numpy.random import default_rng

from hermespy.core import Signal
from hermespy.simulation import SimulatedDevice, SpecificIsolation
from hermespy.tools import db2lin, lin2db


class TestSpecificIsolation(TestCase):

    def setUp(self) -> None:

        self.rng = default_rng(42)

        self.device = SimulatedDevice()
        self.isolation = SpecificIsolation(device=self.device)

    def test_leaking_power(self) -> None:
        """The correct amount of power should be leaked"""

        num_samples = 100
        expected_isolation = 20
        self.isolation.isolation = db2lin(expected_isolation)

        signal = Signal(self.rng.normal(size=(self.device.num_antennas, num_samples)) + 1j * self.rng.normal(size=(self.device.num_antennas, num_samples)), 1.)
        leaking_signal = self.isolation.leak(signal)
        
        realised_isolation = lin2db(signal.power) - lin2db(leaking_signal.power)
        self.assertAlmostEqual(expected_isolation, realised_isolation)
