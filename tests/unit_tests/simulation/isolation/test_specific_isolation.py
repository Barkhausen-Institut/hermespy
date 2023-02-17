# -*- coding: utf-8 -*-

from unittest import TestCase

from numpy.random import default_rng

from hermespy.core import Signal
from hermespy.simulation import SimulatedDevice, SpecificIsolation
from hermespy.tools import db2lin, lin2db

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


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
