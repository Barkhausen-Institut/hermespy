# -*- coding: utf-8 -*-

from unittest import TestCase

from h5py import File

from hermespy.simulation.scenario import SimulationScenario
from hermespy.simulation.drop import SimulatedDrop
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSimulatedDrop(TestCase):
    """Test the simulated drop data structure"""

    def setUp(self) -> None:
        self.scenario = SimulationScenario()
        self.device_alpha = self.scenario.new_device()
        self.device_beta = self.scenario.new_device()

        self.drop: SimulatedDrop = self.scenario.drop()

    def test_channel_realizations(self) -> None:
        """Channel realizations property should return the correct realizations"""

        self.assertEqual(1, len(self.drop.channel_realizations))

    def test_serialization(self) -> None:
        """Test drop serialization"""

        test_roundtrip_serialization(self, self.drop)
