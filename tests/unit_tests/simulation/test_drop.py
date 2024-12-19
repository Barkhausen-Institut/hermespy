# -*- coding: utf-8 -*-

from unittest import TestCase

from h5py import File

from hermespy.simulation.scenario import SimulationScenario
from hermespy.simulation.drop import SimulatedDrop

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
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

    def test_hdf_serialization(self) -> None:
        """Test HDF roundtrip serialization"""

        file = File("test.h5", "w", driver="core", backing_store=False)
        group = file.create_group("group")

        self.drop.to_HDF(group)
        deserialization = SimulatedDrop.from_HDF(group, self.scenario.devices, self.scenario.channels)

        file.close()

        self.assertIsInstance(deserialization, SimulatedDrop)
        self.assertEqual(self.drop.timestamp, deserialization.timestamp)
        self.assertEqual(self.drop.num_device_transmissions, deserialization.num_device_transmissions)
        self.assertEqual(self.drop.num_device_receptions, deserialization.num_device_receptions)
