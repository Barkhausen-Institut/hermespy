# -*- coding: utf-8 -*-

from os.path import join
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from h5py import File, Group

from hermespy.core import Drop, Signal, DeviceTransmission, DeviceReception
from hermespy.core.drop import EvaluatedDrop

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestDrop(TestCase):
    "Test drop class"

    def setUp(self) -> None:
        self.transmission = DeviceTransmission([], Signal.Create(np.random.standard_normal((1, 10)), 1.0))
        self.reception = DeviceReception(Signal.Create(np.random.standard_normal((1, 10)), 1.0), [], [])
        self.timestamp = 1.2345

        self.drop = Drop(self.timestamp, [self.transmission], [self.reception])

    def test_properties(self) -> None:
        """Test properties"""

        self.assertEqual(self.drop.timestamp, self.timestamp)
        self.assertSequenceEqual(self.drop.device_transmissions, [self.transmission])
        self.assertSequenceEqual(self.drop.device_receptions, [self.reception])
        self.assertEqual(1, self.drop.num_device_transmissions)
        self.assertEqual(1, self.drop.num_device_receptions)
        self.assertSequenceEqual([[]], self.drop.operator_inputs)

    def test_hdf_serialization(self) -> None:
        """Test HDF roundtrip serialization"""

        with TemporaryDirectory() as tempdir:
            file_path = join(tempdir, "test.hdf")

            file = File(file_path, "w")
            group = file.create_group("g1")
            self.drop.to_HDF(group)
            file.close()

            file = File(file_path, "r")
            deserialization = Drop.from_HDF(file["g1"])

            mock_scenario = Mock()
            mock_device = Mock()
            mock_device.receivers = []
            mock_device.transmitters = []
            mock_scenario.devices = [mock_device]
            recalled_drop = Drop.from_HDF(file["g1"], mock_scenario.devices)
            file.close()

        self.assertEqual(self.drop.timestamp, deserialization.timestamp)
        self.assertEqual(self.drop.num_device_transmissions, deserialization.num_device_transmissions)
        self.assertEqual(self.drop.num_device_receptions, deserialization.num_device_receptions)
        self.assertEqual(self.drop.timestamp, recalled_drop.timestamp)
        self.assertEqual(self.drop.num_device_transmissions, recalled_drop.num_device_transmissions)
        self.assertEqual(self.drop.num_device_receptions, recalled_drop.num_device_receptions)


class TestEvaluatedDrop(TestCase):
    """Test evaluated drop class"""

    def setUp(self) -> None:
        self.transmission = DeviceTransmission([], Signal.Create(np.random.standard_normal((1, 10)), 1.0))
        self.reception = DeviceReception(Signal.Create(np.random.standard_normal((1, 10)), 1.0), [], [])
        self.timestamp = 1.2345
        self.artifacts = [Mock()]

        self.drop = EvaluatedDrop(self.timestamp, [self.transmission], [self.reception], self.artifacts)

    def test_properties(self) -> None:
        """Test properties"""

        self.assertEqual(1, self.drop.num_artifacts)
        self.assertSequenceEqual(self.drop.artifacts, self.artifacts)
