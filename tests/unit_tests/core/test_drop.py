# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from hermespy.core import Drop, Signal, DeviceTransmission, DeviceReception
from hermespy.core.drop import EvaluatedDrop
from unit_tests.core.test_factory import test_roundtrip_serialization

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

    def test_serialization(self) -> None:
        """Test drop serialization"""

        test_roundtrip_serialization(self, self.drop)


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
