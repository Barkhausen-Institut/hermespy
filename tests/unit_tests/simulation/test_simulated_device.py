# -*- coding: utf-8 -*-
"""Test HermesPy simulated device module."""

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.simulation.simulated_device import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSimulatedDevice(TestCase):
    """Test the simulated device base class."""

    def setUp(self) -> None:

        self.scenario = Mock()
        self.position = np.zeros(3)
        self.orientation = np.zeros(3)
        self.topology = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=float)

        self.device = SimulatedDevice(scenario=self.scenario, position=self.position, orientation=self.orientation,
                                      topology=self.topology)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertIs(self.scenario, self.device.scenario)
        assert_array_equal(self.position, self.device.position)
        assert_array_equal(self.orientation, self.device.orientation)
        assert_array_equal(self.topology, self.device.topology)

    def test_scenario_setget(self) -> None:
        """Scenario property setter should return getter argument."""

        self.device = SimulatedDevice()
        self.device.scenario = self.scenario

        self.assertIs(self.scenario, self.device.scenario)

    def test_scenario_set_validation(self) -> None:
        """Overwriting a scenario property should raise a RuntimeError."""

        with self.assertRaises(RuntimeError):
            self.device.scenario = Mock()

    def test_attached(self) -> None:
        """The attached property should return the proper device attachment state."""

        self.assertTrue(self.device.attached)
        self.assertFalse(SimulatedDevice().attached)

    def test_max_frame_duration(self) -> None:
        """Maximum frame duration property should compute the correct duration."""

        transmitter = Mock()
        transmitter.frame_duration = 10
        self.device.transmitters.add(transmitter)

        receiver = Mock()
        receiver.frame_duration = 4
        self.device.receivers.add(receiver)

        self.assertEqual(10, self.device.max_frame_duration)
