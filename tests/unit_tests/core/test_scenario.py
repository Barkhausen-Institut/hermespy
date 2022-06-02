# -*- coding: utf-8 -*-
"""Test HermesPy scenario description class."""

import numpy as np
import numpy.random as rnd
from typing import List
from unittest import TestCase
from unittest.mock import Mock
from itertools import product
from numpy.testing import assert_array_equal

from hermespy.core.scenario import Scenario

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "3.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestScenario(TestCase):
    """Test scenario base class."""

    def setUp(self) -> None:

        self.rng = rnd.default_rng(42)
        self.random_root = Mock()
        self.random_root._rng = self.rng

        self.drop_duration = 1e-3
        self.scenario = Scenario()
        self.scenario.random_mother = self.random_root

        self.transmitter_alpha = Mock()
        self.transmitter_beta = Mock()
        self.receiver_alpha = Mock()
        self.receiver_beta = Mock()

        self.device_alpha = Mock()
        self.device_beta = Mock()
        self.device_alpha.max_frame_duration = 10
        self.device_beta.max_frame_duration = 2
        self.device_alpha.transmitters = [self.transmitter_alpha]
        self.device_beta.transmitters = [self.transmitter_beta]
        self.device_alpha.receivers = [self.receiver_alpha]
        self.device_beta.receivers = [self.receiver_beta]
        self.scenario.add_device(self.device_alpha)
        self.scenario.add_device(self.device_beta)

    def test_drop_duration_setget(self) -> None:
        """The drop duration property getter should return the setter argument,"""

        drop_duration = 12345
        self.scenario.drop_duration = drop_duration

        self.assertEqual(drop_duration, self.scenario.drop_duration)

    def test_drop_duration_validation(self) -> None:
        """The drop duration property setter should raise a ValueError on negative arguments."""

        with self.assertRaises(ValueError):
            self.scenario.drop_duration = -1

        try:
            self.scenario.drop_duration = 0.0

        except ValueError:
            self.fail("Setting a drop duration of zero should not result in an error throw")

    def test_drop_duration_computation(self) -> None:
        """If the drop duration is set to zero,
        the property getter should return the maximum frame duration as drop duration."""

        max_frame_duration = 10    # Results from the setUp transmit mock
        self.scenario.drop_duration = 0.0

        self.assertEqual(max_frame_duration, self.scenario.drop_duration)

    def test_add_device(self) -> None:
        """Adding a device should register said device to this scenario"""

        device = Mock()
        self.scenario.add_device(device)

        self.assertTrue(self.scenario.device_registered(device))

    def test_add_device_validation(self) -> None:
        """Adding an already registered device should raise a ValueError."""

        with self.assertRaises(ValueError):
            self.scenario.add_device(self.device_alpha)

    def test_transmitters(self) -> None:
        """Transmitters property should return correct list of transmitters"""

        expected_transmitters = [self.transmitter_alpha, self.transmitter_beta]
        self.assertCountEqual(expected_transmitters, self.scenario.transmitters)

    def test_receivers(self) -> None:
        """Receivers property should return correct list of receivers"""

        expected_receivers = [self.receiver_alpha, self.receiver_beta]
        self.assertCountEqual(expected_receivers, self.scenario.receivers)
        
    def test_operators(self) -> None:
        """Receivers property should return correct list of operators"""

        expected_operators = [self.transmitter_alpha, self.transmitter_beta, self.receiver_alpha, self.receiver_beta]
        self.assertCountEqual(expected_operators, self.scenario.operators)
