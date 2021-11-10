# -*- coding: utf-8 -*-
"""Test HermesPy simulation executable."""

import unittest
from unittest.mock import Mock, create_autospec

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.modem import transmitter
from hermespy.channel import Channel
from hermespy.scenario.scenario import Scenario
from hermespy.simulator_core import Simulation, SNRType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSimulation(unittest.TestCase):
    """Test the simulation executable, the base class for simulation operations."""

    def setUp(self) -> None:

        self.simulation = Simulation()

    def test_snr_type_setget(self) -> None:
        """SNR type property getter should return setter argument."""

        for snr_type in SNRType:

            # Enum set
            self.simulation.snr_type = snr_type
            self.assertEqual(snr_type, self.simulation.snr_type)

            # String set
            self.simulation.snr_type = str(snr_type.name)
            self.assertEqual(snr_type, self.simulation.snr_type)

    def test_noise_loop_setget(self) -> None:
        """Noise loop property getter should return setter argument for both arrays and lists."""

        noise_loop = np.arange(20)

        # Set via vector
        self.simulation.noise_loop = noise_loop
        assert_array_equal(noise_loop, self.simulation.noise_loop)

        # Set via list
        self.simulation.noise_loop = noise_loop.tolist()
        assert_array_equal(noise_loop, self.simulation.noise_loop)

    def test_noise_loop_validation(self) -> None:
        """Noise loop property setter should raise ValueErrors on invalid arguments."""

        with self.assertRaises(ValueError):
            self.simulation.noise_loop = np.empty(0, dtype=float)

        with self.assertRaises(ValueError):
            self.simulation.noise_loop = np.zeros((2, 2), dtype=float)

        with self.assertRaises(ValueError):
            self.simulation.noise_loop = []

        with self.assertRaises(ValueError):
            self.simulation.noise_loop = [[2, 3], [4, 5]]

    def test_to_yaml(self) -> None:
        """Test YAML serialization dump validity."""
        pass

    def test_from_yaml(self) -> None:
        """Test YAML serialization recall validity."""
        pass

class TestStoppingCriteria(unittest.TestCase):
    def setUp(self) -> None:
        self.no_rx = 2
        self.no_tx = 3
        self.simulation = Simulation()
        self.simulation.noise_loop = np.arange(30)
        self.simulation.min_num_drops = 1
        self.simulation.max_num_drops = 1


        self.scenario = Scenario()
        self.scenario.drop_duration = 0.1
        for rx_idx in range(self.no_rx):
            self.scenario.add_receiver(Mock())
        for tx_idx in range(self.no_tx):
            self.scenario.add_transmitter(Mock())

        mock_channel = Mock()
        mock_channel.propagate.return_value = (np.array([0]), np.array([0]))

        for rx_idx in range(self.no_rx):
            for tx_idx in range(self.no_tx):
                self.scenario.set_channel(tx_idx, rx_idx, mock_channel)

    def test_propagation_only_from_tx1_2_to_rx0(self) -> None:
        tx_signals = [np.random.randint(low=0, high=2,size=100) for _ in range(self.no_tx)]
        snr_mask = np.ones((self.no_tx, self.no_rx), dtype=bool)
        snr_mask[1, 0] = False
        snr_mask[2, 0] = False
        propagation_matrix = Simulation.propagate(
            self.scenario, tx_signals, snr_mask
        )

        self.assertEqual(propagation_matrix[0][1], tuple())
        self.assertEqual(propagation_matrix[0][2], tuple())