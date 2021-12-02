# -*- coding: utf-8 -*-
"""Test HermesPy simulation executable."""

import unittest
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.channel import ChannelStateInformation
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
            
    def test_min_num_drops_setget(self) -> None:
        """Number of drops property getter should return setter argument."""

        num_drops = 20
        self.simulation.min_num_drops = num_drops

        self.assertEqual(num_drops, self.simulation.min_num_drops)

    def test_min_num_drops_validation(self) -> None:
        """Number of drops property setter should raise ValueError on arguments smaller than one."""

        with self.assertRaises(ValueError):
            self.simulation.min_num_drops = -1

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

        for _ in range(self.no_tx):

            mock_transmitter = Mock()
            mock_transmitter.send.return_value = (np.array([0]), np.array([1j]))
            self.scenario.add_transmitter(mock_transmitter)

        for rx_idx in range(self.no_rx):

            mock_receiver = Mock()
            mock_receiver.receive = lambda signals, noise: signals[0]
            mock_receiver.demodulate = lambda signal, channel, noise: (np.array([0]), np.array([1j]))
            mock_receiver.reference_transmitter.index = rx_idx
            mock_receiver.index = rx_idx

            self.scenario.add_receiver(mock_receiver)

        mock_channel = Mock()
        mock_channel.propagate.return_value = (np.array([0]), ChannelStateInformation.Ideal(num_samples=1))

        for rx_idx in range(self.no_rx):
            for tx_idx in range(self.no_tx):
                self.scenario.set_channel(tx_idx, rx_idx, mock_channel)

        self.drop_run_flag = np.ones((self.no_tx, self.no_rx), dtype=bool)
        self.drop_run_flag[1, 0] = False
        self.drop_run_flag[2, 0] = False
        Simulation.calculate_noise_variance = Mock(return_value=0.0)

    def test_simulation_not_run_for_rx0(self) -> None:

        self.drop_run_flag[:, :] = True
        self.drop_run_flag[:, 0] = False

        tx_signals, _ = Simulation.transmit(self.scenario, self.drop_run_flag)
        propagation_matrix = Simulation.propagate(self.scenario, tx_signals, self.drop_run_flag)
        received_signals = Simulation.receive(self.scenario, propagation_matrix, self.drop_run_flag)
        detected_bits, _ = Simulation.detect(self.scenario, received_signals, self.drop_run_flag)

        self.assertIsNone(detected_bits[0])
        self.assertIsNotNone(detected_bits[1])

    def test_no_flagging_if_no_run_flag_is_passed(self) -> None:

        self.drop_run_flag[:, :] = True
        tx_signals, _ = Simulation.transmit(self.scenario)
        propagation_matrix = Simulation.propagate(self.scenario, tx_signals)
        received_signals = Simulation.receive(self.scenario, propagation_matrix)
        detected_bits = Simulation.detect(self.scenario, received_signals)

        self.assertIsNotNone(detected_bits[0])
        self.assertIsNotNone(detected_bits[1])

    def test_do_not_send_if_tx0_is_flagged(self) -> None:

        self.drop_run_flag[:, :] = True
        self.drop_run_flag[0, :] = False
        transmitted_signals, _ = Simulation.transmit(self.scenario, self.drop_run_flag, None, None)

        self.assertIsNone(transmitted_signals[0])
        self.assertIsNotNone(transmitted_signals[1])
        self.assertIsNotNone(transmitted_signals[2])

    def test_propagation_not_from_tx1_2_to_rx0(self) -> None:

        tx_signals = [np.random.randint(low=0, high=2, size=100) for _ in range(self.no_tx)]

        propagation_matrix = Simulation.propagate(
            self.scenario, tx_signals, self.drop_run_flag
        )
        self.assertEqual(propagation_matrix[0][1], tuple((None, None)))
        self.assertEqual(propagation_matrix[0][2], tuple((None, None)))

    def test_do_not_receive_at_rx0_at_all(self) -> None:

        propagation_matrix = [[(np.random.randint(low=0, high=2, size=(1, 100)),
                                np.random.randint(low=0, high=2, size=(1, 100)))
                               for _ in range(self.no_tx)] for _ in range(self.no_rx)]
        self.drop_run_flag[:, 0] = False
        received_signals = Simulation.receive(
            self.scenario, propagation_matrix, self.drop_run_flag
        )

        self.assertEqual(received_signals[0], (None, None, None))

    def test_no_detection_at_rx0(self) -> None:

        self.drop_run_flag[:, 0] = False
        received_signals = [(Mock(), Mock(), Mock()) for _ in range(self.no_rx)]
        receiver_bits, _ = Simulation.detect(self.scenario, received_signals, self.drop_run_flag)
        self.assertIsNone(receiver_bits[0])
        self.assertIsNotNone(receiver_bits[1])
