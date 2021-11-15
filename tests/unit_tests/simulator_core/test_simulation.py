# -*- coding: utf-8 -*-
"""Test HermesPy simulation executable."""

import unittest
from unittest.mock import Mock
from typing import List, Tuple

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.modem import Receiver
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

class TestReceiver(Receiver):
    """Overrides some methods from Receiver to facilitate testing."""
    def __init__(self, no_antennas: int, scenario: Scenario) -> None:
        self.no_antennas = no_antennas
        self.__scenario = scenario
        self.reference_transmitter = Mock()
        self.reference_transmitter.index = 0

    @property
    def reference_transmitter(self) -> Mock:
        return self.__reference_transmitter

    @reference_transmitter.setter
    def reference_transmitter(self, val: Mock) -> None:
        self.__reference_transmitter = val

    @property
    def scenario(self) -> Scenario:
        return self.__scenario

    @scenario.setter
    def scenario(self, val: Scenario) -> None:
        self.__scenario = val

    def receive(self, rf_signals: List[Tuple[np.ndarray, float]],
                noise_variance: float) -> np.ndarray:
        """Receive signals and just add them.

        Args:
            rf_signals (List[Tuple[np.ndarray, float]]):
                List containing pairs of rf-band samples and their respective carrier frequencies.

            noise_variance (float):
                Variance (i.e. power) of the thermal noise added to the signals during reception.
        """
        no_samples = rf_signals[0][0].shape[1]
        received_signal = np.zeros((self.no_antennas, no_samples))
        for signal, _  in rf_signals:
            if signal is not None:
                for antenna_idx in range(signal.shape[0]):
                    received_signal[antenna_idx, :] += signal[antenna_idx, :]

        return received_signal

    def demodulate(self, signal, channel, noise) -> np.array:
        return np.ones(10)

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
        self.scenario.add_receiver(TestReceiver(1, self.scenario))
        self.scenario.add_receiver(TestReceiver(1, self.scenario))

        mock_transmitter = Mock()
        mock_transmitter.send.return_value = np.array([0])
        self.scenario.add_transmitter(Mock())
        self.scenario.add_transmitter(Mock())
        self.scenario.add_transmitter(Mock())
        
        mock_channel = Mock()
        mock_channel.propagate.return_value = (np.array([0]), np.array([0]))

        for rx_idx in range(self.no_rx):
            for tx_idx in range(self.no_tx):
                self.scenario.set_channel(tx_idx, rx_idx, mock_channel)

        self.snr_mask = np.ones((self.no_tx, self.no_rx), dtype=bool)
        self.snr_mask[1, 0] = False
        self.snr_mask[2, 0] = False

    def test_do_not_send_if_tx0_is_flagged(self) -> None:
        self.snr_mask[:, :] = True
        self.snr_mask[0, :] = False
        transmitted_signals = Simulation.transmit(self.scenario, self.snr_mask, None, None)

        self.assertIsNone(transmitted_signals[0])
        self.assertIsNotNone(transmitted_signals[1])
        self.assertIsNotNone(transmitted_signals[2])

    def test_propagation_not_from_tx1_2_to_rx0(self) -> None:
        tx_signals = [np.random.randint(low=0, high=2,size=100) for _ in range(self.no_tx)]

        propagation_matrix = Simulation.propagate(
            self.scenario, tx_signals, self.snr_mask
        )
        self.assertEqual(propagation_matrix[0][1], tuple((None, None)))
        self.assertEqual(propagation_matrix[0][2], tuple((None, None)))

    def test_do_not_receive_from_tx1_2_to_rx0(self) -> None:
        propagation_matrix = [[(np.random.randint(low=0, high=2, size=(1,100)),
                                np.random.randint(low=0, high=2, size=(1,100)))
                                for _ in range(self.no_tx)] for _ in range(self.no_rx)]

        Simulation.calculate_noise_variance = Mock(return_value=0.0)
        expected_received_signals = [
            propagation_matrix[0][0][0],
            propagation_matrix[1][0][0] + propagation_matrix[1][1][0] + propagation_matrix[1][2][0]
        ]
        received_signals = Simulation.receive(
            self.scenario, propagation_matrix, self.snr_mask)

        np.testing.assert_array_almost_equal(
            received_signals[0][0],
            expected_received_signals[0]
        )
        np.testing.assert_array_almost_equal(
            received_signals[1][0],
            expected_received_signals[1]
        )

    def test_do_not_receive_at_rx0_at_all(self) -> None:
        propagation_matrix = [[(np.random.randint(low=0, high=2, size=(1,100)),
                                np.random.randint(low=0, high=2, size=(1,100)))
                                for _ in range(self.no_tx)] for _ in range(self.no_rx)]
        Simulation.calculate_noise_variance = Mock(return_value=0.0)
        self.snr_mask[:, 0] = False
        received_signals = Simulation.receive(
            self.scenario, propagation_matrix, self.snr_mask
        )

        self.assertEqual(received_signals[0], (None, None, None))

    def test_no_detection_at_rx0(self) -> None:
        self.snr_mask[:, 0] = False
        received_signals = [
            (Mock(), Mock(), Mock()) for _ in range(self.no_rx)
        ]
        receiver_bits = Simulation.detect(self.scenario, received_signals, self.snr_mask)
        self.assertIsNone(receiver_bits[0])
        self.assertIsNotNone(receiver_bits[1])

