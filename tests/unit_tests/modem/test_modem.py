# -*- coding: utf-8 -*-
"""HermesPy testing for modem base class."""

import unittest
from unittest.mock import Mock

import numpy as np
from numpy import random as rnd
from numpy.testing import assert_array_equal, assert_almost_equal
from scipy.constants import speed_of_light

from hermespy.modem.modem import Modem

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestModem(unittest.TestCase):
    """Modem Base Class Test Case"""

    def setUp(self) -> None:

        self.scenario = Mock()
        self.scenario.sampling_rate = 1e6

        self.position = np.zeros(3)
        self.orientation = np.zeros(3)
        self.carrier_frequency = 1e9
        self.num_antennas = 3

        self.encoding = Mock()
        self.precoding = Mock()
        self.waveform = Mock()
        self.rfchain = Mock()

        self.random_generator = rnd.default_rng(42)

        self.modem = Modem(scenario=self.scenario, position=self.position, orientation=self.orientation,
                           carrier_frequency=self.carrier_frequency, num_antennas=self.num_antennas,
                           encoding=self.encoding, precoding=self.precoding, waveform=self.waveform,
                           rfchain=self.rfchain, random_generator=self.random_generator)

    def test_initialization(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertIs(self.scenario, self.modem.scenario)
        self.assertIs(self.position, self.modem.position)
        self.assertIs(self.orientation, self.modem.orientation)
        self.assertIs(self.carrier_frequency, self.modem.carrier_frequency)
        self.assertIs(self.encoding, self.modem.encoder_manager)
        self.assertIs(self.precoding, self.modem.precoding)
        self.assertIs(self.waveform, self.modem.waveform_generator)
        self.assertIs(self.rfchain, self.modem.rf_chain)
        self.assertIs(self.random_generator, self.modem.random_generator)

    def test_init_topology(self) -> None:
        """If no topology is specified, the modem should represent a half-wavelength ULA by default."""

        half_wavelength = .5 * speed_of_light / self.carrier_frequency
        expected_topology = half_wavelength * np.outer(np.arange(self.num_antennas), np.array([1., 0., 0.]))

        assert_almost_equal(expected_topology, self.modem.topology)

    def test_scenario_setget(self) -> None:
        """Scenario property setter should return getter argument."""

        self.modem = Modem()
        self.modem.scenario = self.scenario

        self.assertIs(self.scenario, self.modem.scenario)

    def test_scenario_set_validation(self) -> None:
        """Overwriting a scenario property should raise a RuntimeError."""

        with self.assertRaises(RuntimeError):
            self.modem.scenario = Mock()

    def test_scenario_get_validation(self) -> None:
        """Scenario property getter should raise a RuntimeError if no scenario has been specified."""

        self.modem = Modem()
        with self.assertRaises(RuntimeError):
            _ = self.modem.scenario

    def test_is_attached(self) -> None:
        """The is_attached property should return the proper modem attachment state."""

        self.assertTrue(self.modem.is_attached)
        self.assertFalse(Modem().is_attached)

    def test_random_generator_setget(self) -> None:
        """Random generator property getter should return setter argument."""

        random_generator = Mock()
        self.modem.random_generator = random_generator

        self.assertIs(random_generator, self.modem.random_generator)

    def test_random_generator_get_validation(self) -> None:
        """Random generator getter should raise a RuntimeError if the modem is floating
        and no generator has ben specified."""

        self.modem = Modem()
        with self.assertRaises(RuntimeError):
            _ = self.modem.random_generator

        self.modem.scenario = self.scenario
        try:
            _ = self.modem.random_generator

        except RuntimeError:
            self.fail()

    def test_position_setget(self) -> None:
        """Position property getter should return setter argument."""

        position = np.arange(3)
        self.modem.position = position

        assert_array_equal(position, self.modem.position)

    def test_position_validation(self) -> None:
        """Position property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.modem.position = np.arange(4)

        with self.assertRaises(ValueError):
            self.modem.position = np.array([[1, 2, 3]])

        try:
            self.modem.position = np.arange(1)

        except ValueError:
            self.fail()

    def test_position_expansion(self) -> None:
        """Position property setter should expand vector dimensions if required."""

        position = np.array([1.0])
        expected_position = np.array([1.0, 0.0, 0.0])
        self.modem.position = position

        assert_almost_equal(expected_position, self.modem.position)

    def test_orientation_setget(self) -> None:
        """Modem orientation property getter should return setter argument."""

        orientation = np.arange(3)
        self.modem.orientation = orientation

        assert_array_equal(orientation, self.modem.orientation)

    def test_orientation_validation(self) -> None:
        """Modem orientation property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.modem.orientation = np.array([[1, 2, 3], [4, 5, 6]])

        with self.assertRaises(ValueError):
            self.modem.orientation = np.array([1, 2])

    def test_topology_setget(self) -> None:
        """Modem topology property getter should return setter argument."""

        topology = np.arange(9).reshape((3, 3))
        self.modem.topology = topology

        assert_array_equal(topology, self.modem.topology)

    def test_topology_set_expansion(self) -> None:
        """Topology property setter automatically expands input dimensions."""

        topology = np.arange(3)
        expected_topology = np.zeros((3, 3), dtype=float)
        expected_topology[:, 0] = topology

        self.modem.topology = topology
        assert_array_equal(expected_topology, self.modem.topology)

    def test_topology_validation(self) -> None:
        """Topology property setter should raise ValueErrors on invalid arguments."""

        with self.assertRaises(ValueError):
            self.modem.topology = np.empty(0)

        with self.assertRaises(ValueError):
            self.modem.topology = np.array([[1, 2, 3, 4]])

    def test_carrier_frequency_setget(self) -> None:
        """Carrier frequency property setter should return getter argument."""

        carrier_frequency = 20
        self.modem.carrier_frequency = carrier_frequency

        self.assertEqual(carrier_frequency, self.modem.carrier_frequency)

    def test_carrier_frequency_validation(self) -> None:
        """Carrier frequency property should return ValueError on negative arguments."""

        with self.assertRaises(ValueError):
            self.modem.carrier_frequency = -1.0

        try:
            self.modem.carrier_frequency = 0.0

        except ValueError:
            self.fail()

    def test_linear_topology(self) -> None:
        """Linear topology property flag should return proper topology status."""

        self.modem.topology = np.arange(3)
        self.assertTrue(self.modem.linear_topology)

        self.modem.topology = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        self.assertFalse(self.modem.linear_topology)

    def test_num_antennas(self) -> None:
        """Num antennas property should return proper number of antennas."""

        self.modem.topology = np.arange(3)
        self.assertEqual(3, self.modem.num_antennas)

        self.modem.topology = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        self.assertEqual(2, self.modem.num_antennas)

    def test_num_streams(self) -> None:
        """Number of streams property should return proper number of streams."""

        self.modem.topology = np.arange(3)
        self.assertEqual(3, self.modem.num_streams)

        self.modem.topology = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        self.assertEqual(2, self.modem.num_streams)

    def test_encoder_manager_setget(self) -> None:
        """Encoder manager property getter should return setter argument."""

        encoder_manager = Mock()
        self.modem.encoder_manager = encoder_manager

        self.assertIs(encoder_manager, self.modem.encoder_manager)
        self.assertIs(encoder_manager.modem, self.modem)

    def test_waveform_generator_setget(self) -> None:
        """Waveform generator property getter should return setter argument."""

        waveform_generator = Mock()
        self.modem.waveform_generator = waveform_generator

        self.assertIs(waveform_generator, self.modem.waveform_generator)
        self.assertIs(waveform_generator.modem, self.modem)
        
    def test_rf_chain_setget(self) -> None:
        """Radio frequency chain property getter should return setter argument."""

        rf_chain = Mock()
        self.modem.rf_chain = rf_chain

        self.assertIs(rf_chain, self.modem.rf_chain)
        
    def test_precoding_setget(self) -> None:
        """Precoding configuration property getter should return setter argument."""

        precoding = Mock()
        self.modem.precoding = precoding

        self.assertIs(precoding, self.modem.precoding)
        self.assertIs(precoding.modem, self.modem)
