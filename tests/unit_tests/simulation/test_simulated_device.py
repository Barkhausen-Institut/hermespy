# -*- coding: utf-8 -*-
"""Test HermesPy simulated device module."""

from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from numpy.testing import assert_array_equal

from hermespy.core.device import FloatingError
from hermespy.simulation.simulated_device import SimulatedDevice
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSimulatedDevice(TestCase):
    """Test the simulated device base class."""

    def setUp(self) -> None:

        self.random_generator = np.random.default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.random_generator

        self.scenario = Mock()
        self.position = np.zeros(3)
        self.orientation = np.zeros(3)
        self.antennas = Mock()

        self.device = SimulatedDevice(scenario=self.scenario, antennas=self.antennas, position=self.position, orientation=self.orientation)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertIs(self.scenario, self.device.scenario)
        self.assertIs(self.antennas, self.device.antennas)
        assert_array_equal(self.position, self.device.position)
        assert_array_equal(self.orientation, self.device.orientation)

    def test_transmit(self) -> None:
        """Test modem transmission routine."""

    def test_scenario_setget(self) -> None:
        """Scenario property setter should return getter argument."""

        self.device = SimulatedDevice()
        self.device.scenario = self.scenario

        self.assertIs(self.scenario, self.device.scenario)
            
    def test_attached(self) -> None:
        """The attached property should return the proper device attachment state."""

        self.assertTrue(self.device.attached)
        self.assertFalse(SimulatedDevice().attached)

    def test_noise_setget(self) -> None:
        """Noise property getter should return setter argument."""

        noise = Mock()
        self.device.noise = noise

        self.assertIs(noise, self.device.noise)

    def test_sampling_rate_inference(self) -> None:
        """Sampling rate property should attempt to infer the sampling rate from all possible sources."""

        self.assertEqual(1.0, self.device.sampling_rate)

        receiver = Mock()
        receiver.sampling_rate = 1.23
        self.device.receivers.add(receiver)
        self.assertEqual(1.23, self.device.sampling_rate)

        transmitter = Mock()
        transmitter.sampling_rate = 4.56
        self.device.transmitters.add(transmitter)
        self.assertEqual(4.56, self.device.sampling_rate)

        sampling_rate = 7.89
        self.device.sampling_rate = sampling_rate
        self.assertEqual(sampling_rate, self.device.sampling_rate)

    def test_sampling_rate_validation(self) -> None:
        """Sampling rate property setter should raise ValueError on arguments smaller or equal to zero."""

        with self.assertRaises(ValueError):
            self.device.sampling_rate = -1.

        with self.assertRaises(ValueError):
            self.device.sampling_rate = 0.

    def test_carrier_frequency_setget(self) -> None:
        """Carrier frequency property getter should return setter argument."""

        carrier_frequency = 1.23
        self.device.carrier_frequency = carrier_frequency

        self.assertEqual(carrier_frequency, self.device.carrier_frequency)

    def test_carrier_frequency_validation(self) -> None:
        """Carrier frequency property setter should raise RuntimeError on negative arguments."""

        with self.assertRaises(ValueError):
            self.device.carrier_frequency = -1.

        try:
            self.device.carrier_frequency = 0.

        except RuntimeError:

            self.fail()

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        default_blacklist = self.device.property_blacklist
        default_blacklist.add('scenario')
        
        with patch('hermespy.simulation.simulated_device.SimulatedDevice.property_blacklist', new_callable=PropertyMock) as blacklist:

            blacklist.return_value = default_blacklist
            test_yaml_roundtrip_serialization(self, self.device, {'sampling_rate', 'scenario', 'antenna_positions', 'attached'})
