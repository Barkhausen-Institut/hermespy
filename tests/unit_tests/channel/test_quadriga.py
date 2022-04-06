# -*- coding: utf-8 -*-
"""Tests for the Quadriga Channel Matlab Interface to Hermes."""

from os import environ
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from hermespy.channel import QuadrigaChannel, QuadrigaInterface

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestQuadrigaChannel(TestCase):

    def setUp(self) -> None:

        self.sampling_rate = 1e6
        self.num_samples = 1000
        self.carrier_frequency = 1e9

        self.transmitter = Mock()
        self.receiver = Mock()
        self.transmitter.sampling_rate = self.sampling_rate
        self.receiver.sampling_rate = self.sampling_rate
        self.transmitter.num_antennas = 1
        self.receiver.num_antennas = 1
        self.transmitter.carrier_frequency = self.carrier_frequency
        self.receiver.carrier_frequency = self.carrier_frequency
        self.transmitter.position = np.array([-500., 0., 0.], dtype=float)
        self.receiver.position = np.array([500., 0., 0.], dtype=float)

        self.channel = QuadrigaChannel(self.transmitter, self.receiver)

    def test_channel_registration(self) -> None:
        """Quadriga channel should be properly registered at the interface."""

        self.assertTrue(QuadrigaInterface.GlobalInstance().channel_registered(self.channel))

    def test_impulse_response(self) -> None:
        """Test the Quadriga Channel Impulse Response generation."""

        response = self.channel.impulse_response(self.num_samples, self.sampling_rate)
        self.assertCountEqual([self.num_samples, 1, 1], response.shape[:3])


class TestQuadrigaInterface(TestCase):
    """Test the global quadriga interface."""

    def setUp(self) -> None:

        environ['HERMES_QUADRIGA'] = 'test'
    
        self.interface = QuadrigaInterface()

    def test_environment_quadriga_path(self) -> None:
        """The interface should properly infer the quadriga source path from environment variables."""

        self.assertEqual('test', self.interface.path_quadriga_src)
