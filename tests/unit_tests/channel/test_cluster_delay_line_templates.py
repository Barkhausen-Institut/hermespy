# -*- coding: utf-8 -*-
"""
=====================================
3GPP Cluster Delay Line Model Testing
=====================================
"""

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.random import default_rng

from hermespy.channel.cluster_delay_line_templates import StreetCanyonLineOfSight

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestClusterDelayLine(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9

        self.receiver = Mock()
        self.receiver.num_antennas = 1
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.antenna_positions = np.array([[100., 0., 0.]], dtype=float)
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.num_antennas = 1
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.antenna_positions = np.array([[-100., 0., 0.]], dtype=float)
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.channel = StreetCanyonLineOfSight(receiver=self.receiver,
                                               transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):

        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)
        return
