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
from scipy.constants import pi, speed_of_light

from hermespy.channel.cluster_delay_lines import ClusterDelayLine
from hermespy.simulation.antenna import IdealAntenna, UniformArray

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

        self.num_clusters = 3
        self.delay_spread_mean = -7.49
        self.delay_spread_std = 0.55
        self.delay_scaling = 3.8
        self.carrier_frequency = 1e9

        self.antennas = UniformArray(IdealAntenna(), .5 * self.carrier_frequency / speed_of_light, (2, 2))

        self.receiver = Mock()
        self.receiver.num_antennas = self.antennas.num_antennas
        self.receiver.antennas = self.antennas
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.orientation = np.array([0., 0., 0.])
        self.receiver.antenna_positions = np.array([[100., 0., 0.]], dtype=float)
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.num_antennas = self.antennas.num_antennas
        self.transmitter.antennas = self.antennas
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0., 0., pi])
        self.transmitter.antenna_positions = np.array([[-100., 0., 0.]], dtype=float)
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.channel = ClusterDelayLine(delay_spread_mean=self.delay_spread_mean,
                                        delay_spread_std=self.delay_spread_std,
                                        delay_scaling=self.delay_scaling,
                                        num_clusters=self.num_clusters,
                                        receiver=self.receiver,
                                        transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertEqual(self.delay_spread_mean, self.channel.delay_spread_mean)
        self.assertEqual(self.delay_spread_std, self.channel.delay_spread_std)
        self.assertEqual(self.num_clusters, self.channel.num_clusters)

        self.assertEqual(self.delay_scaling, self.channel.delay_scaling)

    def test_num_clusters_setget(self) -> None:
        """Number of clusters property getter should return setter argument."""

        num_clusters = 123
        self.channel.num_clusters = num_clusters

        self.assertEqual(num_clusters, self.channel.num_clusters)

    def test_num_clusters_validation(self) -> None:
        """Number of clusters property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.channel.num_clusters = -1

        with self.assertRaises(ValueError):
            self.channel.num_clusters = 0

    def test_delay_spread_mean_setget(self) -> None:
        """Delay spread mean property getter should return setter argument."""

        delay_spread = 123
        self.channel.delay_spread_mean = delay_spread

        self.assertEqual(delay_spread, self.channel.delay_spread_mean)

    def test_delay_spread_std_setget(self) -> None:
        """Delay spread mean property getter should return setter argument."""

        std = 123
        self.channel.delay_spread_std = std

        self.assertEqual(std, self.channel.delay_spread_std)

    def test_delay_scaling_setget(self) -> None:
        """Delay scaling property getter should return setter argument."""

        delay_scaling = 123
        self.channel.delay_scaling = delay_scaling

        self.assertEqual(delay_scaling, self.channel.delay_scaling)

    def test_delay_scaling_validation(self) -> None:
        """Delay scaling property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.channel.delay_scaling = -1.

        with self.assertRaises(ValueError):
            self.channel.delay_scaling = 0.5

        try:

            self.channel.delay_scaling = 1.

        except ValueError:
            self.fail()

    def test_rice_factor_mean_setget(self) -> None:
        """Rice factor mean property getter should return setter argument."""

        rice_factor_mean = 123
        self.channel.rice_factor_mean = rice_factor_mean

        self.assertEqual(rice_factor_mean, self.channel.rice_factor_mean)

    def test_rice_factor_mean_validation(self) -> None:
        """Rice factor mean property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.channel.rice_factor_mean = -1.

        try:

            self.channel.rice_factor_mean = 0.

        except ValueError:
            self.fail()
            
    def test_rice_factor_std_setget(self) -> None:
        """Rice factor standard deviation property getter should return setter argument."""

        rice_factor_std = 123
        self.channel.rice_factor_std = rice_factor_std

        self.assertEqual(rice_factor_std, self.channel.rice_factor_std)

    def test_rice_factor_std_validation(self) -> None:
        """Rice factor standard deviation property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.channel.rice_factor_std = -1.

        try:

            self.channel.rice_factor_std = 0.

        except ValueError:
            self.fail()
            
    def test_cluster_shadowing_std_setget(self) -> None:
        """Cluster shadowing standard deviation property getter should return setter argument."""

        cluster_shadowing_std = 123
        self.channel.cluster_shadowing_std = cluster_shadowing_std

        self.assertEqual(cluster_shadowing_std, self.channel.cluster_shadowing_std)

    def test_cluster_shadowing_std_validation(self) -> None:
        """Cluster shadowing standard deviation property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.channel.cluster_shadowing_std = -1.

        try:

            self.channel.cluster_shadowing_std = 0.

        except ValueError:
            self.fail()

    def test_impulse_response_nlos(self):

        self.channel.line_of_sight = False
        num_samples = 100
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])

    def test_impulse_response_los(self):

        self.channel.line_of_sight = True
        num_samples = 100
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])
