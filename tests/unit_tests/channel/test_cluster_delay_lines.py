# -*- coding: utf-8 -*-
"""
=====================================
3GPP Cluster Delay Line Model Testing
=====================================
"""

from unittest import TestCase

from hermespy.channel.cluster_delay_lines import ClusterDelayLine

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ClusterDelayLineMock(ClusterDelayLine):
    """Mock of the abstract cluster delay line base class"""

    @property
    def aoa_spread_mean(self) -> float:
        return 1.73

    @property
    def aoa_spread_std(self) -> float:
        return 0.28

    @property
    def aod_spread_mean(self) -> float:
        return 1.21

    @property
    def aod_spread_std(self) -> float:
        return .41


class TestClusterDelayLine(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.num_clusters = 10
        self.delay_spread = 11e-9
        self.delay_scaling = 1.1

        self.channel = ClusterDelayLineMock(num_clusters=self.num_clusters,
                                            delay_spread=self.delay_spread,
                                            delay_scaling=self.delay_scaling)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertEqual(self.num_clusters, self.channel.num_clusters)
        self.assertEqual(self.delay_spread, self.channel.delay_spread)
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

    def test_delay_spread_setget(self) -> None:
        """Delay spread property getter should return setter argument."""

        delay_spread = 123
        self.channel.delay_spread = delay_spread

        self.assertEqual(delay_spread, self.channel.delay_spread)

    def test_delay_spread_validation(self) -> None:
        """Delay spread property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.channel.delay_spread = -1.

        try:

            self.channel.delay_spread = 0.

        except ValueError:
            self.fail()

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

    def test_xxx(self):

        self.channel.xxxx()