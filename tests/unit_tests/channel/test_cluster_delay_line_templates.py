# -*- coding: utf-8 -*-
"""
=====================================
3GPP Cluster Delay Line Model Testing
=====================================
"""

from typing import Type
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.random import default_rng

from hermespy.core import Transformation
from hermespy.channel import (
    StreetCanyonLineOfSight,
    StreetCanyonNoLineOfSight,
    StreetCanyonOutsideToInside,
    UrbanMacrocellsLineOfSight,
    UrbanMacrocellsNoLineOfSight,
    UrbanMacrocellsOutsideToInside,
    RuralMacrocellsLineOfSight,
    RuralMacrocellsNoLineOfSight,
    RuralMacrocellsOutsideToInside,
    IndoorOfficeLineOfSight,
    IndoorOfficeNoLineOfSight,
    IndoorFactoryNoLineOfSight,
    IndoorFactoryLineOfSight,
    ClusterDelayLine,
    ChannelRealization,
)
from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class __TestClusterDelayLineTemplate(TestCase):
    def _init(self, channel: Type[ClusterDelayLine], **kwargs) -> None:
        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.num_samples = 5000
        self.sampling_rate = 1e5
        self.carrier_frequency = 1e9

        self.alpha_device = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1, (1,)), pose=Transformation.No(), carrier_frequency=self.carrier_frequency)

        self.beta_device = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1, (1,)), pose=Transformation.From_RPY(pos=np.array([100.0, 0.0, 0.0]), rpy=np.array([0.0, 0.0, 0.0])), carrier_frequency=self.carrier_frequency)

        self.channel = channel(alpha_device=self.alpha_device, beta_device=self.beta_device, **kwargs)
        self.channel.random_mother = self.random_node

    def test_realization(self) -> None:
        """Test channel template realization"""

        realization = self.channel.realize()
        self.assertIsInstance(realization, ChannelRealization)


class TestStreetCanyonLOS(__TestClusterDelayLineTemplate):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:
        self._init(StreetCanyonLineOfSight)


class TestStreetCanyonNLOS(__TestClusterDelayLineTemplate):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:
        self._init(StreetCanyonNoLineOfSight)


class TestStreetCanyonO2I(__TestClusterDelayLineTemplate):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:
        self._init(StreetCanyonOutsideToInside)


class TestUrbanMacrocellsLOS(__TestClusterDelayLineTemplate):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:
        self._init(UrbanMacrocellsLineOfSight)


class TestUrbanMacrocellsNLOS(__TestClusterDelayLineTemplate):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:
        self._init(UrbanMacrocellsNoLineOfSight)


class TestUrbanMacrocellsO2I(__TestClusterDelayLineTemplate):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:
        self._init(UrbanMacrocellsOutsideToInside)


class TestRuralMacrocellsLOS(__TestClusterDelayLineTemplate):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:
        self._init(RuralMacrocellsLineOfSight)


class TestRuralMacrocellsNLOS(__TestClusterDelayLineTemplate):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:
        self._init(RuralMacrocellsNoLineOfSight)


class TestRuralMacrocellsO2I(__TestClusterDelayLineTemplate):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:
        self._init(RuralMacrocellsOutsideToInside)


class TestIndoorOfficeLOS(__TestClusterDelayLineTemplate):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:
        self._init(IndoorOfficeLineOfSight)


class TestIndoorOfficeNLOS(__TestClusterDelayLineTemplate):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:
        self._init(IndoorOfficeNoLineOfSight)


class TestIndoorFactory(__TestClusterDelayLineTemplate):
    """Test the indoor factory parameterization base"""

    def setUp(self) -> None:
        self._init(IndoorFactoryLineOfSight, volume=1e5, surface=1e6)

    def test_volume_validation(self) -> None:
        """Volume property setter should raise ValueErrors on arguments smaller or equal to zero"""

        with self.assertRaises(ValueError):
            self.channel.volume = 0.0

        with self.assertRaises(ValueError):
            self.channel.volume = -1.0

    def test_volume_setget(self) -> None:
        """Volume property getter should return setter argument"""  #

        expected_volume = 1.2345
        self.channel.volume = expected_volume

        self.assertEqual(expected_volume, self.channel.volume)

    def test_surface_validation(self) -> None:
        """Surface property setter should raise ValueErrors on arguments smaller or equal to zero"""

        with self.assertRaises(ValueError):
            self.channel.surface = 0.0

        with self.assertRaises(ValueError):
            self.channel.surface = -1.0

    def test_surface_setget(self) -> None:
        """Surface property getter should return setter argument"""  #

        expected_surface = 1.2345
        self.channel.surface = expected_surface

        self.assertEqual(expected_surface, self.channel.surface)


class TestIndoorFactoryLOS(__TestClusterDelayLineTemplate):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:
        self._init(IndoorFactoryLineOfSight, volume=1e5, surface=1e6)


class TestIndoorFactoryNLOS(__TestClusterDelayLineTemplate):
    """Test the 3GPP Cluster Delay Line Model Implementation"""

    def setUp(self) -> None:
        self._init(IndoorFactoryNoLineOfSight, volume=1e5, surface=1e6)


del __TestClusterDelayLineTemplate
