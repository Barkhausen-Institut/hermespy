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
from scipy.constants import pi

from hermespy.core import IdealAntenna, UniformArray
from hermespy.channel import StreetCanyonLineOfSight, StreetCanyonNoLineOfSight,\
    StreetCanyonOutsideToInside, UrbanMacrocellsLineOfSight, UrbanMacrocellsNoLineOfSight, \
    UrbanMacrocellsOutsideToInside, RuralMacrocellsLineOfSight, RuralMacrocellsNoLineOfSight, \
    RuralMacrocellsOutsideToInside, IndoorOfficeLineOfSight, IndoorOfficeNoLineOfSight, IndoorFactoryNoLineOfSight, \
    IndoorFactoryLineOfSight

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestStreetCanyonLOS(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9
        self.antennas = UniformArray(IdealAntenna(), 1, (1,))

        self.receiver = Mock()
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.orientation = np.array([0, 0, pi])
        self.receiver.antennas = self.antennas
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.transmitter.antennas = self.antennas
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.channel = StreetCanyonLineOfSight(receiver=self.receiver,
                                               transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):

        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])


class TestStreetCanyonNLOS(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9
        self.antennas = UniformArray(IdealAntenna(), 1, (1,))

        self.receiver = Mock()
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.antennas = self.antennas
        self.receiver.orientation = np.array([0, 0, 0])
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.transmitter.antennas = self.antennas
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.channel = StreetCanyonNoLineOfSight(receiver=self.receiver,
                                                 transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):

        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])


class TestStreetCanyonO2I(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9
        self.antennas = UniformArray(IdealAntenna(), 1, (1,))

        self.receiver = Mock()
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.antennas = self.antennas
        self.receiver.orientation = np.array([0, 0, 0])
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.transmitter.antennas = self.antennas
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.channel = StreetCanyonOutsideToInside(receiver=self.receiver,
                                                   transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):

        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])


class TestUrbanMacrocellsLOS(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9
        self.antennas = UniformArray(IdealAntenna(), 1, (1,))

        self.receiver = Mock()
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.orientation = np.array([0, 0, 0])
        self.receiver.antennas = self.antennas
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.transmitter.antennas = self.antennas
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.channel = UrbanMacrocellsLineOfSight(receiver=self.receiver,
                                                  transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):

        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])


class TestUrbanMacrocellsNLOS(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9
        self.antennas = UniformArray(IdealAntenna(), 1, (1,))

        self.receiver = Mock()
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.orientation = np.array([0, 0, 0])
        self.receiver.antennas = self.antennas
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.transmitter.antennas = self.antennas
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.channel = UrbanMacrocellsNoLineOfSight(receiver=self.receiver,
                                                    transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):

        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])


class TestUrbanMacrocellsO2I(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9
        self.antennas = UniformArray(IdealAntenna(), 1, (1,))

        self.receiver = Mock()
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.orientation = np.array([0, 0, 0])
        self.receiver.antennas = self.antennas
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.transmitter.antennas = self.antennas
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.channel = UrbanMacrocellsOutsideToInside(receiver=self.receiver,
                                                      transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):

        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])


class TestRuralMacrocellsLOS(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9
        self.antennas = UniformArray(IdealAntenna(), 1, (1,))

        self.receiver = Mock()
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.orientation = np.array([0, 0, 0])
        self.receiver.antennas = self.antennas
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.transmitter.antennas = self.antennas
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.channel = RuralMacrocellsLineOfSight(receiver=self.receiver,
                                                  transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):

        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])


class TestRuralMacrocellsNLOS(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9
        self.antennas = UniformArray(IdealAntenna(), 1, (1,))

        self.receiver = Mock()
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.orientation = np.array([0, 0, 0])
        self.receiver.antennas = self.antennas
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.transmitter.antennas = self.antennas
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.channel = RuralMacrocellsNoLineOfSight(receiver=self.receiver,
                                                    transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):

        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])


class TestRuralMacrocellsO2I(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9
        self.antennas = UniformArray(IdealAntenna(), 1, (1,))

        self.receiver = Mock()
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.orientation = np.array([0, 0, 0])
        self.receiver.antennas = self.antennas
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.transmitter.antennas = self.antennas
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.channel = RuralMacrocellsOutsideToInside(receiver=self.receiver,
                                                      transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):

        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])


class TestIndoorOfficeLOS(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9
        self.antennas = UniformArray(IdealAntenna(), 1, (1,))

        self.receiver = Mock()
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.orientation = np.array([0, 0, 0])
        self.receiver.antennas = self.antennas
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.transmitter.antennas = self.antennas
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.channel = IndoorOfficeLineOfSight(receiver=self.receiver,
                                               transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):

        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])


class TestIndoorOfficeNLOS(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9
        self.antennas = UniformArray(IdealAntenna(), 1, (1,))

        self.receiver = Mock()
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.orientation = np.array([0, 0, 0])
        self.receiver.antennas = self.antennas
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.transmitter.antennas = self.antennas
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.channel = IndoorOfficeNoLineOfSight(receiver=self.receiver,
                                                 transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):

        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])


class TestIndoorFactoryLOS(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:

        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9
        self.antennas = UniformArray(IdealAntenna(), 1, (1,))

        self.receiver = Mock()
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.orientation = np.array([0, 0, 0])
        self.receiver.antennas = self.antennas
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.transmitter.antennas = self.antennas
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.volume = 1e5
        self.surface = 1e6

        self.channel = IndoorFactoryLineOfSight(volume=self.volume,
                                                surface=self.surface,
                                                receiver=self.receiver,
                                                transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):

        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])


class TestIndoorFactoryNLOS(TestCase):
    """Test the 3GPP Cluster Delay Line Model Implementation."""

    def setUp(self) -> None:
        self.rng = default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.rng

        self.carrier_frequency = 1e9
        self.antennas = UniformArray(IdealAntenna(), 1, (1,))

        self.receiver = Mock()
        self.receiver.position = np.array([100., 0., 0.])
        self.receiver.orientation = np.array([0, 0, 0])
        self.receiver.antennas = self.antennas
        self.receiver.velocity = np.array([0., 0., 0.], dtype=float)
        self.receiver.carrier_frequency = self.carrier_frequency

        self.transmitter = Mock()
        self.transmitter.position = np.array([-100., 0., 0.])
        self.transmitter.orientation = np.array([0, 0, 0])
        self.transmitter.antennas = self.antennas
        self.transmitter.velocity = np.array([0., 0., 0.], dtype=float)
        self.transmitter.carrier_frequency = 1e9

        self.volume = 1e5
        self.surface = 1e6

        self.channel = IndoorFactoryNoLineOfSight(volume=self.volume,
                                                  surface=self.surface,
                                                  receiver=self.receiver,
                                                  transmitter=self.transmitter)
        self.channel.random_mother = self.random_node

    def test_impulse_response(self):
        num_samples = 5000
        sampling_rate = 1e5

        impulse_response = self.channel.impulse_response(num_samples, sampling_rate)

        self.assertEqual(num_samples, impulse_response.shape[0])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[1])
        self.assertEqual(self.antennas.num_antennas, impulse_response.shape[2])
