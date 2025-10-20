# -*- coding: utf-8 -*-

from unittest import TestCase

from hermespy.simulation.rf.blocks.mixers import MixerType, IdealMixer, Mixer
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMixer(TestCase):
    """Test the Mixer base class."""

    def setUp(self) -> None:
        self.mixer_type = MixerType.UP
        self.seed = 42
        self.mixer = Mixer(self.mixer_type, seed=42)

    def test_init(self) -> None:
        """Test mixer initialization"""
        
        self.assertEqual(self.mixer_type, self.mixer.mixer_type)
        self.assertEqual(self.seed, self.mixer.seed)

    def test_serialization(self) -> None:
        """Test serialization of Mixer base class"""

        test_roundtrip_serialization(self, self.mixer)


class TestIdealMixer(TestCase):
    """Test the Ideal Mixer block."""

    def setUp(self) -> None:
        self.mixer_type = MixerType.UP
        self.lo_frequency = 2.4e9
        self.seed = 42
        self.mixer = IdealMixer(self.mixer_type, self.lo_frequency, seed=42)

    def test_init(self) -> None:
        """Test ideal mixer initialization"""
        
        self.assertEqual(self.mixer_type, self.mixer.mixer_type)
        self.assertEqual(self.lo_frequency, self.mixer.lo_frequency)
        self.assertEqual(self.seed, self.mixer.seed)

    def test_serialization(self) -> None:
        """Test serialization of Ideal Mixer block"""

        test_roundtrip_serialization(self, self.mixer)
