# -*- coding: utf-8 -*-

from unittest import TestCase

from hermespy.modem.waveforms.ieee_5gnr import NRFrame, NRSlot, NRSubframe, nr_bandwidth
from unit_tests.core.test_factory import test_roundtrip_serialization  # type: ignore

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestNRSlot(TestCase):
    """Test the 5G NR slot mockup implementation."""

    def setUp(self) -> None:
        self.num_resource_blocks = 30
        self.slot = NRSlot(num_resource_blocks=self.num_resource_blocks)

    def test_slot_duration(self) -> None:
        """Ensure the slot duration is 1 ms for numerology 0"""

        bandwidth = nr_bandwidth(0, self.num_resource_blocks)
        nr_slot_duration_numerology_0 = 1e-3

        self.assertAlmostEqual(self.slot.frame_duration(bandwidth), nr_slot_duration_numerology_0, delta=0.01 * nr_slot_duration_numerology_0)

    def test_serialization(self) -> None:
        """Test NR slot serialization."""

        test_roundtrip_serialization(self, self.slot)



class TestNRSubframe(TestCase):
    """Test the 5G NR subframe mockup implementation."""

    def setUp(self) -> None:
        self.numerology = 4
        self.num_resource_blocks = 30
        self.subframe = NRSubframe(numerology=self.numerology, num_resource_blocks=self.num_resource_blocks)

    def test_subframe_duration(self) -> None:
        """Ensure the subframe duration is always 1ms, regardles of numerology"""

        for numerology in range(6):
            subframe = NRSubframe(numerology=numerology, num_resource_blocks=self.num_resource_blocks)
            bandwidth = nr_bandwidth(numerology, self.num_resource_blocks)

            with self.subTest(numerology=numerology):
                self.assertAlmostEqual(subframe.frame_duration(bandwidth), 1e-3, delta=0.01*1e-3)

    def test_serialization(self) -> None:
        """Test NR subframe serialization."""

        test_roundtrip_serialization(self, self.subframe)


class TestNRFrame(TestCase):
    """Test the 5G NR frame mockup implementation."""

    def setUp(self) -> None:
        self.numerology = 4
        self.num_resource_blocks = 30
        self.frame = NRFrame(numerology=self.numerology, num_resource_blocks=self.num_resource_blocks)

    def test_subframe_duration(self) -> None:
        """Ensure the subframe duration is always 1ms, regardles of numerology"""

        for numerology in range(6):
            frame = NRFrame(numerology=numerology, num_resource_blocks=self.num_resource_blocks)
            bandwidth = nr_bandwidth(numerology, self.num_resource_blocks)

            with self.subTest(numerology=numerology):
                self.assertAlmostEqual(frame.frame_duration(bandwidth), 1e-2, delta=0.01*1e-2)

    def test_serialization(self) -> None:
        """Test NR frame serialization."""

        test_roundtrip_serialization(self, self.frame)
