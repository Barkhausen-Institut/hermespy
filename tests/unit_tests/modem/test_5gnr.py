# -*- coding: utf-8 -*-

from unittest import TestCase

from hermespy.modem import NRSlotLink
from unit_tests.core.test_factory import test_roundtrip_serialization  # type: ignore

__author__ = "Jan Adler"
__copyright__ = "Copyright 2026, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestNRSlotLink(TestCase):
    """Test the 5G NR slot link mockup implementation."""

    def setUp(self) -> None:

        self.num_resource_blocks = 42
        self.link = NRSlotLink(num_resource_blocks=self.num_resource_blocks)

    def test_link_configuration(self) -> None:
        """Ensure the link is configured with the correct default parameters."""

        self.assertEqual(self.link.waveform.num_resource_blocks, self.num_resource_blocks)

    def test_serialization(self) -> None:
        """Test NR slot link serialization."""

        test_roundtrip_serialization(self, self.link)
