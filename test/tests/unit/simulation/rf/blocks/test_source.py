# -*- coding: utf-8 -*-

from unittest import TestCase

from hermespy.simulation import ConstantDCPowerModel, Source
from ....core.test_factory import test_roundtrip_serialization

__author__ = "Emre Can Ataş"
__copyright__ = "Copyright 2026, Barkhausen Institut gGmbH"
__credits__ = ["Emre Can Ataş"]
__license__ = "AGPLv3"
__version__ = "1.6.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSource(TestCase):
    """Test the frequency source block"""

    def setUp(self) -> None:
        self.source = Source(carrier_frequency=1e9, amplitude=0.5)

    def test_serialization(self) -> None:
        """Test source serialization"""

        self.source.dc_power_model = ConstantDCPowerModel(2.5)
        test_roundtrip_serialization(self, self.source)