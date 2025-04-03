# -*- coding: utf-8 -*-

from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from h5py import File
from numpy.testing import assert_array_equal

from hermespy.core import Signal
from hermespy.modem import CommunicationReception, CommunicationTransmission
from hermespy.radar import RadarCube, RadarReception
from hermespy.jcas import JCASTransmission, JCASReception
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestJCASTransmission(TestCase):
    """Test JCAS transmission"""

    def setUp(self) -> None:
        self.signal = Signal.Empty(1.0, 1, 0)
        self.transmission = JCASTransmission(CommunicationTransmission(self.signal, []))

    def test_serialization(self) -> None:
        """Test JCAS transmission serialization"""

        test_roundtrip_serialization(self, self.transmission)


class TestJCASReception(TestCase):
    """Test JCAS reception"""

    def setUp(self) -> None:
        self.signal = Signal.Create(np.zeros((1, 10), dtype=np.complex128), 1.0)
        self.communication_reception = CommunicationReception(self.signal)
        self.cube = RadarCube(np.zeros((1, 1, 10)))
        self.radar_reception = RadarReception(self.signal, self.cube)
        self.reception = JCASReception(self.communication_reception, self.radar_reception)

    def test_serialization(self) -> None:
        """Test JCAS reception serialization"""

        test_roundtrip_serialization(self, self.reception)
