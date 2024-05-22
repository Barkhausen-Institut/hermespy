# -*- coding: utf-8 -*-

from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from h5py import File
from numpy.testing import assert_array_equal

from hermespy.core import Signal
from hermespy.modem import CommunicationReception, CommunicationTransmission, DuplexModem, Symbols, CommunicationWaveform
from hermespy.radar import RadarCube, RadarReception
from hermespy.jcas import JCASTransmission, JCASReception

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestJCASTransmission(TestCase):
    """Test JCAS transmission"""

    def setUp(self) -> None:
        self.signal = Signal.Empty(1.0, 1, 0)
        self.transmission = JCASTransmission(CommunicationTransmission(self.signal, []))

    def test_hdf_serialization(self) -> None:
        """Test proper serialization to HDF"""

        transmission: JCASTransmission

        with TemporaryDirectory() as tempdir:
            file_location = path.join(tempdir, "testfile.hdf5")

            with File(file_location, "a") as file:
                group = file.create_group("testgroup")
                self.transmission.to_HDF(group)

            with File(file_location, "r") as file:
                group = file["testgroup"]
                transmission = self.transmission.from_HDF(group)

        assert_array_equal(self.signal[:, :], transmission.signal[:, :])


class TestJCASReception(TestCase):
    """Test JCAS reception"""

    def setUp(self) -> None:
        self.signal = Signal.Create(np.zeros((1, 10), dtype=np.complex_), 1.0)
        self.communication_reception = CommunicationReception(self.signal)
        self.cube = RadarCube(np.zeros((1, 1, 10)))
        self.radar_reception = RadarReception(self.signal, self.cube)
        self.reception = JCASReception(self.communication_reception, self.radar_reception)

    def test_hdf_serialization(self) -> None:
        """Test proper serialization to HDF"""

        with TemporaryDirectory() as tempdir:
            file_location = path.join(tempdir, "testfile.hdf5")

            with File(file_location, "a") as file:
                group = file.create_group("testgroup")
                self.reception.to_HDF(group)

            with File(file_location, "r") as file:
                group = file["testgroup"]
                reception = JCASReception.from_HDF(group)

        assert_array_equal(self.signal[:, :], reception.signal[:, :])
