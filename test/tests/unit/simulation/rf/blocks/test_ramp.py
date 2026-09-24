# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from h5py import File

from hermespy.core import Factory
from hermespy.simulation import ConstantDCPowerModel, RampGenerator, N0
from ....core.test_factory import test_roundtrip_serialization

__author__ = "Emre Can Ataş"
__copyright__ = "Copyright 2026, Barkhausen Institut gGmbH"
__credits__ = ["Emre Can Ataş"]
__license__ = "AGPLv3"
__version__ = "1.6.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRampGenerator(TestCase):
    """Test the FMCW ramp generator block"""

    def setUp(self) -> None:
        self.generator = RampGenerator(10, 1e6, 1e12, 1e-3)

    def test_init_validation(self) -> None:
        """Initialization should raise ValueErrors on invalid arguments"""

        with self.assertRaises(ValueError):
            RampGenerator(-1, 1e6, 1e12, 1e-3)

        with self.assertRaises(ValueError):
            RampGenerator(10, 0.0, 1e12, 1e-3)

        with self.assertRaises(ValueError):
            RampGenerator(10, 1e6, 1e12, 0.0)

    def test_serialization(self) -> None:
        """Test ramp generator serialization"""

        test_roundtrip_serialization(self, self.generator)

    def test_dc_power_model_serialization(self) -> None:
        """Configured power consumption models should survive a serialization roundtrip"""

        self.generator.dc_power_model = ConstantDCPowerModel(2.5)

        file = File("test.h5", "w", driver="core", backing_store=False)
        Factory().to_HDF(file, self.generator)
        deserialization = Factory().from_HDF(file, RampGenerator)
        file.close()

        signal = np.zeros(1, dtype=np.complex128)
        self.assertEqual(2.5, deserialization.dc_power_model.get_power(signal)[0])

    def test_noise_level_serialization(self) -> None:
        """Configured noise levels should survive a serialization roundtrip"""

        generator = RampGenerator(10, 1e6, 1e12, 1e-3, noise_level=N0(1e-9))

        file = File("test.h5", "w", driver="core", backing_store=False)
        Factory().to_HDF(file, generator)
        deserialization = Factory().from_HDF(file, RampGenerator)
        file.close()

        self.assertEqual(1e-9, deserialization.noise_level.get_power(1.0))