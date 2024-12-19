# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from scipy.constants import speed_of_light

from hermespy.tools.math import db2lin, lin2db, marcum_q, rms_value, amplitude_path_loss, DbConversionType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMath(TestCase):
    """Test the math functions"""

    def test_db_roundrip_conversion(self) -> None:
        """Test the roundrip conversion from dB to linear and back"""

        x = 1.2345

        self.assertAlmostEqual(x, lin2db(db2lin(x, DbConversionType.POWER), DbConversionType.POWER))
        self.assertAlmostEqual(x, lin2db(db2lin(x, DbConversionType.AMPLITUDE), DbConversionType.AMPLITUDE))

    def test_db2lin_validation(self) -> None:
        """Decibel to linear conversion should raise ValueError on invalid conversion type"""

        with self.assertRaises(ValueError):
            _ = db2lin(0.0, DbConversionType(2))

    def test_db2lin(self) -> None:
        """Test the conversion from dB to linear"""

        self.assertAlmostEqual(1.0, db2lin(0.0, DbConversionType.POWER))
        self.assertAlmostEqual(1.0, db2lin(0.0, DbConversionType.AMPLITUDE))

        self.assertAlmostEqual(0.1, db2lin(-10.0, DbConversionType.POWER))
        self.assertAlmostEqual(0.1, db2lin(-20.0, DbConversionType.AMPLITUDE))

    def test_lin2db_validation(self) -> None:
        """Linear to decibel conversion should raise ValueError on invalid conversion type"""

        with self.assertRaises(ValueError):
            _ = lin2db(1.0, DbConversionType(2))

    def test_lin2db(self) -> None:
        """Test the conversion from linear to dB"""

        self.assertAlmostEqual(0.0, lin2db(1.0, DbConversionType.POWER))
        self.assertAlmostEqual(0.0, lin2db(1.0, DbConversionType.AMPLITUDE))

        self.assertAlmostEqual(-10.0, lin2db(0.1, DbConversionType.POWER))
        self.assertAlmostEqual(-20.0, lin2db(0.1, DbConversionType.AMPLITUDE))

    def test_marqum_q(self) -> None:
        """Test the marcum_q function"""

        self.assertAlmostEqual(1.0, marcum_q(0.0, 0.0))

    def test_rms_value(self) -> None:
        """Test the root mean square computation function"""

        self.assertAlmostEqual(1.0, rms_value(np.array([1.0, 1.0, 1.0, 1.0])))

    def test_amplitude_path_loss_validation(self) -> None:
        """Amplitude path loss computation should raise ValueError on carrier base band carrier frequency"""

        with self.assertRaises(ValueError):
            _ = amplitude_path_loss(0.0, 1.0)

    def test_amplitude_path_loss(self) -> None:
        """Test the amplitude path loss computation function"""

        carrier_frequency = speed_of_light / (4 * np.pi)
        distance = 1.0

        self.assertAlmostEqual(1.0, amplitude_path_loss(carrier_frequency, distance))
