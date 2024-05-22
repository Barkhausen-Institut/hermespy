# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal

from hermespy.core import Signal
from hermespy.simulation import ImpedanceCoupling, SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestImpedanceCoupling(TestCase):
    """Test impedance coupling model"""

    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.device = SimulatedDevice()
        self.coupling = ImpedanceCoupling(device=self.device)

    def test_transmit_correlation_validation(self) -> None:
        """Transmit correlation property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.coupling.transmit_correlation = np.zeros((1, 2, 3))

        with self.assertRaises(ValueError):
            self.coupling.transmit_correlation = np.zeros((2, 3))

    def test_transmit_correlation_setget(self) -> None:
        """Transmit correlation property getter should return setter argument"""

        expected_correlation = self.rng.standard_normal((2, 2))
        self.coupling.transmit_correlation = expected_correlation

        assert_array_almost_equal(self.coupling.transmit_correlation, expected_correlation)

        self.coupling.transmit_correlation = None
        self.assertIsNone(self.coupling.transmit_correlation)

    def test_receive_correlation_validation(self) -> None:
        """Receive correlation property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.coupling.receive_correlation = np.zeros((1, 2, 3))

        with self.assertRaises(ValueError):
            self.coupling.receive_correlation = np.zeros((2, 3))

    def test_receive_correlation_setget(self) -> None:
        """Receive correlation property getter should return setter argument"""

        expected_correlation = self.rng.standard_normal((2, 2))
        self.coupling.receive_correlation = expected_correlation

        assert_array_almost_equal(self.coupling.receive_correlation, expected_correlation)

        self.coupling.receive_correlation = None
        self.assertIsNone(self.coupling.receive_correlation)

    def test_transmit_impedance_validation(self) -> None:
        """Transmit impedance property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.coupling.transmit_impedance = np.zeros((1, 2, 3))

        with self.assertRaises(ValueError):
            self.coupling.transmit_impedance = np.zeros((2, 3))

    def test_transmit_impedance_setget(self) -> None:
        """Transmit impedance property getter should return setter argument"""

        expected_correlation = self.rng.standard_normal((2, 2))
        self.coupling.transmit_impedance = expected_correlation

        assert_array_almost_equal(self.coupling.transmit_impedance, expected_correlation)

        self.coupling.transmit_impedance = None
        self.assertIsNone(self.coupling.transmit_impedance)

    def test_receive_impedance_validation(self) -> None:
        """Receive impedance property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.coupling.receive_impedance = np.zeros((1, 2, 3))

        with self.assertRaises(ValueError):
            self.coupling.receive_impedance = np.zeros((2, 3))

    def test_receive_impedance_setget(self) -> None:
        """Receive impedance property getter should return setter argument"""

        expected_correlation = self.rng.standard_normal((2, 2))
        self.coupling.receive_impedance = expected_correlation

        assert_array_almost_equal(self.coupling.receive_impedance, expected_correlation)

        self.coupling.receive_impedance = None
        self.assertIsNone(self.coupling.receive_impedance)

    def test_matching_impedance_validation(self) -> None:
        """Matching impedance property should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.coupling.matching_impedance = np.zeros((1, 2, 3))

        with self.assertRaises(ValueError):
            self.coupling.matching_impedance = np.zeros((2, 3))

    def test_matching_impedance_setget(self) -> None:
        """Matching impedance property getter should return setter argument"""

        expected_correlation = self.rng.standard_normal((2, 2))
        self.coupling.matching_impedance = expected_correlation

        assert_array_almost_equal(self.coupling.matching_impedance, expected_correlation)

        self.coupling.matching_impedance = None
        self.assertIsNone(self.coupling.matching_impedance)

    def test_transmit_receive(self) -> None:
        """Transmit and receive routine should be properly called"""

        signal = Signal.Create(self.rng.normal(size=(1, 10)) + 1j * self.rng.normal(size=(1, 10)), 1, carrier_frequency=0.0)

        tx = self.coupling.transmit(signal)
        rx = self.coupling.receive(tx)

        assert_array_almost_equal(signal[:, :], rx[:, :])
