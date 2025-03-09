# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal

from hermespy.core import Signal
from hermespy.simulation import PerfectCoupling, SimulatedDevice
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestPerfectCoupling(TestCase):
    """Test perfect coupling model"""

    def setUp(self) -> None:
        self.rng = default_rng(42)

        self.device = SimulatedDevice()
        self.coupling = PerfectCoupling(self.device)

    def test_transmit_receive(self) -> None:
        """Transmit and receive methods should return the same signal"""

        some_signal = Signal.Create(self.rng.normal(size=10) + 1j * self.rng.normal(size=10), self.device.antennas.num_receive_antennas, carrier_frequency=0.0)
        transmitted_signal = self.coupling.transmit(some_signal)
        received_signal = self.coupling.receive(transmitted_signal)
        
        assert_array_almost_equal(np.abs(some_signal.getitem()), np.abs(received_signal.getitem()))

    def test_serialization(self) -> None:
        """Test serialization"""

        test_roundtrip_serialization(self, self.coupling, {'device'})
