# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.fft import fft

from hermespy.core import Signal
from hermespy.simulation import SelectiveLeakage, SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSelectiveLeakage(TestCase):
    """Test frequency-selective leakage"""

    def setUp(self) -> None:
        self.device = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (3, 2, 1)))

        self.leakage = SelectiveLeakage.Normal(self.device, num_samples=10, mean=1.0, variance=0.0)
        self.device.isolation = self.leakage

    def test_init_validation(self) -> None:
        """Initialization should raise a ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = SelectiveLeakage(np.zeros((1, 2)))

    def test_leakage_response(self) -> None:
        """The leakage response matrix should be the FFT of its impulse response"""

        leakage_response = self.leakage.leakage_response
        leakage_selectivity = fft(leakage_response, axis=2)

        self.assertSequenceEqual((6, 6, 10), leakage_response.shape)
        assert_array_almost_equal(np.ones(leakage_response.shape), np.abs(leakage_selectivity))

    def test_leak(self) -> None:
        """Leaking a signal should result in the expected leak"""

        test_signal = Signal.Create(np.zeros((self.device.antennas.num_transmit_antennas, 100), dtype=np.complex128), self.device.sampling_rate, self.device.carrier_frequency)
        test_signal[:, 0] = 1.0

        leaked_signal = self.leakage.leak(test_signal)
        assert_array_almost_equal(np.abs(6 * test_signal.getitem()), np.abs(leaked_signal.getitem((slice(None, None), slice(None, 100)))))
