# -*- coding: utf-8 -*-

from __future__ import annotations
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.core import Signal, UniformArray, IdealAntenna
from hermespy.simulation import SelectiveLeakage
from hermespy.hardware_loop import PhysicalDeviceDummy, SelectiveLeakageCalibration

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSelectiveLeakageCalibration(TestCase):
    """Test leakage calibration routines for physical devices."""
    
    def setUp(self) -> None:
        
        self.num_samples = 126

        self.device = PhysicalDeviceDummy(carrier_frequency=1e9, sampling_rate=1e8, seed=42, antennas=UniformArray(IdealAntenna, 1e-3, (2, 1, 1)), receive_transmission=False)
    
    def test_mmse_estimation(self):
        """Test estimation of the covariance leakage matrix for a physical device"""

        leakage_model = SelectiveLeakage.Normal(self.device, num_samples=self.num_samples)
        self.device.isolation = leakage_model

        calibration = SelectiveLeakageCalibration.MMSEEstimate(self.device, num_wavelet_samples=self.num_samples)
        
        # Assert calibration validity        
        assert_array_almost_equal(leakage_model.leakage_response, calibration.leakage_response)

    def test_remove_leakage(self):
        """Applying the calibration should result in leakage-free receptions"""

        leakage_model = SelectiveLeakage.Normal(self.device, num_samples=self.num_samples)
        self.device.isolation = leakage_model

        calibration = SelectiveLeakageCalibration(leakage_model.leakage_response)
        self.device.leakage_calibration = calibration

        tx_signal = Signal(np.random.normal(size=(self.device.antennas.num_transmit_antennas, self.num_samples)), self.device.sampling_rate, self.device.carrier_frequency)
        rx_signal = self.device.trigger_direct(tx_signal)

        assert_array_almost_equal(np.zeros(rx_signal.samples.shape), rx_signal.samples)
