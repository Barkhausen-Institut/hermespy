# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_almost_equal

from hermespy.core import Signal, SignalTransmitter, SignalReceiver, UniformArray, IdealAntenna
from hermespy.core.evaluators import ReceivedPowerEvaluator, ReceivedPowerResult, ReceivePowerArtifact, ReceivedPowerEvaluation
from hermespy.simulation import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestReceivedPowerEvaluator(TestCase):
    """Test received power evaluator"""
    
    def setUp(self) -> None:
        
        self.rng = np.random.default_rng(42)
        self.num_samples = 100
        self.sampling_rate = 1e6
        self.num_antennas = 2
        
        self.transmitted_signal = Signal(self.rng.standard_normal((self.num_antennas, self.num_samples)) + 1j * self.rng.standard_normal((self.num_antennas, self.num_samples)),
                                    self.sampling_rate, 0, 0, 0)
        
        self.device = SimulatedDevice(antennas=UniformArray(IdealAntenna, 1., [self.num_antennas, 1, 1]))
        self.transmitter = SignalTransmitter(self.transmitted_signal)
        self.receiver = SignalReceiver(self.num_samples, self.sampling_rate)
        self.transmitter.device = self.device
        self.receiver.device = self.device
        
        self.evaluator = ReceivedPowerEvaluator(self.receiver)
        
    def test_evaluation(self) -> None:
        
        num_drops = 10        
        signal_scales = self.rng.random(num_drops)
        expected_powers = self.transmitted_signal.power * np.sum(signal_scales ** 2) / num_drops
        
        # Collect drop artifacts
        grid = []
        artifacts = np.empty(1, dtype=np.object_)
        artifacts[0] = list()
        
        for signal_scale in signal_scales:
            
            signal = self.transmitted_signal.copy()
            signal.samples *= signal_scale
            
            _ = self.device.receive(signal)
            artifacts[0].append(self.evaluator.evaluate().artifact())
            
        # Generate result
        result = self.evaluator.generate_result(grid, artifacts)        
        assert_almost_equal(result.average_powers[0, :], expected_powers)
