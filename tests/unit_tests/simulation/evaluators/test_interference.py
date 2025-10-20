# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np

from hermespy.core import Signal, SignalTransmitter
from hermespy.simulation import SI, SimulationScenario, SSINR, SpecificIsolation, SimulatedUniformArray, SimulatedIdealAntenna
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSI(TestCase):

    def setUp(self):

        self.rng = np.random.default_rng(42)
        self.isolation = 1e4

        self.scenario = SimulationScenario()
        self.device = self.scenario.new_device(
            carrier_frequency=1e9,
            antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-2, (2, 1, 1)),
            isolation=SpecificIsolation(self.isolation * np.ones((2, 2))),
            oversampling_factor=1,
        )

        self.scenario.channel(self.device, self.device).gain = 1.0
        self.evaluator = SI(self.device)

        self.transmitter = SignalTransmitter(Signal.Create(
            2**-.5 * (self.rng.standard_normal((2, 1000)) + 1j * self.rng.standard_normal((2, 1000))),
            1e6,
        ))
        self.device.transmitters.add(self.transmitter)

    def test_properties(self) -> None:
        """Test static properties of SI evaluators"""

        self.assertEqual(self.evaluator.abbreviation, "SI")
        self.assertEqual(self.evaluator.title, "Self-Interference")

    def test_evaluate(self) -> None:
        """Test SI evaluation routine"""

        _ = self.scenario.drop()
        evaluation = self.evaluator.evaluate()
        self.assertAlmostEqual(np.mean(evaluation.power), 2 / self.isolation, places=4)

    def test_serialization(self) -> None:
        """Test serialization and deserialization of SI evaluators"""

        test_roundtrip_serialization(self, self.evaluator)


class TestSSINR(TestCase):
    """Test signal to self-interference plus noise ratio evaluation."""

    def setUp(self):

        self.rng = np.random.default_rng(42)
        self.isolation = 1e4

        self.scenario = SimulationScenario()
        self.device = self.scenario.new_device(
            oversampling_factor=1,
            carrier_frequency=1e9,
            isolation=SpecificIsolation(self.isolation * np.ones((2, 2))),
            antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-2, (2, 1, 1)),
        )

        self.scenario.channel(self.device, self.device).gain = 1.0
        self.evaluator = SSINR(self.device)

        self.transmitter = SignalTransmitter(Signal.Create(
            2**-.5 * (self.rng.standard_normal((2, 1000)) + 1j * self.rng.standard_normal((2, 1000))),
            self.device.sampling_rate,
        ))
        self.device.transmitters.add(self.transmitter)

    def test_properties(self) -> None:
        """Test static properties of SSINR evaluators"""

        self.assertEqual(self.evaluator.abbreviation, "SSINR")
        self.assertEqual(self.evaluator.title, "Signal to Self-Interference plus Noise Power Ratio")

    def test_evaluate(self) -> None:
        """Test SSINR evaluation routine"""

        drop = self.scenario.drop()
        evaluation = self.evaluator.evaluate()
        self.assertAlmostEqual(np.mean(evaluation.power), .5 * self.isolation, delta=100)
