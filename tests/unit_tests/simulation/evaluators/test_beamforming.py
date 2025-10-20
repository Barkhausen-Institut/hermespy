# -*- coding: utf-8 -*-

import numpy as np

from unittest import TestCase

from unit_tests.core.test_factory import test_roundtrip_serialization
from unit_tests.utils import SimulationTestContext

from hermespy.beamforming import SphericalFocus
from hermespy.core import AntennaMode, Signal, SignalReceiver, SignalTransmitter
from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray
from hermespy.simulation.evaluators.beamforming import SidelobeEvaluation, SidelobeEvaluator

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestSidelobeEvaluation(TestCase):
    """Test the sidelobe evaluation class."""

    def setUp(self) -> None:
        self.powers = np.array([0.1, 0.2, 0.3, 0.4])
        self.focus_power = 0.5
        self.evaluation = SidelobeEvaluation(self.powers, self.focus_power)

    def test_artifact(self) -> None:
        """Test the artifact method"""

        expected_artifact = self.focus_power / np.sum(self.powers)
        self.assertAlmostEqual(self.evaluation.artifact().to_scalar(), expected_artifact, places=4)

    def test_visualization(self) -> None:
        """Test the visualization method"""

        with SimulationTestContext():
            self.evaluation.visualize()


class TestSidelobeEvaluator(TestCase):
    """Test the SidelobeEvaluator class."""

    def setUp(self) -> None:
        devices = [SimulatedDevice(
            carrier_frequency=1e8,
            antennas=SimulatedUniformArray(SimulatedIdealAntenna, .1, [2, 2, 1])
        ) for _ in range(2)]

        self.tx_evaluator, self.rx_evaluator = [SidelobeEvaluator(
            device,
            mode,
            SphericalFocus(0, 0),
        ) for mode, device in zip((AntennaMode.TX, AntennaMode.RX), devices)]

        devices[0].add_dsp(SignalTransmitter(Signal.Create(np.ones((4,60), dtype=np.complex128), devices[0].sampling_rate)))
        devices[1].add_dsp(SignalReceiver(60))

        _ = devices[1].receive(devices[0].transmit())

    def test_static_properties(self) -> None:
        """Test static properties"""

        self.assertEqual(self.tx_evaluator.abbreviation, "SLL")
        self.assertEqual(self.tx_evaluator.title, "Sidelobe Level")

    def test_mode_setget(self) -> None:
        """Mode property getter should return setter argument"""

        expected_mode = AntennaMode.RX
        self.tx_evaluator.mode = expected_mode
        self.assertEqual(self.tx_evaluator.mode, expected_mode)

    def test_desired_focus_setget(self) -> None:
        """Desired focus property getter should return setter argument"""

        expected_focus = SphericalFocus(1, 0)
        self.tx_evaluator.desired_focus = expected_focus
        self.assertIs(self.tx_evaluator.desired_focus, expected_focus)

    def test_evaluate(self) -> None:
        """Test the evaluate method"""

        tx_evaluation = self.tx_evaluator.evaluate()
        self.assertGreater(1.0, tx_evaluation.artifact().to_scalar())

        rx_evaluation = self.rx_evaluator.evaluate()
        self.assertGreater(1.0, rx_evaluation.artifact().to_scalar())

    def test_serialization(self) -> None:
        """Test side lobe evaluator serialization"""

        test_roundtrip_serialization(self, self.tx_evaluator)
