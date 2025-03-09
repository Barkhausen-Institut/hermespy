# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np

from hermespy.core import Signal
from hermespy.simulation import SimulatedDevice
from hermespy.jcas import MatchedFilterJcas
from unit_tests.core.test_factory import test_roundtrip_serialization
from unit_tests.modem.test_waveform import MockCommunicationWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "1.4.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMatchedFilterJoint(TestCase):
    """Matched filter joint testing."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.carrier_frequency = 1e8

        self.joint = MatchedFilterJcas(max_range=10.0)
        self.waveform = MockCommunicationWaveform()
        self.joint.waveform = self.waveform

        self.device = SimulatedDevice(carrier_frequency=self.carrier_frequency)
        self.device._rng = self.rng
        self.device.transmitters.add(self.joint)
        self.device.receivers.add(self.joint)

    def test_receive_validation(self) -> None:
        """Receiving should raise a RuntimeError if there's no cached transmission"""

        with self.assertRaises(RuntimeError):
            self.joint.receive(Signal.Create(np.zeros((1, 10)), 1.0), self.device.state())

    def test_transmit_receive(self) -> None:
        num_delay_samples = 10
        transmission = self.joint.transmit(self.device.state())

        delay_offset = Signal.Create(np.zeros((1, num_delay_samples), dtype=complex), transmission.signal.sampling_rate, carrier_frequency=self.carrier_frequency)
        delay_offset.append_samples(transmission.signal)

        reception = self.joint.receive(delay_offset, self.device.state())
        self.assertTrue(10, reception.cube.data.argmax)

        padded_reception = self.joint.receive(transmission.signal, self.device.state())
        self.assertTrue(10, padded_reception.cube.data.argmax)

    def test_range_resolution_setget(self) -> None:
        """Range resolution property getter should return setter argument."""

        range_resolution = 1e-3
        self.joint.range_resolution = range_resolution

        self.assertEqual(range_resolution, self.joint.range_resolution)

    def test_range_resolution_validation(self) -> None:
        """Range resolution property setter should raise ValueError on non-positive arguments."""

        with self.assertRaises(ValueError):
            self.joint.range_resolution = -1.0

        with self.assertRaises(ValueError):
            self.joint.range_resolution = 0.0

    def test_sampling_rate_validation(self) -> None:
        """Sampling rate property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.joint.sampling_rate = 0.0

    def test_sampling_rate_setget(self) -> None:
        """Sampling rate property getter should return setter argument"""

        self.joint.sampling_rate = 1.234e10
        self.assertEqual(1.234e10, self.joint.sampling_rate)

        self.joint.sampling_rate = None
        self.assertEqual(self.waveform.sampling_rate, self.joint.sampling_rate)

    def test_max_range_validation(self) -> None:
        """Max range property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.joint.max_range = 0.0

    def test_serialization(self) -> None:
        """Test matched filter JCAS serialization"""

        self.joint.waveform = None
        test_roundtrip_serialization(self, self.joint, {'range_resolution', 'sampling_rate'})
