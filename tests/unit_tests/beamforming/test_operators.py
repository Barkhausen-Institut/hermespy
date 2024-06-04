# -*- coding: utf-8 -*-

from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from hermespy.beamforming import BeamformingTransmitter, BeamformingReceiver, ConventionalBeamformer
from hermespy.core import Signal
from hermespy.simulation import SimulatedDevice
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestBeamformingTransmitter(TestCase):
    """Test class for BeamformingTransmitter."""

    def setUp(self) -> None:

        rng = np.random.default_rng(42)
        self.signal = Signal.Create(rng.normal(size=(1, 100)) + 1j * rng.normal(size=(1, 100)), 1e6, 1e9)
        self.beamformer = ConventionalBeamformer()
        self.transmitter = BeamformingTransmitter(self.signal, self.beamformer)
        self.device = SimulatedDevice(carrier_frequency=1e9)
        self.device.transmitters.add(self.transmitter)

    def test_init(self) -> None:
        """Test initialization of the beamforming transmitter"""

        self.assertEqual(self.transmitter.signal, self.signal)
        self.assertEqual(self.transmitter.beamformer, self.beamformer)

    def test_beamformer_setget(self) -> None:
        """Beamformer property getter should return setter argument"""

        beamformer = Mock()
        self.transmitter.beamformer = beamformer

        self.assertIs(beamformer, self.transmitter.beamformer)

    def test_transmit(self) -> None:
        """Transmitting should generate a valid transmission"""

        transmission = self.transmitter.transmit()
        self.assertEqual(100, transmission.signal.num_samples)


class TestBeamformingReceiver(TestCase):
    """Test class for BeamformingReceiver."""

    def setUp(self) -> None:

        self.rng = np.random.default_rng(42)
        self.beamformer = ConventionalBeamformer()
        self.receiver = BeamformingReceiver(self.beamformer, 100, 1e6)
        self.device = SimulatedDevice(carrier_frequency=1e9)
        self.device.receivers.add(self.receiver)

    def test_init(self) -> None:
        """Test initialization of the beamforming receiver"""

        self.assertEqual(self.receiver.beamformer, self.beamformer)

    def test_beamformer_setget(self) -> None:
        """Beamformer property getter should return setter argument"""

        beamformer = Mock()
        self.receiver.beamformer = beamformer

        self.assertIs(beamformer, self.receiver.beamformer)

    def test_receive(self) -> None:
        """Receiving should generate a valid reception"""

        signal = Signal.Create(self.rng.normal(size=(1, 100)) + 1j * self.rng.normal(size=(1, 100)), 1e6, 1e9)
        reception = self.receiver.receive(signal)
        self.assertEqual(100, reception.signal.num_samples)

    def test_yaml_roundtrip_serialization(self) -> None:
        """Test YAML serialization and deserialization"""

        test_yaml_roundtrip_serialization(self, self.receiver)
