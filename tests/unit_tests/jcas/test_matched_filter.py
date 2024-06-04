# -*- coding: utf-8 -*-

from os import path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch, PropertyMock

import numpy as np
from h5py import File

from hermespy.core import Signal
from hermespy.modem import DuplexModem
from hermespy.radar import Radar
from hermespy.simulation import SimulatedDevice
from hermespy.jcas import MatchedFilterJcas
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization
from unit_tests.modem.test_waveform import MockCommunicationWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "Jan Adler"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestMatchedFilterJoint(TestCase):
    """Matched filter joint testing."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)
        self.carrier_frequency = 1e8

        self.joint = MatchedFilterJcas(max_range=10)
        self.waveform = MockCommunicationWaveform()
        self.joint.waveform = self.waveform

        self.device = SimulatedDevice(carrier_frequency=self.carrier_frequency)
        self.device._rng = self.rng
        self.device.transmitters.add(self.joint)
        self.device.receivers.add(self.joint)

    def test_receive_validation(self) -> None:
        """Receiving should raise a RuntimeError if there's no cached transmission"""

        with self.assertRaises(RuntimeError):
            self.joint.receive(Signal.Create(np.zeros((1, 10)), 1.0))

    def test_transmit_receive(self) -> None:
        num_delay_samples = 10
        transmission = self.joint.transmit()

        delay_offset = Signal.Create(np.zeros((1, num_delay_samples), dtype=complex), transmission.signal.sampling_rate, carrier_frequency=self.carrier_frequency)
        delay_offset.append_samples(transmission.signal)

        self.joint.cache_reception(delay_offset)

        reception = self.joint.receive()
        self.assertTrue(10, reception.cube.data.argmax)

        padded_reception = self.joint.receive(transmission.signal)
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

    def test_device_setget(self) -> None:
        """Device property getter should return setter argument"""

        expected_device = SimulatedDevice()
        self.joint.device = expected_device

        self.assertIs(expected_device, self.joint.device)
        self.assertIs(expected_device, DuplexModem.device.fget(self.joint))
        self.assertIs(expected_device, Radar.device.fget(self.joint))

    def test_recall_transmission(self) -> None:
        """Test joint transmission recall from HDF"""

        transmission = self.joint.transmit()

        with TemporaryDirectory() as tempdir:
            file_location = path.join(tempdir, "testfile.hdf5")

            with File(file_location, "w") as file:
                group = file.create_group("testgroup")
                transmission.to_HDF(group)

            with File(file_location, "r") as file:
                recalled_transmission = self.joint._recall_transmission(file["testgroup"])

        self.assertEqual(transmission.signal.num_samples, recalled_transmission.signal.num_samples)

    def test_recall_reception(self) -> None:
        """Test joint reception recall from HDF"""

        transmission = self.joint.transmit()
        reception = self.joint.receive(transmission.signal)

        with TemporaryDirectory() as tempdir:
            file_location = path.join(tempdir, "testfile.hdf5")

            with File(file_location, "w") as file:
                group = file.create_group("testgroup")
                reception.to_HDF(group)

            with File(file_location, "r") as file:
                recalled_reception = self.joint._recall_reception(file["testgroup"])

        self.assertEqual(reception.signal.num_samples, recalled_reception.signal.num_samples)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        with patch("hermespy.jcas.matched_filtering.MatchedFilterJcas.property_blacklist", new_callable=PropertyMock) as blacklist, patch("hermespy.jcas.matched_filtering.MatchedFilterJcas.waveform", new_callable=PropertyMock) as waveform:
            blacklist.return_value = {"slot", "waveform"}
            waveform.return_value = self.waveform

            test_yaml_roundtrip_serialization(self, self.joint)
