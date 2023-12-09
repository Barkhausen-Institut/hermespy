# -*- coding: utf-8 -*-

from os.path import join
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from h5py import File
from numpy.testing import assert_array_equal

from hermespy.core import Signal, SNRType, StaticOperator, SilentTransmitter, SignalTransmitter, SignalReceiver
from hermespy.simulation import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestStaticOperator(TestCase):
    """Test the static device operator base class"""

    def setUp(self) -> None:
        self.device = SimulatedDevice(sampling_rate=1e3)

        self.num_samples = 100
        self.sampling_rate = self.device.sampling_rate
        self.operator = StaticOperator(self.num_samples, self.sampling_rate)
        self.device.transmitters.add(self.operator)

    def test_sampling_rate_validation(self) -> None:
        """Sampling rate property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.operator.sampling_rate = 0.0

    def test_sampling_rate(self) -> None:
        """Sampling rate should be the configured sampling rate"""

        self.assertEqual(self.sampling_rate, self.operator.sampling_rate)

    def test_frame_duration(self) -> None:
        """Silent transmitter should report the correct frame duration"""

        expected_duration = self.num_samples / self.sampling_rate
        self.assertEqual(expected_duration, self.operator.frame_duration)


class TestSilentTransmitter(TestCase):
    """Test static silent signal transmission."""

    def setUp(self) -> None:
        self.device = SimulatedDevice(sampling_rate=1e3)

        self.num_samples = 100
        self.sampling_rate = self.device.sampling_rate
        self.transmitter = SilentTransmitter(self.num_samples, self.sampling_rate)
        self.device.transmitters.add(self.transmitter)

    def test_transmit(self) -> None:
        """Silent transmission should generate a silent signal"""

        default_transmission = self.transmitter.transmit()
        custom_transmission = self.transmitter.transmit(10 / self.device.sampling_rate)

        self.assertEqual(self.num_samples, default_transmission.signal.num_samples)
        self.assertCountEqual([0] * 10, custom_transmission.signal.samples[0, :].tolist())


class TestSignalTransmitter(TestCase):
    """Test static arbitrary signal transmission."""

    def setUp(self) -> None:
        self.device = SimulatedDevice(sampling_rate=1e3)

        self.num_samples = 100
        self.sampling_rate = self.device.sampling_rate
        self.signal = Signal(np.ones((1, 10)), sampling_rate=self.device.sampling_rate, carrier_frequency=self.device.carrier_frequency)
        self.transmitter = SignalTransmitter(self.signal)
        self.device.transmitters.add(self.transmitter)

    def test_signal_setget(self) -> None:
        """Signal property getter should return setter argument"""

        expected_seignal = Signal(np.zeros((1, 10)), sampling_rate=self.device.sampling_rate, carrier_frequency=self.device.carrier_frequency)
        self.transmitter.signal = expected_seignal

        self.assertIs(expected_seignal, self.transmitter.signal)

    def test_transmit(self) -> None:
        """Transmit routine should transmit the submitted signal samples"""

        transmission = self.transmitter.transmit()

        assert_array_equal(self.signal.samples, transmission.signal.samples)

    def test_recall_transmission(self) -> None:
        """Recall transmission should recall the last transmission"""

        transmission = self.transmitter.transmit()

        with TemporaryDirectory() as temp:
            file_location = join(temp, "test.h5")
            with File(file_location, "w") as file:
                transmission.to_HDF(file.create_group("transmission"))

            with File(file_location, "r") as file:
                recalled_transmission = self.transmitter.recall_transmission(file["transmission"])

        assert_array_equal(transmission.signal.samples, recalled_transmission.signal.samples)


class TestSignalReceiver(TestCase):
    def setUp(self) -> None:
        self.device = SimulatedDevice(sampling_rate=1e3)

        self.num_samples = 100
        self.sampling_rate = self.device.sampling_rate
        self.expected_power = 1.234
        self.receiver = SignalReceiver(self.num_samples, self.sampling_rate, self.expected_power)
        self.device.receivers.add(self.receiver)

    def test_init_validation(self) -> None:
        """Init should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            SignalReceiver(1, self.sampling_rate, -1.234)

    def test_energy(self) -> None:
        """Reported energy should be zero"""

        self.assertEqual(123.4, self.receiver.energy)

    def test_receive(self) -> None:
        """Receiver should receive a signal"""

        power_signal = Signal(np.ones((1, 10)), sampling_rate=self.device.sampling_rate, carrier_frequency=self.device.carrier_frequency)
        self.device.process_input(power_signal)

        received_signal = self.receiver.receive().signal

        assert_array_equal(received_signal.samples, power_signal.samples)

    def test_noise_power_validation(self) -> None:
        """Noise power routine should raise ValueError on invalid noise types"""

        with self.assertRaises(ValueError):
            self.receiver.noise_power(1, "invalid")

    def test_noise_power(self) -> None:
        """Noise power routine should compute the correct noise power"""

        noise_power = self.receiver.noise_power(1, SNRType.PN0)
        self.assertEqual(self.expected_power, noise_power)

        noise_energy = self.receiver.noise_power(1, SNRType.EN0)
        self.assertEqual(self.expected_power * self.num_samples, noise_energy)
