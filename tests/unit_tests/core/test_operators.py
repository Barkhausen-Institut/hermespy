# -*- coding: utf-8 -*-

from os.path import join
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from h5py import File
from numpy.testing import assert_array_equal

from hermespy.core import Signal, StaticOperator, SilentTransmitter, SignalTransmitter, SignalReceiver
from hermespy.simulation import SimulatedDevice

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.4.0"
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

    def test_power(self) -> None:
        """Power should be zero"""

        self.assertEqual(0.0, self.transmitter.power)

    def test_transmit(self) -> None:
        """Silent transmission should generate a silent signal"""

        default_transmission = self.transmitter.transmit(self.device.state())
        custom_transmission = self.transmitter.transmit(self.device.state(), duration=10 / self.device.sampling_rate)

        self.assertEqual(self.num_samples, default_transmission.signal.num_samples)
        self.assertCountEqual([0] * 10, custom_transmission.signal.getitem(0).flatten().tolist())


class TestSignalTransmitter(TestCase):
    """Test static arbitrary signal transmission."""

    def setUp(self) -> None:
        self.device = SimulatedDevice(sampling_rate=1e3)

        self.num_samples = 100
        self.sampling_rate = self.device.sampling_rate
        self.signal = Signal.Create(np.ones((1, 10)), sampling_rate=self.device.sampling_rate, carrier_frequency=self.device.carrier_frequency)
        self.transmitter = SignalTransmitter(self.signal)
        self.device.transmitters.add(self.transmitter)

    def test_power(self) -> None:
        """Power should be one"""

        self.assertEqual(1.0, self.transmitter.power)

    def test_signal_setget(self) -> None:
        """Signal property getter should return setter argument"""

        expected_seignal = Signal.Create(np.zeros((1, 10)), sampling_rate=self.device.sampling_rate, carrier_frequency=self.device.carrier_frequency)
        self.transmitter.signal = expected_seignal

        self.assertIs(expected_seignal, self.transmitter.signal)

    def test_transmit(self) -> None:
        """Transmit routine should transmit the submitted signal samples"""

        transmission = self.transmitter.transmit(self.device.state())

        assert_array_equal(self.signal.getitem(), transmission.signal.getitem())

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

    def test_power(self) -> None:
        """Reported power should be the expected power"""

        self.assertEqual(self.expected_power, self.receiver.power)

    def test_receive(self) -> None:
        """Receiver should receive a signal"""

        power_signal = Signal.Create(np.ones((1, 10)), sampling_rate=self.device.sampling_rate, carrier_frequency=self.device.carrier_frequency)
        device_state = self.device.state()
        processed_input = self.device.process_input(power_signal, device_state)
        reception = self.receiver.receive(processed_input.operator_inputs[0], device_state)

        assert_array_equal(reception.signal.getitem(), power_signal.getitem())
