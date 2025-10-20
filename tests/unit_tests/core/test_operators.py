# -*- coding: utf-8 -*-

from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal

from hermespy.core import DenseSignal, StaticOperator, SilentTransmitter, SignalTransmitter, SignalReceiver
from hermespy.simulation import SimulatedDevice
from unit_tests.utils import assert_signals_equal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestStaticOperator(TestCase):
    """Test the static device operator base class"""

    def setUp(self) -> None:

        self.num_samples = 100
        self.operator = StaticOperator(self.num_samples)

    def test_properties(self) -> None:
        """Properties should return the submitted arguments"""

        self.assertEqual(self.num_samples, self.operator.num_samples)


class TestSilentTransmitter(TestCase):
    """Test static silent signal transmission."""

    def setUp(self) -> None:
        self.device = SimulatedDevice(bandwidth=1e3)

        self.num_samples = 100
        self.transmitter = SilentTransmitter(self.num_samples)
        self.device.transmitters.add(self.transmitter)

    def test_power(self) -> None:
        """Power should be zero"""

        self.assertEqual(0.0, self.transmitter.power)

    def test_transmit(self) -> None:
        """Silent transmission should generate a silent signal"""

        default_transmission = self.transmitter.transmit(self.device.state())
        custom_transmission = self.transmitter.transmit(self.device.state(), duration=10 / self.device.sampling_rate)

        self.assertEqual(self.num_samples, default_transmission.signal.num_samples)
        assert_array_almost_equal(np.zeros((1, 10)), custom_transmission.signal.view(np.ndarray))


class TestSignalTransmitter(TestCase):
    """Test static arbitrary signal transmission."""

    def setUp(self) -> None:
        self.device = SimulatedDevice(bandwidth=1e3)

        self.num_samples = 100
        self.signal = DenseSignal.FromNDArray(np.ones((1, 10)), sampling_rate=self.device.sampling_rate, carrier_frequency=self.device.carrier_frequency)
        self.transmitter = SignalTransmitter(self.signal)
        self.device.transmitters.add(self.transmitter)

    def test_power(self) -> None:
        """Power should be one"""

        self.assertAlmostEqual(1.0, self.transmitter.power)

    def test_signal_setget(self) -> None:
        """Signal property getter should return setter argument"""

        expected_seignal = DenseSignal.FromNDArray(np.zeros((1, 10)), sampling_rate=self.device.sampling_rate, carrier_frequency=self.device.carrier_frequency)
        self.transmitter.signal = expected_seignal

        self.assertIs(expected_seignal, self.transmitter.signal)

    def test_transmit(self) -> None:
        """Transmit routine should transmit the submitted signal samples"""

        transmission = self.transmitter.transmit(self.device.state())
        assert_signals_equal(self, self.signal, transmission.signal)


class TestSignalReceiver(TestCase):
    def setUp(self) -> None:
        self.device = SimulatedDevice(bandwidth=1e3)

        self.num_samples = 100
        self.expected_power = 1.234
        self.receiver = SignalReceiver(self.num_samples, expected_power=self.expected_power)
        self.device.receivers.add(self.receiver)

    def test_energy(self) -> None:
        """Reported energy should be zero"""

        self.assertEqual(123.4, self.receiver.energy)

    def test_power(self) -> None:
        """Reported power should be the expected power"""

        self.assertEqual(self.expected_power, self.receiver.power)

    def test_receive(self) -> None:
        """Receiver should receive a signal"""

        power_signal = DenseSignal.FromNDArray(np.ones((1, 100)), sampling_rate=self.device.sampling_rate, carrier_frequency=self.device.carrier_frequency)
        device_state = self.device.state()
        processed_input = self.device.process_input(power_signal, device_state)
        reception = self.receiver.receive(processed_input.operator_inputs[0], device_state)

        assert_signals_equal(self, reception.signal, power_signal)
