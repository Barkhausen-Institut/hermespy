# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Type
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from h5py import File, Group
from numpy.testing import assert_array_equal

from hermespy.channel.channel import Channel, ChannelPropagation, ChannelRealization, DirectiveChannelRealization, InterpolationMode
from hermespy.core import ChannelStateInformation, Device, DeviceOutput, Signal
from hermespy.simulation import SimulatedDevice, SimulatedIdealAntenna, SimulatedUniformArray

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ChannelRealizationMock(ChannelRealization):
    """Implementation of the abstract channel realization base calss for testing purposes"""

    def _propagate(self, signal: Signal, transmitter: Device, receiver: Device, interpolation: InterpolationMode) -> Signal:
        return signal

    def state(self, transmitter: Device, receiver: Device, delay: float, sampling_rate: float, num_samples: int, max_num_taps: int) -> ChannelStateInformation:
        return ChannelStateInformation.Ideal(num_samples=num_samples, num_receive_streams=1, num_transmit_streams=1)

    @classmethod
    def From_HDF(cls: Type[ChannelRealizationMock], group: Group, alpha_device: Device, beta_device: Device) -> ChannelRealizationMock:
        return ChannelRealizationMock(alpha_device, beta_device, **cls._parameters_from_HDF(group))


class ChannelMock(Channel[ChannelRealizationMock]):
    """Implementation of the abstract channel base class for testing purposes only"""

    def _realize(self) -> ChannelRealizationMock:
        return ChannelRealizationMock(self.alpha_device, self.beta_device, self.gain)

    def recall_realization(self, group: Group):
        return ChannelRealizationMock.From_HDF(group, self.alpha_device, self.beta_device)


class TestChannelRealization(TestCase):
    """Test base class for channel realizations"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.alpha_device = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (2, 1, 1)))
        self.beta_device = SimulatedDevice()
        self.gain = 0.9

        self.channel = ChannelMock(self.alpha_device, self.beta_device, self.gain)
        self.realization = ChannelRealizationMock(self.alpha_device, self.beta_device, self.gain)

    def test_initialization(self) -> None:
        """Initialization arguments should be properly stored as properties"""

        self.assertIs(self.alpha_device, self.realization.alpha_device)
        self.assertIs(self.beta_device, self.realization.beta_device)
        self.assertEqual(self.gain, self.realization.gain)

    def test_propagate_validation(self) -> None:
        """Propagate routine should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            _ = self.realization.propagate(Mock())

    def test_propagate_deviceoutput(self) -> None:
        """Test propgation of device outputs"""

        signal = Signal(self.rng.random((2, 10)), 1.0)
        device_output = Mock(spec=DeviceOutput)
        device_output.mixed_signal = signal

        propagation = self.realization.propagate(device_output)

        self.assertIsInstance(propagation, ChannelPropagation)

    def test_propagate_signal(self) -> None:
        """Test propagation of signals"""

        signal = Signal(self.rng.random((2, 10)), 1.0)
        propagation = self.realization.propagate(signal)

        self.assertIsInstance(propagation, ChannelPropagation)

    def test_hdf_serialization(self) -> None:
        """Test HDF roundtrip serialization"""

        file = File("test.h5", "w", driver="core", backing_store=False)
        group = file.create_group("group")

        self.realization.to_HDF(group)
        recalled_realization = self.realization.From_HDF(file["group"], self.alpha_device, self.beta_device)

        file.close()

        self.assertIsInstance(recalled_realization, ChannelRealizationMock)
        self.assertIs(self.alpha_device, recalled_realization.alpha_device)
        self.assertIs(self.beta_device, recalled_realization.beta_device)
        self.assertEqual(self.gain, recalled_realization.gain)


class TestDirectiveChannelRealization(TestCase):
    """Test the directive channel realization wrapper"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.transmitter = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (2, 1, 1)))
        self.receiver = SimulatedDevice()
        self.gain = 0.9

        self.channel = ChannelMock(self.transmitter, self.receiver, self.gain)
        self.realization = ChannelRealizationMock(self.transmitter, self.receiver, self.gain)
        self.directive_realization = DirectiveChannelRealization(self.transmitter, self.receiver, self.realization)

    def test_init(self) -> None:
        """Initialization should properly store arguments as properties"""

        self.assertIs(self.transmitter, self.directive_realization.transmitter)
        self.assertIs(self.receiver, self.directive_realization.receiver)

    def test_propagate(self) -> None:
        """Propagation routine should properly call the wrapped realization's propagation routine"""

        signal = Signal(self.rng.random((2, 10)), 1.0)

        directive_propagation = self.directive_realization.propagate(signal)
        specific_propagation = self.realization.propagate(signal, self.transmitter, self.receiver)

        assert_array_equal(directive_propagation.signal.samples, specific_propagation.signal.samples)

    def test_state(self) -> None:
        """State routine should properly call the wrapped realization's state routine"""

        directive_state = self.directive_realization.state(0, 1.0, 10, 1)
        specific_state = self.realization.state(self.transmitter, self.receiver, 0, 1.0, 10, 1)

        assert_array_equal(directive_state.dense_state(), specific_state.dense_state())


class TestChannelPropagation(TestCase):
    """Test channel propagation dataclass"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.transmitter = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (2, 1, 1)))
        self.receiver = SimulatedDevice()
        self.gain = 0.9
        self.realization = ChannelRealizationMock(self.transmitter, self.receiver, self.gain)
        self.signal = Signal(self.rng.random((2, 10)), 1.0)
        self.interpolation_mode = InterpolationMode.NEAREST

        self.propagation = ChannelPropagation[ChannelRealizationMock](self.realization, self.signal, self.transmitter, self.receiver, self.interpolation_mode)

    def test_properties(self) -> None:
        """Initialization should properly initialize object properties"""

        self.assertIs(self.signal, self.propagation.signal)
        self.assertIs(self.transmitter, self.propagation.transmitter)
        self.assertIs(self.receiver, self.propagation.receiver)
        self.assertEqual(self.interpolation_mode, self.propagation.interpolation_mode)

    def test_state(self) -> None:
        """State method should correctly call the realization's state computation routine"""

        with patch.object(self.realization, "state") as state_method_mock:
            expected_state = Mock()
            state_method_mock.return_value = expected_state

            _ = self.propagation.state(0, 1.0, 1, 1)
            state_method_mock.assert_called_once_with(self.transmitter, self.receiver, 0, 1.0, 1, 1)


class TestChannel(TestCase):
    """Test channel base class"""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(42)

        self.alpha_device = SimulatedDevice(antennas=SimulatedUniformArray(SimulatedIdealAntenna, 1e-3, (2, 1, 1)))
        self.beta_device = SimulatedDevice()
        self.gain = 0.8

        self.channel = ChannelMock(self.alpha_device, self.beta_device, 0.8)

    def test_devices_init_validation(self) -> None:
        """Specifying transmitter / receiver and devices is forbidden"""

        with self.assertRaises(ValueError):
            ChannelMock(self.alpha_device, self.beta_device, devices=(Mock(), Mock()))

    def test_devices_init(self) -> None:
        """Specifiying devices insteand of transmitter / receiver should properly initialize channel"""

        self.channel = ChannelMock(devices=(self.alpha_device, self.beta_device))

        self.assertIs(self.alpha_device, self.channel.alpha_device)
        self.assertIs(self.beta_device, self.channel.beta_device)

    def test_alpha_device_setget(self) -> None:
        """Alpha device property getter should return setter argument"""

        expected_alpha_device = Mock()
        self.channel.alpha_device = expected_alpha_device

        self.assertIs(expected_alpha_device, self.channel.alpha_device)

    def test_beta_device_setget(self) -> None:
        """Beta device property getter should return setter argument"""

        expected_beta_device = Mock()
        self.channel.beta_device = expected_beta_device

        self.assertIs(expected_beta_device, self.channel.beta_device)

    def test_scenario_setget(self) -> None:
        """Scenario property setter should correctly configure channel"""

        scenario = Mock()
        self.channel.scenario = scenario

        self.assertIs(scenario, self.channel.scenario)
        self.assertIs(scenario, self.channel.random_mother)

    def test_gain_setget(self) -> None:
        """Gain property getter must return setter parameter"""

        gain = 5.0
        self.channel.gain = 5.0

        self.assertIs(gain, self.channel.gain, "Gain property set/get produced unexpected result")

    def test_gain_validation(self) -> None:
        """Gain property setter must raise exception on arguments smaller than zero"""

        with self.assertRaises(ValueError):
            self.channel.gain = -1.0

        try:
            self.channel.gain = 0.0

        except ValueError:
            self.fail("Gain property set to zero raised unexpected exception")

    def test_interpolation_mode_setget(self) -> None:
        """Interpolation mode property getter should return setter argument"""

        expected_mode = Mock()
        self.channel.interpolation_mode = expected_mode

        self.assertIs(expected_mode, self.channel.interpolation_mode)

    def test_realize(self) -> None:
        """Realizing a channel should generate and cache a new realization"""

        realization = self.channel.realize(cache=True)
        self.assertIs(realization, self.channel.realization)
        self.assertIsInstance(realization, ChannelRealizationMock)

        _ = self.channel.realize(cache=False)
        self.assertIs(realization, self.channel.realization)

    def test_propagate_validation(self) -> None:
        """Propagate routine should raise ValueError on invalid signal stream counts"""

        signal = Signal(self.rng.random((3, 10)), 1.0)

        with self.assertRaises(ValueError):
            self.channel.propagate(signal)

    def test_propagate(self) -> None:
        """Propagation routine should properly realize the channel and propagate the signal"""

        signal = Signal(self.rng.random((2, 10)), 1.0)
        propagation = self.channel.propagate(signal)

        self.assertIsInstance(propagation, ChannelPropagation)
