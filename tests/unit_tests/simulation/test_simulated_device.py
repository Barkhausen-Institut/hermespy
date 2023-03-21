# -*- coding: utf-8 -*-
"""Test HermesPy simulated device module."""

from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from hermespy.core import dB, Signal, IdealAntenna, Transformation, UniformArray
from hermespy.simulation import SimulatedDevice, SpecificIsolation, TriggerModel, TriggerRealization, RandomTrigger, StaticTrigger, OffsetTrigger
from unit_tests.core.test_factory import test_yaml_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class MockTriggerModel(TriggerModel):
    """Trigger model implementation only for base class testing purposes"""

    def realize(self) -> TriggerRealization:

        return TriggerRealization(0, 1.)


class TesTriggerModel(TestCase):
    """Test the trigger model base class."""

    def setUp(self) -> None:
        
        self.trigger_model = MockTriggerModel()

    def test_add_remove_devices(self) -> None:
        """Test the device registration mechanism"""

        alpha = SimulatedDevice()
        beta = SimulatedDevice()

        self.trigger_model.add_device(alpha)
        self.trigger_model.add_device(beta)

        self.assertCountEqual((alpha, beta), self.trigger_model.devices)
        self.assertIs(self.trigger_model, alpha.trigger_model)
        self.assertIs(self.trigger_model, beta.trigger_model)

        self.trigger_model.remove_device(beta)

        self.assertCountEqual((alpha,), self.trigger_model.devices)


class TestStaticTrigger(TestCase):
    """Test the static trigger implementation"""

    def setUp(self) -> None:
        
        self.trigger_model = StaticTrigger()

    def test_realize(self) -> None:
        """Realization of a static trigger should always return zero"""

        trigger_realization = self.trigger_model.realize()
        self.assertEqual(0, trigger_realization.num_offset_samples)
        self.assertEqual(1., trigger_realization.sampling_rate)

        self.trigger_model.add_device(SimulatedDevice(sampling_rate=1.234))

        trigger_realization = self.trigger_model.realize()
        self.assertEqual(0, trigger_realization.num_offset_samples)
        self.assertEqual(1.234, trigger_realization.sampling_rate)

class TestOffsetTrigger(TestCase):
    """Test the offset trigger implementation"""

    def setUp(self) -> None:
        
        self.trigger_model = OffsetTrigger(3)
        self.trigger_model.add_device(SimulatedDevice(sampling_rate=1.234))

    def test_realize(self) -> None:
        """Realizing the offset trigger should always return the offset value"""

        realization = self.trigger_model.realize()

        self.assertEqual(3, realization.num_offset_samples)
        self.assertEqual(1.234, realization.sampling_rate)

    def test_realize_validation(self) -> None:
        """Realizing the offset trigger without a controlled dveice should raise a RuntimeError"""

        trigger_model = OffsetTrigger(3)

        with self.assertRaises(RuntimeError):
            _ = trigger_model.realize()

    def test_offset_setget(self) -> None:
        """Offset property getter should return setter argument"""

        self.trigger_model.num_offset_samples = 11
        self.assertEqual(11, self.trigger_model.num_offset_samples)

    def test_offset_validation(self) -> None:
        """Offset property setter should raise ValueError on negative arguments"""

        with self.assertRaises(ValueError):
            self.trigger_model.num_offset_samples = -1


class TestRandomTrigger(TestCase):
    """Test the random trigger implementation"""

    def setUp(self) -> None:
        
        self.trigger_model = RandomTrigger()

        self.devices = [SimulatedDevice(sampling_rate=1.23), SimulatedDevice(sampling_rate=1.23)]
        for device in self.devices:
            self.trigger_model.add_device(device)

    def test_realize_validation(self) -> None:
        """The realization routine should raise RuntimeErrors on invalid arguments"""

        for device in self.devices:
            self.trigger_model.remove_device(device)

        with self.assertRaises(RuntimeError):
            _ = self.trigger_model.realize()

        self.trigger_model.add_device(SimulatedDevice(sampling_rate=1.23))
        self.trigger_model.add_device(SimulatedDevice(sampling_rate=3.45))

        with self.assertRaises(RuntimeError):
            _ = self.trigger_model.realize()

    def test_realize(self) -> None:
        """The realization routine should properly generated a random delay"""

        device = SimulatedDevice()
        self.trigger_model.add_device(device)

        operator = Mock()
        operator.sampling_rate = 1.23
        operator.frame_duration = 10 / operator.sampling_rate
        device.transmitters.add(operator)

        for _ in range(10):

            trigger_realization = self.trigger_model.realize()

            self.assertEqual(1.23, trigger_realization.sampling_rate)
            self.assertGreater(11, trigger_realization.num_offset_samples)
            self.assertLessEqual(0, trigger_realization.num_offset_samples)


class TestSimulatedDevice(TestCase):
    """Test the simulated device base class."""

    def setUp(self) -> None:

        self.random_generator = np.random.default_rng(42)
        self.random_node = Mock()
        self.random_node._rng = self.random_generator

        self.scenario = Mock()
        self.position = np.zeros(3)
        self.orientation = np.zeros(3)
        self.antennas = UniformArray(IdealAntenna(), 1., (1, 1, 1))

        self.device = SimulatedDevice(scenario=self.scenario, antennas=self.antennas, pose=Transformation.From_RPY(self.orientation, self.position))

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertIs(self.scenario, self.device.scenario)
        self.assertIs(self.antennas, self.device.antennas)
        assert_array_equal(self.position, self.device.position)
        assert_array_equal(self.orientation, self.device.orientation)

    def test_transmit(self) -> None:
        """Test modem transmission routine."""

    def test_scenario_setget(self) -> None:
        """Scenario property setter should return getter argument."""

        self.device = SimulatedDevice()
        self.device.scenario = self.scenario

        self.assertIs(self.scenario, self.device.scenario)
            
    def test_attached(self) -> None:
        """The attached property should return the proper device attachment state."""

        self.assertTrue(self.device.attached)
        self.assertFalse(SimulatedDevice().attached)

    def test_noise_setget(self) -> None:
        """Noise property getter should return setter argument."""

        noise = Mock()
        self.device.noise = noise

        self.assertIs(noise, self.device.noise)

    def test_sampling_rate_inference(self) -> None:
        """Sampling rate property should attempt to infer the sampling rate from all possible sources."""

        self.assertEqual(1.0, self.device.sampling_rate)

        receiver = Mock()
        receiver.sampling_rate = 1.23
        self.device.receivers.add(receiver)
        self.assertEqual(1.23, self.device.sampling_rate)

        transmitter = Mock()
        transmitter.sampling_rate = 4.56
        self.device.transmitters.add(transmitter)
        self.assertEqual(4.56, self.device.sampling_rate)

        sampling_rate = 7.89
        self.device.sampling_rate = sampling_rate
        self.assertEqual(sampling_rate, self.device.sampling_rate)

    def test_sampling_rate_validation(self) -> None:
        """Sampling rate property setter should raise ValueError on arguments smaller or equal to zero."""

        with self.assertRaises(ValueError):
            self.device.sampling_rate = -1.

        with self.assertRaises(ValueError):
            self.device.sampling_rate = 0.

    def test_carrier_frequency_setget(self) -> None:
        """Carrier frequency property getter should return setter argument."""

        carrier_frequency = 1.23
        self.device.carrier_frequency = carrier_frequency

        self.assertEqual(carrier_frequency, self.device.carrier_frequency)

    def test_carrier_frequency_validation(self) -> None:
        """Carrier frequency property setter should raise RuntimeError on negative arguments."""

        with self.assertRaises(ValueError):
            self.device.carrier_frequency = -1.

        try:
            self.device.carrier_frequency = 0.

        except RuntimeError:

            self.fail()
            
    def test_simulate_input_perfect(self) -> None:
        """Test input modeling without imperfections"""
        
        mixed_signal = Signal(self.random_generator.normal(size=(self.device.num_antennas, 10)) + 1j * self.random_generator.normal(size=(self.device.num_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)

        perfect_input, _ = self.device._simulate_input(mixed_signal)
        assert_array_almost_equal(mixed_signal.samples, perfect_input.samples)

    def test_simulate_input_leakage(self) -> None:
        """Test leakage input modeling"""
         
        self.device.isolation = SpecificIsolation(dB(0))
        
        mixed_signal = Signal(self.random_generator.normal(size=(self.device.num_antennas, 10)) + 1j * self.random_generator.normal(size=(self.device.num_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)
        leaking_signal = Signal(self.random_generator.normal(size=(self.device.num_antennas, 10)) + 1j * self.random_generator.normal(size=(self.device.num_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)
        
        expected_leaking_input = Signal(mixed_signal.samples + leaking_signal.samples, self.device.sampling_rate, self.device.carrier_frequency)
        leaking_input, _ = self.device._simulate_input(mixed_signal, leaking_signal)
        
        assert_array_almost_equal(expected_leaking_input.samples, leaking_input.samples)

    def test_simulate_input_rf_modeling(self) -> None:
        """Test rf impairments input modeling"""
        
        phase_offset = .5 * np.pi
        self.device.rf_chain.phase_offset = phase_offset
        
        mixed_signal = Signal(self.random_generator.normal(size=(self.device.num_antennas, 10)) + 1j * self.random_generator.normal(size=(self.device.num_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)

        expected_impaired_input = self.device.rf_chain.receive(mixed_signal)
        impaired_input, _ = self.device._simulate_input(mixed_signal)

        assert_array_almost_equal(expected_impaired_input.samples, impaired_input.samples)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        default_blacklist = self.device.property_blacklist
        default_blacklist.add('scenario')
        default_blacklist.add('antennas')
        
        with patch('hermespy.simulation.simulated_device.SimulatedDevice.property_blacklist', new_callable=PropertyMock) as blacklist:

            blacklist.return_value = default_blacklist
            test_yaml_roundtrip_serialization(self, self.device, {'sampling_rate', 'scenario', 'antennas', 'attached'})
