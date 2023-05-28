# -*- coding: utf-8 -*-
"""Test HermesPy simulated device module"""

from unittest import TestCase
from unittest.mock import Mock, patch, PropertyMock

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from hermespy.core import dB, DeviceInput, RandomNode, Signal, SignalReceiver, SignalTransmitter, IdealAntenna, Transformation, UniformArray
from hermespy.simulation import ProcessedSimulatedDeviceInput, SimulatedDevice, SimulatedDeviceOutput, SpecificIsolation, TriggerModel, TriggerRealization, RandomTrigger, StaticTrigger, OffsetTrigger
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
    
    
class TestTriggerRealiuation(TestCase):
    """Test the trigger realization class"""
    
    def setUp(self) -> None:
        
        self.trigger_realization = TriggerRealization(3, 1.23)
        
    def test_init_validation(self) -> None:
        """Initialization parameters should be properly validated"""
        
        with self.assertRaises(ValueError):
            TriggerRealization(-1, 1.23)
            
        with self.assertRaises(ValueError):
            TriggerRealization(3, -1.23)
        
    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""
        
        self.assertEqual(3, self.trigger_realization.num_offset_samples)
        self.assertEqual(1.23, self.trigger_realization.sampling_rate)
        
    def test_trigger_delay(self) -> None:
        """The trigger delay should be properly calculated"""
        
        self.assertEqual(3 / 1.23, self.trigger_realization.trigger_delay)
        
    def test_compute_num_offset_samples_validation(self) -> None:
        """The offset samples computation should raise a ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            _ =self.trigger_realization.compute_num_offset_samples(-1)
            
        with self.assertRaises(ValueError):
            _ = self.trigger_realization.compute_num_offset_samples(0.)
            
    def test_compute_num_offset_samples(self) -> None:
        """The offset samples computation should return the proper value"""
        
        a = self.trigger_realization.compute_num_offset_samples(2 * self.trigger_realization.sampling_rate)
        b = self.trigger_realization.compute_num_offset_samples(self.trigger_realization.sampling_rate)
        
        self.assertEqual(.5 * a, b)


class TestTriggerModel(TestCase):
    """Test the trigger model base class"""

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
            
    def test_default_realize(self) -> None:
        """The realization routine should generate a zero dealy if the frame duration is zero"""""
        
        with patch('hermespy.core.device.Device.max_frame_duration', new_callable=PropertyMock) as duration_mock:
            
            duration_mock.return_value = 0.
            realization = self.trigger_model.realize()
            
            self.assertEqual(0, realization.trigger_delay)
            
            
class TestSimulatedDeviceOutput(TestCase):
    """Test the simulated device output class"""
    
    def setUp(self) -> None:
        
        self.sampling_rate = 1.23
        self.num_antennas = 1
        self.carrier_frequency = 4.56
        self.emerging_signals = [Signal(np.zeros((1, 10)), self.sampling_rate, self.carrier_frequency) for _ in range(3)]
        self.trigger_realization = TriggerRealization(3, 1.)

        self.output = SimulatedDeviceOutput(self.emerging_signals, self.trigger_realization, self.sampling_rate, self.num_antennas, self.carrier_frequency)
            
    def test_properties(self) -> None:
        """The properties should return the proper values"""
        
        self.assertCountEqual(self.emerging_signals, self.output.emerging_signals)
        self.assertIs(self.trigger_realization, self.output.trigger_realization)
        self.assertEqual(self.sampling_rate, self.output.sampling_rate)
        self.assertEqual(self.num_antennas, self.output.num_antennas)
        self.assertEqual(self.carrier_frequency, self.output.carrier_frequency)
        
    def test_init_validation(self) -> None:
        """Initialization parameters should be properly validated"""
        
        with self.assertRaises(ValueError):
            _ = SimulatedDeviceOutput(self.emerging_signals, self.trigger_realization, 10.2345, self.num_antennas, self.carrier_frequency)

        with self.assertRaises(ValueError):
            _ = SimulatedDeviceOutput(self.emerging_signals, self.trigger_realization, self.sampling_rate, 3, self.carrier_frequency)
            
        with self.assertRaises(ValueError):
            _ = SimulatedDeviceOutput(self.emerging_signals, self.trigger_realization, self.sampling_rate, self.num_antennas, 10.456)

    def test_operator_separation_get(self) -> None:
        """The operator separation getter should return the proper value"""
        
        self.assertTrue(self.output.operator_separation)
    
    
class TestProcessedSimulatedDeviceInput(TestCase):
    """Test the processed simulated device input class"""
    
    def setUp(self) -> None:
        
        self.sampling_rate = 1.23
        self.num_antennas = 1
        self.carrier_frequency = 4.56
        self.impinging_signals = [Signal(np.zeros((1, 10)), self.sampling_rate, self.carrier_frequency) for _ in range(3)]
        self.leaking_signal = Signal(np.zeros((1, 10)), self.sampling_rate, self.carrier_frequency)
        self.baseband_signal = Signal(np.zeros((1, 10)), self.sampling_rate, self.carrier_frequency)
        self.operator_separation = True
        self.operator_inputs = [(s, None) for s in self.impinging_signals]
        self.noise_realizations = [Mock() for _ in range(3)]
        self.trigger_realization = TriggerRealization(0, self.sampling_rate)
        
        self.processed_input = ProcessedSimulatedDeviceInput([self.impinging_signals], self.leaking_signal, self.baseband_signal,
                                                             self.operator_separation, self.operator_inputs, self.noise_realizations, self.trigger_realization)
        
    def test_init_validation(self) -> None:
        """Initialization parameters should be properly validated"""
        
        with self.assertRaises(ValueError):
            ProcessedSimulatedDeviceInput([Mock()], self.leaking_signal, self.baseband_signal,
                                          self.operator_separation, self.operator_inputs, self.noise_realizations, self.trigger_realization)
        
    def test_properties(self) -> None:
        """The properties should return the proper values"""
        
        self.assertIs(self.leaking_signal, self.processed_input.leaking_signal)
        self.assertIs(self.baseband_signal, self.processed_input.baseband_signal)
        self.assertTrue(self.processed_input.operator_separation)
        self.assertCountEqual(self.operator_inputs, self.processed_input.operator_inputs)
        self.assertCountEqual(self.noise_realizations, self.processed_input.noise_realizations)
        self.assertIs(self.trigger_realization, self.processed_input.trigger_realization)


class TestSimulatedDevice(TestCase):
    """Test the simulated device base class"""

    def setUp(self) -> None:

        self.random_node = RandomNode(seed=42)

        self.scenario = Mock()
        self.position = np.zeros(3)
        self.orientation = np.zeros(3)
        self.antennas = UniformArray(IdealAntenna(), 1., (1, 1, 1))

        self.device = SimulatedDevice(scenario=self.scenario, antennas=self.antennas, pose=Transformation.From_RPY(self.orientation, self.position))
        self.device.random_mother = self.random_node

        self.transmitter_alpha = SignalTransmitter(Signal(np.zeros((1, 10)), 1., 0.))
        self.transmitter_beta = SignalTransmitter(Signal(np.zeros((1, 10)), 1., 0.))
        self.device.transmitters.add(self.transmitter_alpha)
        self.device.transmitters.add(self.transmitter_beta)
        
        self.receiver = SignalReceiver(10, 1.)
        self.device.receivers.add(self.receiver)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertIs(self.scenario, self.device.scenario)
        self.assertIs(self.antennas, self.device.antennas)
        assert_array_equal(self.position, self.device.position)
        assert_array_equal(self.orientation, self.device.orientation)

    def test_transmit(self) -> None:
        """Test modem transmission routine"""

    def test_scenario_setget(self) -> None:
        """Scenario property setter should return getter argument"""

        self.device = SimulatedDevice()
        self.device.scenario = self.scenario

        self.assertIs(self.scenario, self.device.scenario)
            
    def test_attached(self) -> None:
        """The attached property should return the proper device attachment state"""

        self.assertTrue(self.device.attached)
        self.assertFalse(SimulatedDevice().attached)

    def test_noise_setget(self) -> None:
        """Noise property getter should return setter argument"""

        noise = Mock()
        self.device.noise = noise

        self.assertIs(noise, self.device.noise)

    def test_sampling_rate_inference(self) -> None:
        """Sampling rate property should attempt to infer the sampling rate from all possible sources"""

        self.assertEqual(1.0, self.device.sampling_rate)

        self.transmitter_alpha.sampling_rate = 1.23
        self.assertEqual(1.23, self.device.sampling_rate)
        
        self.device.sampling_rate = 4.5678
        self.assertEqual(4.5678, self.device.sampling_rate)

    def test_sampling_rate_validation(self) -> None:
        """Sampling rate property setter should raise ValueError on arguments smaller or equal to zero"""

        with self.assertRaises(ValueError):
            self.device.sampling_rate = -1.

        with self.assertRaises(ValueError):
            self.device.sampling_rate = 0.

    def test_carrier_frequency_setget(self) -> None:
        """Carrier frequency property getter should return setter argument"""

        carrier_frequency = 1.23
        self.device.carrier_frequency = carrier_frequency

        self.assertEqual(carrier_frequency, self.device.carrier_frequency)

    def test_carrier_frequency_validation(self) -> None:
        """Carrier frequency property setter should raise RuntimeError on negative arguments"""

        with self.assertRaises(ValueError):
            self.device.carrier_frequency = -1.

        try:
            self.device.carrier_frequency = 0.

        except RuntimeError:

            self.fail()
            
    def test_velocity_validation(self) -> None:
        """Velocity property setter should raise ValueError on non-cartesian vectors"""
        
        with self.assertRaises(ValueError):
            self.device.velocity = np.zeros(2)
            
    def test_velocity_setget(self) -> None:
        """Velocity property getter should return setter argument"""
        
        velocity = np.array([1, 2, 3])
        self.device.velocity = velocity
        
        assert_array_equal(velocity, self.device.velocity)
        
    def test_snr_validation(self) -> None:
        """SNR property setter should raise ValueError on negative arguments"""
        
        with self.assertRaises(ValueError):
            self.device.snr = -1.
            
    def test_snr_setget(self) -> None:
        """SNR property getter should return setter argument"""
        
        snr = 1.23
        self.device.snr = snr
        
        self.assertEqual(snr, self.device.snr)
            
    def test_simulate_input_perfect(self) -> None:
        """Test input modeling without imperfections"""
        
        mixed_signal = Signal(self.random_node._rng.normal(size=(self.device.num_antennas, 10)) + 1j * self.random_node._rng.normal(size=(self.device.num_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)

        perfect_input, _ = self.device._simulate_input(mixed_signal)
        assert_array_almost_equal(mixed_signal.samples, perfect_input.samples)

    def test_simulate_input_leakage(self) -> None:
        """Test leakage input modeling"""
         
        self.device.isolation = SpecificIsolation(dB(0))
        
        mixed_signal = Signal(self.random_node._rng.normal(size=(self.device.num_antennas, 10)) + 1j * self.random_node._rng.normal(size=(self.device.num_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)
        leaking_signal = Signal(self.random_node._rng.normal(size=(self.device.num_antennas, 10)) + 1j * self.random_node._rng.normal(size=(self.device.num_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)
        
        expected_leaking_input = Signal(mixed_signal.samples + leaking_signal.samples, self.device.sampling_rate, self.device.carrier_frequency)
        leaking_input, _ = self.device._simulate_input(mixed_signal, leaking_signal)
        
        assert_array_almost_equal(expected_leaking_input.samples, leaking_input.samples)

    def test_simulate_input_rf_modeling(self) -> None:
        """Test rf impairments input modeling"""
        
        phase_offset = .5 * np.pi
        self.device.rf_chain.phase_offset = phase_offset
        
        mixed_signal = Signal(self.random_node._rng.normal(size=(self.device.num_antennas, 10)) + 1j * self.random_node._rng.normal(size=(self.device.num_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)

        expected_impaired_input = self.device.rf_chain.receive(mixed_signal)
        impaired_input, _ = self.device._simulate_input(mixed_signal)

        assert_array_almost_equal(expected_impaired_input.samples, impaired_input.samples)

    def test_generate_output_validation(self) -> None:
        """The generate output routine should raise a ValueError on invalid arguments"""

        operator_transmissions = [Mock() for _ in range(1 + self.device.transmitters.num_operators)]

        with self.assertRaises(ValueError):
            _ = self.device.generate_output(operator_transmissions=operator_transmissions)

    def test_generate_ouput_operator_separation(self) -> None:
        """The generate output routine should properly separate operator signals"""
        
        self.device.operator_separation = True
        
        transmissions = self.device.transmit_operators()
        ouput = self.device.generate_output(operator_transmissions=transmissions)
        
        self.assertTrue(ouput.operator_separation)

    def test_realize_reception(self) -> None:
        """The realize reception routine should properly realize the reception of a signal"""
        
        realization = self.device.realize_reception()
        self.assertIs(realization, self.device.realization)
        
    def test_process_from_realization_validation(self) -> None:
        """The process from realization routine should raise a ValueError on invalid arguments"""
        
        with self.assertRaises(ValueError):
            _ = self.device.process_from_realization(Mock(), Mock())
            
        with self.assertRaises(ValueError):
            _ = self.device.process_from_realization([Mock()], Mock())
        
    def test_process_output_from_realization(self) -> None:
        """The process from realization routine should properly process a device output"""
        
        device_transmission = self.device.transmit()
        device_realization = self.device.realize_reception()
        
        processed_input = self.device.process_from_realization(device_transmission, device_realization)
        
        self.assertIs(processed_input, self.device.input)
        
    def test_process_device_input_from_realization(self) -> None:
        """The process from realization routine should properly process a device input"""
        
        impinging_signals = [Signal(np.zeros((1, 10)), 1., 0.)]
        input = DeviceInput(impinging_signals=impinging_signals)
        device_realization = self.device.realize_reception()
        
        processed_input = self.device.process_from_realization(input, device_realization)
        
        self.assertIs(processed_input, self.device.input)
        
    def test_process_signal_from_realization(self) -> None:
        """The process from realization routine should properly process a signal"""
        
        signal = Signal(np.zeros((1, 10)), 1., 0.)
        device_realization = self.device.realize_reception()
        
        processed_input = self.device.process_from_realization(signal, device_realization)
        
        self.assertIs(processed_input, self.device.input)
        
    def test_process_impinging_signals_from_realization(self) -> None:
        """The process from realization routine should properly process impinging signals"""
        
        impinging_signals = [Signal(np.zeros((1, 10)), 1., 0.)]
        device_realization = self.device.realize_reception()
        
        processed_input = self.device.process_from_realization(impinging_signals, device_realization)
        
        self.assertIs(processed_input, self.device.input)

    def test_process_input(self) -> None:
        """The process input routine should properly process a device input"""

        impinging_signals = [Signal(np.zeros((1, 10)), 1., 0.)]
        input = DeviceInput(impinging_signals=impinging_signals)
        processed_input = self.device.process_input(input)

        self.assertIs(processed_input, self.device.input)
        
    def test_receive(self) -> None:
        """Test the device reception routine"""
        
        impinging_signals = [Signal(np.zeros((1, 10)), 1., 0.)]
        reception = self.device.receive(impinging_signals)
        
        self.assertEqual(1, reception.num_operator_receptions)

    def test_serialization(self) -> None:
        """Test YAML serialization"""

        default_blacklist = self.device.property_blacklist
        default_blacklist.add('scenario')
        default_blacklist.add('antennas')
        
        with patch('hermespy.simulation.simulated_device.SimulatedDevice.property_blacklist', new_callable=PropertyMock) as blacklist:

            blacklist.return_value = default_blacklist
            test_yaml_roundtrip_serialization(self, self.device, {'sampling_rate', 'scenario', 'antennas', 'attached'})
