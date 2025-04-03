# -*- coding: utf-8 -*-
"""Test HermesPy device module"""

from os.path import join
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from h5py import File
from scipy.constants import speed_of_light
from numpy.testing import assert_array_equal

from hermespy.core import AntennaArray, IdealAntenna, Signal, Transformation, UniformArray
from hermespy.core.device import Device, DeviceInput, DeviceState, ProcessedDeviceInput, DeviceReception, DeviceOutput, DeviceTransmission, MixingOperator, OperationResult, Operator, OperatorSlot, Receiver, Reception, Transmission, TransmitState, Transmitter, UnsupportedSlot
from unit_tests.core.test_factory import test_roundtrip_serialization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DeviceMock(Device[DeviceState]):
    """Mock of the device base class"""

    def __init__(self, *args, **kwargs) -> None:
        self.__carrier_frequency = 1e9
        Device.__init__(self, *args, **kwargs)

    def state(self):
        return DeviceState(
            id(self),
            0.0,
            self.pose,
            self.velocity,
            self.carrier_frequency,
            self.sampling_rate,
            self.num_digital_transmit_ports,
            self.num_digital_receive_ports,
            self.antennas.state(self.pose),
        )

    @property
    def antennas(self) -> AntennaArray:
        return UniformArray(IdealAntenna, 1.0, [2, 1, 1])

    @property
    def velocity(self) -> np.ndarray:
        return np.zeros(3, dtype=float)

    @property
    def sampling_rate(self) -> float:
        return 1.0

    @property
    def carrier_frequency(self) -> float:
        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> float:
        self.__carrier_frequency = value


class OperatorMock(Operator):
    """Mock of the operator base class"""

    def __init__(self, *args, **kwargs) -> None:
        self.__carrier_frequency = 1e9
        Operator.__init__(self, *args, **kwargs)

    @property
    def sampling_rate(self) -> float:
        return 1.0

    @property
    def carrier_frequency(self) -> float:
        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> float:
        self.__carrier_frequency = value

    @property
    def frame_duration(self) -> float:
        return 0.0


class TestOperationResult(TestCase):
    """Test operation result class"""

    def setUp(self) -> None:
        self.signal = Signal.Create(np.random.standard_normal((2, 10)), 1.0)
        self.result = OperationResult(self.signal)

    def test_serialization(self) -> None:
        """Test operation result serialization"""

        test_roundtrip_serialization(self, self.result)


class TestDeviceOutput(TestCase):
    """Test device output base class"""

    def setUp(self) -> None:
        self.signal = Signal.Create(np.random.standard_normal((2, 10)), 1.0)
        self.output = DeviceOutput(self.signal)

    def test_properties(self) -> None:
        """Test the properties of the device output class"""

        self.assertIs(self.signal, self.output.mixed_signal)
        self.assertEqual(self.signal.sampling_rate, self.output.sampling_rate)
        self.assertEqual(self.signal.num_streams, self.output.num_antennas)
        self.assertEqual(self.signal.carrier_frequency, self.output.carrier_frequency)
        self.assertSequenceEqual([self.signal], self.output.emerging_signals)
        self.assertEqual(1, self.output.num_emerging_signals)

    def test_serialization(self) -> None:
        """Test device output serialization"""

        test_roundtrip_serialization(self, self.output)


class TestDeviceTransmission(TestCase):
    """Test device transmission base class"""

    def setUp(self) -> None:
        self.mixed_signal = Signal.Create(np.random.standard_normal((2, 10)), 1.0)
        self.operator_transmissions = [Transmission(self.mixed_signal)]

        self.transmission = DeviceTransmission(self.operator_transmissions, self.mixed_signal)

        self.transmitter = TransmitterMock()
        self.device = DeviceMock()
        self.device.transmitters.add(self.transmitter)

    def test_properties(self) -> None:
        """Test the properties of the device transmission class"""

        self.assertSequenceEqual(self.operator_transmissions, self.operator_transmissions)
        self.assertEqual(1, self.transmission.num_operator_transmissions)

    def test_serialization(self) -> None:
        """Test device transmission serialization"""

        test_roundtrip_serialization(self, self.transmission)


class TestDeviceInput(TestCase):
    """Test device input base class"""

    def setUp(self) -> None:
        self.impinging_signals = [Signal.Create(np.random.standard_normal((2, 10)), 1.0)]
        self.input = DeviceInput(self.impinging_signals)

    def test_properties(self) -> None:
        """Test the properties of the device input class"""

        self.assertSequenceEqual(self.impinging_signals, self.input.impinging_signals)
        self.assertEqual(1, self.input.num_impinging_signals)

    def test_serialization(self) -> None:
        """Test device input serialization"""

        test_roundtrip_serialization(self, self.input)


class TestProcessedDeviceInput(TestCase):
    """Test processed device input base class"""

    def setUp(self) -> None:
        self.impinging_signals = [Signal.Create(np.random.standard_normal((2, 10)), 1.0)]
        self.operator_inputs = [self.impinging_signals[0]]

        self.input = ProcessedDeviceInput(self.impinging_signals, self.operator_inputs)

    def test_properties(self) -> None:
        """Test the properties of the processed device input class"""

        self.assertSequenceEqual(self.impinging_signals, self.input.impinging_signals)
        self.assertEqual(1, self.input.num_impinging_signals)
        self.assertSequenceEqual(self.operator_inputs, self.input.operator_inputs)
        self.assertEqual(1, self.input.num_operator_inputs)

    def test_serialization(self) -> None:
        """Test processed device input serialization"""

        test_roundtrip_serialization(self, self.input)


class TestDeviceReception(TestCase):
    """Test device reception base class"""

    def setUp(self) -> None:
        self.impinging_signals = [Signal.Create(np.random.standard_normal((2, 10)), 1.0)]
        self.operator_inputs = [self.impinging_signals[0]]
        self.operator_receptions = [Reception(Signal.Create(np.random.standard_normal((2, 10)), 1.0))]
        self.reception = DeviceReception(self.impinging_signals, self.operator_inputs, self.operator_receptions)

        self.receiver = ReceiverMock()
        self.device = DeviceMock()
        self.device.receivers.add(self.receiver)

    def test_properties(self) -> None:
        """Test the properties of the device reception class"""

        self.assertSequenceEqual(self.operator_receptions, self.reception.operator_receptions)
        self.assertEqual(1, self.reception.num_operator_receptions)

    def test_serialization(self) -> None:
        """Test device reception serialization"""

        test_roundtrip_serialization(self, self.reception)


class MixingOperatorMock(MixingOperator):
    """Mock of the mixing operator base class"""

    def __init__(self, *args, **kwargs) -> None:
        MixingOperator.__init__(self, *args, **kwargs)

    @property
    def frame_duration(self) -> float:
        return 1.0

    @property
    def sampling_rate(self) -> float:
        return 1.0


class TestMixingOperator(TestCase):
    """Test the base class for mixing operators"""

    def setUp(self) -> None:
        self.mixing_operator = MixingOperatorMock()

    def test_carrier_frequency_validation(self) -> None:
        """Carrier frequency property setter should raise ValueError on negative arguments"""

        with self.assertRaises(ValueError):
            self.mixing_operator.carrier_frequency = -1.0

        try:
            self.mixing_operator.carrier_frequency = 0.0

        except ValueError:
            self.fail()

    def test_carrier_frequency_setget(self) -> None:
        """Carrier frequency property getter should return setter argument"""

        carrier_frequency = 2
        self.mixing_operator.carrier_frequency = carrier_frequency

        self.assertEqual(carrier_frequency, self.mixing_operator.carrier_frequency)


class ReceiverMock(Receiver):
    """Mock of the receiving device operator base class"""

    def __init__(self, *args, **kwargs):
        Receiver.__init__(self, *args, **kwargs)

    def _receive(self, signal: Signal, device: DeviceState) -> Reception:
        return Reception(signal)

    @property
    def frame_duration(self) -> float:
        return 10.0

    @property
    def sampling_rate(self) -> float:
        return 1.0

    @property
    def energy(self) -> float:
        return 1.0

    @property
    def power(self) -> float:
        return 1.0

    def _noise_power(self, strength, snr_type=...) -> float:
        return strength

    def _recall_reception(self, group) -> Reception:
        return Reception.from_HDF(group)


class TestReceiver(TestCase):
    """Test the base class for receiving operators"""

    def setUp(self) -> None:
        self.device = DeviceMock()
        self.seed = 42
        self.receiver = ReceiverMock(seed=42)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertEqual(self.seed, self.receiver.seed)

    def test_slot_setget(self) -> None:
        """Operator slot getter should return setter argument"""

        slot = Mock()
        self.receiver.slot = slot

        self.assertIs(slot, self.receiver.slot)

    def test_reference_transmitter_setget(self) -> None:
        """Reference transmitter property getter should return setter argument"""

        reference = Mock()
        self.receiver.reference = reference

        self.assertIs(reference, self.receiver.reference)

    def test_selected_receive_ports_setget(self) -> None:
        """Selectred receive ports property getter should return setter argument"""

        self.receiver.selected_receive_ports = [0, 1]
        self.assertSequenceEqual([0, 1], self.receiver.selected_receive_ports)

        self.receiver.selected_receive_ports = None
        self.assertIsNone(self.receiver.selected_receive_ports)

    def test_receive_validation(self) -> None:
        """Receiving with an invalid signal model shoud raise a ValueError"""

        with self.assertRaises(ValueError):
            _ = self.receiver.receive(Signal.Empty(1.0, 3), DeviceState.Basic(2, 2))

    def test_receive(self) -> None:
        signal = Mock()
        signal.num_streams = 2
        reception = self.receiver.receive(signal, DeviceState.Basic(2, 2))
        self.assertIs(reception.signal, signal)


class TestOperatorSlot(TestCase):
    """Test device operator slots"""

    def setUp(self) -> None:
        self.device = DeviceMock()
        self.slot = OperatorSlot(self.device)
        self.operator = OperatorMock()

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertIs(self.device, self.slot.device)

    def test_operator_index(self) -> None:
        """Operator index lookup should return the proper operator index"""

        operator = Mock()
        self.slot.add(operator)
        self.slot.add(Mock())

        self.assertEqual(0, self.slot.operator_index(operator))

    def test_add(self) -> None:
        """Adding an operator should register said operator at the slot"""

        operator = Mock()

        self.slot.add(operator)
        self.assertTrue(self.slot.registered(operator))
        self.assertEqual(1, self.slot.num_operators)

        self.slot.add(operator)
        self.assertTrue(self.slot.registered(operator))
        self.assertEqual(1, self.slot.num_operators)

    def test_remove(self) -> None:
        """Operators should be properly removed from the list"""

        self.slot.add(Mock())
        self.slot.remove(self.operator)

        self.assertFalse(self.slot.registered(self.operator))
        self.assertEqual(1, self.slot.num_operators)

    def test_registered(self) -> None:
        """Registered check should return the proper registration state for slots"""

        registered_slot = Mock()
        unregistered_slot = Mock()
        self.slot.add(registered_slot)

        self.assertTrue(self.slot.registered(registered_slot))
        self.assertFalse(self.slot.registered(unregistered_slot))

    def test_num_operators(self) -> None:
        """Number of operators property should compute the correct number of registered operators"""

        self.slot.add(Mock())
        self.assertEqual(1, self.slot.num_operators)

        self.slot.add(Mock())
        self.assertEqual(2, self.slot.num_operators)

    def test_max_sampling_rate(self) -> None:
        """Maximum sampling rate property should compute the correct sampling rate"""

        operator = Mock()
        operator.sampling_rate = 10
        self.slot.add(operator)

        self.assertEqual(10, self.slot.max_sampling_rate)

    def test_min_frame_duration(self) -> None:
        """Minimum frame duration property should compute the correct duration"""

        operator = Mock()
        operator.frame_duration = 10
        self.slot.add(operator)

        self.assertEqual(10, self.slot.min_frame_duration)

    def test_min_num_samples_per_frame(self) -> None:
        """Minimum number of samples per frame property should compute the correct number"""

        operator = Mock()
        operator.sampling_rate = 10
        operator.frame_duration = 10
        self.slot.add(operator)

        self.assertEqual(100, self.slot.min_num_samples_per_frame)

    def test_getitem(self) -> None:
        """Getitem should return the proper operator"""

        self.slot.add(Mock())
        self.slot.add(self.operator)

        self.assertIs(self.operator, self.slot[1])

    def test_iteration(self) -> None:
        """Iteration should yield the proper order of operators"""

        operator = Mock()
        self.slot.add(self.operator)
        self.slot.add(operator)

        expected_iterator_elements = [self.operator, operator]

        for element, expected_element in zip(self.slot.__iter__(), expected_iterator_elements):
            self.assertIs(expected_element, element)

    def test_contains(self) -> None:
        """Contains should return the proper registration state for operators"""

        registered_operator = Mock()
        unregistered_operator = Mock()
        self.slot.add(registered_operator)

        self.assertTrue(registered_operator in self.slot)
        self.assertFalse(unregistered_operator in self.slot)

    def test_len(self) -> None:
        """Len should return the proper number of registered operators"""

        self.slot.add(self.operator)
        self.assertEqual(1, len(self.slot))


class TransmitterMock(Transmitter):
    """Mock of the base class for transmitting operators"""

    def __init__(self, *args, **kwargs) -> None:
        Transmitter.__init__(self, *args, **kwargs)

    def mock_transmission(self, device: TransmitState) -> Transmission:
        rng = np.random.default_rng(42)
        return Transmission(Signal.Create(rng.standard_normal((device.num_digital_transmit_ports, 10)), self.sampling_rate, device.carrier_frequency))

    @property
    def frame_duration(self) -> float:
        return 10.0

    @property
    def power(self) -> float:
        return 1.0

    def _transmit(self, device: DeviceState, duration: float) -> Transmission:
        return self.mock_transmission(device)

    @property
    def sampling_rate(self) -> float:
        return 1.0

    def _recall_transmission(self, group) -> Transmission:
        return Transmission.from_HDF(group)


class TestTransmitter(TestCase):
    """Test transmitting operator base class"""

    def setUp(self) -> None:
        self.device = DeviceMock()
        self.seed = 42
        self.transmitter = TransmitterMock(seed=42)

    def test_init(self) -> None:
        """Initialization parameters should properly be stored as class attributes"""

        self.assertEqual(self.seed, self.transmitter.seed)

    def test_selected_transmit_ports_setget(self) -> None:
        """Selectred transmit ports property getter should return setter argument"""

        self.transmitter.selected_transmit_ports = [0, 1]
        self.assertSequenceEqual([0, 1], self.transmitter.selected_transmit_ports)

        self.transmitter.selected_transmit_ports = None
        self.assertIsNone(self.transmitter.selected_transmit_ports)

    def test_transmission_recall(self) -> None:
        """Test transmission recall from HDF"""
        ...


class TestUnsupportedSlot(TestCase):
    """Test unsupported operator device slot"""

    def setUp(self) -> None:
        self.device = DeviceMock()
        self.slot = UnsupportedSlot(self.device)

    def test_add(self) -> None:
        """Adding to an sunsupported slot should raise a RuntimeError"""

        with self.assertRaises(RuntimeError):
            self.slot.add(Mock())


class TestDevice(TestCase):
    """Test device base class"""

    def setUp(self) -> None:
        self.power = 1.5
        self.position = np.zeros(3)
        self.orientation = np.zeros(3)

        self.device = DeviceMock(power=self.power, pose=Transformation.From_RPY(rpy=self.orientation, pos=self.position))

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertEqual(self.power, self.device.power)
        assert_array_equal(self.position, self.device.position)
        assert_array_equal(self.orientation, self.device.orientation)

    def test_power_setget(self) -> None:
        """Power property getter should return setter argument"""

        power = 1.23
        self.device.power = power

        self.assertEqual(power, self.device.power)

    def test_power_validation(self) -> None:
        """Power property setter should raise ValueError on negative arguments"""

        with self.assertRaises(ValueError):
            self.device.power = -1.0

        try:
            self.device.power = 0.0

        except ValueError:
            self.fail()

    def test_snr(self) -> None:
        """SNR property should return the proper SNR"""

        self.assertEqual(float("inf"), self.device.snr)

    def test_num_antennas(self) -> None:
        """Number of antennas property should return the proper antenna count"""

        self.assertEqual(2, self.device.num_antennas)

    def test_max_frame_duration(self) -> None:
        """Maximum frame duration property should compute the correct duration"""

        transmitter = Mock()
        transmitter.frame_duration = 10
        self.device.transmitters.add(transmitter)

        receiver = Mock()
        receiver.frame_duration = 4
        self.device.receivers.add(receiver)

        self.assertEqual(10, self.device.max_frame_duration)

    def test_wavelength(self) -> None:
        """Wavelength property should return the proper wavelength"""

        self.assertEqual(speed_of_light / self.device.carrier_frequency, self.device.wavelength)

    def test_transmit_operators(self) -> None:
        """Transmit operators property should return the proper operators"""

        expected_transmission = Transmission(Signal.Create(np.random.standard_normal((1, 10)), self.device.sampling_rate, self.device.carrier_frequency))
        transmitter = Mock()
        transmitter.transmit.return_value = expected_transmission
        self.device.transmitters.add(transmitter)

        transmissions = self.device.transmit_operators()
        self.assertSequenceEqual([expected_transmission], transmissions)

    def test_generate_output(self) -> None:
        """Generate output should return the proper output"""

        transmitter = TransmitterMock()
        self.device.transmitters.add(transmitter)
        transmitter.transmit(self.device.state())

        state = self.device.state()
        transmissions = self.device.transmit_operators(state)
        output = self.device.generate_output(transmissions, state)
        mock_transmission = transmitter.mock_transmission(state)

        assert_array_equal(mock_transmission.signal.getitem(), output.mixed_signal.getitem())

    def test_transmit(self) -> None:
        """Transmit should return the proper transmission"""

        transmitter = TransmitterMock()
        self.device.transmitters.add(transmitter)

        transmission = self.device.transmit(self.device.state())
        mock_transmission = transmitter.mock_transmission(self.device.state())
        assert_array_equal(mock_transmission.signal.getitem(), transmission.mixed_signal.getitem())

    def test_process_input(self) -> None:
        """Process input should return the proper processed input"""

        impinging_signal = Signal.Create(np.random.standard_normal((self.device.num_receive_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)

        receiver = Mock()
        receiver.selected_receive_ports = [0]
        self.device.receivers.add(receiver)

        processed_input = self.device.process_input(impinging_signal)
        assert_array_equal(impinging_signal.getitem(), processed_input.impinging_signals[0].getitem())

        processed_input = self.device.process_input([impinging_signal, impinging_signal])
        assert_array_equal(impinging_signal.getitem(), processed_input.impinging_signals[0].getitem())
        assert_array_equal(impinging_signal.getitem(), processed_input.impinging_signals[1].getitem())

    def test_receive_operators_validation(self) -> None:
        """Receive operators should raise a ValueError if number of signals doesn't match number of receivers"""

        with self.assertRaises(ValueError):
            _ = self.device.receive_operators([Mock()])

    def test_receive_operators(self) -> None:
        """Receive operators property should return the proper operators"""

        signal = Mock()
        signal.num_streams = 2
        self.receiver = ReceiverMock()
        self.device.receivers.add(self.receiver)
        state = self.device.state()

        operator_receptions = self.device.receive_operators([signal], state)
        self.assertEqual(1, len(operator_receptions))

        impinging_signals = [Signal.Create(np.random.standard_normal((2, 10)), self.device.sampling_rate, self.device.carrier_frequency)]
        operator_receptions = self.device.receive_operators(self.device.process_input(impinging_signals, state), state)
        self.assertEqual(1, len(operator_receptions))

    def test_receive(self) -> None:
        """Receive should return the proper reception"""

        impinging_signal = Signal.Create(np.random.standard_normal((self.device.num_receive_antennas, 10)), self.device.sampling_rate, self.device.carrier_frequency)

        receiver = Mock()
        receiver.selected_receive_ports = [0]
        self.device.receivers.add(receiver)

        reception = self.device.receive(impinging_signal)
        assert_array_equal(impinging_signal.getitem(), reception.impinging_signals[0].getitem())

    def test_serialization(self) -> None:
        """"Test device roundtrip serialization"""

        test_roundtrip_serialization(self, self.device)
