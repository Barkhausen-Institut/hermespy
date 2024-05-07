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
from hermespy.core.device import Device, DeviceInput, ProcessedDeviceInput, DeviceReception, DeviceOutput, DeviceTransmission, MixingOperator, OperationResult, Operator, OperatorSlot, Receiver, ReceiverSlot, Reception, Transmission, Transmitter, TransmitterSlot, UnsupportedSlot

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DeviceMock(Device):
    """Mock of the device base class"""

    def __init__(self, *args, **kwargs) -> None:
        self.__carrier_frequency = 1e9
        Device.__init__(self, *args, **kwargs)

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
        self.signal = Signal(np.random.standard_normal((2, 10)), 1.0)
        self.result = OperationResult(self.signal)

    def test_hdf_serialization(self) -> None:
        """Test HDF roundtrip serialization"""

        with TemporaryDirectory() as tempdir:
            file_path = join(tempdir, "test.hdf")

            file = File(file_path, "w")
            group = file.create_group("g1")
            self.result.to_HDF(group)
            file.close()

            file = File(file_path, "r")
            recalled_result = OperationResult.from_HDF(file["g1"])
            file.close()

            assert_array_equal(self.result.signal.samples, recalled_result.signal.samples)


class TestOperator(TestCase):
    """Test device slot operators"""

    def setUp(self) -> None:
        self.device = DeviceMock()
        self.slot = OperatorSlot(device=self.device)
        self.operator = Operator()
        self.slot.add(self.operator)

    def test_slot_setget(self) -> None:
        """Slot property getter should return setter argument"""

        slot = OperatorSlot(self.device)
        self.operator.slot = slot
        self.assertIs(slot, self.operator.slot)

        self.operator.slot = None
        self.assertEqual(None, self.operator.slot)

    def test_slot_set(self) -> None:
        """Setting an operator slot should result in the operator being registered at the slot"""

        slot = OperatorSlot(self.device)
        self.operator.slot = slot

        self.assertTrue(slot.registered(self.operator))

    def test_slot_index(self) -> None:
        """The slot index property should return the proper slot index"""

        self.assertEqual(0, self.operator.slot_index)
        self.assertEqual(1, self.slot.num_operators)

        self.slot.add(Operator())
        self.slot.add(Operator())

        new_operator = Operator()
        self.slot.add(new_operator)
        self.assertEqual(3, new_operator.slot_index)

        unbound_operator = Operator()
        self.assertIsNone(unbound_operator.slot_index)

    def test_device(self) -> None:
        """The device property should return the proper handle"""

        self.assertIs(self.device, self.operator.device)

        self.operator.slot = None
        self.assertEqual(None, self.operator.device)

    def test_attached(self) -> None:
        """Attached property should return correct attachment status"""

        self.assertTrue(self.operator.attached)

        self.operator.slot = None
        self.assertFalse(self.operator.attached)


class TestDeviceOutput(TestCase):
    """Test device output base class"""

    def setUp(self) -> None:
        self.signal = Signal(np.random.standard_normal((2, 10)), 1.0)
        self.output = DeviceOutput(self.signal)

    def test_properties(self) -> None:
        """Test the properties of the device output class"""

        self.assertIs(self.signal, self.output.mixed_signal)
        self.assertEqual(self.signal.sampling_rate, self.output.sampling_rate)
        self.assertEqual(self.signal.num_streams, self.output.num_antennas)
        self.assertEqual(self.signal.carrier_frequency, self.output.carrier_frequency)
        self.assertSequenceEqual([self.signal], self.output.emerging_signals)
        self.assertEqual(1, self.output.num_emerging_signals)

    def test_hdf_serialization(self) -> None:
        """Test HDF roundtrip serialization"""

        with TemporaryDirectory() as tempdir:
            file_path = join(tempdir, "test.hdf")

            file = File(file_path, "w")
            group = file.create_group("g1")
            self.output.to_HDF(group)
            file.close()

            file = File(file_path, "r")
            recalled_output = DeviceOutput.from_HDF(file["g1"])
            file.close()

            assert_array_equal(self.signal.samples, recalled_output.mixed_signal.samples)


class TestDeviceTransmission(TestCase):
    """Test device transmission base class"""

    def setUp(self) -> None:
        self.mixed_signal = Signal(np.random.standard_normal((2, 10)), 1.0)
        self.operator_transmissions = [Transmission(self.mixed_signal)]

        self.transmission = DeviceTransmission(self.operator_transmissions, self.mixed_signal)

        self.transmitter = TransmitterMock()
        self.device = DeviceMock()
        self.device.transmitters.add(self.transmitter)

    def test_properties(self) -> None:
        """Test the properties of the device transmission class"""

        self.assertSequenceEqual(self.operator_transmissions, self.operator_transmissions)
        self.assertEqual(1, self.transmission.num_operator_transmissions)

    def test_hdf_serialization(self) -> None:
        """Test HDF roundtrip serialization"""

        with TemporaryDirectory() as tempdir:
            file_path = join(tempdir, "test.hdf")

            file = File(file_path, "w")
            group = file.create_group("g1")
            self.transmission.to_HDF(group)
            file.close()

            file = File(file_path, "r")
            deserialized_transmission = DeviceTransmission.from_HDF(file["g1"])
            recalled_transmission = DeviceTransmission.Recall(file["g1"], self.device)
            file.close()

            assert_array_equal(self.mixed_signal.samples, deserialized_transmission.mixed_signal.samples)
            assert_array_equal(self.mixed_signal.samples, recalled_transmission.mixed_signal.samples)


class TestDeviceInput(TestCase):
    """Test device input base class"""

    def setUp(self) -> None:
        self.impinging_signals = [Signal(np.random.standard_normal((2, 10)), 1.0)]
        self.input = DeviceInput(self.impinging_signals)

    def test_properties(self) -> None:
        """Test the properties of the device input class"""

        self.assertSequenceEqual(self.impinging_signals, self.input.impinging_signals)
        self.assertEqual(1, self.input.num_impinging_signals)

    def test_hdf_serialization(self) -> None:
        """Test HDF roundtrip serialization"""

        with TemporaryDirectory() as tempdir:
            file_path = join(tempdir, "test.hdf")

            file = File(file_path, "w")
            group = file.create_group("g1")
            self.input.to_HDF(group)
            file.close()

            file = File(file_path, "r")
            recalled_input = DeviceInput.from_HDF(file["g1"])
            file.close()

            assert_array_equal(self.impinging_signals[0].samples, recalled_input.impinging_signals[0].samples)


class TestProcessedDeviceInput(TestCase):
    """Test processed device input base class"""

    def setUp(self) -> None:
        self.impinging_signals = [Signal(np.random.standard_normal((2, 10)), 1.0)]
        self.operator_inputs = [self.impinging_signals[0]]

        self.input = ProcessedDeviceInput(self.impinging_signals, self.operator_inputs)

    def test_properties(self) -> None:
        """Test the properties of the processed device input class"""

        self.assertSequenceEqual(self.impinging_signals, self.input.impinging_signals)
        self.assertEqual(1, self.input.num_impinging_signals)
        self.assertSequenceEqual(self.operator_inputs, self.input.operator_inputs)
        self.assertEqual(1, self.input.num_operator_inputs)

    def test_hdf_serialization(self) -> None:
        """Test HDF roundtrip serialization"""

        with TemporaryDirectory() as tempdir:
            file_path = join(tempdir, "test.hdf")

            file = File(file_path, "w")
            group = file.create_group("g1")
            self.input.to_HDF(group)
            file.close()

            file = File(file_path, "r")
            recalled_input = ProcessedDeviceInput.from_HDF(file["g1"])
            file.close()

            assert_array_equal(self.impinging_signals[0].samples, recalled_input.impinging_signals[0].samples)


class TestDeviceReception(TestCase):
    """Test device reception base class"""

    def setUp(self) -> None:
        self.impinging_signals = [Signal(np.random.standard_normal((2, 10)), 1.0)]
        self.operator_inputs = [self.impinging_signals[0]]
        self.operator_receptions = [Reception(Signal(np.random.standard_normal((2, 10)), 1.0))]
        self.reception = DeviceReception(self.impinging_signals, self.operator_inputs, self.operator_receptions)

        self.receiver = ReceiverMock()
        self.device = DeviceMock()
        self.device.receivers.add(self.receiver)

    def test_properties(self) -> None:
        """Test the properties of the device reception class"""

        self.assertSequenceEqual(self.operator_receptions, self.reception.operator_receptions)
        self.assertEqual(1, self.reception.num_operator_receptions)

    def test_hdf_serialization(self) -> None:
        """Test HDF roundtrip serialization"""

        file = File("test.h5", "w", driver="core", backing_store=False)
        group = file.create_group("g1")

        self.reception.to_HDF(group)
        deserialized_reception = DeviceReception.from_HDF(file["g1"])
        recalled_reception = DeviceReception.Recall(file["g1"], self.device)

        file.close()

        assert_array_equal(self.operator_receptions[0].signal.samples, deserialized_reception.operator_receptions[0].signal.samples)
        assert_array_equal(self.operator_receptions[0].signal.samples, recalled_reception.operator_receptions[0].signal.samples)


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
        self.slot = Mock()
        self.slot.device = Mock()
        self.mixing_operator = MixingOperatorMock(slot=self.slot)

    def test_carrier_frequency_setget(self) -> None:
        """Carrier frequency property getter should return setter argument"""

        carrier_frequency = 2
        self.mixing_operator.carrier_frequency = carrier_frequency

        self.assertEqual(carrier_frequency, self.mixing_operator.carrier_frequency)

    def test_carrier_frequency_device_get(self) -> None:
        """Carrier frequency property should return the device's carrier frequency by default"""

        carrier_frequency = 10
        self.slot.device.carrier_frequency = carrier_frequency
        self.mixing_operator.carrier_frequency = None

        self.assertEqual(carrier_frequency, self.mixing_operator.carrier_frequency)

    def test_carrier_frequency_default_get(self) -> None:
        """Operator should return base band carrier frequency if no device is attached"""

        operator = MixingOperator()
        self.assertEqual(0.0, operator.carrier_frequency)

    def test_carrier_frequency_validation(self) -> None:
        """Carrier frequency property setter should raise ValueError on negative arguments"""

        with self.assertRaises(ValueError):
            self.mixing_operator.carrier_frequency = -1.0

        try:
            self.mixing_operator.carrier_frequency = 0.0

        except ValueError:
            self.fail()


class ReceiverMock(Receiver):
    """Mock of the receiving device operator base class"""

    def __init__(self, *args, **kwargs):
        Receiver.__init__(self, *args, **kwargs)

    def _receive(self, signal) -> Reception:
        return Reception(Signal)

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
        self.device = Mock()
        self.device.antennas = UniformArray(IdealAntenna, 1., [2, 1, 1])
        self.slot = ReceiverSlot(device=self.device)
        self.seed = 42
        self.receiver = ReceiverMock(seed=42, slot=self.slot)

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

    def test_selected_receive_ports_validation(self) -> None:
        """Selected receive ports property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.receiver.selected_receive_ports = [11, 12]

    def test_selected_receive_ports_setget(self) -> None:
        """Selectred receive ports property getter should return setter argument"""

        self.receiver.selected_receive_ports = [0, 1]
        self.assertSequenceEqual([0, 1], self.receiver.selected_receive_ports)

        self.receiver.selected_receive_ports = None
        self.assertIsNone(self.receiver.selected_receive_ports)

    def test_num_receive_ports(self) -> None:
        """Number of receive ports property should return the correct number"""

        self.assertEqual(2, self.receiver.num_receive_ports)

        self.receiver.selected_receive_ports = [0, 1]
        self.assertEqual(2, self.receiver.num_receive_ports)

        detached_receiver = ReceiverMock()
        self.assertEqual(0, detached_receiver.num_receive_ports)

    def test_num_receive_antennas(self) -> None:
        """Number of receive antennas property should return the correct number"""

        self.assertEqual(2, self.receiver.num_receive_antennas)

        self.receiver.selected_receive_ports = [0, 1]
        self.assertEqual(2, self.receiver.num_receive_antennas)

        detached_receiver = ReceiverMock()
        self.assertEqual(0, detached_receiver.num_receive_antennas)

        detached_receiver.selected_receive_ports = [0, 1, 2]
        self.assertEqual(0, detached_receiver.num_receive_antennas)

    def test_cache_reception(self) -> None:
        """Cached receptions should be returned by the signal and channel realization properties"""

        signal = Mock()
        self.receiver.cache_reception(signal)

        self.assertIs(signal, self.receiver.signal)

    def test_receive_validation(self) -> None:
        """Receiving without cached reception should raise RuntimeError"""

        with self.assertRaises(RuntimeError):
            _ = self.receiver.receive()

        with self.assertRaises(ValueError):
            _ = self.receiver.receive(Signal.empty(1.0, 3), cache=True)

    def test_receive(self) -> None:
        signal = Mock()
        signal.num_streams = 2
        self.receiver.cache_reception(signal)
        reception = self.receiver.receive(cache=True)

        self.assertIs(reception, self.receiver.reception)


class TestOperatorSlot(TestCase):
    """Test device operator slots"""

    def setUp(self) -> None:
        self.device = DeviceMock()
        self.slot = OperatorSlot(device=self.device)
        self.operator = OperatorMock(slot=self.slot)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes"""

        self.assertIs(self.device, self.slot.device)

    def test_operator_index(self) -> None:
        """Operator index lookup should return the proper operator index"""

        operator = Mock()
        self.slot.add(operator)
        self.slot.add(Mock())

        self.assertEqual(1, self.slot.operator_index(operator))

    def test_add(self) -> None:
        """Adding an operator should register said operator at the slot"""

        operator = Mock()
        self.slot.add(operator)

        self.assertTrue(self.slot.registered(operator))
        self.assertIs(operator.slot, self.slot)

        self.slot.add(operator)
        self.assertEqual(2, self.slot.num_operators)

    def test_remove(self) -> None:
        """Operators should be properly removed from the list"""

        self.slot.add(Mock())
        self.slot.remove(self.operator)

        self.assertFalse(self.slot.registered(self.operator))
        self.assertEqual(None, self.operator.slot)

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
        self.assertEqual(2, self.slot.num_operators)

        self.slot.add(Mock())
        self.assertEqual(3, self.slot.num_operators)

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

        operator = Mock()
        self.slot.add(operator)

        self.assertIs(operator, self.slot[1])

    def test_iteration(self) -> None:
        """Iteration should yield the proper order of operators"""

        operator = Mock()
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

        self.assertEqual(1, len(self.slot))


class TransmitterMock(Transmitter):
    """Mock of the base class for transmitting operators"""

    def __init__(self, *args, **kwargs) -> None:
        Transmitter.__init__(self, *args, **kwargs)

    @property
    def mock_transmission(self) -> Transmission:
        rng = np.random.default_rng(42)
        return Transmission(Signal(rng.standard_normal((self.device.antennas.num_transmit_antennas, 10)), self.sampling_rate, self.device.carrier_frequency))

    @property
    def power(self) -> float:
        return 1.0

    def _transmit(self, _: float) -> Transmission:
        return self.mock_transmission

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
        self.transmitter = TransmitterMock(seed=42, slot=self.device.transmitters)

    def test_init(self) -> None:
        """Initialization parameters should properly be stored as class attributes"""

        self.assertEqual(self.seed, self.transmitter.seed)

    def test_slot_setget(self) -> None:
        """Slot property getter should return setter argument"""

        slot = Mock()
        self.transmitter.slot = slot

        self.assertIs(slot, self.transmitter.slot)

    def test_selected_transmit_ports_validation(self) -> None:
        """Selected transmit ports property setter should raise ValueError on invalid arguments"""

        with self.assertRaises(ValueError):
            self.transmitter.selected_transmit_ports = [11, 12]

    def test_selected_transmit_ports_setget(self) -> None:
        """Selectred transmit ports property getter should return setter argument"""

        self.transmitter.selected_transmit_ports = [0, 1]
        self.assertSequenceEqual([0, 1], self.transmitter.selected_transmit_ports)

        self.transmitter.selected_transmit_ports = None
        self.assertIsNone(self.transmitter.selected_transmit_ports)

    def test_num_transmit_ports(self) -> None:
        """Number of transmit ports property should return the correct number"""

        self.assertEqual(2, self.transmitter.num_transmit_ports)

        self.transmitter.selected_transmit_ports = [0, 1]
        self.assertEqual(2, self.transmitter.num_transmit_ports)

        teached_transmitter = TransmitterMock()
        self.assertEqual(0, teached_transmitter.num_transmit_ports)

        teached_transmitter.selected_transmit_ports = [0, 1, 2]
        self.assertEqual(3, teached_transmitter.num_transmit_ports)

    def test_num_transmit_antennas(self) -> None:
        """Number of transmit antennas property should return the correct number"""

        self.assertEqual(2, self.transmitter.num_transmit_antennas)

        self.transmitter.selected_transmit_ports = [0, 1]
        self.assertEqual(2, self.transmitter.num_transmit_antennas)

        detached_transmitter = TransmitterMock()
        self.assertEqual(0, detached_transmitter.num_transmit_antennas)

    def test_transmission_caching(self) -> None:
        """Transmission caching should be properly handled"""

        transmission = self.transmitter.transmit(0.0)
        self.assertIs(transmission, self.transmitter.transmission)

        expected_transmission = Transmission(Signal(np.random.standard_normal((self.transmitter.device.antennas.num_transmit_antennas, 10)), 1.0))
        self.transmitter.cache_transmission(expected_transmission)
        self.assertIs(expected_transmission, self.transmitter.transmission)

    def test_transmission_recall(self) -> None:
        """Test transmission recall from HDF"""
        ...


class TestTransmitterSlot(TestCase):
    """Test transmitting operator device slot"""

    def setUp(self) -> None:
        self.device = DeviceMock()
        self.slot = TransmitterSlot(device=self.device)
        self.transmitter = TransmitterMock()
        self.slot.add(self.transmitter)

    def test_add_validation(self) -> None:
        """Adding a transmission should raise ValueErrors for invalid arguments"""

        with self.assertRaises(ValueError):
            self.slot.add_transmission(Mock(), Mock())

        with self.assertRaises(ValueError):
            self.slot.add_transmission(self.transmitter, Transmission(Signal(np.random.standard_normal((3, 10)), 1.0)))

    def test_get_transmissions(self) -> None:
        """Getting transmissions should return the proper transmissions"""

        expected_transmissions = Transmission(Signal(np.random.standard_normal((self.device.num_transmit_antennas, 10)), 1.0))
        self.slot.add_transmission(self.transmitter, expected_transmissions)

        self.assertSequenceEqual([expected_transmissions], self.slot.get_transmissions(clear_cache=True))


class TestReceiverSlot(TestCase):
    """Test transmitting operator device slot"""

    def setUp(self) -> None:
        self.device = DeviceMock()
        self.slot = ReceiverSlot(device=self.device)
        self.receiver = ReceiverMock()
        self.slot.add(self.receiver)


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

        expected_transmission = Transmission(Signal(np.random.standard_normal((1, 10)), self.device.sampling_rate, self.device.carrier_frequency))
        transmitter = Mock()
        transmitter.transmit.return_value = expected_transmission
        self.device.transmitters.add(transmitter)

        transmissions = self.device.transmit_operators()
        self.assertSequenceEqual([expected_transmission], transmissions)

    def test_generate_output(self) -> None:
        """Generate output should return the proper output"""

        transmitter = TransmitterMock()
        self.device.transmitters.add(transmitter)
        transmitter.transmit()

        output = self.device.generate_output()
        assert_array_equal(transmitter.mock_transmission.signal.samples, output.mixed_signal.samples)

    def test_transmit(self) -> None:
        """Transmit should return the proper transmission"""

        transmitter = TransmitterMock()
        self.device.transmitters.add(transmitter)

        transmission = self.device.transmit()
        assert_array_equal(transmitter.mock_transmission.signal.samples, transmission.mixed_signal.samples)

    def test_cache_transmission(self) -> None:
        """Cache device transmission should properly cache operator transmissions"""

        expected_operator_transmission = Transmission(Signal(np.random.standard_normal((1, 10)), self.device.sampling_rate, self.device.carrier_frequency))
        expected_device_transmission = DeviceTransmission([expected_operator_transmission], expected_operator_transmission.signal)

        transmitter = Mock()
        self.device.transmitters.add(transmitter)

        self.device.cache_transmission(expected_device_transmission)
        transmitter.cache_transmission.assert_called()

    def test_process_input(self) -> None:
        """Process input should return the proper processed input"""

        impinging_signal = Signal(np.random.standard_normal((1, 10)), self.device.sampling_rate, self.device.carrier_frequency)

        receiver = Mock()
        receiver.selected_receive_ports = [0]
        self.device.receivers.add(receiver)

        processed_input = self.device.process_input(impinging_signal)
        receiver.cache_reception.assert_called()
        assert_array_equal(impinging_signal.samples, processed_input.impinging_signals[0].samples)

        processed_input = self.device.process_input([impinging_signal, impinging_signal])
        assert_array_equal(impinging_signal.samples, processed_input.impinging_signals[0].samples)
        assert_array_equal(impinging_signal.samples, processed_input.impinging_signals[1].samples)

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
        self.receiver.cache_reception(signal)

        operator_receptions = self.device.receive_operators()
        self.assertSequenceEqual([self.receiver.reception], operator_receptions)

        operator_receptions = self.device.receive_operators([signal])
        self.assertSequenceEqual([self.receiver.reception], operator_receptions)

        impinging_signals = [Signal(np.random.standard_normal((2, 10)), self.device.sampling_rate, self.device.carrier_frequency)]
        operator_receptions = self.device.receive_operators(self.device.process_input(impinging_signals))
        self.assertSequenceEqual([self.receiver.reception], operator_receptions)

    def test_receive(self) -> None:
        """Receive should return the proper reception"""

        impinging_signal = Signal(np.random.standard_normal((1, 10)), self.device.sampling_rate, self.device.carrier_frequency)

        receiver = Mock()
        receiver.selected_receive_ports = [0]
        self.device.receivers.add(receiver)

        reception = self.device.receive(impinging_signal)
        assert_array_equal(impinging_signal.samples, reception.impinging_signals[0].samples)
