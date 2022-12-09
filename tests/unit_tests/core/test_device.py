# -*- coding: utf-8 -*-
"""Test HermesPy device module."""

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from hermespy.core.device import Device, MixingOperator,  Operator, OperatorSlot, Receiver, ReceiverSlot, Transmitter,\
    TransmitterSlot

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DeviceMock(Device):
    """Mock of the device base class."""
    
    def __init__(self, *args, **kwargs) -> None:

        self.__carrier_frequency = 1e9
        Device.__init__(self, *args, **kwargs)

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


class TestOperator(TestCase):
    """Test device slot operators."""

    def setUp(self) -> None:

        self.device = DeviceMock()
        self.slot = OperatorSlot(device=self.device)
        self.operator = Operator()
        self.slot.add(self.operator)

    def test_slot_setget(self) -> None:
        """Slot property getter should return setter argument."""

        slot = OperatorSlot(self.device)
        self.operator.slot = slot
        self.assertIs(slot, self.operator.slot)

        self.operator.slot = None
        self.assertEqual(None, self.operator.slot)

    def test_slot_set(self) -> None:
        """Setting an operator slot should result in the operator being registered at the slot."""

        slot = OperatorSlot(self.device)
        self.operator.slot = slot

        self.assertTrue(slot.registered(self.operator))

    def test_slot_index(self) -> None:
        """The slot index property should return the proper slot index."""

        self.assertEqual(0, self.operator.slot_index)
        self.assertEqual(1, self.slot.num_operators)

        self.slot.add(Operator())
        self.slot.add(Operator())
        
        new_operator = Operator()
        self.slot.add(new_operator)
        self.assertEqual(3, new_operator.slot_index)

    def test_device(self) -> None:
        """The device property should return the proper handle."""

        self.assertIs(self.device, self.operator.device)

        self.operator.slot = None
        self.assertEqual(None, self.operator.device)

    def test_attached(self) -> None:
        """Attached property should return correct attachment status."""

        self.assertTrue(self.operator.attached)

        self.operator.slot = None
        self.assertFalse(self.operator.attached)


class MixingOperatorMock(MixingOperator):
    """Mock of the mixing operator base class."""

    def __init__(self, *args, **kwargs) -> None:

        MixingOperator.__init__(self, *args, **kwargs)

    @property
    def frame_duration(self) -> float:

        return 1.0

    @property
    def sampling_rate(self) -> float:

        return 1.0


class TestMixingOperator(TestCase):
    """Test the base class for mixing operators."""

    def setUp(self) -> None:

        self.slot = Mock()
        self.slot.device = Mock()
        self.mixing_operator = MixingOperatorMock(slot=self.slot)

    def test_carrier_frequency_setget(self) -> None:
        """Carrier frequency property getter should return setter argument."""

        carrier_frequency = 2
        self.mixing_operator.carrier_frequency = carrier_frequency

        self.assertEqual(carrier_frequency, self.mixing_operator.carrier_frequency)

    def test_carrier_frequency_device_get(self) -> None:
        """Carrier frequency property should return the device's carrier frequency by default."""

        carrier_frequency = 10
        self.slot.device.carrier_frequency = carrier_frequency
        self.mixing_operator.carrier_frequency = None

        self.assertEqual(carrier_frequency, self.mixing_operator.carrier_frequency)

    def test_carrier_frequency_validation(self) -> None:
        """Carrier frequency property setter should raise ValueError on negative arguments."""

        with self.assertRaises(ValueError):
            self.mixing_operator.carrier_frequency = -1.

        try:
            self.mixing_operator.carrier_frequency = 0.

        except ValueError:
            self.fail()


class ReceiverMock(Receiver):
    """Mock of the receiving device operator base class."""

    def __init__(self, *args, **kwargs):

        Receiver.__init__(self, *args, **kwargs)

    def receive(self) -> None:

        pass

    @property
    def sampling_rate(self) -> float:

        return 1.0

    @property
    def energy(self) -> float:

        return 1.0


class TestReceiver(TestCase):
    """Test the base class for receiving operators."""

    def setUp(self) -> None:

        self.device = Mock()
        self.slot = ReceiverSlot(device=self.device)
        self.receiver = ReceiverMock(slot=self.slot)

    def test_slot_setget(self) -> None:
        """Operator slot getter should return setter argument."""

        slot = Mock()
        self.receiver.slot = slot

        self.assertIs(slot, self.receiver.slot)

    def test_reference_transmitter_setget(self) -> None:
        """Reference transmitter property getter should return setter argument."""

        reference = Mock()
        self.receiver.reference_transmitter = reference

        self.assertIs(reference, self.receiver.reference_transmitter)

    def test_cache_reception(self) -> None:
        """Cached receptions should be returned by the signal and csi properties."""

        signal = Mock()
        csi = Mock()
        self.receiver.cache_reception(signal, csi)

        self.assertIs(signal, self.receiver.signal)
        self.assertIs(csi, self.receiver.csi)


class TestOperatorSlot(TestCase):
    """Test device operator slots."""

    def setUp(self) -> None:

        self.device = DeviceMock()
        self.slot = OperatorSlot(device=self.device)
        self.operator = Operator(slot=self.slot)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertIs(self.device, self.slot.device)

    def test_operator_index(self) -> None:
        """Operator index lookup should return the proper operator index."""

        operator = Mock()
        self.slot.add(operator)
        self.slot.add(Mock())

        self.assertEqual(1, self.slot.operator_index(operator))

    def test_add(self) -> None:
        """Adding an operator should register said operator at the slot."""

        operator = Mock()
        self.slot.add(operator)

        self.assertTrue(self.slot.registered(operator))
        self.assertIs(operator.slot, self.slot)

    def test_remove(self) -> None:
        """Operators should be properly removed from the list."""

        self.slot.add(Mock())
        self.slot.remove(self.operator)

        self.assertFalse(self.slot.registered(self.operator))
        self.assertEqual(None, self.operator.slot)

    def test_registered(self) -> None:
        """Registered check should return the proper registration state for slots."""

        registered_slot = Mock()
        unregistered_slot = Mock()
        self.slot.add(registered_slot)

        self.assertTrue(self.slot.registered(registered_slot))
        self.assertFalse(self.slot.registered(unregistered_slot))

    def test_num_operators(self):
        """Number of operators property should compute the correct number of registered operators."""

        self.slot.add(Mock())
        self.assertEqual(2, self.slot.num_operators)

        self.slot.add(Mock())
        self.assertEqual(3, self.slot.num_operators)

    def test_iteration(self) -> None:
        """Iteration should yield the proper order of operators."""

        operator = Mock()
        self.slot.add(operator)

        expected_iterator_elements = [self.operator, operator]

        for element, expected_element in zip(self.slot.__iter__(), expected_iterator_elements):
            self.assertIs(expected_element, element)

    def test_contains(self) -> None:
        """Contains should return the proper registration state for operators."""

        registered_operator = Mock()
        unregistered_operator = Mock()
        self.slot.add(registered_operator)

        self.assertTrue(registered_operator in self.slot)
        self.assertFalse(unregistered_operator in self.slot)


class TransmitterMock(Transmitter):
    """Mock of the base class for transmitting operators."""

    def __init__(self, *args, **kwargs) -> None:

        Transmitter.__init__(self, *args, **kwargs)

    def transmit(self) -> None:
        pass

    @property
    def sampling_rate(self) -> float:
        return 1.0


class TestTransmitter(TestCase):
    """Test transmitting operator base class."""

    def setUp(self) -> None:

        self.slot = Mock()
        self.transmitter = TransmitterMock(slot=self.slot)

    def test_slot_setget(self) -> None:
        """Slot property getter should return setter argument."""

        slot = Mock()
        self.transmitter.slot = slot

        self.assertIs(slot, self.transmitter.slot)


class TestTransmitterSlot(TestCase):
    """Test transmitting operator device slot."""

    def setUp(self) -> None:

        self.device = Mock()
        self.slot = TransmitterSlot(device=self.device)


class TestReceiverSlot(TestCase):
    """Test transmitting operator device slot."""

    def setUp(self) -> None:

        self.device = Mock()
        self.slot = TransmitterSlot(device=self.device)


class TestDevice(TestCase):
    """Test device base class."""

    def setUp(self) -> None:

        self.power = 1.5
        self.position = np.zeros(3)
        self.orientation = np.zeros(3)
        self.antennas = Mock()

        self.device = DeviceMock(power=self.power, position=self.position, orientation=self.orientation,
                                 antennas=self.antennas)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertIs(self.antennas, self.device.antennas)
        self.assertEqual(self.power, self.device.power)
        assert_array_equal(self.position, self.device.position)
        assert_array_equal(self.orientation, self.device.orientation)

    def test_power_setget(self) -> None:
        """Power property getter should return setter argument."""

        power = 1.23
        self.device.power = power

        self.assertEqual(power, self.device.power)

    def test_power_validation(self) -> None:
        """Power property setter should raise ValueError on negative arguments."""

        with self.assertRaises(ValueError):
            self.device.power = -1.

        try:
            self.device.power = 0.

        except ValueError:
            self.fail()

    def test_position_setget(self) -> None:
        """Position property getter should return setter argument."""

        position = np.arange(3)
        self.device.position = position

        assert_array_equal(position, self.device.position)

    def test_position_validation(self) -> None:
        """Position property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.device.position = np.arange(4)

        with self.assertRaises(ValueError):
            self.device.position = np.array([[1, 2, 3]])

        try:
            self.device.position = np.arange(1)

        except ValueError:
            self.fail()

    def test_position_expansion(self) -> None:
        """Position property setter should expand vector dimensions if required."""

        position = np.array([1.0])
        expected_position = np.array([1.0, 0.0, 0.0])
        self.device.position = position

        assert_array_almost_equal(expected_position, self.device.position)

    def test_orientation_setget(self) -> None:
        """Device orientation property getter should return setter argument."""

        orientation = np.arange(3)
        self.device.orientation = orientation

        assert_array_equal(orientation, self.device.orientation)

    def test_orientation_validation(self) -> None:
        """Device orientation property setter should raise ValueError on invalid arguments."""

        with self.assertRaises(ValueError):
            self.device.orientation = np.array([[1, 2, 3], [4, 5, 6]])

        with self.assertRaises(ValueError):
            self.device.orientation = np.array([1, 2])

    def test_max_frame_duration(self) -> None:
        """Maximum frame duration property should compute the correct duration."""

        transmitter = Mock()
        transmitter.frame_duration = 10
        self.device.transmitters.add(transmitter)

        receiver = Mock()
        receiver.frame_duration = 4
        self.device.receivers.add(receiver)

        self.assertEqual(10, self.device.max_frame_duration)
        
    def test_wavelength_validation(self) -> None:
        """Device wavelength property setter should raise ValueError on invalid arguments."""
        
        with self.assertRaises(ValueError):
            self.device.wavelength = -1.
            
        with self.assertRaises(ValueError):
            self.device.wavelength = 0.
            
    def test_wavelength_setget(self) -> None:
        """Wavelength property getter should return setter argument"""
        
        expected_wavelength = 1.234
        self.device.wavelength = expected_wavelength
        
        self.assertAlmostEqual(expected_wavelength, self.device.wavelength)
