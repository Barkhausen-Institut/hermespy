# -*- coding: utf-8 -*-
"""Test HermesPy device module."""

from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from hermespy.core.device import Device, PhysicalDevice, SimulatedDevice, Operator, OperatorSlot

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.3"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestOperator(TestCase):
    """Test device slot operators."""

    def setUp(self) -> None:

        self.device = Device()
        self.slot = OperatorSlot(device=self.device)
        self.operator = Operator(slot=self.slot)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertIs(self.slot, self.operator.slot)

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

        self.operator.slot = None
        self.assertEqual(None, self.operator.slot_index)
        self.assertEqual(0, self.slot.num_operators)

        self.slot.add(Operator())
        self.operator.slot = self.slot
        self.slot.add(Operator())
        self.assertEqual(self.operator.slot_index, 1)

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


class TestOperatorSlot(TestCase):
    """Test device operator slots."""

    def setUp(self) -> None:

        self.device = Device()
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

    def test_add_validation(self) -> None:
        """Adding an already existing operator should raise a RuntimeError."""

        operator = Mock()
        self.slot.add(operator)

        with self.assertRaises(RuntimeError):
            self.slot.add(operator)

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


class TestDevice(TestCase):
    """Test device base class."""

    def setUp(self) -> None:

        self.position = np.zeros(3)
        self.orientation = np.zeros(3)
        self.topology = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=float)

        self.device = Device(position=self.position, orientation=self.orientation,
                             topology=self.topology)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        assert_array_equal(self.position, self.device.position)
        assert_array_equal(self.orientation, self.device.orientation)
        assert_array_equal(self.topology, self.device.topology)

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

    def test_topology_setget(self) -> None:
        """Device topology property getter should return setter argument."""

        topology = np.arange(9).reshape((3, 3))
        self.device.topology = topology

        assert_array_equal(topology, self.device.topology)

    def test_topology_set_expansion(self) -> None:
        """Topology property setter automatically expands input dimensions."""

        topology = np.arange(3)
        expected_topology = np.zeros((3, 3), dtype=float)
        expected_topology[:, 0] = topology

        self.device.topology = topology
        assert_array_equal(expected_topology, self.device.topology)

    def test_topology_validation(self) -> None:
        """Topology property setter should raise ValueErrors on invalid arguments."""

        with self.assertRaises(ValueError):
            self.device.topology = np.empty(0)

        with self.assertRaises(ValueError):
            self.device.topology = np.array([[1, 2, 3, 4]])


class TestPhysicalDevice(TestCase):
    """Test the physical device base class."""

    def setUp(self) -> None:

        self.device = PhysicalDevice()


class TestSimulatedDevice(TestCase):
    """Test the simulated device base class."""

    def setUp(self) -> None:

        self.scenario = Mock()
        self.position = np.zeros(3)
        self.orientation = np.zeros(3)
        self.topology = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=float)

        self.device = SimulatedDevice(scenario=self.scenario, position=self.position, orientation=self.orientation,
                                      topology=self.topology)

    def test_init(self) -> None:
        """Initialization parameters should be properly stored as class attributes."""

        self.assertIs(self.scenario, self.device.scenario)
        assert_array_equal(self.position, self.device.position)
        assert_array_equal(self.orientation, self.device.orientation)
        assert_array_equal(self.topology, self.device.topology)

    def test_scenario_setget(self) -> None:
        """Scenario property setter should return getter argument."""

        self.device = SimulatedDevice()
        self.device.scenario = self.scenario

        self.assertIs(self.scenario, self.device.scenario)

    def test_scenario_set_validation(self) -> None:
        """Overwriting a scenario property should raise a RuntimeError."""

        with self.assertRaises(RuntimeError):
            self.device.scenario = Mock()

    def test_attached(self) -> None:
        """The attached property should return the proper device attachment state."""

        self.assertTrue(self.device.attached)
        self.assertFalse(SimulatedDevice().attached)

    def test_max_frame_duration(self) -> None:
        """Maximum frame duration property should compute the correct duration."""

        transmitter = Mock()
        transmitter.frame_duration = 10
        self.device.transmitters.add(transmitter)

        receiver = Mock()
        receiver.frame_duration = 4
        self.device.receivers.add(receiver)

        self.assertEqual(10, self.device.max_frame_duration)
