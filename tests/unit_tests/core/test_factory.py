# -*- coding: utf-8 -*-
"""Test HermesPy serialization factory"""

from __future__ import annotations
from inspect import getmembers
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
from h5py import File
from numpy.testing import assert_array_almost_equal, assert_array_equal

from hermespy.core import Serializable, SerializationProcess, DeserializationProcess
from hermespy.channel import MultipathFadingChannel, IdealChannel
from hermespy.core.factory import Factory, SerializableEnum, HDFDeserializationProcess, HDFSerializationProcess

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


def test_roundtrip_serialization(case: TestCase, serializable: Serializable, property_blacklist: set[str] | None = None) -> None:
    """Test the serialization and deserialization of a serializable class instance.

    Fails the `case` if the serializable properties don't match the deserialization.

    Args:

        case (TestCase): Unit test case on which the testing routine is executed.
        serializable (Serializable): The serializable object to be tested.
        property_blacklist (Set[str], optional): Set of property names to be ignored during testing.

    Raises:

        ValueError: If `serializable` does not inherit from :class:`hermespy.core.factory.Serializable`.
    """

    # Make sure the object to be tested is serializable
    if not issubclass(type(serializable), Serializable):
        raise ValueError("The object under test does not inherit from the Serializable base class")

    _property_blacklist: set[str] = set() if property_blacklist is None else property_blacklist
    serializable_class = serializable.__class__

    # Query serializable properties
    attributes: set[str] = set()
    for attribute_key, attribute_type in getmembers(serializable_class):
        # Prevent the access to protected or private attributes
        if attribute_key.startswith("_"):
            continue

        # Only add attribute if it isn't blacklisted
        if attribute_key in _property_blacklist:
            continue

        # Make sure the attribute is a property
        if not isinstance(attribute_type, property):
            continue

        # Don't serialize if the property isn't settable
        if attribute_type.fset is None:
            continue

        attributes.add(attribute_key)

    # Serialize the serialzable object configuration to text
    factory = Factory()
    file = File("test.h5", "w", driver="core", backing_store=False)
    factory.to_HDF(file, serializable)

    # De-serialize the serializable object configuration from text
    deserialization: Serializable = factory.from_HDF(file, serializable_class)
    file.close()

    # Assert property equality
    for attribute_key in attributes:
        try:
            serialized_value = getattr(serializable, attribute_key)
            serialized_type = type(serialized_value)
            deserialized_value = getattr(deserialization, attribute_key)

        except Exception as e:
            case.fail(f"Roundtrip serialization testing of {serializable.__class__.__name__}.{attribute_key}: {str(e)}")

        # Skip blacklisted properties
        if attribute_key in _property_blacklist:
            continue

        # Both values should have identical type
        case.assertEqual(serialized_type, type(deserialized_value), f"Roundtrip serialization of {serializable.__class__.__name__}.{attribute_key} resulted in wrong type ({type(deserialized_value)} instead of {type(serialized_value)})")

        if serialized_type in (int, str, bool):
            case.assertEqual(serialized_value, deserialized_value, f"Roundtrip serialization of {serializable.__class__.__name__}.{attribute_key}  failed for attribute {attribute_key}")

        elif serialized_type is np.ndarray:
            # Type hinting
            serialized_value: np.ndarray

            # Assert with strict equality for integer arrays
            if serialized_value.dtype is np.dtype(int):
                assert_array_equal(serialized_value, deserialized_value, f"Roundtrip serialization of {serializable.__class__.__name__}.{attribute_key}  failed for attribute {attribute_key}")

            # Assert with approximate equality for floating point arrays
            else:
                assert_array_almost_equal(serialized_value, deserialized_value, decimal=6, err_msg=f"Roundtrip serialization of {serializable.__class__.__name__}.{attribute_key}  failed for attribute {attribute_key}")

        elif serialized_type is float:
            case.assertAlmostEqual(serialized_value, deserialized_value, msg=f"Roundtrip serialization of {serializable.__class__.__name__}  failed for attribute {attribute_key}")


class SerializableMock(Serializable):
    """Mock serializable for testing purposes"""

    standard_attribute: int
    additionl_attribute: int
    blacklisted_attribute: int

    def __init__(self, standard_property: str = "default_standard_value", standard_attribute: int = 0, additional_attribute: int = 5) -> None:
        self.standard_property = standard_property
        self.standard_attribute = standard_attribute
        self.additional_attribute = additional_attribute

    @property
    def standard_property(self) -> str:
        return self.__standard_property

    @standard_property.setter
    def standard_property(self, value: str) -> None:
        self.__standard_property = value

    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_string(self.standard_property, "standard_property")
        process.serialize_integer(self.standard_attribute, "standard_attribute")
        process.serialize_integer(self.additional_attribute, "additional_attribute")

    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> SerializableMock:
        standard_property = process.deserialize_string("standard_property")
        standard_attribute = process.deserialize_integer("standard_attribute")
        additional_attribute = process.deserialize_integer("additional_attribute")
        return cls(standard_property, standard_attribute, additional_attribute)


class SerializableEnumMock(SerializableEnum):
    """Mock serializable enumeration for testing purposes"""
    A = 1
    B = 2


class TestSerializableEnum(TestCase):
    """Test the serializable enum class"""

    def test_from_parameters_validation(self) -> None:
        """Test the validation of the from_parameters method"""

        with self.assertRaises(ValueError):
            SerializableEnumMock.from_parameters(Mock())

    def test_from_parameters(self) -> None:
        """Test the from_parameters method"""

        self.assertEqual(SerializableEnumMock.from_parameters(1), SerializableEnumMock.A)
        self.assertEqual(SerializableEnumMock.from_parameters("B"), SerializableEnumMock.B)
        self.assertEqual(SerializableEnumMock.from_parameters(SerializableEnumMock.A), SerializableEnumMock.A)

    def test_serialization(self) -> None:
        """Test serialization of the serializable enum"""

        test_roundtrip_serialization(self, SerializableEnumMock.A)


class TestFactory(TestCase):
    """Test the factory responsible to convert config files to executable simulations"""

    def setUp(self) -> None:
        self.factory = Factory()

    def test_registered_classes(self) -> None:
        """Registered classes should contain all serializable classes"""

        expected_classes = [IdealChannel, MultipathFadingChannel]
        registered_classes = self.factory.registered_classes

        for expected_class in expected_classes:
            self.assertTrue(expected_class in registered_classes)

    def test_registered_tags(self) -> None:
        """Test the serializable classes registration / discovery mechanism"""

        expected_tags = ["hermespy.channel.ideal.IdealChannel", "hermespy.channel.fading.fading.MultipathFadingChannel"]
        registered_tags = self.factory.registered_tags

        for expected_tag in expected_tags:
            self.assertTrue(expected_tag in registered_tags)

    def test_tag_registry(self) -> None:
        """Test the tag registry"""

        tag_registy = self.factory.tag_registry
        self.assertIs(tag_registy["hermespy.channel.ideal.IdealChannel"], IdealChannel)

    def test_HDF_serialization(self) -> None:
        """Test the HDF serialization routines"""

        memory_file = File("test.h5", "w", driver="core", backing_store=False)

        serialized_mock = SerializableMock()
        self.factory.to_HDF(memory_file, serialized_mock)
        deserialized_mock: SerializableMock = self.factory.from_HDF(memory_file, SerializableMock)

        memory_file.close()

        self.assertEqual(serialized_mock.standard_attribute, deserialized_mock.standard_attribute)
        self.assertEqual(serialized_mock.standard_property, deserialized_mock.standard_property)


class TestSerializationProcess(object):

    serialization: SerializationProcess
    deserialization: DeserializationProcess

    def test_range_serialization(self) -> None:
        """Test serialization of ranges"""

        scalar_value = 5.0
        range_value = (1.0, 3.0)

        self.serialization.serialize_range(scalar_value, "scalar_range")
        self.serialization.serialize_range(range_value, "range")
        self.serialization.serialize_range(None, "missing_value")

        with self.assertRaises(ValueError):
            self.serialization.serialize_range(Mock(), "wrong_type")

        deserialize_scalar_value = self.deserialization.deserialize_range("scalar_range")
        deserialize_range_value = self.deserialization.deserialize_range("range")
        deserialized_default_value = self.deserialization.deserialize_range("missing_value", 4.0)
        self.assertEqual(scalar_value, deserialize_scalar_value)
        self.assertSequenceEqual(deserialize_range_value, range_value)
        self.assertEqual(deserialized_default_value, 4.0)

        with self.assertRaises(RuntimeError):
            self.deserialization.deserialize_range("missing_value")


class TestHDFSerializationProcess(TestSerializationProcess, TestCase):
    """Test the HDF serialization process"""
    
    def setUp(self):    
        self.file = File("test.h5", "w", driver="core", backing_store=False)
        tag_registry = {'unit_tests.core.test_factory.SerializableMock': SerializableMock}
        self.serialization = HDFSerializationProcess.New(tag_registry, self.file)
        self.deserialization = HDFDeserializationProcess.New(tag_registry, self.file)
    
    def tearDown(self):
        self.file.close()

    def test_serialize_object_sequence(self) -> None:
        """Test the serialization of object sequences"""
    
        expected_objects = [SerializableMock() for _ in range(5)]
        self.serialization.serialize_object_sequence(expected_objects, "objects")

        # Test full deserilization
        deserialized_objects = self.deserialization.deserialize_object_sequence("objects", SerializableMock)        
        for serialized, deserialized_sequence in zip(expected_objects, deserialized_objects):
            self.assertEqual(serialized.standard_attribute, deserialized_sequence.standard_attribute)
            self.assertEqual(serialized.standard_property, deserialized_sequence.standard_property)

        # Test fetching a specific index
        for index in range(5):
            deserialized_sequence = self.deserialization.deserialize_object_sequence("objects", SerializableMock, index, index + 1)
            self.assertEqual(len(deserialized_sequence), 1)
            
            deserialized_object = deserialized_sequence[0]
            self.assertEqual(expected_objects[index].standard_attribute, deserialized_object.standard_attribute)
            self.assertEqual(expected_objects[index].standard_property, deserialized_object.standard_property)
