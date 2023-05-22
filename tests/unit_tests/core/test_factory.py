# -*- coding: utf-8 -*-
"""Test HermesPy serialization factory"""

from collections.abc import Sequence
from typing import Optional, List, Set
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from hermespy.channel import Channel, MultipathFadingChannel, IdealChannel
from hermespy.core.factory import Factory, Serializable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


def test_yaml_roundtrip_serialization(case: TestCase, 
                                      serializable: Serializable,
                                      property_blacklist: Optional[Set[str]] = None) -> None:
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
    
    # Generate the property blacklist for class serialization
    if property_blacklist:
        property_blacklist.update(serializable.property_blacklist)
        
    else:
        property_blacklist = serializable.property_blacklist

    # Query serializable properties
    attributes = serializable._serializable_attributes()

    # Serialize the serialzable object configuration to text
    factory = Factory()
    serialization = factory.to_str(serializable)

    # De-serialize the serializable object configuration from text
    deserialization_list = factory.from_str(serialization)

    if isinstance(deserialization_list, Sequence) and not isinstance(serializable, Sequence):
    
        if len(deserialization_list) != 1:
            case.fail(f"Deserialization of {serializable.__name__} resulted in incorrect amount of objects ({len(deserialization_list)})")

        deserialization = deserialization_list[0]
        
    else:
        deserialization = deserialization_list

    # Assert property equality
    for attribute_key in attributes:
        
        try:

            serialized_value = getattr(serializable, attribute_key)
            serialized_type = type(serialized_value)
            deserialized_value = getattr(deserialization, attribute_key)
            
        except Exception as e:
            case.fail(f"Roundtrip serialization testing of {serializable.__class__.__name__}.{attribute_key}: {str(e)}")

        # Both values should have identical type
        case.assertEqual(serialized_type, type(deserialized_value),
                         f"Roundtrip serialization of {serializable.__class__.__name__}.{attribute_key} resulted in wrong type ({type(deserialized_value)} instead of {type(serialized_value)})")

        if serialized_type in (int, str, bool):
            case.assertEqual(serialized_value, deserialized_value,
                             f"Roundtrip serialization of {serializable.__class__.__name__}.{attribute_key}  failed for attribute {attribute_key}")

        elif serialized_type is np.ndarray:
            
            # Type hinting
            serialized_value: np.ndarray
            
            # Assert with strict equality for integer arrays
            if serialized_value.dtype is np.dtype(int):
                assert_array_equal(serialized_value, deserialized_value,
                                   f"Roundtrip serialization of {serializable.__class__.__name__}.{attribute_key}  failed for attribute {attribute_key}")

            # Assert with approximate equality for floating point arrays
            else:
                assert_array_almost_equal(serialized_value, deserialized_value, decimal=6,
                                          err_msg=f"Roundtrip serialization of {serializable.__class__.__name__}.{attribute_key}  failed for attribute {attribute_key}")

        elif serialized_type is float:
            case.assertAlmostEqual(serialized_value, deserialized_value,
                                   f"Roundtrip serialization of {serializable.__class__.__name__}.{attribute_key}  failed for attribute {attribute_key}")


class SerializableMock(Serializable):
    """Mock serializable for testing purposes"""

    yaml_tag = 'SerializableMock'
    property_blacklist = {'blacklisted_property', 'blacklisted_attribute'}
    serialized_attributes = {'standard_attribute'}
    
    standard_attribute: int
    blacklisted_attribute: int
    
    def __init__(self,
                 standard_property: str,
                 standard_attribute: int) -> None:
        
        self.standard_property = standard_property
        self.standard_attribute = standard_attribute
        
        self.blacklisted_property = 'blacklisted_value'
        self.blacklisted_attribute = 0
        
    @property
    def standard_property(self) -> str:
        return self.__standard_property
    
    @standard_property.setter
    def standard_property(self, value: str) -> None:
        self.__standard_property = value
        
    @property
    def blacklisted_property(self) -> str:
        return self.__blacklisted_property
    
    @blacklisted_property.setter
    def blacklisted_property(self, value: str) -> None:
        self.__blacklisted_property = value
    
    
class TestSerializable(TestCase):
    
    def setUp(self) -> None:
        
        self.serializable = SerializableMock('properta_value', 1)
    
    def test_serializable_attributes(self) -> None:
        """Subroutine for serializable attribute collection should detect correct attributes"""

        expected_serializable_attributes = {'standard_property', 'standard_attribute'}
        serializable_attributes = self.serializable._serializable_attributes()

        self.assertCountEqual(expected_serializable_attributes, serializable_attributes)


class TestFactory(TestCase):
    """Test the factory responsible to convert config files to executable simulations"""

    def setUp(self) -> None:

        self.factory = Factory()

    def test_clean_set_get(self) -> None:
        """Test that the clean getter returns the setter argument"""

        self.factory.clean = True
        self.assertEqual(self.factory.clean, True, "Clean set/get produced unexpected result")

        self.factory.clean = False
        self.assertEqual(self.factory.clean, False, "Clean set/get produced unexpected result")

    def test_registered_classes(self) -> None:
        """Registered classes should contain all serializable classes"""

        expected_classes = [IdealChannel, MultipathFadingChannel]
        registered_classes = self.factory.registered_classes

        for expected_class in expected_classes:
            self.assertTrue(expected_class in registered_classes)

    def test_registered_tags(self) -> None:
        """Test the serializable classes registration / discovery mechanism"""

        expected_tags = [u'Channel', u'MultipathFading']
        registered_tags = self.factory.registered_tags

        for expected_tag in expected_tags:
            self.assertTrue(expected_tag in registered_tags)

    def test_complex_serialization(self) -> None:
        """Test serialization of complex numbers"""
        
        expected_number = 1 + 2j
        
        serialized_number = self.factory.to_str(expected_number)
        deserialized_number = self.factory.from_str(serialized_number)
        
        self.assertEqual(expected_number, deserialized_number)
        
    def test_array_serialization(self) -> None:
        """Test serialization of numpy arrays"""
        
        array_candidates = [np.random.normal(size=(2, 3)), np.arange(10)]
        for expected_array  in array_candidates:
            
            serialized_array = self.factory.to_str(expected_array)
            deserialized_array = self.factory.from_str(serialized_array)
            
            assert_array_equal(expected_array, deserialized_array)

    def test_complex_array_serialization(self) -> None:
        """Test serialization of complex numpy arrays"""
        
        expected_array = np.random.normal(size=(2, 3)) + 1j * np.random.normal(size=(2, 3))
        expected_array[0] = 0
        
        serialized_array = self.factory.to_str(expected_array)
        deserialized_array = self.factory.from_str(serialized_array)
        
        assert_array_equal(expected_array, deserialized_array)
