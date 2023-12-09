# -*- coding: utf-8 -*-
"""Test HermesPy serialization factory"""

from __future__ import annotations
from collections.abc import Sequence
from os.path import join
from tempfile import TemporaryDirectory
from typing import Optional, Set
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ruamel.yaml import ScalarNode, SafeConstructor, SafeRepresenter
from ruamel.yaml.constructor import ConstructorError

from hermespy.core import Logarithmic, LogarithmicSequence
from hermespy.channel import MultipathFadingChannel, IdealChannel
from hermespy.core.factory import Factory, Serializable, SerializableEnum

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


def test_yaml_roundtrip_serialization(case: TestCase, serializable: Serializable, property_blacklist: Optional[Set[str]] = None) -> None:
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
            case.assertAlmostEqual(serialized_value, deserialized_value, f"Roundtrip serialization of {serializable.__class__.__name__}.{attribute_key}  failed for attribute {attribute_key}")


class SerializableMock(Serializable):
    """Mock serializable for testing purposes"""

    yaml_tag = "SerializableMock"
    property_blacklist = {"blacklisted_property", "blacklisted_attribute"}
    serialized_attributes = {"standard_attribute", "nonsettable_property"}

    standard_attribute: int
    additionl_attribute: int
    blacklisted_attribute: int

    def __init__(self, standard_property: str = "default_standard_value", standard_attribute: int = 0, additional_attribute: int = 5) -> None:
        self.standard_property = standard_property
        self.standard_attribute = standard_attribute
        self.additional_attribute = additional_attribute

        self.blacklisted_property = "blacklisted_value"
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

    @property
    def nonsettable_property(self) -> str:
        return "nonsettable_value"


class TestSerializable(TestCase):
    def setUp(self) -> None:
        self.serializable = SerializableMock("properta_value", 1)

    def test_serializable_attributes(self) -> None:
        """Subroutine for serializable attribute collection should detect correct attributes"""

        expected_serializable_attributes = {"nonsettable_property", "standard_property", "standard_attribute"}
        serializable_attributes = self.serializable._serializable_attributes()

        self.assertCountEqual(expected_serializable_attributes, serializable_attributes)

        property_blacklist = {"standard_property"}
        expected_attributes_with_blacklist = {"standard_attribute", "nonsettable_property"}

        self.assertCountEqual(expected_attributes_with_blacklist, self.serializable._serializable_attributes(property_blacklist))

    def test_mapping_serialization_wrapper(self) -> None:
        """Test the mapping serialization wrapper"""

        representer_mock = Mock(spec=SafeRepresenter)
        additional_fields = {"additional_field": "additional_value"}

        _ = self.serializable._mapping_serialization_wrapper(representer_mock, additional_fields=additional_fields)
        call_args = representer_mock.represent_mapping.call_args[0][1]

        self.assertEqual(call_args["additional_field"], "additional_value")

    def test_from_scalar_node(self) -> None:
        """Test the deserialization of scalar nodes"""

        constructor_mock = Mock(spec=SafeConstructor)
        node = Mock(spec=ScalarNode)
        deserialized_value = self.serializable.from_yaml(constructor_mock, node)

        self.assertEqual(deserialized_value.standard_property, "default_standard_value")

    def test_initialization_wrapper_validation(self) -> None:
        """Initialization wrapper should raise errors on initialization exceptions"""

        with self.assertRaises(TypeError):
            SerializableMock.InitializationWrapper({"nonexisting_property": "some_value"})

        with self.assertRaises(AttributeError):
            SerializableMock.InitializationWrapper({"nonsettable_property": "some_value"})


class SerializableEnumMock(SerializableEnum):
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

    def test_from_yaml(self) -> None:
        """Test deserialization from YAML"""

        constructor = Mock()
        node = Mock()
        node.value = "A"

        deserialization = SerializableEnumMock.from_yaml(constructor, node)

        self.assertEqual(deserialization, SerializableEnumMock.A)

    def test_to_yaml(self) -> None:
        """Test serialization to YAML"""

        representer = Mock()
        SerializableEnumMock.to_yaml(representer, SerializableEnumMock.A)

        representer.represent_scalar.assert_called_once_with(SerializableEnumMock.yaml_tag, "A")


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

        expected_tags = ["Channel", "MultipathFading"]
        registered_tags = self.factory.registered_tags

        for expected_tag in expected_tags:
            self.assertTrue(expected_tag in registered_tags)

    def test_tag_registry(self) -> None:
        """Test the tag registry"""

        tag_registy = self.factory.tag_registry
        self.assertIs(tag_registy["Channel"], IdealChannel)

    def test_complex_serialization(self) -> None:
        """Test serialization of complex numbers"""

        expected_number = 1 + 2j

        serialized_number = self.factory.to_str(expected_number)
        deserialized_number = self.factory.from_str(serialized_number)

        self.assertEqual(expected_number, deserialized_number)

    def test_array_serialization(self) -> None:
        """Test serialization of numpy arrays"""

        array_candidates = [np.random.normal(size=(2, 3)), np.arange(10)]
        for expected_array in array_candidates:
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

    def test_numpy_float_serialization(self) -> None:
        """Test serialization of numpy floats"""

        expected_float = np.float_(1.0)

        serialized_float = self.factory.to_str(expected_float)
        deserialized_float = self.factory.from_str(serialized_float)

        self.assertEqual(expected_float, deserialized_float)

    def test_logarithmic_serialization(self) -> None:
        """Test serialization of logarithmic values"""

        expected_logarithmic = Logarithmic(10)
        expected_logarithmic_sequence = LogarithmicSequence((10, 20, 30))

        deserialized_logarithmic = self.factory.from_str("!<dB> 10")
        deserialized_logarithmic_sequence = self.factory.from_str("!<dB> [10, 20, 30]")

        self.assertEqual(expected_logarithmic, deserialized_logarithmic)
        assert_array_equal(expected_logarithmic_sequence, deserialized_logarithmic_sequence)

    def test_decibel_conversion(self) -> None:
        """Test the decibel conversion routine"""

        deserialized_db = self.factory.from_str("[1, 2, ..., 5] dB")
        assert_array_equal(LogarithmicSequence((1, 2, 3, 4, 5)), deserialized_db)

    def test_from_path_validation(self) -> None:
        """Test the validation of the from_path method"""

        with self.assertRaises(ValueError):
            self.factory.from_path("nonexisting_path")

    def test_from_path(self) -> None:
        """Test the from_path method"""

        with TemporaryDirectory() as temp_dir:
            file_location = join(temp_dir, "test.yml")

            with open(file_location, "w") as f:
                f.write("1")

            single_objects = self.factory.from_path(temp_dir)

            with open(file_location, "w") as f:
                f.write("[1, 2, 3]")

            multiple_objects = self.factory.from_path(file_location)

        self.assertCountEqual(single_objects, [1])
        self.assertCountEqual(multiple_objects, [1, 2, 3])

    def test_from_folder_validation(self) -> None:
        """Test the validation of the from_folder method"""

        with self.assertRaises(ValueError):
            _ = self.factory.from_folder("nonexisting_path")

        with TemporaryDirectory() as temp_dir:
            file_location = join(temp_dir, "test.yml")

            with open(file_location, "w") as f:
                f.write("1")

            with self.assertRaises(ValueError):
                _ = self.factory.from_folder(file_location)

    def test_from_folder(self) -> None:
        """Test factory deserialization from folder"""

        with TemporaryDirectory() as temp_dir:
            file_location = join(temp_dir, "test.yml")

            with open(file_location, "w") as f:
                f.write("1")

            single_objects = self.factory.from_folder(temp_dir, recurse=False)

        self.assertCountEqual(single_objects, [1])

    def test_from_file_error_handling(self) -> None:
        """Construct errors should be transformed by deserialization routine"""

        def construct_error(*args):
            raise ConstructorError(problem_mark=Mock())

        with patch("hermespy.core.factory.Factory.from_stream") as from_stream_mock:
            from_stream_mock.side_effect = construct_error

            with TemporaryDirectory() as temp_dir:
                file_location = join(temp_dir, "test.yml")

                with open(file_location, "w") as f:
                    f.write("1")

                with self.assertRaises(ConstructorError):
                    _ = self.factory.from_file(file_location)

    def test_clean_from_file(self) -> None:
        """Test the clean flag of the from_file method"""

        self.factory.clean = False

        with TemporaryDirectory() as temp_dir:
            file_location = join(temp_dir, "test.yml")

            with open(file_location, "w") as f:
                f.write("1")

            single_object = self.factory.from_file(file_location)

        self.assertEqual(1, single_object)

    def test_empty_from_file(self) -> None:
        """Test the from_file method with empty file"""

        with TemporaryDirectory() as temp_dir:
            file_location = join(temp_dir, "test.yml")

            with open(file_location, "w") as f:
                f.write("")

            no_objects = self.factory.from_file(file_location)

        self.assertEqual([], no_objects)

    def test_range_restore_callback(self) -> None:
        """Test the range restore callback"""

        recalled_range = self.factory.from_str("[1, 2, ..., 10]")
        self.assertSequenceEqual([i for i in range(1, 11)], recalled_range)
