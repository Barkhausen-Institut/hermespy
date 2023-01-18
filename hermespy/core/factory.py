# -*- coding: utf-8 -*-
"""
=====================
Serialization Factory
=====================

This module implements the main interface for loading / dumping HermesPy configurations from / to `YAML`_ files.
Every mutable object that is expected to have its state represented as a text-section within configuration files
must inherit from the :class:`.Serializable` base class.

All :class:`.Serializable` classes within the `hermespy` namespace are detected automatically by the :class:`.Factory`
managing the serialization process.
As a result, dumping any :class:`.Serializable` object state to a `.yml` text file is as easy as

.. code-block:: python

   factory = Factory()
   factory.to_file("dump.yml", serializable)

and can be loaded again just as easily via

.. code-block::  python

        factory = Factory()
        serializable = factory.from_file("dump.yml")

from any context.


.. _YAML: https://yaml.org/
"""

from __future__ import annotations

import re
from abc import ABCMeta, abstractclassmethod, abstractmethod
from collections.abc import Iterable
from enum import Enum
from functools import partial
from inspect import getmembers, isclass, signature
from importlib import import_module
from io import TextIOBase, StringIO
import os
from pkgutil import iter_modules
from re import compile, Pattern, Match
from typing import Any, Dict, Set, Sequence, Mapping, Union, List, Optional, Tuple, Type

import numpy as np
from h5py import Group
from ruamel.yaml import YAML, SafeConstructor, SafeRepresenter, ScalarNode, Node, MappingNode, SequenceNode
from ruamel.yaml.constructor import ConstructorError

import hermespy
from .logarithmic import Logarithmic, LogarithmicSequence

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Serializable(object):
    """Base class for serializable classes.

    Only classes inheriting from `Serializable` will be serialized by the factory.
    """

    yaml_tag: Optional[str] = None
    """YAML serialization tag."""

    property_blacklist: Set[str] = set()
    """Set of properties to be ignored during serialization."""

    serialized_attributes: Set[str] = set()
    """Set of object attributes to be serialized."""

    @staticmethod
    def _arg_signature() -> Set[str]:
        """Argument signature.

        Returns: Additional arguments not inferable from the init signature.
        """

        return set()

    @classmethod
    def _serializable_attributes(cls: Type[Serializable], blacklist: Optional[Set[str]] = None) -> Set[str]:
        """Extract the set of serializable class attributes.

        Args:
            cls (Type[Serializable]): Class of the object to be serialized.
            blacklist (Set[str], optional): List of attribute names to be ignored during extraction.

        Returns: Set of serializable attribute names.
        """

        if blacklist:
            blacklist = blacklist.copy()
            blacklist.update(cls.property_blacklist)

        else:
            blacklist = cls.property_blacklist

        # Extract initialization signature
        init_signature = set(signature(cls.__init__).parameters.keys())

        # Query serializable properties
        attributes = set()
        for attribute_key, attribute_type in getmembers(cls):

            # Prevent the access to protected or private attributes
            if attribute_key.startswith("_"):
                continue

            # Only add attribute if it isn't blacklisted
            if attribute_key in blacklist:
                continue

            # Make sure the attribute is a property
            if not isinstance(attribute_type, property):
                continue

            # Don't serialize if the property isn't settable
            if attribute_type.fset is None and attribute_key not in init_signature:
                continue

            attributes.add(attribute_key)

        # Add forced attributes
        attributes.update(cls.serialized_attributes)

        return attributes

    @classmethod
    def to_yaml(cls: Type[Serializable], representer: SafeRepresenter, node: Serializable) -> Node:
        """Serialize a serializable object to YAML.

        Args:

            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Serializable):
                The channel instance to be serialized.

        Returns: The serialized YAML node.
        """

        return node._mapping_serialization_wrapper(representer)

    def _mapping_serialization_wrapper(self, representer: SafeRepresenter, blacklist: Optional[Set[str]] = None, additional_fields: Optional[Dict[str, Any]] = None) -> MappingNode:
        """Conveniently serializes the class to a YAML mapping node.

        Args:

            blacklist (Set[str], optional): Properties to be ignored during serialization.
            additional_fields (Dict[str, Any], optional): Additional fields to be serialized.

        Returns: A YAML mapping node representing this object.
        """

        # Init additional fields
        additional_fields = additional_fields if additional_fields else {}

        # Query serializable properties
        serializable_atrributes = self._serializable_attributes(blacklist)

        # Construct state dictionary by querying serializable attributes
        state: Dict[str, Any] = {}
        for attribute_key in serializable_atrributes:

            attribute_value = getattr(self, attribute_key)

            # Don't serialize attribute if it is None
            if attribute_value is None:
                continue

            state[attribute_key] = attribute_value

        # Add additional fields to state
        if additional_fields:
            state.update(additional_fields)

        # Create YAML mapping
        return representer.represent_mapping(self.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[Serializable], constructor: SafeConstructor, node: Node) -> Serializable:
        """Recall a new serializable class instance from YAML.

        Args:

            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Serializable` serialization.

        Returns: The de-serialized object.
        """

        # Handle empty yaml nodes
        if isinstance(node, ScalarNode):
            return cls()

        return cls.InitializationWrapper(constructor.construct_mapping(node, deep=True))

    @classmethod
    def InitializationWrapper(cls, configuration: Dict[str, Any]) -> Serializable:
        """Conveniently initializes serializable classes.

        Args:

            configuration (Dict[str, Any]):
                Configuration parameter dictionary.

        Returns:
            SerializableArray: Initialized class instance.
        """

        # Extract initialization signature
        init_signature = list(signature(cls.__init__).parameters.keys())
        arg_signature = cls._arg_signature()
        init_signature.remove("self")

        # Extract settable class properties
        properties = cls._serializable_attributes()

        init_parameters: Dict[str, Any] = {}
        init_properties: Dict[str, Any] = {}

        for configuration_key in list(configuration.keys()):

            if configuration_key in init_signature or configuration_key in arg_signature:

                init_parameters[configuration_key] = configuration.pop(configuration_key)
                continue

            lower_key = configuration_key.lower()

            if lower_key in init_signature or lower_key in arg_signature:

                init_parameters[lower_key] = configuration.pop(configuration_key)
                continue

            if configuration_key in properties:

                init_properties[configuration_key] = configuration.pop(configuration_key)
                continue

            if lower_key in properties:

                init_properties[lower_key] = configuration.pop(configuration_key)
                continue

        # Initialize class
        # Remaining configuration fields get treated as kwargs
        init_parameters.update(configuration)

        try:
            instance = cls(**init_parameters)

        except TypeError as e:
            raise TypeError(f"Error while attempting to initialize '{cls.__name__}', {str(e)}")

        # Configure properties
        for property_name, property_value in init_properties.items():

            try:
                setattr(instance, property_name, property_value)

            except AttributeError as e:
                raise AttributeError(f"Error while attempting to configure '{property_name}', {str(e)}")

        # Return configured class instance
        return instance


class SerializableEnum(Serializable, Enum):
    """Base class for serializable enumerations."""

    @classmethod
    def from_yaml(cls: Type[SerializableEnum], _: SafeConstructor, node: ScalarNode) -> SerializableEnum:

        # Convert scalar string representation back to enum
        return cls[node.value]

    @classmethod
    def to_yaml(cls: Type[SerializableEnum], representer: SafeRepresenter, node: SerializableEnum) -> ScalarNode:

        # Convert enum to scalar string representation
        return representer.represent_scalar(cls.yaml_tag, "{.name}".format(node))

    @classmethod
    @property
    def yaml_tag(cls) -> str:

        return cls.__name__


class SerializableArray(Serializable, metaclass=ABCMeta):
    """Base class for serializable classes within an array-like structure.

    Only classes inheriting from `Serializable` will be serialized by the factory.
    Additionally, `SerializableArray` nodes may be annotated with a tuple of non-negative
    integers indicating their location within the array grid.
    """

    @staticmethod
    def Set_Array(matrix: Union[Mapping, Sequence], deserialized_data: List[Tuple[SerializableArray, Tuple[int, ...]]]) -> None:
        """Set matrix fields from deserialized array data.

        Args:

            matrix (Union[Mapping, Sequence]):
                The matrix to be set.

            deserialized_data (
        """

        # Skip if no data was provided
        if not deserialized_data or len(deserialized_data) < 1:
            return

        for deserialized_object, position in deserialized_data:
            matrix[position] = deserialized_object


class Factory:
    """Helper class to load HermesPy simulation scenarios from YAML configuration files."""

    extensions: Set[str] = [".yml", ".yaml", ".cfg"]
    """List of recognized filename extensions for serialization files."""

    __yaml: YAML
    __clean: bool
    __db_regex: Pattern
    __registered_classes: Set[Type[Serializable]]
    __registered_tags: Set[str]

    def __init__(self) -> None:

        # YAML dumper configuration
        self.__yaml = YAML(typ="safe", pure=True)
        self.__yaml.default_flow_style = False
        self.__yaml.compact(seq_seq=False, seq_map=False)
        self.__yaml.encoding = None
        self.__yaml.indent(mapping=4, sequence=4, offset=2)
        self.__clean = True
        self.__registered_classes = set()
        self.__registered_tags = set()

        # Add custom representers
        self.__yaml.representer.add_representer(complex, Factory.__complex_representer)
        self.__yaml.representer.add_representer(np.ndarray, Factory.__array_representer)

        # Add custom constructors
        self.__yaml.constructor.add_constructor("complex", Factory.__complex_constructor)
        self.__yaml.constructor.add_constructor("array", Factory.__array_constructor)
        self.__yaml.constructor.add_constructor("dB", Factory.__logarithmic_constructor)

        # Iterate over all modules within the hermespy namespace
        # Scan for serializable classes

        lookup_paths = list(hermespy.__path__) + [os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))]
        for _, name, is_module in iter_modules(lookup_paths, hermespy.__name__ + "."):

            if not is_module:
                continue

            module = import_module(name)

            for _, serializable_class in getmembers(module):

                if not isclass(serializable_class) or not issubclass(serializable_class, Serializable):
                    continue

                self.__registered_classes.add(serializable_class)
                self.__yaml.register_class(serializable_class)

                if serializable_class.yaml_tag is not None:

                    self.__registered_tags.add(serializable_class.yaml_tag)

                    if issubclass(serializable_class, SerializableArray):

                        array_constructor = partial(Factory.__construct_matrix, serializable_class)
                        self.__yaml.constructor.add_multi_constructor(serializable_class.yaml_tag, array_constructor)

        # Add constructors for untagged classes
        self.__yaml.constructor.add_constructor("tag:yaml.org,2002:map", self.__construct_map)
        # self.__yaml.constructor.add_constructor('tag:yaml.org,2002:seq', self.__construct_sequence)

        # Construct regular expressions for purging
        self.__range_regex = compile(r"([0-9.e-]*)[ ]*,[ ]*([0-9.e-]*)[ ]*,[ ]*\.\.\.[ ]*,[ ]*([0-9.e-]*)")
        self.__db_regex = compile(r"\[([ 0-9.,-]*)\][ ]*dB")

    @property
    def clean(self) -> bool:
        """Use clean YAML standard.

        Disabling the clean flag will deactivate additional text processing
        to the YAML configuration files done by Hermes, such as dB conversion or linear
        number spaces.

        Returns: Clean flag.
        """

        return self.__clean

    @clean.setter
    def clean(self, flag: bool) -> None:

        self.__clean = flag

    @property
    def registered_classes(self) -> Set[Type[Serializable]]:
        """Classes registered for serialization within the factory."""

        return self.__registered_classes.copy()

    @property
    def registered_tags(self) -> Set[str]:
        """Read registered YAML tags.

        Returns:
            Set[str]: Set of registered YAML tags.
        """

        return self.__registered_tags

    def load(self, path: str) -> List[Serializable]:
        """Load a serialized executable configuration from a filesystem location.

        Args:
            path (str): Path to a file or a folder featuring serialization files.

        Returns:
            executables (List[Serializable]):
                Serializable HermesPy objects.

        Raises:
            RuntimeError: If `path` does not contain an executable object.
            RuntimeError: If `path` contains more than one executable object.
        """

        # Recover serialized objects
        hermes_objects: List[Any] = self.from_path(path)

        executables: List[Serializable] = []

        for hermes_object in hermes_objects:

            if isinstance(hermes_object, Serializable):
                executables.append(hermes_object)

        # Return fully configured executable
        return executables

    @staticmethod
    def __complex_representer(representer: SafeRepresenter, value: complex) -> ScalarNode:
        """Represent complex numbers as strings.

        Args:

            representer (SafeRepresenter): YAML representer.
            value (complex): The complex number to be transformed to a string.

        Returns: Scalar yaml node.
        """

        complex_string = str(value)[1:-1]
        return representer.represent_scalar("complex", complex_string)

    @staticmethod
    def __complex_constructor(constructor: SafeConstructor, node: ScalarNode) -> complex:
        """Construct a complex number from YAML.

        Args:

            constructor (SafeConstructor): YAML constructor.
            node (ScalarNode): The YAML node representing the complex number.


        Returns: A complex number.
        """

        complex_number = complex(constructor.construct_scalar(node))
        return complex_number

    @staticmethod
    def __array_representer(representer: SafeRepresenter, array: np.ndarray) -> SequenceNode:
        """Represent numpy arrays as lists.

        Args:

            representer (SafeRepresenter): YAML representer.
            array (np.ndarray): The array to be transformed to a sequence.

        Returns: Sequence yaml node.
        """

        # Transform complex numpy arrays to their string representation
        if array.dtype in [np.complex64, np.complex128]:

            object_array = np.empty(array.shape, dtype=object)
            for index, number in np.ndenumerate(array):
                object_array[index] = str(number)[1:-1]

            list = object_array.tolist()

        else:
            list = array.tolist()

        sequence = representer.represent_sequence("array", list, flow_style=True)
        return sequence

    @staticmethod
    def __array_constructor(constructor: SafeConstructor, node: SequenceNode) -> np.ndarray:
        """Construct a numpy array from YAML.

        Args:

            constructor (SafeConstructor): YAML constructor.
            node (ScalarNode): The YAML node representing the array.

        Returns: A numpy array.
        """

        if isinstance(node, SequenceNode):
            return np.array([Factory.__array_constructor(constructor, n) for n in node.value])

        if "j" in node.value:
            return Factory.__complex_constructor(constructor, node)

        else:
            return constructor.construct_object(node)

    @staticmethod
    def __logarithmic_constructor(constructor: SafeConstructor, node: Union[ScalarNode, SequenceNode]) -> Union[Logarithmic, LogarithmicSequence]:
        """Construct a logarithmic value or sequence from YAML.

        Args:

            constructor (SafeConstructor): YAML constructor.
            node (Union[ScalarNode, SequenceNode]): The YAML node representing the array.

        Returns: A logarithmic representation.
        """

        if isinstance(node, ScalarNode):
            return Logarithmic(constructor.construct_scalar(node))

        if isinstance(node, SequenceNode):
            return LogarithmicSequence(constructor.construct_sequence(node))

    @staticmethod
    def __construct_map(constructor: SafeConstructor, node: MappingNode) -> Mapping[MappingNode, Any]:
        """A custom map generator.

        Hacks ruamel to accept node names as tags.

        Args:
            constructor (SafeConstructor): Handle to the constructor.
            node (MappingNode): A YAML map node.

        Returns:
            Mapping[MappingNode, Any]: A sequence of objects created from `node`.
        """

        tag = node.value[0][0].value

        if tag in constructor.yaml_constructors:
            return constructor.yaml_constructors[tag](constructor, node.value[0][1])

        else:
            return constructor.construct_mapping(node, deep=True)

    @staticmethod
    def __decibel_conversion(match: re.Match) -> str:
        """Convert YAML sequences with dB annotations to tagged sequences.

        Args:
            match (re.Match): The serialization sequence to be converted.

        Returns:
            str: The purged sequence.
        """

        linear_values = [float(str_rep) for str_rep in match[1].replace(" ", "").split(",")]

        string_replacement = "!<dB> ["
        for linear_value in linear_values:
            string_replacement += str(linear_value) + ", "

        string_replacement += "]"
        return string_replacement

    def from_path(self, paths: Union[str, Set[str]]) -> List[Any]:
        """Load a configuration from an arbitrary file system path.

        Args:
            paths (Union[str, Set[str]]): Paths to a file or a folder featuring .yml config files.

        Returns:
            List[Any]: List of serializable objects recalled from `paths`.

        Raises:
            ValueError: If the provided `path` does not exist on the filesystem.
        """

        # Convert single path to a set if required
        if isinstance(paths, str):
            paths = [paths]

        hermes_objects = []
        for path in paths:

            if not os.path.exists(path):
                raise ValueError(f"Lookup path '{path}' not found")

            if os.path.isdir(path):
                hermes_objects += self.from_folder(path)

            elif os.path.isfile(path):
                hermes_objects += self.from_file(path)

            else:
                raise ValueError("Lookup location '{}' not recognized".format(path))

        return hermes_objects

    def from_folder(self, path: str, recurse: bool = True, follow_links: bool = False) -> List[Any]:
        """Load a configuration from a folder.

        Args:
            path (str): Path to the folder configuration.
            recurse (bool, optional): Recurse into sub-folders within `path`.
            follow_links (bool, optional): Follow links within `path`.

        Returns:
            List[Any]: List of serializable objects recalled from `path`.

        Raises:
            ValueError: If `path` is not a directory.
        """

        if not os.path.exists(path):
            raise ValueError("Lookup path '{}' not found".format(path))

        if not os.path.isdir(path):
            raise ValueError("Lookup path '{}' is not a directory".format(path))

        hermes_objects: List[Any] = []

        for directory, _, files in os.walk(path, followlinks=follow_links):
            for file in files:

                _, extension = os.path.splitext(file)
                if extension in self.extensions:
                    hermes_objects += self.from_file(os.path.join(directory, file))

            if not recurse:
                break

        return hermes_objects

    def to_folder(self, path: str, *args: Any) -> None:
        """Dump a configuration to a folder.

        Args:
            path (str): Path to the folder configuration.
            *args (Any):
                Configuration objects to be dumped.
        """
        pass

    def from_str(self, config: str) -> List[Any]:
        """Load a configuration from a string object.

        Args:
            config (str): The configuration to be loaded.

        Returns:
            List[Any]: List of serialized objects within `path`.
        """

        stream = StringIO(config)
        return self.from_stream(stream)

    def to_str(self, *args: Any) -> str:
        """Dump a configuration to a folder.

        Args:
            *args (Any): Configuration objects to be dumped.

        Returns:
            str: String containing full YAML configuration.

        Raises:
            RepresenterError: If objects in ``*args`` are unregistered classes.
        """

        stream = StringIO()
        self.to_stream(stream, args)
        return stream.getvalue()

    def from_file(self, file: str) -> List[Any]:
        """Load a configuration from a single YAML file.

        Args:
            file (str): Path to the folder configuration.

        Returns:
            List[Any]: List of serialized objects within `path`.
        """

        with open(file, mode="r") as file_stream:

            try:
                return self.from_stream(file_stream)

            # Re-raise constructor errors with the correct file name
            except ConstructorError as constructor_error:

                constructor_error.problem_mark.name = file
                raise constructor_error

    def to_file(self, path: str, *args: Any) -> None:
        """Dump a configuration to a single YML file.

        Args:
            path (str): Path to the configuration file.
            *args (Any): Configuration objects to be dumped.

        Raises:
            RepresenterError: If objects in ``*args`` are unregistered classes.
        """
        pass

    @staticmethod
    def __range_restore_callback(m: Match) -> str:
        """Internal regular expression callback.

        Args:
            m (Match): Regular expression match.

        Returns:
            str: The processed match line.
        """

        # Extract range parameters
        start = float(m.group(1))
        step = float(m.group(2)) - start
        stop = float(m.group(3)) + step

        range = np.arange(start=start, stop=stop, step=step)

        replacement = ""
        for step in range[:-1]:
            replacement += str(step) + ", "

        replacement += str(range[-1])
        return replacement

    def from_stream(self, stream: TextIOBase) -> List[Any]:
        """Load a configuration from an arbitrary text stream.

        Args:
            stream (TextIOBase): Text stream containing the configuration.

        Returns:
            List[Any]: List of serialized objects within `stream`.

        Raises:
            ConstructorError: If YAML parsing fails.
        """

        if not self.__clean:
            return self.__yaml.load(stream)

        clean_stream = ""
        for line in stream.readlines():

            clean_line = self.__range_regex.sub(self.__range_restore_callback, line)
            clean_line = self.__db_regex.sub(self.__decibel_conversion, clean_line)
            clean_stream += clean_line

        hermes_objects = self.__yaml.load(StringIO(clean_stream))

        if hermes_objects is None:
            return []

        if isinstance(hermes_objects, Iterable):
            return hermes_objects

        else:
            return [hermes_objects]

    def to_stream(self, stream: TextIOBase, *args: Any) -> None:
        """Dump a configuration to an arbitrary text stream.

        Args:
            stream (TextIOBase): Text stream to the configuration.
            *args (Any): Configuration objects to be dumped.

        Raises:
            RepresenterError: If objects in ``*args`` are unregistered classes.
        """

        for serializable_object in args:
            self.__yaml.dump(*serializable_object, stream)


class HDFSerializable(metaclass=ABCMeta):
    """Base class for object serializable to the HDF5 format.

    Structures are serialized to HDF5 files by the :meth:`.to_HDF` routine and
    de-serialized by the :meth:`.from_HDF` method, respectively.
    """

    @abstractmethod
    def to_HDF(self, group: Group) -> None:
        """Serialize the object state to HDF5.

        Dumps the object's state and additional information to a HDF5 group.

        Args:

            group (Group):
                The HDF5 group to which the object is serialized.
        """
        ...  # pragma no cover

    @abstractclassmethod
    def from_HDF(cls: Type[HDFSerializable], group: Group) -> HDFSerializable:
        """De-Serialized the object state from HDF5.

        Recalls the object's state from a HDF5 group.

        Args:

            group (Group):
                The HDF5 group from which the object state is recalled.

        Returns: The object initialized from the HDF5 group state.
        """
        ...  # pragma no cover
