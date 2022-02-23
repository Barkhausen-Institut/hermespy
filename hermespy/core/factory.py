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
from abc import ABCMeta
from collections.abc import Iterable
from functools import partial
from inspect import getmembers, isclass, signature
from importlib import import_module
from io import TextIOBase, StringIO
import os
from pkgutil import iter_modules
from re import compile, Pattern, Match
from typing import Any, Dict, Set, Sequence, Mapping, Union, List, Optional, Tuple, Type

import numpy as np
from ruamel.yaml import YAML, SafeConstructor, SafeRepresenter, ScalarNode, Node, MappingNode, SequenceNode
from ruamel.yaml.constructor import ConstructorError

import hermespy as hermes
from ..tools import db2lin

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.6"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Serializable(metaclass=ABCMeta):
    """Base class for serializable classes.

    Only classes inheriting from `Serializable` will be serialized by the factory.
    """

    yaml_tag: Optional[str] = None
    """YAML serialization tag."""

    @classmethod
    def to_yaml(cls: Type[Serializable], representer: SafeRepresenter, node: Serializable) -> Node:
        """Serialize a serializable object to YAML.

        Args:

            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (Serializable):
                The channel instance to be serialized.

        Returns:

            Node:
                The serialized YAML node.
        """

        return ScalarNode(cls.yaml_tag, None)

    @classmethod
    def from_yaml(cls: Type[Serializable],
                  constructor: SafeConstructor,
                  node: Node) -> Serializable:
        """Recall a new serializable class instance from YAML.

        Args:

            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `Channel` serialization.

        Returns:

            Serializable:
                The de-serialized object.
        """

        # Handle empty yaml nodes
        if isinstance(node, ScalarNode):
            return cls()

        return cls.InitializationWrapper(constructor.construct_mapping(node))

    @classmethod
    def InitializationWrapper(cls,
                              configuration: Dict[str, Any]) -> Serializable:
        """Conveniently initializes serializable classes.

        Args:

            configuration (Dict[str, Any]):
                Configuration parameter dictionary.

        Returns:
            SerializableArray: Initialized class instance.
        """

        # Extract initialization signature
        init_signature = list(signature(cls.__init__).parameters.keys())
        init_signature.remove('self')

        # Extract settable class properties
        properties: List[str] = []
        for attribute_key, attribute_type in getmembers(cls):

            # Prevent the access to protected or private attributes
            if attribute_key.startswith('_'):
                continue

            # Make sure the attribute is a property and settable
            if isinstance(attribute_type, property) and attribute_type.setter:
                properties.append(attribute_key)

        init_parameters: Dict[str, Any] = {}
        init_properties: Dict[str, Any] = {}

        for configuration_key in list(configuration.keys()):

            if configuration_key in init_signature:

                init_parameters[configuration_key] = configuration.pop(configuration_key)
                continue

            if configuration_key in properties:
                init_properties[configuration_key] = configuration.pop(configuration_key)

        # Initialize class
        init_parameters.update(configuration)       # Remaining configuration fields get treated as kwargs
        instance = cls(**init_parameters)

        # Configure properties
        for property_name, property_value in init_properties.items():
            setattr(instance, property_name, property_value)

        # Return configured class instance
        return instance


class SerializableArray(Serializable, metaclass=ABCMeta):
    """Base class for serializable classes within an array-like structure.

    Only classes inheriting from `Serializable` will be serialized by the factory.
    Additionally, `SerializableArray` nodes may be annotated with a tuple of non-negative
    integers indicating their location within the array grid.
    """

    @staticmethod
    def Set_Array(matrix: Union[Mapping, Sequence],
                  deserialized_data: List[Tuple[SerializableArray, Tuple[int, ...]]]) -> None:
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

    extensions: Set[str] = ['.yml', '.yaml', '.cfg']
    """List of recognized filename extensions for serialization files."""

    __yaml: YAML
    __clean: bool
    __purge_regex_alpha: Pattern
    __purge_regex_beta: Pattern
    __db_regex: Pattern
    __restore_regex_alpha: Pattern
    __registered_classes: Set[Type[Serializable]]
    __registered_tags: Set[str]

    def __init__(self) -> None:

        # YAML dumper configuration
        self.__yaml = YAML(typ='safe', pure=True)
        self.__yaml.default_flow_style = False
        self.__yaml.compact(seq_seq=False, seq_map=False)
        self.__yaml.encoding = None
        self.__yaml.indent(mapping=4, sequence=4, offset=2)
        self.__clean = True
        self.__registered_classes = set()
        self.__registered_tags = set()

        # Browse the current environment for packages within the 'hermespy' namespace
        for finder, name, ispkg in iter_modules(hermes.__path__, "hermespy."):

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
        self.__yaml.constructor.add_constructor('tag:yaml.org,2002:map', self.__construct_map)
        # self.__yaml.constructor.add_constructor('tag:yaml.org,2002:seq', self.__construct_sequence)

        # Construct regular expressions for purging
        self.__purge_regex_alpha = compile(r': !<.*')
        self.__purge_regex_beta = compile(r"- !<([^']+)>")
        self.__restore_regex_alpha = compile(r"([ ]*)([a-zA-Z]+):\n$")
        self.__restore_regex_beta = compile(r"([ ]*)- ([^\s]+)([^']*)\n$")
        self.__range_regex = compile(r'([0-9.e-]*)[ ]*,[ ]*([0-9.e-]*)[ ]*,[ ]*\.\.\.[ ]*,[ ]*([0-9.e-]*)')
        self.__db_regex = compile(r"\[([ 0-9.,-]*)\][ ]*dB")

    @property
    def clean(self) -> bool:
        """Access clean flag.

        Returns:
            bool: Clean flag.
        """

        return self.__clean

    @clean.setter
    def clean(self, flag: bool) -> None:
        """Modify clean flag.

        Args:
            flag (bool): New clean flag.
        """

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
    def __construct_matrix(cls: Any, constructor: SafeConstructor, tag_suffix: str, node: Any)\
            -> Tuple[Any, Tuple[int, ...]]:
        """Construct a matrix node from YAML.

        Args:

            cls (Any):
                The type of class to be constructed. This argument will be managed by ruamel.
                The class `cls` must define a `from_yaml` routine.

            constructor (SafeConstructor):
                A handle to the constructor extracting the YAML information.

            tag_suffix (str):
                Tag suffix in the YAML config describing the channel position within the matrix.

            node (Node):
                YAML node representing the `cls` serialization.

        Returns:
            cls:
                Newly created `cls` instance.

            int:
                First dimension position within the matrix.

            int:
                Second dimension within the matrix.
            """

        indices: List[str] = re.split(' |_', tag_suffix)
        if indices[0] == '':
            indices.pop(0)

        indices: Tuple[int] = tuple([int(idx) for idx in indices])

        return cls.from_yaml(constructor, node), indices

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
    def __construct_sequence(constructor: SafeConstructor, node: SequenceNode) -> Sequence[Any]:
        """A custom sequence generator.

        Hacks ruamel to accept node names as tags.

        Args:
            constructor (SafeConstructor): Handle to the constructor.
            node (SequenceNode): A YAML sequence node.

        Returns:
            Sequence[Any]: A sequence of objects created from `node`.
        """

        sequence = []
        for node in node.value:

            if node.tag in constructor.yaml_constructors:
                sequence.append(constructor.yaml_constructors[node.tag](constructor, node))

            else:
                sequence.append(constructor.construct_non_recursive_object(node))

        return sequence

    def __purge_tags(self, serialization: str) -> str:
        """Callback to remove explicit YAML tags from serialization stream.

        Args:
            serialization (str): The serialization sequence to be purged.

        Returns:
            str: The purged sequence.
        """

        cleaned_sequence = ''
        for line in serialization.splitlines(True):

            cleaned_line = self.__purge_regex_alpha.sub(r':', line)
            cleaned_line = self.__purge_regex_beta.sub(r'- \1', cleaned_line)
            cleaned_line = cleaned_line.replace('%20', " ")

            cleaned_sequence += cleaned_line

        return cleaned_sequence

    def refurbish_tags(self, serialization: str) -> str:
        """Callback to restore explicit YAML tags to serialization streams."""
        pass

    @staticmethod
    def __decibel_conversion(match: re.Match) -> str:
        """Convert linear series to decibel series.

        Args:
            match (re.Match): The serialization sequence to be converted.

        Returns:
            str: The purged sequence.
        """

        linear_values = [db2lin(float(str_rep)) for str_rep in match[1].replace(' ', '').split(',')]

        string_replacement = "["
        for linear_value in linear_values:
            string_replacement += str(linear_value) + ', '

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

        with open(file, mode='r') as file_stream:

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

    def __restore_callback_alpha(self, m: Match) -> str:
        """Internal regular expression callback.

        Args:
            m (Match): Regular expression match.

        Returns:
            str: The processed match line.
        """

        if m.group(2) in self.registered_tags:
            return m.group(1) + m.group(2) + ": !<" + m.group(2) + ">\n"

        else:
            return m.string

    def __restore_callback_beta(self, m: Match) -> str:
        """Internal regular expression callback.

        Args:
            m (Match): Regular expression match.

        Returns:
            str: The processed match line.
        """

        if m.group(2) in self.registered_tags:

            indices = m.group(3).replace(" ", "%20")
            return m.group(1) + "- !<" + m.group(2) + indices + ">\n"

        else:
            return m.string

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

        replacement = ''
        for step in range[:-1]:
            replacement += str(step) + ', '

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

        clean_stream = ''
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

            if self.__clean:
                self.__yaml.dump(*serializable_object, stream, transform=self.__purge_tags)

            else:
                self.__yaml.dump(*serializable_object, stream)
