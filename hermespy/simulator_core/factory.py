# -*- coding: utf-8 -*-
"""HermesPy serialization factory.

This module implements the main interface for loading and dumping HermesPy configurations.

    Dumping to text file:

        factory = Factory()
        factory.to_file("config.yml", simulation)

    Loading from text file:

        factory = Factory()
        simulation = factory.from_file("config.yml")

    Attributes:
        SerializableClasses (List): List of classes permitted to be dumped to / loaded from YAML config files.
"""

from __future__ import annotations
from ruamel.yaml import YAML, SafeConstructor, MappingNode, SequenceNode
from ruamel.yaml.constructor import ConstructorError
from typing import Any, Set, Sequence, Mapping, Union, List, Optional, Tuple
from io import TextIOBase, StringIO
from re import compile, Pattern, Match
from collections.abc import Iterable
from functools import partial
import os

from .executable import Executable
from .simulation import Simulation
from hermespy.scenario import Scenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


SerializableClasses: Set = set()


class Factory:
    """Helper class to load HermesPy simulation scenarios from YAML configuration files.

    Attributes:
        extensions (Set[str]): List of recognized filename extensions for serialization files.
    """

    extensions: Set[str] = ['.yml', '.yaml', '.cfg']
    __yaml: YAML
    __clean: bool
    __purge_regex_alpha: Pattern
    __purge_regex_beta: Pattern
    __restore_regex_alpha: Pattern
    __registered_tags: Set[str]

    def __init__(self) -> None:
        """Object initialization.
        """

        # YAML dumper configuration
        self.__yaml = YAML(typ='safe', pure=True)
        self.__yaml.default_flow_style = False
        self.__yaml.compact(seq_seq=False, seq_map=False)
        self.__yaml.encoding = None
        self.__yaml.indent(mapping=4, sequence=4, offset=2)
        self.__clean = True
        self.__registered_tags = set()

        # Register serializable classes for safe recall / dumping
        for serializable_class in SerializableClasses:

            self.__yaml.register_class(serializable_class)

            if hasattr(serializable_class, 'yaml_tag'):    # Dirty. Very dirty.

                self.__registered_tags.add(serializable_class.yaml_tag)

                if hasattr(serializable_class, 'yaml_matrix') and serializable_class.yaml_matrix is True:

                    matrix_constructor = partial(Factory.__construct_matrix, serializable_class)
                    self.__yaml.constructor.add_multi_constructor(serializable_class.yaml_tag, matrix_constructor)

        # Add constructors for untagged classes
        self.__yaml.constructor.add_constructor('tag:yaml.org,2002:map', self.__construct_map)
        self.__yaml.constructor.add_constructor('tag:yaml.org,2002:seq', self.__construct_sequence)

        # Construct regular expressions for purging
        self.__purge_regex_alpha = compile(r': !<.*')
        self.__purge_regex_beta = compile(r"- !<([^']+)>")
        self.__restore_regex_alpha = compile(r"([ ]*)([a-zA-Z]+):\n$")
        self.__restore_regex_beta = compile(r"([ ]*)- ([^\s]+)([^']*)\n$")

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
    def registered_tags(self) -> Set[str]:
        """Read registered YAML tags.

        Returns:
            Set[str]: Set of registered YAML tags.
        """

        return self.__registered_tags

    def load(self, path: str) -> Executable:
        """Load a serialized executable configuration from a filesystem location.

        Args:
            path (str): Path to a file or a folder featuring serialization files.

        Returns:
            Executable: An executable HermesPy object.

        Raises:
            RuntimeError: If `path` does not contain an executable object.
            RuntimeError: If `path` contains more than one executable object.
        """

        # Recover serialized objects
        hermes_objects: List[Any] = self.from_path(path)

        executable: Optional[Executable] = None
        scenarios: List[Scenario] = []

        for hermes_object in hermes_objects:

            if isinstance(hermes_object, Executable):

                if executable is None:
                    executable = hermes_object

                else:
                    raise RuntimeError("Ambiguous configuration containing more than one executable")

            elif isinstance(hermes_object, Scenario):
                scenarios.append(hermes_object)

            else:
                raise RuntimeError("Unsupported class type in configuration")

        # Default executable is a simulation
        if executable is None:
            executable = Simulation()

        # Register scenarios with executable
        for scenario in scenarios:
            executable.add_scenario(scenario)

        # Return fully configured executable
        return executable

    @staticmethod
    def __construct_matrix(cls: Any, constructor: SafeConstructor, tag_suffix: str, node: Any)\
            -> Tuple[Any, int, int]:
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

        indices = tag_suffix.split(' ')
        if indices[0] == '':
            indices.pop(0)

        return cls.from_yaml(constructor, node), int(indices[0]), int(indices[1])

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
            clean_line = self.__restore_regex_alpha.sub(self.__restore_callback_alpha, line)
            clean_line = self.__restore_regex_beta.sub(self.__restore_callback_beta, clean_line)
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
