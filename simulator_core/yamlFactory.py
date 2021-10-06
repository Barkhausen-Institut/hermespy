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
from typing import Any, Set, Sequence, Mapping
from io import TextIOBase, StringIO
from re import compile, Pattern, Match

from . import Simulation

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


SerializableClasses: Set = set()


class Factory:
    """Helper class to load HermesPy simulation scenarios from YAML configuration files.
    """

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

            if hasattr(serializable_class, 'yaml_tag'):                 # Dirty. Very dirty.
                self.__registered_tags.add(serializable_class.yaml_tag)

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
    def clean(self, new: bool) -> None:
        """Modify clean flag.

        Args:
            new (bool): New clean flag.
        """

        self.__clean = new

    @property
    def registered_tags(self) -> Set[str]:
        """Read registered YAML tags.

        Returns:
            Set[str]: Set of registered YAML tags.
        """

        return self.__registered_tags

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

    def from_folder(self, path: str) -> Simulation:
        """Load a configuration from a folder.

        Args:
            path (str): Path to the folder configuration.

        Returns:
            Simulation: A configured simulation.
        """
        pass

    def to_folder(self, path: str, *args: Any) -> None:
        """Dump a configuration to a folder.

        Args:
            path (str): Path to the folder configuration.
            *args (Any):
                Configuration objects to be dumped.
        """
        pass

    def from_str(self, config: str) -> Simulation:
        """Load a configuration from a string object.

        Args:
            config (str): The configuration to be loaded.

        Returns:
            Simulation: A configured simulation.
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

    def from_file(self, config: str) -> Simulation:
        """Load a configuration from a single YAML file.

        Args:
            config (str): The configuration to be loaded.

        Returns:
            Simulation: A configured simulation.
        """
        pass

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

    def from_stream(self, stream: TextIOBase) -> Simulation:
        """Load a configuration from an arbitrary text stream.

        Args:
            stream (TextIOBase): Text stream containing the configuration.

        Returns:
            Simulation: A configured simulation.
        """

        if not self.__clean:
            return self.__yaml.load(stream)

        clean_stream = ''
        for line in stream.readlines():
            clean_line = self.__restore_regex_alpha.sub(self.__restore_callback_alpha, line)
            clean_line = self.__restore_regex_beta.sub(self.__restore_callback_beta, clean_line)
            clean_stream += clean_line

        return self.__yaml.load(StringIO(clean_stream))

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
