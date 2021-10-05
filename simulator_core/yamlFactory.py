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
from typing import Any, Set
from io import TextIOBase, StringIO

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

    def __init__(self) -> None:
        """Object initialization.
        """

        # YAML dumper configuration
        self.__yaml = YAML(typ='safe')
        self.__yaml.default_flow_style = False
        self.__yaml.compact(seq_seq=False, seq_map=False)
        self.__yaml.encoding = None

        # Register serializable classes for safe recall / dumping
        for serializable_class in SerializableClasses:
            self.__yaml.register_class(serializable_class)

        # Add constructors for untagged classes
        self.__yaml.constructor.add_constructor('tag:yaml.org,2002:map', self.__construct_map)
        self.__yaml.constructor.add_constructor('tag:yaml.org,2002:seq', self.__construct_sequence)

        self.__yaml.representer

    @staticmethod
    def __construct_map(constructor: SafeConstructor, node: MappingNode) -> Any:

        tag = node.value[0][0].value

        if tag in constructor.yaml_constructors:
            return constructor.yaml_constructors[tag](constructor, node.value[0][1])

        else:
            return constructor.construct_mapping(node, deep=True)

    @staticmethod
    def __construct_sequence(constructor: SafeConstructor, node: SequenceNode) -> Any:

        tag = node.value[0].value

        if tag in constructor.yaml_constructors:
            return constructor.yaml_constructors[tag](constructor, node.value[0])

        else:
            return constructor.construct_sqeuence(node, deep=True)

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

    def from_stream(self, stream: TextIOBase) -> Simulation:
        """Load a configuration from an arbitrary text stream.

        Args:
            stream (TextIOBase): Text stream containing the configuration.

        Returns:
            Simulation: A configured simulation.
        """

        return self.__yaml.load(stream)

    def to_stream(self, stream: TextIOBase, *args: Any) -> None:
        """Dump a configuration to an arbitrary text stream.

        Args:
            stream (TextIOBase): Text stream to the configuration.
            *args (Any): Configuration objects to be dumped.

        Raises:
            RepresenterError: If objects in ``*args`` are unregistered classes.
        """

        for serializable_object in args:
            self.__yaml.dump(serializable_object, stream)
