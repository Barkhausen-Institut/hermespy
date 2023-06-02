# -*- coding: utf-8 -*-
"""
===========
Executable
===========

HermesPy base for executable configurations.
"""

from __future__ import annotations
import os.path as path
import datetime
from abc import ABC, abstractmethod
from contextlib import contextmanager
from glob import glob
from os import getcwd, mkdir
from sys import exit
from typing import Any, Generator, List, Optional, Union

import matplotlib.pyplot as plt
from rich.console import Console
from rich.prompt import Confirm

from .definitions import ConsoleMode
from .factory import SerializableEnum

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Verbosity(SerializableEnum):
    """Information output behaviour configuration of an executable"""

    ALL = 0
    """Print absolutely everything"""

    INFO = 1
    """Print general information"""

    WARNING = 2
    """Print only warnings and errors"""

    ERROR = 3
    """Print only errors"""

    NONE = 4
    """Print absolutely nothing"""


class Executable(ABC):
    """Base Class for HermesPy Entry Points.

    All executables are required to implement the :meth:`.run` method.
    """

    __results_dir: Optional[str]  # Directory in which all execution artifacts will be dropped.
    __verbosity: Verbosity  # Information output behaviour during execution.
    __style: str = "dark"  # Plotting color scheme
    __console: Console  # Rich console instance for text output
    __console_mode: ConsoleMode  # Output format during execution

    def __init__(self, results_dir: Optional[str] = None, verbosity: Union[Verbosity, str] = Verbosity.INFO, console: Optional[Console] = None, console_mode: ConsoleMode = ConsoleMode.INTERACTIVE) -> None:
        """
        Args:

            results_dir(str, optional):
                Directory in which all execution artifacts will be dropped.

            verbosity (Union[str, Verbosity], optional):
                Information output behaviour during execution.

            console (Console, optional):
                The console instance the executable will operate on.

            console_mode (ConsoleMode, optional):
                Output behaviour of the information printed to the console.
                Interactive by default.
        """

        # Default parameters
        self.results_dir = results_dir
        self.verbosity = verbosity if isinstance(verbosity, Verbosity) else Verbosity[verbosity]
        self.__console = Console(record=False) if console is None else console
        self.console_mode = console_mode

    def execute(self) -> None:
        """Execute the executable.

        Sets up the environment to the implemented :meth:`.run` routine.
        """

        with self.style_context():
            _ = self.run()

    @abstractmethod
    def run(self) -> Any:
        """Execute the configuration.

        Returns: The result of the run.
        """
        ...  # pragma no cover

    @property
    def results_dir(self) -> str:
        """Directory in which the execution results will be saved.

        Returns:
            str: The directory.
        """

        return self.__results_dir

    @results_dir.setter
    def results_dir(self, directory: Optional[str]) -> None:
        """Modify the directory in which the execution results will be saved.

        Args:
            directory (str): New directory.

        Raises:
            ValueError: If `directory` does not exist within the filesystem.
        """

        if directory is None:
            self.__results_dir = None
            return

        if not path.exists(directory):
            raise ValueError("The provided results directory does not exist")

        if not path.isdir(directory):
            raise ValueError("The provided results directory path is not a directory")

        self.__results_dir = directory

    @property
    def verbosity(self) -> Verbosity:
        """Information output behaviour during execution.

        Returns:
            Verbosity: Configuration flag.
        """

        return self.__verbosity

    @verbosity.setter
    def verbosity(self, new_verbosity: Union[str, Verbosity]) -> None:
        """Modify the information output behaviour during execution.

        Args:
            new_verbosity (Union[str, Verbosity]): The new output behaviour.
        """

        # Convert string arguments to verbosity enum fields
        if isinstance(new_verbosity, str):
            self.__verbosity = Verbosity[new_verbosity.upper()]

        else:
            self.__verbosity = new_verbosity

    @staticmethod
    def default_results_dir() -> str:
        """Create a default directory to store execution results.

        Returns:
            str: Path to the newly created directory.
        """

        today = str(datetime.date.today())

        dir_index = 0

        base_directory = path.join(getcwd(), "results")

        # Create results directory within the current working directory if it does not exist yet
        if not path.exists(base_directory):
            mkdir(base_directory)

        results_dir = path.join(base_directory, today + "_" + "{:03d}".format(dir_index))

        while path.exists(results_dir):
            dir_index += 1
            results_dir = path.join(base_directory, today + "_" + "{:03d}".format(dir_index))

        # Create the results directory
        mkdir(results_dir)

        return results_dir

    @property
    def style(self) -> str:
        """Matplotlib color scheme.

        Returns:
            str: Color scheme.

        Raises:
            ValueError: If the `style` is not available.
        """

        return self.__style

    @style.setter
    def style(self, value: str) -> None:
        hermes_styles = self.__hermes_styles()
        if value in hermes_styles:
            self.__style = value
            return

        matplotlib_styles = plt.style.available
        if value in matplotlib_styles:
            self.__style = value
            return

        raise ValueError("Requested style identifier not available")

    @staticmethod
    def __hermes_styles() -> List[str]:
        """Styles available in Hermes only.

        Returns:
            List[str]: List of style identifiers.
        """

        return [path.splitext(path.basename(x))[0] for x in glob(path.join(Executable.__hermes_root_dir(), "core", "styles", "*.mplstyle"))]

    @staticmethod
    @contextmanager
    def style_context() -> Generator:  # pragma: no cover
        """Context for the configured style.

        Returns:  Style context manager generator.
        """

        if Executable.__style in Executable.__hermes_styles():
            yield plt.style.use(path.join(Executable.__hermes_root_dir(), "core", "styles", Executable.__style + ".mplstyle"))

        else:
            yield plt.style.use(Executable.__style)

    @staticmethod
    def __hermes_root_dir() -> str:
        """HermesPy Package Root Directory.

        Returns:
            str: Path to the package root.
        """

        return path.dirname(path.dirname(path.abspath(__file__)))

    @property
    def console(self) -> Console:
        """Console the Simulation writes to.

        Returns:
            Console: Handle to the console.
        """

        return self.__console

    @console.setter
    def console(self, value: Console) -> None:
        self.__console = value

    @property
    def console_mode(self) -> ConsoleMode:
        """Console mode during runtime.

        Returms: The current console mode.
        """

        return self.__console_mode

    @console_mode.setter
    def console_mode(self, value: Union[ConsoleMode, str]) -> None:
        # Convert string arguments to iterable
        if isinstance(value, str):
            value = ConsoleMode[value]

        self.__console_mode = value

    def _handle_exception(self, force: bool = False, show_locals: bool = True, confirm: bool = True) -> None:
        """Print an exception traceback if Verbosity is ALL or higher.

        Args:

            force (bool): If True, print the traceback regardless of Verbosity level
            show_locals (bool): Output the local variables.
            confirm (bool): Confirm for continuing execution.
        """

        # Check if the exception should be ignored
        if (self.verbosity.value < Verbosity.NONE.value and self.console_mode != ConsoleMode.SILENT) or force:
            # Resort to rich's exception tracing
            self.console.print_exception(show_locals=show_locals)

            # If the confirmation flag is enabled, ask to conntinue excetion and abort script if not confirmed
            if confirm:
                if not Confirm.ask("Continue execution?", console=self.console, choices=["y", "n"]):
                    exit(0)
