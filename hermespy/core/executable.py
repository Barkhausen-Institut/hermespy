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
from os import getcwd, mkdir, makedirs
from sys import exit
from typing import Any, Generator, List, Union

import matplotlib.pyplot as plt
from rich.console import Console
from rich.prompt import Confirm

from .definitions import ConsoleMode
from .factory import SerializableEnum

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
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

    __results_dir: str | None  # Directory in which all execution artifacts will be dropped.
    __verbosity: Verbosity  # Information output behaviour during execution.
    __style: str = "dark"  # Plotting color scheme
    __console: Console  # Rich console instance for text output
    __console_mode: ConsoleMode  # Output format during execution
    __debug: bool  # Debug mode flag

    def __init__(
        self,
        results_dir: str | None = None,
        verbosity: Union[Verbosity, str] = Verbosity.INFO,
        console: Console | None = None,
        console_mode: ConsoleMode = ConsoleMode.INTERACTIVE,
        debug: bool = False,
    ) -> None:
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

            debug (bool, optional):
                If enabled, the executable will be run in debug mode.
                In this case, the exception handler will re-raise exceptions
                and stop the execution.
        """

        # Default parameters
        self.results_dir = results_dir
        self.verbosity = verbosity if isinstance(verbosity, Verbosity) else Verbosity[verbosity]
        self.__console = Console(record=False) if console is None else console
        self.console_mode = console_mode
        self.__debug = debug

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
    def results_dir(self, directory: str | None) -> None:
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

    @property
    def debug(self) -> bool:
        """Debug mode flag.

        If enabled, the executable will be run in debug mode.
        In this case, the exception handler will re-raise exceptions
        and stop the execution.
        """

        return self.__debug

    @staticmethod
    def default_results_dir(experiment: str | None = None, overwrite_results: bool = False) -> str:
        """Create a default directory to store execution results.

        .. warning::
           If `overwrite_results` is set to True, the current results directory will be erased.
           Proceed with caution as to not lose any important data.

        Args:

            experiment(str, optional):
                Name of the experiment.
                If specified, will generate a subdirectory with the experiment name.

            overwrite_results(bool, optional):
                If False, a new dated directory will be created with a unique index.
                If True, executing this function will erase the current results directory.

        Returns: Path to the newly created directory.
        """

        # Select the base directory
        base_directory = path.join(getcwd(), "results")

        if experiment is not None:
            base_directory = path.join(base_directory, experiment)

        # Create results directory within the current working directory if it does not exist yet
        makedirs(base_directory, 511, True)

        # Select the current base directory as the results directory if the overwrite flag is set
        if overwrite_results:
            results_dir = base_directory

        # Otherwise, create a new dated directory with a unique index
        else:

            today = str(datetime.date.today())
            dir_index = 0

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

        return [
            path.splitext(path.basename(x))[0]
            for x in glob(path.join(Executable.__hermes_root_dir(), "core", "styles", "*.mplstyle"))
        ]

    @staticmethod
    @contextmanager
    def style_context() -> Generator:  # pragma: no cover
        """Context for the configured style.

        Returns:  Style context manager generator.
        """

        style_path = Executable.__style
        if style_path in Executable.__hermes_styles():
            style_path = path.join(
                Executable.__hermes_root_dir(), "core", "styles", Executable.__style + ".mplstyle"
            )
        with plt.style.context(style_path):
            yield

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

    def _handle_exception(
        self,
        exception: Exception,
        force: bool = False,
        show_locals: bool = True,
        confirm: bool = True,
    ) -> None:
        """Print an exception traceback if Verbosity is ALL or higher.

        Args:

            exception (Exception): The exception to be handled.
            force (bool): If True, print the traceback regardless of Verbosity level
            show_locals (bool): Output the local variables.
            confirm (bool): Confirm for continuing execution.

        Raises: The original exception if debug mode is enabled.
        """

        # If debug mode is enabled, re-raise the exception without any additional handling
        if self.debug:
            raise exception

        # Check if the exception should be ignored
        if (
            self.verbosity.value < Verbosity.NONE.value and self.console_mode != ConsoleMode.SILENT
        ) or force:
            # Resort to rich's exception tracing
            self.console.print_exception(show_locals=show_locals)

            # If the confirmation flag is enabled, ask to conntinue excetion and abort script if not confirmed
            if confirm:
                if not Confirm.ask("Continue execution?", console=self.console, choices=["y", "n"]):
                    exit(0)
