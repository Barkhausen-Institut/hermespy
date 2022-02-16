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
from enum import Enum
from glob import glob
from os import getcwd, mkdir
from typing import ContextManager, List, Optional, Union

import matplotlib.pyplot as plt

from hermespy.core.factory import Serializable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.5"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Verbosity(Enum):
    """Information output behaviour configuration of an executable."""

    ALL = 0      # Print absolutely everything
    INFO = 1     # Information
    WARNING = 2  # Warnings only
    ERROR = 3    # Errors only
    NONE = 4     # Print absolutely nothing


class Executable(ABC, Serializable):
    """Base Class for HermesPy Entry Points.

    All executables are required to implement the :meth:`.run` method.
    """

    yaml_tag = u'Executable'
    """YAML serialization tag."""

    __spectrum_fft_size: int        # Number of FFT bins considered during computation.
    __max_num_drops: int            # Number of maximum executions per scenario.
    __results_dir: Optional[str]    # Directory in which all execution artifacts will be dropped.
    __verbosity: Verbosity          # Information output behaviour during execution.
    __style: str = 'dark'           # Color scheme

    def __init__(self,
                 results_dir: Optional[str] = None,
                 verbosity: Union[Verbosity, str] = Verbosity.INFO) -> None:
        """
        Args:

            results_dir(str, optional):
                Directory in which all execution artifacts will be dropped.

            verbosity (Union[str, Verbosity], optional):
                Information output behaviour during execution.
                By default, the verbosity level is `INFO`.
        """

        # Default parameters
        self.__scenarios = []
        self.results_dir = results_dir
        self.verbosity = verbosity

    def execute(self) -> None:
        """Execute the executable.

        Sets up the environment to the implemented :meth:`.run` routine.
        """

        with self.style_context():
            self.run()

    @abstractmethod
    def run(self) -> None:
        """Execute the configuration."""
        ...

    @property
    def spectrum_fft_size(self) -> int:
        """Number of discrete frequency bins considered during Fast Fourier Transform.

        Returns:
            int: The number of bins.
        """

        return self.__spectrum_fft_size

    @spectrum_fft_size.setter
    def spectrum_fft_size(self, bins: int) -> None:
        """Modify the configured number of discrete frequency bins considered during Fast Fourier Transform.

        Args:
            bins (int): The new number of bins.

        Raises:
            ValueError: If `bins` is negative.
        """

        if bins < 0:
            raise ValueError("Number of bins must be greater or equal to zero")

        self.__spectrum_fft_size = bins

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

        results_dir = path.join(base_directory, today + '_' + '{:03d}'.format(dir_index))

        while path.exists(results_dir):

            dir_index += 1
            results_dir = path.join(base_directory, today + '_' + '{:03d}'.format(dir_index))

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

        return [path.splitext(path.basename(x))[0] for x in
                glob(path.join(Executable.__hermes_root_dir(), 'resources', 'styles', '*.mplstyle'))]

    @staticmethod
    @contextmanager
    def style_context() -> ContextManager:
        """Context for the configured style.

        Returns:
            ContextManager:
                Style context manager.
        """

        if Executable.__style in Executable.__hermes_styles():
            yield plt.style.use(path.join(Executable.__hermes_root_dir(), 'resources', 'styles',
                                          Executable.__style + '.mplstyle'))

        else:
            yield plt.style.use(Executable.__style)

    @staticmethod
    def __hermes_root_dir() -> str:
        """HermesPy Package Root Directory.

        Returns:
            str: Path to the package root.
        """

        return path.dirname(path.dirname(path.abspath(__file__)))
