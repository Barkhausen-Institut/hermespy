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
    """Abstract base class for executable configurations."""

    yaml_tag = u'Executable'
    """YAML serialization tag."""

    plot_drop: bool
    """Plot each drop during execution of scenarios."""

    calc_transmit_spectrum: bool
    """Compute the transmitted signals frequency domain spectra."""

    calc_receive_spectrum: bool
    """Compute the received signals frequency domain spectra."""

    calc_transmit_stft: bool
    """Compute the short time Fourier transform of transmitted signals."""

    calc_receive_stft: bool
    """Compute the short time Fourier transform of received signals."""

    __scenarios: List[Scenario]     # List of registered scenarios to be executed
    __spectrum_fft_size: int        # Number of FFT bins considered during computation.
    __max_num_drops: int            # Number of maximum executions per scenario.
    __results_dir: Optional[str]    # Directory in which all execution artifacts will be dropped.
    __verbosity: Verbosity          # Information output behaviour during execution.
    __style: str                    # Color scheme

    def __init__(self,
                 plot_drop: bool = False,
                 calc_transmit_spectrum: bool = False,
                 calc_receive_spectrum: bool = False,
                 calc_transmit_stft: bool = False,
                 calc_receive_stft: bool = False,
                 spectrum_fft_size: int = 0,
                 max_num_drops: int = 1,
                 results_dir: Optional[str] = None,
                 verbosity: Union[Verbosity, str] = Verbosity.INFO,
                 style: str = 'dark') -> None:
        """Object initialization.

        Args:

            plot_drop (bool):
                Plot each drop during execution of scenarios.

            calc_transmit_spectrum (bool):
                Compute the transmitted signals frequency domain spectra.

            calc_receive_spectrum (bool):
                Compute the received signals frequency domain spectra.

            calc_transmit_stft (bool):
                Compute the short time Fourier transform of transmitted signals.

            calc_receive_stft (bool):
                Compute the short time Fourier transform of received signals.

            spectrum_fft_size (int):
                Number of discrete frequency bins computed within the Fast Fourier Transforms.

            max_num_drops (int):
                Maximum Number of drops per executed scenario.

            results_dir(str, optional):
                Directory in which all execution artifacts will be dropped.

            verbosity (Union[str, Verbosity], optional):
                Information output behaviour during execution.

            style (str. optional):
                Color scheme. Dark by default.
        """

        # Default parameters
        self.__scenarios = []
        self.plot_drop = plot_drop
        self.calc_transmit_spectrum = calc_transmit_spectrum
        self.calc_receive_spectrum = calc_receive_spectrum
        self.calc_transmit_stft = calc_transmit_stft
        self.calc_receive_stft = calc_receive_stft
        self.spectrum_fft_size = spectrum_fft_size
        self.max_num_drops = max_num_drops
        self.results_dir = results_dir
        self.verbosity = verbosity
        self.style = style

    def execute(self) -> None:
        """Execute the executable."""

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
    def max_num_drops(self) -> int:
        """Access number of drops per executed scenario.

        Returns:
            int: Number of drops.
        """

        return self.__max_num_drops

    @max_num_drops.setter
    def max_num_drops(self, num: int) -> None:
        """Modify maximum number of drops per executed scenario.

        Args:
            num (int): New number of drops.

        Raises:
            ValueError: If `num` is smaller than one.
        """

        if num < 1:
            raise ValueError("Number of drops must be greater than zero")

        self .__max_num_drops = num

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
        results_dir = path.join(getcwd(), "results", today + '_' + '{:03d}'.format(dir_index))

        while path.exists(results_dir):

            dir_index += 1
            results_dir = path.join(getcwd(), "results", today + '_' + '{:03d}'.format(dir_index))

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
        """Set the Matplotlib color scheme."""

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

    @contextmanager
    def style_context(self) -> ContextManager:
        """Context for the configured style.

        Returns:
            ContextManager: Style context manager.
        """

        if self.__style in self.__hermes_styles():
            yield plt.style.use(path.join(self.__hermes_root_dir(), 'resources', 'styles', self.__style + '.mplstyle'))

        else:
            yield plt.style.use(self.__style)

    @staticmethod
    def __hermes_root_dir() -> str:
        """HermesPy package root directory.

        Returns:
            str: Path to the package root.
        """

        return path.dirname(path.dirname(path.abspath(__file__)))
