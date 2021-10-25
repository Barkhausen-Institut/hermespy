# -*- coding: utf-8 -*-
"""HermesPy base for executable configurations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
from os import getcwd, mkdir
import os.path as path
import datetime

from scenario import Scenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Executable(ABC):
    """Abstract base class for executable configurations.

    Attributes:
        plot_drop (bool): Plot each drop during execution of scenarios.
        calc_transmit_spectrum (bool): Compute the transmitted signals frequency domain spectra.
        calc_receive_spectrum (bool): Compute the received signals frequency domain spectra.
        calc_transmit_stft (bool): Compute the short time Fourier transform of transmitted signals.
        calc_receive_stft (bool): Compute the short time Fourier transform of received signals.
        __spectrum_fft_size (int): Number of FFT bins considered during computation.
        __num_drops (int): Number of executions per scenario.
        __results_dir (str): Directory in which execution results will be saved.
    """

    yaml_tag = u'Executable'
    __scenarios: List[Scenario]
    plot_drop: bool
    calc_transmit_spectrum: bool
    calc_receive_spectrum: bool
    calc_transmit_stft: bool
    calc_receive_stft: bool
    __spectrum_fft_size: int
    __num_drops: int
    __results_dir: str

    def __init__(self,
                 plot_drop: bool = False,
                 calc_transmit_spectrum: bool = False,
                 calc_receive_spectrum: bool = False,
                 calc_transmit_stft: bool = False,
                 calc_receive_stft: bool = False,
                 spectrum_fft_size: int = 0,
                 num_drops: int = 1,
                 results_dir: Optional[str] = None) -> None:
        """Object initialization.

        Args:
            plot_drop (bool): Plot each drop during execution of scenarios.
            calc_transmit_spectrum (bool): Compute the transmitted signals frequency domain spectra.
            calc_receive_spectrum (bool): Compute the received signals frequency domain spectra.
            calc_transmit_stft (bool): Compute the short time Fourier transform of transmitted signals.
            calc_receive_stft (bool): Compute the short time Fourier transform of received signals.
            spectrum_fft_size (int): Number of discrete frequency bins computed within the Fast Fourier Transforms.
            num_drops (int): Number of drops per executed scenario.
        """

        # Default parameters
        self.__scenarios = []
        self.plot_drop = plot_drop
        self.calc_transmit_spectrum = calc_transmit_spectrum
        self.calc_receive_spectrum = calc_receive_spectrum
        self.calc_transmit_stft = calc_transmit_stft
        self.calc_receive_stft = calc_receive_stft
        self.spectrum_fft_size = spectrum_fft_size
        self.num_drops = num_drops

        if results_dir is None:
            results_dir = Executable.__default_results_dir()

        self.results_dir = results_dir

    @abstractmethod
    def run(self) -> None:
        """Execute the configuration."""
        ...

    def add_scenario(self, scenario: Scenario) -> None:
        """Add a new scenario description to this executable.

        Args:
            scenario (Scenario): The scenario description to be added.
        """

        self.__scenarios.append(scenario)

    @property
    def scenarios(self) -> List[Scenario]:
        """Access scenarios within this executable.

        Returns:
            List[Scenario]: Scenarios within this executable.
        """

        return self.__scenarios

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
    def num_drops(self) -> int:
        """Access number of drops per executed scenario.

        Returns:
            int: Number of drops.
        """

        return self.__num_drops

    @num_drops.setter
    def num_drops(self, num: int) -> None:
        """Modify number of drops per executed scenario.

        Args:
            num (int): New number of drops.

        Raises:
            ValueError: If `num` is smaller than one.
        """

        if num < 1:
            raise ValueError("Number of drops must be greater than zero")

        self .__num_drops = num

    @property
    def results_dir(self) -> str:
        """Directory in which the execution results will be saved.

        Returns:
            str: The directory.
        """

        return self.__results_dir

    @results_dir.setter
    def results_dir(self, directory: str) -> None:
        """Modify the directory in which the execution results will be saved.

        Args:
            directory (str): New directory.

        Raises:
            ValueError: If `directory` does not exist within the filesystem.
        """

        if not path.exists(directory):
            raise ValueError("The provided results directory does not exist")

        if not path.isdir(directory):
            raise ValueError("The provided results directory path is not a directory")

        self.__results_dir = directory

    @staticmethod
    def __default_results_dir() -> str:
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
