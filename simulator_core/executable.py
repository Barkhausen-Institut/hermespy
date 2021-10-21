# -*- coding: utf-8 -*-
"""HermesPy base for executable configurations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
from enum import Enum

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
    """

    yaml_tag = u'Executable'
    __scenarios: List[Scenario]
    plot_drop: bool
    calc_transmit_spectrum: bool
    calc_receive_spectrum: bool
    calc_transmit_stft: bool
    calc_receive_stft: bool

    def __init__(self,
                 plot_drop: bool = False,
                 calc_transmit_spectrum: bool = False,
                 calc_receive_spectrum: bool = False,
                 calc_transmit_stft: bool = False,
                 calc_receive_stft: bool = False) -> None:
        """Object initialization.

        Args:
            plot_drop (bool): Plot each drop during execution of scenarios.
            calc_transmit_spectrum (bool): Compute the transmitted signals frequency domain spectra.
            calc_receive_spectrum (bool): Compute the received signals frequency domain spectra.
            calc_transmit_stft (bool): Compute the short time Fourier transform of transmitted signals.
            calc_receive_stft (bool): Compute the short time Fourier transform of received signals.
        """

        # Default parameters
        self.__scenarios = []
        self.plot_drop = plot_drop
        self.calc_transmit_spectrum = calc_transmit_spectrum
        self.calc_receive_spectrum = calc_receive_spectrum
        self.calc_transmit_stft = calc_transmit_stft
        self.calc_receive_stft = calc_receive_stft

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
