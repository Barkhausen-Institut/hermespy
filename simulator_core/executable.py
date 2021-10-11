# -*- coding: utf-8 -*-
"""HermesPy base for executable configurations."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional

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
    """Abstract base class for executable configurations."""

    yaml_tag = u'Executable'
    __scenarios: List[Scenario]
    plot_drop: bool

    def __init__(self,
                 plot_drop: bool = False) -> None:
        """Object initialization.

        Args:
            plot_drop (bool): Pause to plot each drop during execution.
        """

        # Default parameters
        self.__scenarios = []
        self.plot_drop = plot_drop

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
