# -*- coding: utf-8 -*-
"""HermesPy base for executable configurations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

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

    def __init__(self) -> None:
        """Object initialization."""

        self.__scenarios = []

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
