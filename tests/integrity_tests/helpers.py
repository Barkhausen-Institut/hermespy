# -*- coding: utf-8 -*-

from __future__ import annotations
from os import getenv
from types import TracebackType
from typing import Any, List
from unittest.mock import patch

from hermespy.core import ConsoleMode, MonteCarlo, GridDimension, Verbosity
from hermespy.simulation import Simulation, SimulationScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


monte_carlo_init = MonteCarlo.__init__
simulation_init = Simulation.__init__


def monte_carlo_init_mock(cls: MonteCarlo, *args, **kwargs) -> None:
    args = list(args)
    args[1] = 1
    kwargs["num_actors"] = 1  # Only spawn a single actor
    kwargs["catch_exceptions"] = False  # Don't catch exceptions during runtime
    kwargs["min_num_samples"] = 1  # Only generate a single sample

    monte_carlo_init(cls, *args, **kwargs)


def simulation_init_mock(self: Simulation, scenario: None | SimulationScenario = None, num_samples: int = 100, drop_duration: float = 0.0, plot_results: bool = False, dump_results: bool = True, console_mode: ConsoleMode = ConsoleMode.INTERACTIVE, ray_address=None, results_dir=None, verbosity=Verbosity.INFO, seed=None, num_actors=None) -> None:
    num_samples = 1
    simulation_init(self, scenario, num_samples, drop_duration, plot_results, dump_results, console_mode, ray_address, results_dir, verbosity, seed, num_actors)


def new_dimension_mock(cls: MonteCarlo, dimension: str, sample_points: List[Any], considered_object: Any | None = None) -> GridDimension:
    _considered_object = cls.investigated_object if considered_object is None else considered_object

    # Only take a single sample point into account to speed up simulations
    dimension = GridDimension(_considered_object, dimension, [sample_points[0]])
    cls.add_dimension(dimension)

    return dimension


class SimulationTestContext(object):
    __plot: bool

    __std_patch: Any
    __monte_carlo_patch: Any
    __simulation_patch: Any
    __new_dimension_patch: Any
    __figure_patch: Any

    def __init__(self) -> None:
        self.__patch_plot = getenv("HERMES_TEST_PLOT", "False").lower() != "true"

        # Initialize context stack of patches
        self.__std_patch = patch("sys.stdout")
        self.__monte_carlo_patch = patch.object(MonteCarlo, "__init__", new=monte_carlo_init_mock)
        self.__simulation_patch = patch.object(Simulation, "__init__", new=simulation_init_mock)
        self.__new_dimension_patch = patch.object(MonteCarlo, "new_dimension", new=new_dimension_mock)
        self.__figure_patch = patch("matplotlib.pyplot.figure")

    def __enter__(self) -> None:
        # Enter context stack of patches if active
        if self.__patch_plot:
            self.__new_dimension_patch.__enter__()
            self.__figure_patch.__enter__()
            self.__std_patch.__enter__()

        self.__monte_carlo_patch.__enter__()
        self.__simulation_patch.__enter__()

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        # Exit context stack of patches if active
        if self.__patch_plot:
            self.__std_patch.__exit__(exc_type, exc_val, exc_tb)
            self.__new_dimension_patch.__exit__(exc_type, exc_val, exc_tb)
            self.__figure_patch.__exit__(exc_type, exc_val, exc_tb)

        self.__monte_carlo_patch.__exit__(exc_type, exc_val, exc_tb)
        self.__simulation_patch.__exit__(exc_type, exc_val, exc_tb)
