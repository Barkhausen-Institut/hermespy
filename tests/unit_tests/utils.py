# -*- coding: utf-8 -*-

from __future__ import annotations
from os import getenv
from types import TracebackType
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import StemContainer

from hermespy.core import ConsoleMode, MonteCarlo, GridDimension, Verbosity
from hermespy.simulation import Simulation, SimulationScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


def flatten_blocks(bits: List[np.array]) -> np.array:
    flattened_blocks = np.array([])
    for block in bits:
        flattened_blocks = np.append(flattened_blocks, block)
    return flattened_blocks


def assert_frame_equality(data_bits: List[np.array], encoded_bits: List[np.array]) -> None:
    for data_block, encoded_block in zip(data_bits, encoded_bits):
        np.testing.assert_array_equal(data_block, encoded_block)


def yaml_str_contains_element(yaml_str: str, key: float, value: float) -> bool:
    regex = re.compile(f"^\s*{key}: {value}\s*$", re.MULTILINE)

    return re.search(regex, yaml_str) is not None


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


def new_dimension_mock(cls: MonteCarlo, dimension: str, sample_points: List[Any], *args) -> GridDimension:
    _considered_object = cls.investigated_object if len(args) < 1 else args

    # Only take a single sample point into account to speed up simulations
    dimension = GridDimension(_considered_object, dimension, [sample_points[0]])
    cls.add_dimension(dimension)

    return dimension


def subplots_mock(x=1, y=1, *args, squeeze=True, **kwargs) -> Tuple[MagicMock, MagicMock | np.ndarray]:

    figure_mock = MagicMock(spec=plt.Figure)
    figure_mock.canvas = MagicMock(spec=plt.FigureCanvasBase)

    if x == 1 and y == 1 and squeeze:
        axes_mock = MagicMock()
        axes_mock.stem.return_value = MagicMock(spec=StemContainer)

    else:
        axes_mock = np.empty((x, y), dtype=np.object_)
        for x, y in np.ndindex(axes_mock.shape):
            mock_element = MagicMock()
            container_mock = MagicMock(spec=StemContainer)
            container_mock.markerline = MagicMock(spec=plt.Line2D)
            container_mock.stemlines = MagicMock(spec=list)
            mock_element.stem.return_value = container_mock
            mock_element.plot.return_value = [MagicMock(spec=plt.Line2D)]
            axes_mock[x, y] = mock_element

    return figure_mock, axes_mock


class SimulationTestContext(object):
    """Context manager for testing HermesPy simulations.

    Patches stdio and plot functions to suppress output and speed up tests.
    """

    __patch_plot: bool
    __std_patch: Any
    __monte_carlo_patch: Any
    __simulation_patch: Any
    __new_dimension_patch: Any
    __figure_patch: Any

    def __init__(self, *, patch_plot: bool = True) -> None:
        """
        Args:

            patch_plot (bool):
                Whether to patch the plot functions. Defaults to True.
                Ignored when the `HERMES_TEST_PLOT` environment variable is set.
        """

        self.__patch_plot = patch_plot
        patch_env = getenv("HERMES_TEST_PLOT", None)
        if patch_env is not None:
            self.__patch_plot = patch_env.lower() != "true"

        # Initialize context stack of patches
        self.__std_patch = patch("sys.stdout")
        self.__monte_carlo_patch = patch.object(MonteCarlo, "__init__", new=monte_carlo_init_mock)
        self.__simulation_patch = patch.object(Simulation, "__init__", new=simulation_init_mock)
        self.__new_dimension_patch = patch.object(MonteCarlo, "new_dimension", new=new_dimension_mock)
        self.__figure_patch = patch("matplotlib.pyplot.figure")
        self.__subplots_patch = patch("matplotlib.pyplot.subplots", side_effect=subplots_mock)

    def __enter__(self) -> None:
        # Enter context stack of patches if active
        if self.__patch_plot:
            self.__new_dimension_patch.__enter__()
            self.__figure_patch.__enter__()
            self.__subplots_patch.__enter__()
            self.__std_patch.__enter__()

        self.__monte_carlo_patch.__enter__()
        self.__simulation_patch.__enter__()

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None) -> None:
        # Exit context stack of patches if active
        if self.__patch_plot:
            self.__std_patch.__exit__(exc_type, exc_val, exc_tb)
            self.__new_dimension_patch.__exit__(exc_type, exc_val, exc_tb)
            self.__figure_patch.__exit__(exc_type, exc_val, exc_tb)
            self.__subplots_patch.__exit__(exc_type, exc_val, exc_tb)

        self.__monte_carlo_patch.__exit__(exc_type, exc_val, exc_tb)
        self.__simulation_patch.__exit__(exc_type, exc_val, exc_tb)

    @property
    def patch_plot(self) -> bool:
        """Whether the plot functions are patched."""

        return self.__patch_plot
