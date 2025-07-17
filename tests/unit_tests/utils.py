# -*- coding: utf-8 -*-

from __future__ import annotations
from os import getenv
from types import TracebackType
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch
from unittest import TestCase
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.container import StemContainer
from mpl_toolkits.mplot3d.art3d import Line3D
from numpy.testing import assert_array_almost_equal

from hermespy.core import ConsoleMode, MonteCarlo, GridDimension, Signal, Verbosity
from hermespy.simulation import Simulation, SimulationScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
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
    args[1] = 1  # Only generate a single sample
    kwargs["num_actors"] = 3  # Only spawn a few actors to speed up tests
    kwargs["catch_exceptions"] = False  # Don't catch exceptions during runtime

    monte_carlo_init(cls, *args, **kwargs)


def simulation_init_mock(self: Simulation, scenario: None | SimulationScenario = None, num_samples: int = 100, drop_duration: float = 0.0, drop_interval: float = float('inf'), plot_results: bool = False, dump_results: bool = True, console_mode: ConsoleMode = ConsoleMode.INTERACTIVE, ray_address=None, results_dir=None, verbosity=Verbosity.INFO, seed=None, num_actors=None) -> None:
    num_samples = 1
    drop_duration = float('inf')
    simulation_init(self, scenario, num_samples, drop_duration, drop_interval, plot_results, dump_results, console_mode, ray_address, results_dir, verbosity, seed, num_actors)


def new_dimension_mock(cls: MonteCarlo, dimension: str, sample_points: List[Any], *args, **kwargs) -> GridDimension:
    _considered_object = cls.investigated_object if len(args) < 1 else args

    # Only take a single sample point into account to speed up simulations
    dimension = GridDimension(_considered_object, dimension, [sample_points[0]], **kwargs)
    cls.add_dimension(dimension)

    return dimension


def subplots_mock(x=1, y=1, *args, squeeze=True, **kwargs) -> Tuple[MagicMock, MagicMock | np.ndarray]:

    figure_mock = MagicMock(spec=plt.Figure)
    figure_mock.canvas = MagicMock(spec=plt.FigureCanvasBase)

    # Detect 3D mode
    mode_3d = False
    if "subplot_kw" in kwargs:
        if "projection" in kwargs["subplot_kw"]:
            mode_3d = True

    line_spec = Line3D if mode_3d else plt.Line2D

    if x == 1 and y == 1 and squeeze:
        axes_mock = MagicMock()
        axes_mock.stem.return_value = MagicMock(spec=StemContainer)

    else:
        axes_mock = np.empty((x, y), dtype=np.object_)
        for x, y in np.ndindex(axes_mock.shape):
            mock_element = MagicMock()
            container_mock = MagicMock(spec=StemContainer)
            container_mock.markerline = MagicMock(spec=line_spec)
            container_mock.stemlines = MagicMock(spec=list)
            mock_element.stem.return_value = container_mock  
            plot_lines_mock = MagicMock(spec=line_spec)
            plot_lines_mock.set_3d_properties = MagicMock()
            mock_element.plot.return_value = [plot_lines_mock]
            mock_element.semilogy.return_value = [MagicMock(spec=line_spec)]
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

def assert_signals_equal(test: TestCase, expected_signal: Signal, actual_signal: Signal) -> None:
    
    test.assertEqual(expected_signal.sampling_rate, actual_signal.sampling_rate, msg="Sampling rate mismatch")
    test.assertEqual(expected_signal.num_samples, actual_signal.num_samples, msg="Number of samples mismatch")
    test.assertEqual(expected_signal.num_streams, actual_signal.num_streams, msg="Number of streams mismatch")
    test.assertEqual(len(expected_signal), len(actual_signal), msg="Number of blovks mismatch")
    
    for expected_block, actual_block in zip(expected_signal, actual_signal):
        assert_array_almost_equal(expected_block, actual_block)
