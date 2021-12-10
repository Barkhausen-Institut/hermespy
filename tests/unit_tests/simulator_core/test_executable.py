# -*- coding: utf-8 -*-
"""Test HermesPy base executable."""

import unittest
import tempfile
from contextlib import _GeneratorContextManager
from unittest.mock import Mock, patch

from hermespy.simulator_core import Executable, Verbosity

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ExecutableStub(Executable):
    """Testing instantiation of the abstract Executable prototype, mock implementing all abstract functions."""

    def __init__(self, *args) -> None:
        Executable.__init__(self, *args)

    def run(self) -> None:
        pass


class TestExecutable(unittest.TestCase):
    """Test the base executable prototype, the base class for hermes operations."""

    def setUp(self) -> None:

        self.plot_drop = True
        self.calc_transmit_spectrum = True
        self.calc_receive_spectrum = True
        self.calc_transmit_stft = True
        self.calc_receive_stft = True
        self.spectrum_fft_size = 20
        self.max_num_drops = 1
        self.verbosity = Verbosity.NONE

        with tempfile.TemporaryDirectory() as tempdir:
            self.executable = ExecutableStub(self.plot_drop, self.calc_transmit_spectrum, self.calc_receive_spectrum,
                                             self.calc_transmit_stft, self.calc_receive_stft, self.spectrum_fft_size,
                                             self.max_num_drops, tempdir, self.verbosity)

    def test_init(self) -> None:
        """Executable initialization parameters should be properly stored."""

        self.assertEqual(self.plot_drop, self.executable.plot_drop)
        self.assertEqual(self.calc_transmit_spectrum, self.executable.calc_transmit_spectrum)
        self.assertEqual(self.calc_receive_spectrum, self.executable.calc_receive_spectrum)
        self.assertEqual(self.calc_transmit_stft, self.executable.calc_transmit_stft)
        self.assertEqual(self.calc_receive_stft, self.executable.calc_receive_stft)
        self.assertEqual(self.spectrum_fft_size, self.executable.spectrum_fft_size)
        self.assertEqual(self.max_num_drops, self.executable.max_num_drops)
        self.assertEqual(self.verbosity, self.executable.verbosity)

    def test_execute(self) -> None:
        """Executing the executable should call the run routine."""

        with patch.object(self.executable, 'run') as run:

            self.executable.execute()
            self.assertTrue(run.called)

    def test_add_scenario(self) -> None:
        """Scenario property should return scenarios added by the add_scenario function."""

        scenarios = [Mock() for _ in range(10)]

        for scenario in scenarios:
            self.executable.add_scenario(scenario)

        self.assertCountEqual(scenarios, self.executable.scenarios)

    def test_spectrum_fft_size_setget(self) -> None:
        """Spectrum FFT size property getter should return setter argument."""

        fft_size = 50
        self.executable.spectrum_fft_size = fft_size

        self.assertEqual(fft_size, self.executable.spectrum_fft_size)

    def test_spectrum_fft_size_validation(self) -> None:
        """Spectrum FFT size property setter should raise ValueError on negative arguments,"""

        with self.assertRaises(ValueError):
            self.executable.spectrum_fft_size = -1

        try:
            self.executable.spectrum_fft_size = 0

        except ValueError:
            self.fail("Spectrum FFT size setter should not raise ValueError on zero argument")

    def test_max_num_drops_setget(self) -> None:
        """Number of drops property getter should return setter argument."""

        num_drops = 20
        self.executable.max_num_drops = num_drops

        self.assertEqual(num_drops, self.executable.max_num_drops)

    def test_max_num_drops_validation(self) -> None:
        """Number of drops property setter should raise ValueError on arguments smaller than one."""

        with self.assertRaises(ValueError):
            self.executable.max_num_drops = 0

        with self.assertRaises(ValueError):
            self.executable.max_num_drops = -1

    def test_results_dir_setget(self) -> None:
        """Results directory property getter should return setter argument."""

        with tempfile.TemporaryDirectory() as dirname:

            self.executable.results_dir = dirname
            self.assertEqual(dirname, self.executable.results_dir)

    def test_results_dir_validation(self) -> None:
        """Results directory property setter should throw ValueError on invalid arguments."""

        with tempfile.NamedTemporaryFile() as file:
            with self.assertRaises(ValueError):
                self.executable.results_dir = file.name

        with self.assertRaises(ValueError):
            self.executable.results_dir = "ad213ijt0923h1o2i3hnjqnda"

    def test_verbosity_setget(self) -> None:
        """Verbosity property getter should return setter argument."""

        for verbosity_option in Verbosity:

            self.executable.verbosity = verbosity_option
            self.assertEqual(verbosity_option, self.executable.verbosity)

    def test_verbosity_setget_str(self) -> None:
        """Verbosity property getter should return setter argument for strings."""

        for verbosity_option in Verbosity:

            self.executable.verbosity = verbosity_option.name
            self.assertEqual(verbosity_option, self.executable.verbosity)

    def test_style_setget(self) -> None:
        """Style property getter should return setter argument."""

        style = 'light'
        self.executable.style = style

        self.assertEqual(style, self.executable.style)

    def test_style_validation(self) -> None:
        """Style property setter should raise ValueError on invalid styles."""

        with self.assertRaises(ValueError):
            self.executable.style = "131241251"

    def test_style_context(self) -> None:
        """Style context should return PyPlot style context."""

        self.assertTrue(isinstance(self.executable.style_context(), _GeneratorContextManager))
