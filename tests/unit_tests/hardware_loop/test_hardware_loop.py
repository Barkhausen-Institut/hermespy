# -*- coding: utf-8 -*-
"""Test HermesPy physical device module."""

from contextlib import ExitStack
from io import StringIO
from os import getenv
from os.path import join
from pathlib import Path
from typing import Tuple
from unittest import TestCase
from unittest.mock import MagicMock, Mock, patch
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from hermespy.core import VAT, ConsoleMode, GridDimension, Visualization
from hermespy.hardware_loop.hardware_loop import HardwareLoopSample
from hermespy.modem import BitErrorEvaluator, DuplexModem, RRCWaveform
from hermespy.hardware_loop import DeviceTransmissionPlot, EvaluatorPlotMode, EvaluatorRegistration, HardwareLoop, HardwareLoopPlot, IterationPriority, PhysicalScenarioDummy, PhysicalDeviceDummy
from unit_tests.core.test_factory import test_roundtrip_serialization
from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class HardwareLoopPlotMock(HardwareLoopPlot[Visualization]):
    """Hardware loop plot implementation for testing purposes."""

    @property
    def _default_title(self) -> str:
        return "Hardware Loop Plot Mock"

    def _prepare_plot(self) -> Tuple[plt.Figure, plt.Axes]:
        figure_mock = Mock(spec=plt.Figure)
        figure_mock.canvas = Mock(spec=plt.FigureCanvasBase)
        return (figure_mock, np.array([[Mock(spec=plt.Axes)]], dtype=np.object_))

    def _initial_plot(self, sample: HardwareLoopSample, axes: VAT) -> Visualization:
        return Mock(spec=Visualization)

    def _update_plot(self, sample: HardwareLoopSample, visualization: Visualization) -> None:
        pass


class TestEvaluatorRegistration(TestCase):
    """Test the evaluator registration routines."""

    def setUp(self) -> None:
        modem = DuplexModem()
        self.evaluator = BitErrorEvaluator(modem, modem)
        self.registration = EvaluatorRegistration(self.evaluator, EvaluatorPlotMode.HIDE)

    def test_init(self) -> None:
        """Test the evaluator registration initialization."""

        self.assertIs(self.evaluator, self.registration.evaluator)
        self.assertEqual(EvaluatorPlotMode.HIDE, self.registration.plot_mode)

    def test_evaluator_properties(self) -> None:
        """Test the evaluator properties"""

        self.assertEqual(self.evaluator.abbreviation, self.registration.abbreviation)
        self.assertEqual(self.evaluator.title, self.registration.title)
        self.assertEqual(self.evaluator.confidence, self.registration.confidence)
        self.assertEqual(self.evaluator.tolerance, self.registration.tolerance)

    def test_confidence_setget(self) -> None:
        """Confidence property getter should return setter argument"""

        self.registration.confidence = 0.0
        self.assertEqual(0.0, self.registration.confidence)

    def test_tolerance_setget(self) -> None:
        """Tolerance property getter should return setter argument"""

        self.registration.tolerance = 0.0
        self.assertEqual(0.0, self.registration.tolerance)

    @patch("hermespy.modem.evaluators.BitErrorEvaluator.evaluate")
    def test_evaluate(self, mock_evaluate: MagicMock) -> None:
        """Test the evaluate routine"""

        self.registration.evaluate()
        mock_evaluate.assert_called_once()

    @patch("hermespy.modem.evaluators.BitErrorEvaluator.generate_result")
    def test_generate_result(self, mock_generate: MagicMock) -> None:
        """Test the generate result routine"""

        self.registration.generate_result(Mock(), Mock())
        mock_generate.assert_called_once()


class TestHardwareLoopSample(TestCase):
    """Test the hardware loop sample container class."""

    def setUp(self) -> None:
        self.drop = Mock()
        self.evaluations = [Mock()]
        self.artifacts = [Mock()]

        self.sample = HardwareLoopSample(self.drop, self.evaluations, self.artifacts)

    def test_init(self) -> None:
        """Test the sample initialization"""

        self.assertIs(self.drop, self.sample.drop)
        self.assertIs(self.evaluations, self.sample.evaluations)
        self.assertIs(self.artifacts, self.sample.artifacts)


class TestHardwareLoopPlot(TestCase):
    """Test the hardware loop plot class."""

    def setUp(self) -> None:
        self.title = "Test Plot"
        self.plot = HardwareLoopPlotMock(self.title)
        self.figure, self.axes = self.plot._prepare_plot()

    def test_hardware_loop_valiadtion(self) -> None:
        """Hardware loop property setter should raise RuntimeError if already set"""

        loop = HardwareLoop[PhysicalScenarioDummy, PhysicalDeviceDummy](PhysicalScenarioDummy())
        self.plot.hardware_loop = loop

        with self.assertRaises(RuntimeError):
            self.plot.hardware_loop = loop

    def test_hardware_loop_setget(self) -> None:
        """Hardware loop getter should return setter argument"""

        loop = HardwareLoop[PhysicalScenarioDummy, PhysicalDeviceDummy](PhysicalScenarioDummy())
        self.plot.hardware_loop = loop

        self.assertIs(loop, self.plot.hardware_loop)

    def test_title(self) -> None:
        """Title property should return correct plot title"""

        self.assertEqual(self.title, self.plot.title)

    def test_prepare_plot(self) -> None:
        """Test the prepare plot routine"""

        canvas, axes, figure = self.plot.prepare_plot()
        self.assertIsInstance(canvas, plt.FigureCanvasBase)
        self.assertIsInstance(axes, np.ndarray)
        self.assertIsInstance(figure, plt.Figure)

    def test_update_plot_prepared(self) -> None:
        """Test updateing the plot with a previous call to prepare"""

        loop = HardwareLoop[PhysicalScenarioDummy, PhysicalDeviceDummy](PhysicalScenarioDummy())
        self.plot._update_plot = MagicMock()
        self.plot._initial_plot = MagicMock()
        self.plot.hardware_loop = loop
        self.plot.prepare_plot()

        # First call should call initial plot
        self.plot.update_plot(HardwareLoopSample(Mock(), [], []))
        self.plot._initial_plot.assert_called_once()
        self.plot._update_plot.assert_not_called()

        # Second call should not call initial plot
        self.plot.update_plot(HardwareLoopSample(Mock(), [], []))
        self.plot._initial_plot.assert_called_once()
        self.plot._update_plot.assert_called_once()

    def test_update_plot_unprepared(self) -> None:
        """Updating a plot unprepared should result in a call to prepare plot"""

        loop = HardwareLoop[PhysicalScenarioDummy, PhysicalDeviceDummy](PhysicalScenarioDummy())
        self.plot._prepare_plot = MagicMock(return_value=(self.figure, self.axes))
        self.plot.hardware_loop = loop

        self.plot.update_plot(HardwareLoopSample(Mock(), [], []))
        self.plot._prepare_plot.assert_called_once()


class TestHardwareLoop(TestCase):
    """Test the hardware loop pipeline executable class."""

    def setUp(self) -> None:
        self.scenario = PhysicalScenarioDummy()
        self.hardware_loop = HardwareLoop[PhysicalScenarioDummy, PhysicalDeviceDummy](self.scenario, debug=True)
        self.hardware_loop.plot_information = False  # Disable plotting for testing

    def test_init(self) -> None:
        """Physical device class should be properly initialized"""

        self.assertIs(self.scenario, self.hardware_loop.scenario)

    def test_add_dimension_validation(self) -> None:
        """Add dimension routine should raise ValueError if dimension already registered"""

        grid_dimension = GridDimension(PhysicalDeviceDummy(), "sampling_rate", (1, 2, 3))
        self.hardware_loop.add_dimension(grid_dimension)

        with self.assertRaises(ValueError):
            self.hardware_loop.add_dimension(grid_dimension)

    def test_evaluator_index(self) -> None:
        modem = DuplexModem()
        evaluators = [BitErrorEvaluator(modem, modem) for _ in range(3)]

        for evaluator in evaluators:
            self.hardware_loop.add_evaluator(evaluator)

        for evaluator_index, evaluator in enumerate(evaluators):
            self.assertEqual(evaluator_index, self.hardware_loop.evaluator_index(evaluator))

    def test_run(self) -> None:
        """Test the run routine"""

        # Only output stuff if in a debugging session
        generate_output = getenv("HERMES_TEST_PLOT", "False").lower() == "true"
        if not generate_output:
            self.hardware_loop.console = Console(file=StringIO())

        # Add devices of all plot modes
        device = self.hardware_loop.new_device()

        waveform = RRCWaveform(symbol_rate=1e8, oversampling_factor=4, num_preamble_symbols=0, num_data_symbols=20)
        modem = DuplexModem(waveform=waveform)
        device.transmitters.add(modem)
        device.receivers.add(modem)

        # Add evaluators of all plot modes
        self.hardware_loop.add_evaluator(BitErrorEvaluator(modem, modem))
        self.hardware_loop.new_dimension("carrier_frequency", [0.0, 1e6, 1e9], device)

        # Add a visualizer
        self.hardware_loop.add_plot(DeviceTransmissionPlot(device, "Device Transmission"))

        temp = TemporaryDirectory()

        self.hardware_loop.results_dir = temp.name
        self.hardware_loop.num_drops = 10 if generate_output else 2

        with SimulationTestContext():
            # Make a verbose run
            self.hardware_loop.run()

            # Make a silent run
            self.hardware_loop.console_mode = ConsoleMode.LINEAR
            self.hardware_loop.num_drops = 1
            self.hardware_loop.run()

        # Make sure the loop generated a drops file
        self.assertTrue(Path(join(temp.name, "drops.h5")).is_file())

        temp.cleanup()

    def test_run_no_recording_warning(self) -> None:
        """Hardware loop should warn if drops are not recorded but a directory is set"""

        temp = TemporaryDirectory()
        self.hardware_loop.results_dir = temp.name
        self.hardware_loop.record_drops = False
        console_mock = Mock()
        self.hardware_loop.console = console_mock

        with patch.object(self.hardware_loop, '_HardwareLoop__run'):
            self.hardware_loop.run()

        console_mock.print.assert_called_once()

        temp.cleanup()

    def test_run_grid_priority(self) -> None:
        """Running a hardware loop with grid priority should prioritize grid dimensions"""

        self.hardware_loop.plot_information = False
        self.hardware_loop.num_drops = 1
        self.hardware_loop.console_mode = ConsoleMode.SILENT
        self.hardware_loop.iteration_priority = IterationPriority.GRID

        # Add devices of all plot modes
        device = self.hardware_loop.new_device()
        self.hardware_loop.new_dimension("carrier_frequency", [0.0, 1e6, 1e9], device)

        _ = self.hardware_loop.run()

    def test_run_invald_grid_flag(self) -> None:
        """Run rountine should raise a RuntimeError if an invalid grid flag is set"""

        self.hardware_loop.plot_information = False
        self.hardware_loop.num_drops = 1
        self.hardware_loop.console_mode = ConsoleMode.SILENT
        self.hardware_loop.iteration_priority = 'wrong'

        with self.assertRaises(RuntimeError):
            self.hardware_loop.run()

    def test_run_dummy_serialization(self) -> None:
        """Test the run routine with dummy serialization"""

        generate_output = getenv("HERMES_TEST_PLOT", "False").lower() == "true"
        if not generate_output:
            self.hardware_loop.console = Console(file=StringIO())

        temp = TemporaryDirectory()
        self.hardware_loop.results_dir = temp.name

        # Add devices of all plot modes
        device = self.hardware_loop.new_device()

        waveform = RRCWaveform(symbol_rate=1e8, oversampling_factor=4, num_preamble_symbols=0, num_data_symbols=20)
        modem = DuplexModem(waveform=waveform)
        device.transmitters.add(modem)
        device.receivers.add(modem)

        with ExitStack() as stack:
            isinstance_mock = stack.enter_context(patch("hermespy.hardware_loop.hardware_loop.isinstance"))
            isinstance_mock.return_value = False
            stack.enter_context(patch("matplotlib.pyplot.figure"))

            self.hardware_loop.console_mode = ConsoleMode.LINEAR
            self.hardware_loop.num_drops = 1
            self.hardware_loop.run()

        temp.cleanup()

        self.assertSequenceEqual([device], self.hardware_loop.scenario.devices)
        self.assertCountEqual([modem], self.hardware_loop.scenario.operators)

    def test_record_replay(self) -> None:
        """Test recording a dataset from a hardware loop and replaying it"""

        # Add devices of all plot modes
        device = self.hardware_loop.new_device()
        waveform = RRCWaveform(symbol_rate=1e8, oversampling_factor=4, num_preamble_symbols=0, num_data_symbols=20)
        modem = DuplexModem(waveform=waveform)
        device.transmitters.add(modem)
        device.receivers.add(modem)

        # Add evaluators of all plot modes
        self.hardware_loop.add_evaluator(BitErrorEvaluator(modem, modem))

        # Configure hardware loop
        self.hardware_loop.num_drops = 2
        self.hardware_loop.console_mode = ConsoleMode.SILENT
        self.hardware_loop.plot_information = False

        with TemporaryDirectory() as temp:
            self.hardware_loop.results_dir = temp

            self.hardware_loop.run()
            self.hardware_loop.replay(join(temp, "drops.h5"))

    def test_run_exception_handling(self) -> None:
        """Test exception handling in the run routine"""

        self.hardware_loop.console_mode = ConsoleMode.SILENT
        self.hardware_loop.plot_information = False
        self.hardware_loop.num_drops = 1

        with ExitStack() as stack:

            def raiseError():
                raise RuntimeError()

            add_samples_patch = stack.enter_context(patch("hermespy.core.monte_carlo.GridSection.add_samples"))
            add_samples_patch.side_effect = raiseError

            exception_patch = stack.enter_context(patch.object(self.hardware_loop, "_handle_exception"))

            self.hardware_loop.run()
            exception_patch.assert_called_once()

    def test_add_evaluator(self) -> None:
        """Test the evaluator adding method"""

        modem = DuplexModem()
        evaluator = BitErrorEvaluator(modem, modem)

        self.hardware_loop.add_evaluator(evaluator)

        self.assertEqual(1, self.hardware_loop.num_evaluators)
        self.assertIn(evaluator, self.hardware_loop.evaluators)
