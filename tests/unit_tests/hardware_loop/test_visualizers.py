# -*- coding: utf-8 -*-

from typing import Type
from unittest import TestCase
from unittest.mock import MagicMock

import matplotlib.pyplot as plt

from hermespy.core import Executable
from hermespy.hardware_loop import DeviceReceptionPlot, DeviceTransmissionPlot, EyePlot, ReceivedConstellationPlot, RadarRangePlot, EvaluationPlot, ArtifactPlot, PhysicalDeviceDummy, HardwareLoopPlot, PhysicalScenarioDummy, HardwareLoop
from hermespy.hardware_loop.hardware_loop import HardwareLoopSample
from hermespy.modem import DuplexModem, RRCWaveform, BitErrorEvaluator
from hermespy.radar import Radar, FMCW

from unit_tests.utils import SimulationTestContext

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class _HardwareLoopPlotTest(TestCase):
    """Test the hardware loop plot"""

    def setUp(self) -> None:
        self.scenario = PhysicalScenarioDummy()
        self.loop = HardwareLoop[PhysicalDeviceDummy, PhysicalScenarioDummy](self.scenario)
        self.device = self.scenario.new_device()

    def _prepare_plot(self, plot: Type[HardwareLoopPlot], *args, **kwargs) -> None:

        self.plot = plot(*args, **kwargs)
        self.loop.add_plot(self.plot)

        self._test_context = SimulationTestContext(patch_plot=True)
        with self._test_context, Executable.style_context():
            self.figure, self.axes = self.plot.prepare_plot()

    def test_visualization(self) -> None:
        """Test the visualization of the hardware loop plot"""

        # Generate a new drop (triggers the hardware)
        drop = self.scenario.drop()

        # Generate evaluations and artifacts
        evaluations = [e.evaluate() for e in self.loop.evaluators]
        artifacts = [e.artifact() for e in evaluations]

        # Compute sample
        sample = HardwareLoopSample(drop, evaluations, artifacts)

        # Only test for a call to the axes if the plot routines are patched
        if self._test_context.patch_plot:
            
            # Initial plot
            call_count = len(self.axes[0, 0].mock_calls)
            self.plot.update_plot(sample)
            self.assertGreater(len(self.axes[0, 0].mock_calls), call_count)

            # Updated plot
            self.plot.update_plot(sample)

        # Otherwise, simply plot the sample for debugging purposes
        else:
            self.plot.update_plot(sample)
            plt.show()

        return


class TestDeviceReceptionPlot(_HardwareLoopPlotTest, TestCase):
    """Test the device reception plot"""

    def setUp(self) -> None:
        super().setUp()
        self._prepare_plot(DeviceReceptionPlot, self.device)


class TestDeviceTransmissionPlot(_HardwareLoopPlotTest, TestCase):
    """Test the device transmission plot"""

    def setUp(self) -> None:
        super().setUp()
        self._prepare_plot(DeviceTransmissionPlot, self.device)


class TestEyePlot(_HardwareLoopPlotTest, TestCase):
    """Test the eye plot"""

    def setUp(self) -> None:
        super().setUp()

        self.modem = DuplexModem()
        self.modem.device = self.device
        self.modem.waveform = RRCWaveform(oversampling_factor=4, symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=50)

        self._prepare_plot(EyePlot, self.modem)
        
    def test_initial_plot_validation(self) -> None:
        """Initial plot should raise a RuntimeError if no synchronized frame is available"""
        
        plot = EyePlot(DuplexModem())
        self.loop.add_plot(plot)
        with self.assertRaises(RuntimeError):
            with SimulationTestContext():
                plot.update_plot(MagicMock(spec=HardwareLoopSample))


class TestReceivedConstellationPlot(_HardwareLoopPlotTest, TestCase):
    """Test the received constellation plot"""

    def setUp(self) -> None:
        super().setUp()

        self.modem = DuplexModem()
        self.modem.device = self.device
        self.modem.waveform = RRCWaveform(oversampling_factor=4, symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=10)

        self._prepare_plot(ReceivedConstellationPlot, self.modem)


class TestRadarRangePlot(_HardwareLoopPlotTest, TestCase):
    """Test the radar range plot"""

    def setUp(self) -> None:
        super().setUp()

        self.radar = Radar()
        self.radar.device = self.device
        self.radar.waveform = FMCW()

        self._prepare_plot(RadarRangePlot, self.radar)
        
    def test_update_plot_validation(self) -> None:
        """Updating the plot should raise a RuntimeError if no cube is available"""
        
        radar = MagicMock(spec=Radar)
        radar.reception = None
        
        with SimulationTestContext():
            with self.assertRaises(RuntimeError):
                self.plot.update_plot(MagicMock(spec=HardwareLoopSample))


class TestEvaluationPlot(_HardwareLoopPlotTest, TestCase):
    """Test the evaluator plot"""

    def setUp(self) -> None:
        super().setUp()

        self.modem = DuplexModem()
        self.modem.device = self.device
        self.modem.waveform = RRCWaveform(oversampling_factor=4, symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=10)

        self.evaluator = BitErrorEvaluator(self.modem, self.modem)
        self.loop.add_evaluator(self.evaluator)

        self._prepare_plot(EvaluationPlot, self.evaluator)


class TestArtifactPlot(_HardwareLoopPlotTest, TestCase):
    """Test the artifact plot"""

    def setUp(self) -> None:
        super().setUp()

        self.modem = DuplexModem()
        self.modem.device = self.device
        self.modem.waveform = RRCWaveform(oversampling_factor=4, symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=10)

        self.evaluator = BitErrorEvaluator(self.modem, self.modem)
        self.loop.add_evaluator(self.evaluator)

        self._prepare_plot(ArtifactPlot, self.evaluator)


del _HardwareLoopPlotTest
