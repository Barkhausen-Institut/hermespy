# -*- coding: utf-8 -*-

from contextlib import ExitStack
from os import getenv
from typing import Type
from unittest import TestCase
from unittest.mock import MagicMock, patch

from hermespy.hardware_loop import DeviceReceptionPlot, DeviceTransmissionPlot, EyePlot, ReceivedConstellationPlot, RadarRangePlot, EvaluationPlot, ArtifactPlot, PhysicalDeviceDummy, HardwareLoopPlot, PhysicalScenarioDummy, HardwareLoop
from hermespy.hardware_loop.visualizers import SignalPlot, HardwareLoopDevicePlot
from hermespy.hardware_loop.hardware_loop import HardwareLoopSample
from hermespy.modem import DuplexModem, RRCWaveform, BitErrorEvaluator
from hermespy.radar import Radar, FMCW

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


VISUALIZE_PLOTS = getenv('HERMES_TEST_PLOT', 'False').lower() == 'true'


class HardwareLoopPlotTest(TestCase):
    """Test the hardware loop plot"""
    
    def setUp(self) -> None:
        
        self.scenario = PhysicalScenarioDummy()
        self.loop = HardwareLoop[PhysicalDeviceDummy, PhysicalScenarioDummy](self.scenario)
        self.device = self.scenario.new_device()

    def _prepare_plot(self, plot: Type[HardwareLoopPlot], *args, **kwargs) -> None:
        
        with ExitStack() as stack:
            
            if not VISUALIZE_PLOTS:

                subplots_patch = stack.enter_context(patch('matplotlib.pyplot.subplots'))
                subplots_patch.return_value = MagicMock(), MagicMock()
            
            self.plot = plot(*args, **kwargs)            
            self.loop.add_plot(self.plot)
            
            self.figure, self.axes = self.plot.prepare_figure()

    def _test_update_plot(self) -> None:
        """Subroutine for testing the plotting behaviour"""
        
        # Generate a new drop (triggers the hardware)
        drop = self.scenario.drop()
        
        # Generate evaluations and artifacts
        evaluations = [e.evaluate() for e in self.loop.evaluators]
        artifacts = [e.artifact() for e in evaluations]
        
        # Compute sample
        sample = HardwareLoopSample(drop, evaluations, artifacts)

        # Update the plot
        if VISUALIZE_PLOTS:
            self.plot.update_plot(sample)

        else:
            call_count = len(self.axes.mock_calls)
            self.plot.update_plot(sample)
            self.assertGreater(len(self.axes.mock_calls), call_count)


class TestDeviceReceptionPlot(HardwareLoopPlotTest, TestCase):
    """Test the device reception plot"""

    def setUp(self) -> None:
        
        HardwareLoopPlotTest.setUp(self)
        self._prepare_plot(DeviceReceptionPlot, self.device)

    def test_update_plot(self) -> None:
        """Test the update plot routine"""
        
        self._test_update_plot()


class TestDeviceTransmissionPlot(HardwareLoopPlotTest, TestCase):
    """Test the device transmission plot"""

    def setUp(self) -> None:
        
        HardwareLoopPlotTest.setUp(self)
        self._prepare_plot(DeviceTransmissionPlot, self.device)

    def test_update_plot(self) -> None:
        """Test the update plot routine"""
        
        self._test_update_plot()


class TestEyePlot(HardwareLoopPlotTest, TestCase):
    """Test the eye plot"""

    def setUp(self) -> None:
        
        HardwareLoopPlotTest.setUp(self)
        
        self.modem = DuplexModem()
        self.modem.device = self.device
        self.modem.waveform_generator = RRCWaveform(oversampling_factor=4, symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=10)
        
        self._prepare_plot(EyePlot, self.modem)

    def test_update_plot(self) -> None:
        """Test the update plot routine"""
        
        self._test_update_plot()
    
    
class TestReceivedConstellationPlot(HardwareLoopPlotTest, TestCase):
    """Test the received constellation plot"""

    def setUp(self) -> None:
        
        HardwareLoopPlotTest.setUp(self)
        
        self.modem = DuplexModem()
        self.modem.device = self.device
        self.modem.waveform_generator = RRCWaveform(oversampling_factor=4, symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=10)
        
        self._prepare_plot(ReceivedConstellationPlot, self.modem)

    def test_update_plot(self) -> None:
        """Test the update plot routine"""
        
        self._test_update_plot()


class TestRadarRangePlot(HardwareLoopPlotTest, TestCase):
    """Test the radar range plot"""

    def setUp(self) -> None:
        
        HardwareLoopPlotTest.setUp(self)
        
        self.radar = Radar()
        self.radar.device = self.device
        self.radar.waveform = FMCW()
        
        self._prepare_plot(RadarRangePlot, self.radar)

    def test_update_plot(self) -> None:
        """Test the update plot routine"""
        
        self._test_update_plot()
        
        
class TestEvaluationPlot(HardwareLoopPlotTest, TestCase):
    """Test the evaluator plot"""
    
    def setUp(self) -> None:
        
        HardwareLoopPlotTest.setUp(self)
        
        self.modem = DuplexModem()
        self.modem.device = self.device
        self.modem.waveform_generator = RRCWaveform(oversampling_factor=4, symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=10)
         
        self.evaluator = BitErrorEvaluator(self.modem, self.modem)
        self.loop.add_evaluator(self.evaluator)

        self._prepare_plot(EvaluationPlot, self.evaluator)
        
    def test_update_plot(self) -> None:
        """Test the update plot routine"""
        
        self._test_update_plot()


class TestArtifactPlot(HardwareLoopPlotTest, TestCase):
    """Test the artifact plot"""
    
    def setUp(self) -> None:
        
        HardwareLoopPlotTest.setUp(self)
        
        self.modem = DuplexModem()
        self.modem.device = self.device
        self.modem.waveform_generator = RRCWaveform(oversampling_factor=4, symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=10)
         
        self.evaluator = BitErrorEvaluator(self.modem, self.modem)
        self.loop.add_evaluator(self.evaluator)

        self._prepare_plot(ArtifactPlot, self.evaluator)
        
    def test_update_plot(self) -> None:
        """Test the update plot routine"""
        
        self._test_update_plot()
