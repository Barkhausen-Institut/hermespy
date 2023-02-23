# -*- coding: utf-8 -*-
"""Test HermesPy physical device module."""


from os.path import join
from pathlib import Path
from sys import gettrace
from unittest import TestCase
from unittest.mock import patch
from tempfile import TemporaryDirectory

from hermespy.core import ConsoleMode
from hermespy.hardware_loop import HardwareLoop, PhysicalScenarioDummy, PhysicalDeviceDummy
from hermespy.modem import BitErrorEvaluator, DuplexModem, RRCWaveform

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"



class TestHardwareLoop(TestCase):
    """Test the hardware loop pipeline executable class."""

    def setUp(self) -> None:

        self.scenario = PhysicalScenarioDummy()
        self.hardware_loop = HardwareLoop[PhysicalScenarioDummy, PhysicalDeviceDummy](self.scenario)
        self.device = self.hardware_loop.new_device()

    def test_init(self) -> None:
        """Physical device class should be properly initialized"""
        
        self.assertIs(self.scenario, self.hardware_loop.scenario)

    def test_run(self) -> None:
        """Test the run routine"""
        
        # Only output stuff if in a debugging session
        manual_debug = gettrace() is not None
        self.hardware_loop.console_mode = ConsoleMode.INTERACTIVE if manual_debug else ConsoleMode.SILENT
        self.hardware_loop.plot_information = manual_debug
        
        waveform = RRCWaveform(symbol_rate=1e8, oversampling_factor=4, num_preamble_symbols=0, num_data_symbols=20)
        modem = DuplexModem(waveform=waveform)
        modem.device = self.device
        self.hardware_loop.add_evaluator(BitErrorEvaluator(modem ,modem))
        
        self.hardware_loop.new_dimension('carrier_frequency', [0., 1e6, 1e9], self.device)
        
        temp = TemporaryDirectory()

        self.hardware_loop.results_dir = temp.name
        self.hardware_loop.run()    

        # Make sure the loop generated a drops file
        self.assertTrue(Path(join(temp.name, 'drops.h5')).is_file())
        
        temp.cleanup()
