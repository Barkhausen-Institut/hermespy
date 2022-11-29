# -*- coding: utf-8 -*-

from os import path
from unittest import TestCase
from tempfile import TemporaryDirectory

from hermespy.core import SNRType
from hermespy.channel import RadarChannel
from hermespy.radar import Radar, FMCW, ReceiverOperatingCharacteristic
from hermespy.simulation import SimulationScenario

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRocFromMeasurements(TestCase):
    
    def setUp(self) -> None:
        
        self.num_drops = 10
        
        self.h0_scenario = SimulationScenario(snr=10., snr_type=SNRType.PN0)
        self.h1_scenario = SimulationScenario(snr=10., snr_type=SNRType.PN0)
        
        h0_device = self.h0_scenario.new_device()
        h1_device = self.h1_scenario.new_device()
        
        h0_operator = Radar()
        h1_operator = Radar()
        h0_operator.waveform = FMCW()
        h1_operator.waveform = FMCW()
        h0_operator.device = h0_device
        h1_operator.device = h1_device
        
        self.h0_channel = RadarChannel((0, h0_operator.waveform.max_range), 0., attenuate=True, transmitter=h0_device, receiver=h0_device)
        self.h1_channel = RadarChannel((0, h1_operator.waveform.max_range), 1., attenuate=False, transmitter=h1_device, receiver=h1_device)
        
    def test_roc_generation(self) -> None:
        
        dir = TemporaryDirectory()
        
        h0_path = path.join(dir.name, 'h0.h5')
        h1_path = path.join(dir.name, 'h1.h5')
    
        self.h0_scenario.record(h0_path)
        self.h1_scenario.record(h1_path)
    
        for _ in range(self.num_drops):
            
            self.h0_scenario.drop()
            self.h1_scenario.drop()
            
        self.h0_scenario.stop()
        self.h1_scenario.stop()
        self.h0_scenario.replay(h0_path)
        self.h1_scenario.replay(h1_path)
        
        roc = ReceiverOperatingCharacteristic.from_scenarios(self.h0_scenario, self.h1_scenario)
        dir.cleanup()
