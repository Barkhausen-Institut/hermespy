# -*- coding: utf-8 -*-

from os import path
from tempfile import TemporaryDirectory
from typing import List
from unittest import TestCase

from numpy.testing import assert_array_almost_equal

from hermespy.core import Drop
from hermespy.simulation import SimulationScenario
from hermespy.modem import TransmittingModem, ReceivingModem, RaisedCosineWaveform


__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRecordReplay(TestCase):
    """Test recording and replaying of scenario drops"""

    def setUp(self) -> None:


        self.scenario = SimulationScenario()
        self.num_drops = 3

        modem_alpha = TransmittingModem()
        modem_alpha.waveform_generator = RaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=20)

        modem_beta = ReceivingModem()
        modem_beta.waveform_generator = RaisedCosineWaveform(symbol_rate=1e6, num_preamble_symbols=0, num_data_symbols=20)

        device_alpha = self.scenario.new_device()
        device_beta = self.scenario.new_device()
        device_alpha.transmitters.add(modem_alpha)
        device_beta.receivers.add(modem_beta)

        self.scenario.channel(device_alpha, device_alpha).gain = 0.
        self.scenario.channel(device_beta, device_beta).gain = 0.

        self.tempdir = TemporaryDirectory()
        self.file = path.join(self.tempdir.name, 'test.h5')

    def tearDown(self) -> None:

        self.scenario.stop()
        self.tempdir.cleanup()

    def _record(self) -> List[Drop]:
        """Record some drops for testing.
        
        Returns: List of recorded drops.
        """

        # Start recording
        self.scenario.record(self.file)

        # Save drops
        expected_drops = [self.scenario.drop() for _ in range(self.num_drops)]

        # Stop recording
        self.scenario.stop()

        # Return generated drops
        return expected_drops

    def test_record_replay(self) -> None:
        """Test recording and replaying of drops"""

        # Record drops
        expected_drops = self._record()

        # Replay drops
        self.scenario.replay(self.file)
        replayed_drops = [self.scenario.drop() for _ in range(self.num_drops)]

        for expected_drop, replayed_drop in zip(expected_drops, replayed_drops):

            self.assertEqual(expected_drop.timestamp, replayed_drop.timestamp)
            self.assertEqual(expected_drop.num_device_receptions, replayed_drop.num_device_transmissions)

    def test_record_replay_from_dataset(self) -> None:
        """Test recording and replaying datasets directly from the filesystem"""
        
        self.scenario.record(self.file)

        expected_drops = [self.scenario.drop() for _ in range(self.num_drops)]
        
        self.scenario.stop()
        self.scenario.replay(self.file)
        
        replay_scenario = SimulationScenario.From_Dataset(self.file)
        replayed_drops = [replay_scenario.drop() for _ in range(self.num_drops)]
        
        # Compare the expected and replayed drops to make sure the generated information is identical
        for expected_drop, replayed_drop in zip(expected_drops, replayed_drops):
            
            self.assertEqual(expected_drop.timestamp, replayed_drop.timestamp)
            self.assertEqual(expected_drop.num_device_receptions, replayed_drop.num_device_transmissions)

    def test_record_replay_reinitialize(self) -> None:
        """Test recording and reinitializing a scenario from a savefile"""

        # Record drops
        expected_drops = self._record()

        # Initialize scenario from recording and replay drops
        replay_scenario = SimulationScenario.Replay(self.file)
        replayed_drops = [replay_scenario.drop() for _ in range(self.num_drops)]

        for expected_drop, replayed_drop in zip(expected_drops, replayed_drops):

            self.assertEqual(expected_drop.timestamp, replayed_drop.timestamp)
            self.assertEqual(expected_drop.num_device_receptions, replayed_drop.num_device_transmissions)
