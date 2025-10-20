# -*- coding: utf-8 -*-

from os import path
from tempfile import TemporaryDirectory
from typing import List
from unittest import TestCase

from numpy.testing import assert_array_almost_equal

from hermespy.simulation import SimulatedDrop, SimulationScenario
from hermespy.modem import TransmittingModem, ReceivingModem, RaisedCosineWaveform
from unit_tests.utils import assert_signals_equal

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestRecordReplay(TestCase):
    """Test recording and replaying of scenario drops"""

    def setUp(self) -> None:
        self.scenario = SimulationScenario()
        self.num_drops = 3
        self.bandwidth = 1e6

        modem_alpha = TransmittingModem()
        modem_alpha.waveform = RaisedCosineWaveform(num_preamble_symbols=0, num_data_symbols=20)

        modem_beta = ReceivingModem()
        modem_beta.waveform = RaisedCosineWaveform(num_preamble_symbols=0, num_data_symbols=20)

        device_alpha = self.scenario.new_device(bandwidth=self.bandwidth)
        device_beta = self.scenario.new_device(bandwidth=self.bandwidth)
        device_alpha.transmitters.add(modem_alpha)
        device_beta.receivers.add(modem_beta)

        self.scenario.channel(device_alpha, device_alpha).gain = 0.0
        self.scenario.channel(device_beta, device_beta).gain = 0.0

        self.tempdir = TemporaryDirectory()
        self.file = path.join(self.tempdir.name, "test.h5")

    def tearDown(self) -> None:
        self.scenario.stop()
        self.tempdir.cleanup()

    def _record(self) -> List[SimulatedDrop]:
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

    def test_record_replay_from_dataset(self) -> None:
        """Test recording and replaying datasets directly from the filesystem"""

        # Record drops
        expected_drops = self._record()

        # Compare the expected and replayed drops to make sure the generated information is identical
        self.scenario.replay(self.file)
        for expected_drop in expected_drops:

            replayed_drop = self.scenario.drop()

            self.assertEqual(expected_drop.timestamp, replayed_drop.timestamp)
            self.assertEqual(expected_drop.num_device_receptions, replayed_drop.num_device_transmissions)
            assert_array_almost_equal(expected_drop.device_receptions[1].operator_receptions[0].equalized_symbols.raw, replayed_drop.device_receptions[1].operator_receptions[0].equalized_symbols.raw)

        self.scenario.stop()

    def test_record_replay_reinitialize(self) -> None:
        """Test recording and reinitializing a scenario from a savefile"""

        # Record drops
        expected_drops = self._record()

        # Initialize scenario from recording and replay drops
        replay_scenario, _ = SimulationScenario.Replay(self.file)
        for expected_drop in expected_drops:

            replayed_drop = replay_scenario.drop()

            try:
                self.assertEqual(expected_drop.timestamp, replayed_drop.timestamp)
                self.assertEqual(expected_drop.num_device_receptions, replayed_drop.num_device_transmissions)

                # Make sure the operator inputs are identical
                assert_signals_equal(self, expected_drop.operator_inputs[1][0], replayed_drop.operator_inputs[1][0])

                # Assert that operators have identical input signals
                assert_signals_equal(self, expected_drop.device_receptions[1].operator_receptions[0].signal, replayed_drop.device_receptions[1].operator_receptions[0].signal)

                # Assert that operators have identical equalized symbols
                assert_array_almost_equal(expected_drop.device_receptions[1].operator_receptions[0].equalized_symbols.raw, replayed_drop.device_receptions[1].operator_receptions[0].equalized_symbols.raw)

            except AssertionError:
                replay_scenario.stop()
                raise

        replay_scenario.stop()
