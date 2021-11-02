# -*- coding: utf-8 -*-
"""Test HermesPy serialization factory."""

import unittest
from unittest.mock import Mock
from io import StringIO
from typing import List, Union

import numpy as np

from simulator_core import Factory, SerializableClasses

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestFactory(unittest.TestCase):
    """Test the factory responsible to convert config files to executable simulations."""

    def setUp(self) -> None:

        self.factory = Factory()

    def test_clean_set_get(self) -> None:
        """Test that the clean getter returns the setter argument."""

        self.factory.clean = True
        self.assertEqual(self.factory.clean, True, "Clean set/get produced unexpected result")

        self.factory.clean = False
        self.assertEqual(self.factory.clean, False, "Clean set/get produced unexpected result")

    def test_registered_tags(self) -> None:
        """Test the serializable classes registration / discovery mechanism."""

        MockClass = Mock()
        MockClass.yaml_tag = "MockTag"
        MockClass.__name__ = "MockClass"

        SerializableClasses.add(MockClass)
        self.factory.__init__()     # Re-run init to discover new class

        self.assertTrue(MockClass.yaml_tag in self.factory.registered_tags,
                        "Mock class tag not registered as expected for serialization")


class TestChannelTimeoffsetScenarioCreation(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario_stream = self._create_scenario_stream()
        self.scenario_stream = self._append_channel(self.scenario_stream, 0, 0)
        self.factory = Factory()

    def test_setup_single_offset_correct_initialization_with_correct_values(self) -> None:
        LOW = 1
        HIGH = 5
        self.scenario_stream = self._append_sync_offset(self.scenario_stream, LOW, HIGH)
        scenario = self.factory.from_str(self.scenario_stream)

        self.assertEqual(scenario[0].channels[0, 0].sync_offset_low, LOW)
        self.assertEqual(scenario[0].channels[0, 0].sync_offset_high, HIGH)

    def test_no_parameters_result_in_default_values(self) -> None:
        scenario = self.factory.from_str(self.scenario_stream)

        self.assertEqual(scenario[0].channels[0, 0].sync_offset_low, 0)
        self.assertEqual(scenario[0].channels[0, 0].sync_offset_high, 0)

    def test_exception_raised_if_high_smaller_than_low(self) -> None:
        LOW = 2
        HIGH = 1

        scenario_stream = self._append_sync_offset(self.scenario_stream, LOW, HIGH)
        with self.assertRaises(ValueError):
            scenario = self.factory.from_str(scenario_stream)

    def _create_scenario_stream(self) -> str:
        return """
!<Scenario>

sampling_rate: 2e6

Modems:

  - Transmitter
    carrier_frequency: 1e9
    position: [0, 0, 0]
    WaveformChirpFsk:
        chirp_bandwidth: 5e5
        chirp_duration: 512e-6
        freq_difference: 1953.125
        num_data_chirps: 12
        modulation_order: 32

  - Receiver
    carrier_frequency: 1e9
    position: [10, 10, 10]
    WaveformChirpFsk:
        chirp_bandwidth: 5e5
        chirp_duration: 512e-6
        freq_difference: 1953.125
        num_data_chirps: 12
        modulation_order: 32

Channels:
"""

    def _append_channel(self, scenario_stream: str, tx: int, rx: int) -> str:
        return scenario_stream + f"""
  - Channel {tx} {rx}
    active: true

"""
    def _append_sync_offset(self, scenario_stream: str,
                            low: float, high: float) -> str:

        return scenario_stream + f"""
    sync_offset_low: {low}
    sync_offset_high: {high}
"""

if __name__ == '__main__':
    unittest.main()
