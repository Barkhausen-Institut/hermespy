# -*- coding: utf-8 -*-
"""Test HermesPy serialization factory."""

import unittest
from unittest.mock import Mock
import re

import numpy as np
from scenario.scenario import Scenario

from simulator_core import Factory, SerializableClasses
from channel import Channel

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

class TestChannelTimeoffsetScenarioDumping(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = Factory()

    def test_dumping_low_high(self) -> None:
        LOW = 0
        HIGH = 3
        ch = Channel(transmitter=Mock(), receiver=Mock(),
                     active=True, gain=1,
                     sync_offset_low=LOW, sync_offset_high=HIGH)
        serialized_ch = self.factory.to_str(ch)
        self.assertTrue(
            self._yaml_str_contains_sync_offsets(
                yaml_str=serialized_ch,
                sync_offset_low=LOW,
                sync_offset_high=HIGH
            ))

    def test_dumping_default_parameters_are_printed(self) -> None:
        ch = Channel(transmitter=Mock(), receiver=Mock(),
                     active=True, gain=1)
        serialized_ch = self.factory.to_str(ch)
        self.assertTrue(
            self._yaml_str_contains_sync_offsets(
                yaml_str=serialized_ch,
                sync_offset_low=0,
                sync_offset_high=0
            ))

    def _yaml_str_contains_sync_offsets(self, yaml_str: str, 
                                           sync_offset_low: float,
                                           sync_offset_high: float) -> bool:
        regex_low = re.compile(
            f'^sync_offset_low: {sync_offset_low}$',
            re.MULTILINE)
        regex_high = re.compile(
            f'^sync_offset_high: {sync_offset_high}$',
            re.MULTILINE)


        correctly_parsed = (
            re.search(regex_low, yaml_str) is not None and
            re.search(regex_high, yaml_str) is not None
        )
        return correctly_parsed

class TestChannelTimeoffsetScenarioCreation(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario_str = self._create_scenario_stream_header()
        self.scenario_str = self._append_section(self.scenario_str, "Modems")
        self.scenario_str = self._append_random_modem(self.scenario_str, "Transmitter")
        self.scenario_str = self._append_random_modem(self.scenario_str, "Receiver")

        self.scenario_str = self._append_section(self.scenario_str, "Channels")
        self.scenario_str = self._append_channel(self.scenario_str, 0, 0)
        self.factory = Factory()

    def test_setup_single_offset_correct_initialization_with_correct_values(self) -> None:
        LOW = 1
        HIGH = 5
        self.scenario_str = self._append_sync_offset(self.scenario_str, LOW, HIGH)
        scenario = self.factory.from_str(self.scenario_str)

        self.assertEqual(scenario[0].channels[0, 0].sync_offset_low, LOW)
        self.assertEqual(scenario[0].channels[0, 0].sync_offset_high, HIGH)

    def test_no_parameters_result_in_default_values(self) -> None:
        scenario = self.factory.from_str(self.scenario_str)

        self.assertEqual(scenario[0].channels[0, 0].sync_offset_low, 0)
        self.assertEqual(scenario[0].channels[0, 0].sync_offset_high, 0)

    def test_exception_raised_if_high_smaller_than_low(self) -> None:
        LOW = 2
        HIGH = 1

        scenario_str = self._append_sync_offset(self.scenario_str, LOW, HIGH)
        with self.assertRaises(ValueError):
            scenario = self.factory.from_str(scenario_str)

    def test_exception_raised_if_low_smaller_than_zero(self) -> None:
        LOW = -1
        HIGH = 0

        scenario_str = self._append_sync_offset(self.scenario_str, LOW, HIGH)
        with self.assertRaises(ValueError):
            scenario = self.factory.from_str(scenario_str)

    def test_exception_raised_if_high_smaller_than_zero(self) -> None:
        LOW = -1
        HIGH = -5

        scenario_str = self._append_sync_offset(self.scenario_str, LOW, HIGH)
        with self.assertRaises(ValueError):
            scenario = self.factory.from_str(scenario_str)


    def _create_scenario_stream_header(self) -> str:
        return """
!<Scenario>

sampling_rate: 2e6
"""

    def _append_random_modem(self, scenario_stream: str, modem_type: str) -> str:
        if modem_type.upper() not in ["TRANSMITTER", "RECEIVER"]:
            raise ValueError("Modem type not supported")

        scenario_stream += f"""
  - {modem_type}
    carrier_frequency: 1e9
    position: [0, 0, 0]
    WaveformChirpFsk:
        chirp_bandwidth: 5e5
        chirp_duration: 512e-6
        freq_difference: 1953.125
        num_data_chirps: 12
        modulation_order: 32
"""
        return scenario_stream

    def _append_section(self, scenario_stream: str, section: str) -> str:
        return scenario_stream + f"""
{section}:"""

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
