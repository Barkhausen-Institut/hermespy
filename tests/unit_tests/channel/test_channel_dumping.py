import unittest
from typing import List

import numpy as np

from hermespy.channel.channel import Channel
from hermespy.core.factory import Factory

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


def create_simulation_stream_header() -> str:
    return """"""


def create_section_yaml_str(section: str) -> str:
    return f"""
{section}:"""


def create_channel_yaml_str() -> str:
    return f"""
  - !<Channel>
    active: true

"""


def create_sync_offset_yaml_str(low: float, high: float) -> str:

    return f"""
    sync_offset_low: {low}
    sync_offset_high: {high}
"""


class TestChannelTimeOffsetScenarioCreation(unittest.TestCase):

    def setUp(self) -> None:

        self.scenario_str = ""

        self.scenario_str += create_section_yaml_str("Channels")
        self.scenario_str += create_channel_yaml_str()
        self.factory = Factory()

    def test_setup_single_offset_correct_initialization_with_correct_values(self) -> None:
        
        LOW = 1
        HIGH = 5
        self.scenario_str += create_sync_offset_yaml_str(LOW, HIGH)
        scenario = self.factory.from_str(self.scenario_str)
        channel = scenario['Channels'][0]

        self.assertEqual(channel.sync_offset_low, LOW)
        self.assertEqual(channel.sync_offset_high, HIGH)

    def test_no_parameters_result_in_default_values(self) -> None:

        scenario = self.factory.from_str(self.scenario_str)

        channel = scenario['Channels'][0]
        self.assertEqual(channel.sync_offset_low, 0)
        self.assertEqual(channel.sync_offset_high, 0)

    def test_multiple_channels_creation_all_sync_offsets_defined(self) -> None:
        
        sync_offsets = {
            'ch0_0': {'LOW': 0, 'HIGH': 3},
            'ch1_0': {'LOW': 2, 'HIGH': 5},
            'ch0_1': {'LOW': 1, 'HIGH': 4},
            'ch1_1': {'LOW': 5, 'HIGH': 10}
        }

        s = create_simulation_stream_header()

        s += create_section_yaml_str("Channels")
        s += create_channel_yaml_str()
        s += create_sync_offset_yaml_str(low=sync_offsets['ch0_0']['LOW'], high=sync_offsets['ch0_0']['HIGH'])
        s += create_channel_yaml_str()
        s += create_sync_offset_yaml_str(low=sync_offsets['ch1_0']['LOW'], high=sync_offsets['ch1_0']['HIGH'])
        s += create_channel_yaml_str()
        s += create_sync_offset_yaml_str(low=sync_offsets['ch0_1']['LOW'], high=sync_offsets['ch0_1']['HIGH'])
        s += create_channel_yaml_str()
        s += create_sync_offset_yaml_str(low=sync_offsets['ch1_1']['LOW'], high=sync_offsets['ch1_1']['HIGH'])

        channels: List[Channel] = self.factory.from_str(s)['Channels']

        for sync, ch in zip(sync_offsets.values(), channels):
            
            self.assertEqual(ch.sync_offset_low, sync['LOW'])
            self.assertEqual(ch.sync_offset_high, sync['HIGH'])
