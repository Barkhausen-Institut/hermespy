import unittest
from unittest.mock import Mock
from typing import Dict, Any, Tuple

import numpy as np

from channel import (
    Channel, MultipathFadingChannel
)
from channel.multipath_fading_templates import (
    MultipathFading5GTDL, MultipathFadingExponential,
    MultipathFadingCost256)
from simulator_core import Factory
from tests.unit_tests.utils import yaml_str_contains_element


class SyncOffset(unittest.TestCase):
    def setUp(self) -> None:
        self.channel_params = {
            'transmitter': Mock(),
            'receiver': Mock()
        }
        self.factory = Factory()

    def test_channel_base(self) -> None:
        LOW, HIGH = self.add_sync_offsets_to_params()
        ch = Channel(**self.channel_params)
        self.serialized_channel_contains_sync_offsets(ch, LOW, HIGH)

    def test_multipath_fading(self) -> None:
        self.channel_params['delays'] = np.zeros(1, dtype=float)
        self.channel_params['power_profile'] = np.ones(1, dtype=float)
        self.channel_params['rice_factors'] = np.zeros(1, dtype=float)

        LOW, HIGH = self.add_sync_offsets_to_params()
        ch = MultipathFadingChannel(**self.channel_params)
        self.serialized_channel_contains_sync_offsets(ch, LOW, HIGH)

    def test_cost256(self) -> None:
        LOW, HIGH = self.add_sync_offsets_to_params()
        ch = MultipathFadingCost256(**self.channel_params)
        self.serialized_channel_contains_sync_offsets(ch, LOW, HIGH)

    def test_5gtdl(self) -> None:
        LOW, HIGH = self.add_sync_offsets_to_params()
        ch = MultipathFading5GTDL(**self.channel_params)
        self.serialized_channel_contains_sync_offsets(ch, LOW, HIGH)

    def test_5gtdl(self) -> None:
        LOW, HIGH = self.add_sync_offsets_to_params()
        ch = MultipathFadingExponential(**self.channel_params)
        self.serialized_channel_contains_sync_offsets(ch, LOW, HIGH)

    def add_sync_offsets_to_params(self) -> Tuple[float, float]:
        LOW = 0
        HIGH = 5

        self.channel_params['sync_offset_low'] = LOW
        self.channel_params['sync_offset_high'] = HIGH

        return (LOW, HIGH)

    def serialized_channel_contains_sync_offsets(self, ch: Channel, low: float, high: float) -> bool:
        serialized_ch = self.factory.to_str(ch)
        self.assertTrue(
            yaml_str_contains_element(
                yaml_str=serialized_ch,
                key="sync_offset_low",
                value=low
            )
        )

        self.assertTrue(
            yaml_str_contains_element(
                yaml_str=serialized_ch,
                key="sync_offset_high",
                value=high
            )
        )