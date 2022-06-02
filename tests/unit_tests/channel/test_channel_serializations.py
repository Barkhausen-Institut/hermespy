import unittest
from unittest.mock import Mock
from typing import Tuple

import numpy as np

from hermespy.channel import (
    Channel, MultipathFadingChannel
)
from hermespy.channel.multipath_fading_templates import (
    MultipathFading5GTDL, MultipathFadingExponential,
    MultipathFadingCost256)
from hermespy.channel.quadriga_channel import QuadrigaChannel
from hermespy.core import Factory
from tests.unit_tests.utils import yaml_str_contains_element

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


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
        self.channel_params['model_type'] = MultipathFading5GTDL.TYPE.A
        ch = MultipathFading5GTDL(**self.channel_params)
        self.serialized_channel_contains_sync_offsets(ch, LOW, HIGH)

    def test_exponential(self) -> None:
        LOW, HIGH = self.add_sync_offsets_to_params()
        self.channel_params['tap_interval'] = 0.1
        self.channel_params['rms_delay'] = 1e-9
        ch = MultipathFadingExponential(**self.channel_params)
        self.serialized_channel_contains_sync_offsets(ch, LOW, HIGH)

    def test_quadriga(self) -> None:
        LOW, HIGH = self.add_sync_offsets_to_params()
        ch = QuadrigaChannel(**self.channel_params)
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