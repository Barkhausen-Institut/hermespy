# -*- coding: utf-8 -*-
"""Test HermesPy serialization factory."""

import unittest
from unittest.mock import Mock
from io import StringIO

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
        self.channel_time_offset = 3e-6
        self.scenario_stream = self._create_scenario_stream(self.channel_time_offset)
        self.factory = Factory()

    def test_setup(self) -> None:
        scenario = self.factory.from_stream(self.scenario_stream)
        self.assertAlmostEqual(
              scenario[0].channel_time_offset,
              self.channel_time_offset)

    def _create_scenario_stream(self, time_offset: float) -> StringIO:
        return StringIO(f"""
!<Scenario>

sampling_rate: 2e6
channel_time_offset: {time_offset}
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
  - Channel 0 0
    active: true
""")

if __name__ == '__main__':
    unittest.main()
