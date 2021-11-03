# -*- coding: utf-8 -*-
"""Test HermesPy serialization factory."""

import unittest
from unittest.mock import Mock
import re

from ruamel import yaml

from simulator_core import Factory, SerializableClasses
from modem.rf_chain import RfChain

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

class TestIqImbalanceCreationAndSerialization(unittest.TestCase):
    def setUp(self) -> None:
        self.factory = Factory()

    def test_creation_proper_values(self) -> None:
        AMPLITUDE_IMBALANCE = 0.5
        PHASE_OFFSET = 3
        yaml_str = f"""
!<Scenario>

Modems:
  - Transmitter
    RfChain:
       amplitude_imbalance: {AMPLITUDE_IMBALANCE}
       phase_offset: {PHASE_OFFSET}
"""
        scenarios = self.factory.from_str(yaml_str)
        self.assertAlmostEqual(
            scenarios[0].transmitters[0].rf_chain.amplitude_imbalance,
            AMPLITUDE_IMBALANCE
        )
        self.assertEqual(
            scenarios[0].transmitters[0].rf_chain.phase_offset,
            PHASE_OFFSET
        )

    def test_iq_imbalance_serialisation(self) -> None:
        PHASE_OFFSET = 10
        AMPLITUDE_IMBALANCE = 0.5
        rf_chain = RfChain(phase_offset=PHASE_OFFSET,
                           amplitude_imbalance=AMPLITUDE_IMBALANCE)

        serialized_rf_chain = self.factory.to_str(rf_chain)
        phase_offset_regex = re.compile(
            f'^phase_offset: {PHASE_OFFSET}$', re.MULTILINE)
        amplitude_imbalance_regex = re.compile(
            f'^amplitude_imbalance: {AMPLITUDE_IMBALANCE}$',
            re.MULTILINE)

        self.assertTrue(re.search(phase_offset_regex, serialized_rf_chain) is not None)
        self.assertTrue(re.search(amplitude_imbalance_regex, serialized_rf_chain) is not None)