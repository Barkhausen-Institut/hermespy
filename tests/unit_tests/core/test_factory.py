# -*- coding: utf-8 -*-
"""Test HermesPy serialization factory."""

import unittest
from unittest.mock import Mock

from hermespy.core.factory import Factory

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
