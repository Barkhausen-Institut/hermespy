# -*- coding: utf-8 -*-
"""Test HermesPy serialization factory."""

import unittest
from unittest.mock import Mock

from hermespy.channel import Channel, MultipathFadingChannel
from hermespy.core.factory import Factory


__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "3.0.0"
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

    def test_registered_classes(self) -> None:
        """Registered classes should contain all serializable classes."""

        expected_classes = [Channel, MultipathFadingChannel]
        registered_classes = self.factory.registered_classes

        for expected_class in expected_classes:
            self.assertTrue(expected_class in registered_classes)

    def test_registered_tags(self) -> None:
        """Test the serializable classes registration / discovery mechanism."""

        expected_tags = [u'Channel', u'MultipathFading']
        registered_tags = self.factory.registered_tags

        for expected_tag in expected_tags:
            self.assertTrue(expected_tag in registered_tags)
