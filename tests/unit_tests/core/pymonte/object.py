# -*- coding: utf-8 -*-

from hermespy.core.pymonte.grid import register

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class TestObjectMock(object):
    """Mock of a tested object"""

    def __init__(self) -> None:
        self.property_a = None
        self.property_b = 0
        self.property_c = 0

    @register(first_impact="init_stage", last_impact="exit_stage")
    @property
    def property_a(self):
        return self.__property_a

    @property_a.setter
    def property_a(self, value) -> None:
        self.__property_a = value

    @property
    def property_b(self):
        return self.__property_b

    @property_b.setter
    def property_b(self, value):
        self.__property_b = value

    @property
    def property_c(self):
        return self.__property_c

    @property_c.setter
    def property_c(self, value):
        self.__property_c = value

    def some_operation(self):
        return 2 * self.__property_a + self.__property_b + self.__property_c
