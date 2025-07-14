# -*- coding: utf-8 -*-

from typing import TypeVar

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


MO = TypeVar("MO")
"""Type of Monte Carlo object under investigation."""


class UnmatchableException(Exception):
    """An exception that can never get caught."""
    ...  # pragma: no cover
