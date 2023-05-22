# -*- coding: utf-8 -*-
"""
============
Logarithmics
============
"""

from __future__ import annotations
from collections.abc import Sequence
from enum import Enum
from math import isclose
from typing import Any, overload, List, Optional, Tuple, Type, Union

import numpy as np

from hermespy.tools import db2lin, lin2db, DbConversionType

__author__ = "Jan Adler"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ValueType(Enum):
    LIN = 0
    """Linear number."""

    DB = 1
    """Logarithmic number."""


class Logarithmic(float):
    """Representation of a logarithmic number.

    Logarithmic numbers represent Decibel (dB) parameters within Hermes' API.
    However, they will always act as their linear value when being interacted with,
    in order to preserve compatibility with any internal equation,
    since equations internally assume all parameters to be linear.

    Note that therefore,

    .. code-block::

       a = Logarithmic(10)
       b = Logarithmic(20)

       c = a + b
       print(c)
       >>> 20dB

    will return in the output :math:`20.41392685158225` instead of :math:`30`,
    since internally the linear representations will be summed.
    Instead, use the multiplication operator to sum Logarithmics, i.e.

    .. code-block::

       a = Logarithmic(10)
       b = Logarithmic(20)

       c = a * b
       print(c)
       >>> 30dB
    """

    __value_db: float  # Logarithmic value in dB
    __conversion: DbConversionType  # Conversion type of the logarithmic scale

    def __init__(self, value: Union[float, int], value_type: ValueType = ValueType.DB, conversion: DbConversionType = DbConversionType.POWER) -> None:
        """
        Args:

            value (Union[float, int]):
                Value of the logarithmic number.

            value_type (ValueType, optional):
                Assumed type of `value`.
                Decibels by default.

            conversion (DbConversionType, optional):
                Conversion of logarithmic scale.
                Power by default.
        """

        value_db: float

        if value_type is ValueType.DB:
            value_db = value

        elif value_type is ValueType.LIN:
            value_db = lin2db(value, conversion)

        else:
            raise ValueError("Unknown value type")

        if float(self) <= 0.0:
            raise ValueError("Logarithmic scales can't express values smaller or equal to zero")

        if np.isnan(value_db):
            raise ValueError("Numerical error computing logarithmic value")

        # Save attributes
        self.__value_db = value_db
        self.__conversion = conversion

    def __new__(cls: Type[Logarithmic], value: Union[float, int], value_type: ValueType = ValueType.DB, conversion: DbConversionType = DbConversionType.POWER) -> Logarithmic:
        """
        Args:

            value (Union[float, int]):
                Value of the logarithmic number.

            value_type (ValueType, optional):
                Assumed type of `value`.
                Decibels by default.

            conversion (DbConversionType, optional):
                Conversion of logarithmic scale.
                Power by default.
        """

        if value_type is ValueType.LIN:
            return float.__new__(cls, value)

        if value_type is ValueType.DB:
            return float.__new__(cls, db2lin(value, conversion))

        raise ValueError("Unknown value type")

    @classmethod
    def From_Tuple(cls: Type[Logarithmic], linear: float, logarithmic: float, conversion: DbConversionType = DbConversionType.POWER) -> Logarithmic:
        instance = cls.__new__(cls, linear, ValueType.LIN)
        cls.__init__(instance, logarithmic, ValueType.DB, conversion)

        return instance

    @property
    def value_db(self) -> float:
        """Logarithmic value of represented number.

        Returns: Logarithmic value.
        """

        return self.__value_db

    @property
    def conversion(self) -> DbConversionType:
        """Logarithmic conversion type.

        Returns: Conversion type.
        """

        return self.__conversion

    def __add__(self, value: Union[Logarithmic, float, int]) -> Union[Logarithmic, float]:
        """Summing operation overload.

        Args:

            value (Union[Logarithmic, float, int]):
                Value to be added to the represented logarithmic number.

        Returns: Logarithmic sum representation.
        """

        if isinstance(value, Logarithmic):
            sum = float(self) + float(value)
            return Logarithmic(sum, ValueType.LIN, self.conversion)

        else:
            return float.__add__(self, value)

    def __sub__(self, value: Union[Logarithmic, float, int]) -> Union[Logarithmic, float]:
        """Substraction operation overload.

        Args:

            value (Union[Logarithmic, float, int]):
                Value to be substracted from the represented logarithmic number.

        Returns: Logarithmic substraction representation.
        """

        if isinstance(value, Logarithmic):
            sum = float(self) - float(value)
            return Logarithmic(sum, ValueType.LIN, self.conversion)

        else:
            return float.__sub__(self, value)

    def __mul__(self, value: Union[Logarithmic, float, int]) -> Union[Logarithmic, float]:
        """Multiplication operation overload.

        Args:

            value (Union[Logarithmic, float, int]):
                Value to be multiplied to the represented logarithmic number.

        Returns: Logarithmic multiplication representation.
        """

        if isinstance(value, Logarithmic):
            product = float(self) * float(value)
            return Logarithmic(product, ValueType.LIN, self.conversion)

        else:
            return float.__mul__(self, value)

    def __truediv__(self, value: Union[Logarithmic, float, int]) -> Union[Logarithmic, float]:
        """Division operation overload.

        Args:

            value (Union[Logarithmic, float, int]):
                Value to be divided from the represented logarithmic number.

        Returns: Logarithmic division representation.
        """

        if isinstance(value, Logarithmic):
            division = float(self) / float(value)
            return Logarithmic(division, ValueType.LIN, self.conversion)

        else:
            return float.__truediv__(self, value)

    def __str__(self) -> str:
        """Explicit type conversion to string.

        Returns: Text representation of the represented :class:`Logarithmic` value.
        """

        # Check if a pretty integer notation is possible
        integer_representation = int(self.value_db)
        if isclose(integer_representation, self.value_db):
            return f"{integer_representation}dB"

        # Resort to an ugly scientific notation otherwise
        return f"{self.value_db:.2g}dB"

    def __repr__(self) -> str:
        """Object representation overload.

        Returns: Text representation of the :class:`Logarithmic` object.
        """

        return f"<Log {str(self)}>"


class LogarithmicSequence(np.ndarray):
    """A sequence of logarithmic numbers."""

    __values_db: List[float]
    __conversion: DbConversionType

    def __new__(cls: Type[LogarithmicSequence], values: Optional[Sequence[Union[float, int]]] = None, value_type: ValueType = ValueType.DB, conversion: DbConversionType = DbConversionType.POWER) -> LogarithmicSequence:
        """
        Args:

            values (Sequence[Union[float, int]], optional):
                Initial content of the represented sequence.
                If not provided, the sequence will be initialized as empty.

            value_type (ValueType, optional):
                Assumed type of `value`.
                Decibels by default.

            conversion (DbConversionType, optional):
                Conversion of logarithmic scale.
                Power by default.
        """

        values = [] if values is None else values
        scalar_values: List[float]

        if value_type is ValueType.LIN:
            scalar_values = [float(value) for value in values]

        elif value_type is ValueType.DB:
            scalar_values = [db2lin(float(value), conversion) for value in values]

        else:
            raise ValueError("Unknown value type")

        cast = np.asarray(scalar_values, dtype=float).view(cls)
        cast.__conversion = conversion

        return cast

    def __array_finalize__(self, instance: Union[np.ndarray, None]) -> None:
        # Do nothing if the view is on None
        if instance is None:
            return

        # Convert view
        view = self.view(np.ndarray)

        # Abort if a numpy boolean array is represented
        # This is required to work around a strange bug in assert_array_almost_equal
        if not issubclass(view.dtype.type, np.floating) or self.ndim < 1:
            np.ndarray.__array_finalize__(self, instance)
            return

        # Recover initialization attributes
        conversion = getattr(instance, "conversion", DbConversionType.POWER)

        # Configure class atributes
        self.__conversion = conversion
        self.__values_db = [lin2db(float(value), conversion) for value in view]

    @property
    def conversion(self) -> DbConversionType:
        """Logarithmic conversion type.

        Returns: Conversion type.
        """

        return self.__conversion

    def tolist(self) -> List[Logarithmic]:
        """Convert to list representation.

        Returns: List of logarithmics.
        """

        return [Logarithmic.From_Tuple(lin, log, self.conversion) for lin, log in zip(self.view(np.ndarray), self.__values_db)]

    def __getitem__(self, i: Any) -> Union[Logarithmic, np.ndarray]:  # type: ignore
        if isinstance(i, (int, np.integer)):
            return Logarithmic.From_Tuple(np.ndarray.__getitem__(self, i), self.__values_db[i], self.conversion)

        return np.ndarray.__getitem__(self, i)

    def __setitem__(self, i: int, item: Union[Logarithmic, float, int]) -> None:
        # Convert non-logarithmic items to logarithmic
        if isinstance(item, Logarithmic):
            np.ndarray.__setitem__(self, i, float(item))
            self.__values_db[i] = item.value_db

        else:
            np.ndarray.__setitem__(self, i, item)
            self.__values_db[i] = lin2db(item, self.conversion)

    def __reduce__(self) -> Tuple[Type[LogarithmicSequence], Tuple[np.ndarray, ValueType, DbConversionType]]:
        """Serialization callback for the Ray framework."""

        deserializer = LogarithmicSequence
        serialized_data = (self.view(np.ndarray), ValueType.LIN, self.conversion)

        return deserializer, serialized_data


@overload
def dB(*values: Sequence[Union[int, float]], conversion: DbConversionType = DbConversionType.POWER) -> LogarithmicSequence:
    ...  # pragma no cover


@overload
def dB(*values: Union[int, float], conversion: DbConversionType = DbConversionType.POWER) -> Union[Logarithmic, LogarithmicSequence]:
    ...  # pragma no cover


def dB(*values: Union[int, float, Sequence[Union[int, float]]], conversion: DbConversionType = DbConversionType.POWER) -> Union[Logarithmic, LogarithmicSequence]:
    """Represent scalar value as logarithmic number.

    Args:

        *values (Tuple[Union[int, float]]):
            Value or sequence of values to be represented as logarithmic.

        conversion (DbConversionType, optional):
            Conversion of logarithmic scale.
            Power by default.

    Returns: The logarithmic representation of `*values`.
    """

    if isinstance(values[0], Sequence):
        return LogarithmicSequence(values[0], value_type=ValueType.DB, conversion=conversion)

    if any(isinstance(value, Sequence) for value in values):
        raise ValueError("Only the first argument may be a sequence")

    if len(values) == 1:
        return Logarithmic(values[0], value_type=ValueType.DB, conversion=conversion)

    return LogarithmicSequence(values, value_type=ValueType.DB, conversion=conversion)  # type: ignore
