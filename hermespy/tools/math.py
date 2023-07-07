# -*- coding: utf-8 -*-
"""
===========
Math Tools
===========

Implementations of basic maths equations.

"""

from enum import Enum
from typing import Optional

import numpy as np
from scipy import stats
from scipy.constants import speed_of_light
from numba import jit

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DbConversionType(Enum):
    """Supported db conversion types."""

    POWER = 0
    AMPLITUDE = 1


@jit(nopython=True)
def db2lin(db_val: float, conversion_type: Optional[DbConversionType] = DbConversionType.POWER):  # pragma: no cover
    """
    Converts from dB to linear

    Args:
        db_val (float): value in dB
        conversion_type (DbConversionType, optional): if POWER then it converts from dB to a power ratio
                                                      if AMPLITUDE, then it converts from dB to an amplitude ratio
                                                      default = POWER
    Returns:
        (float): the equivalent value in linear scale
    """
    if conversion_type == DbConversionType.POWER:
        output = 10 ** (db_val / 10)
    elif conversion_type == DbConversionType.AMPLITUDE:
        output = 10 ** (db_val / 20)
    else:
        raise ValueError("dB conversion type not supported")

    return output


@jit(nopython=True)
def lin2db(val: float, conversion_type: Optional[DbConversionType] = DbConversionType.POWER):  # pragma: no cover
    """
    Converts from linear to dB

    Args:
        val (float): value in linear scale
        conversion_type (DbConversionType, optional): if POWER then it converts from a power ratio to dB
                                                      if AMPLITUDE, then it converts from an amplitude ratio to dB
                                                      default = POWER
    Returns:
        (float) the equivalent value in linear scale
    """
    if conversion_type == DbConversionType.POWER:
        output = 10 * np.log10(val)
    elif conversion_type == DbConversionType.AMPLITUDE:
        output = 20 * np.log10(val)
    else:
        raise ValueError("dB conversion type not supported")

    return output


def marcum_q(a: float, b: np.ndarray, m: Optional[float] = 1):
    """Calculates the Marcum-Q function Q_m(a, b)

    This method uses the relationship between Marcum-Q function and the chi-squared distribution.

    Args:
        a (float):
        b (np.array):
        m (float):

    Returns:
        (np.ndarray): the approximated Marcum-Q function for the desired parameters
    """

    q = stats.ncx2.sf(b**2, 2 * m, a**2)

    return q


@jit(nopython=True)
def rms_value(x: np.ndarray) -> float:  # pragma: no cover
    """Returns the root-mean-square value of a given input vector"""

    return np.linalg.norm(x, 2) / np.sqrt(x.size)


def amplitude_path_loss(carrier_frequency: float, distance: float) -> float:
    """Compute the free space propagation loss of a wave in vacuum.

    Args:

        carrier_frequency (float):
            The wave's carrier frequency in Hz.

        distance (float):
            The traveled distance in m.

    Raises:

        ValueError: If the absolute value of `carrier_frequency` is zero.
    """

    absolute_carrier = abs(carrier_frequency)

    if absolute_carrier == 0.0:
        raise ValueError("Carrier frequency may not be zero for free space propagation path loss modeling")

    wavelength = speed_of_light / absolute_carrier
    amplitude_scale = wavelength / (4 * np.pi * distance)

    return amplitude_scale
