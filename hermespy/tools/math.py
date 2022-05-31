# -*- coding: utf-8 -*-
"""
===========
Math Tools
===========

Implementations of basic maths equations.

"""

from enum import Enum
from math import cos, sin
from typing import Optional

import numpy as np
from scipy import stats
from numba import jit

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class DbConversionType(Enum):
    """Supported db conversion types."""
    POWER = 0
    AMPLITUDE = 1
    HILLY = 2

@jit
def db2lin(db_val: float,
           conversion_type: Optional[DbConversionType] = DbConversionType.POWER):
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
        output = 10**(db_val/10)
    elif conversion_type == DbConversionType.AMPLITUDE:
        output = 10**(db_val/20)
    else:
        raise ValueError(f"dB conversion type not supported")

    return output


@jit
def lin2db(val: float,
           conversion_type: Optional[DbConversionType] = DbConversionType.POWER):
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
        raise ValueError(f"dB conversion type not supported")

    return output


def marcum_q(a: float,
             b: np.ndarray,
             m: Optional[float] = 1):
    """Calculates the Marcum-Q function Q_m(a, b)

    This method uses the relationship between Marcum-Q function and the chi-squared distribution

    Args:
        a (float):
        b (np.array):
        m (float):

    Returns:
        (np.ndarray): the approximated Marcum-Q function for the desired parameters
    """

    q = stats.ncx2.sf(b**2, 2 * m, a**2)

    return q


def rotation_matrix(orientation: np.ndarray) -> np.ndarray:

    a = orientation[2]  # Pitch: Rotation around the z-axis
    b = orientation[1]  # Yaw:   Rotation around the y-axis
    c = orientation[0]  # Roll:  Rotation around the x-axis      

    R = np.array([[cos(a)*cos(b), cos(a)*sin(b)*sin(c) - sin(a)*cos(c), cos(a)*sin(b)*cos(c) + sin(a)*sin(c)],
                  [sin(a)*cos(b), sin(a)*sin(b)*sin(c) + cos(a)*cos(c), sin(a)*sin(b)*cos(c) - cos(a)*sin(c)],
                  [-sin(b), cos(b)*sin(c), cos(b)*cos(c)]])

    return R


def transform_vector(vector: np.ndarray,
                     position: np.ndarray,
                     orientation: np.ndarray) -> np.ndarray:

    R = rotation_matrix(orientation)
    return R @ vector + position


def transform_coordinates(coordinates: np.ndarray,
                          position: np.ndarray,
                          orientation: np.ndarray) -> np.ndarray:

    R = rotation_matrix(orientation)
    return (R @ coordinates.T + position[:, np.newaxis]).T


@jit(nopython=True)
def rms_value(x: np.ndarray) -> float:
    """Returns the root-mean-square value of a given input vector
    """

    return np.linalg.norm(x, 2) / np.sqrt(x.size)
