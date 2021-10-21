import numpy as np
from scipy import stats

from typing import Optional


def db2lin(db_val: float,
           conversion_type: Optional[str] = 'power'):
    """
    Converts from dB to linear

    Args:
        db_val (float): value in dB
        conversion_type (str, optional): if 'power' then it converts from dB to a power ratio
                                         if 'amplitude', then it converts from dB to an amplitude ratio
                                         default = 'power'
    Returns:
        (float): the equivalent value in linear scale
    """
    if conversion_type == 'power':
        output = 10**(db_val/10)
    elif conversion_type == 'amplitude':
        output = 10**(db_val/20)
    else:
        raise ValueError(f"dB conversion type ({conversion_type}) not supported")

    return output


def lin2db(val: float,
           conversion_type: Optional[str] = 'power'):
    """
    Converts from linear to dB

    Args:
        val (float): value in linear scale
        conversion_type (str, optional): if 'power' then it converts from a power ratio to dB
                                         if 'amplitude', then it converts from an amplitude ratio to dB
                                         default = 'power'
    Returns:
        (float) the equivalent value in linear scale
    """
    if conversion_type == 'power':
        output = 10 * np.log10(val)
    elif conversion_type == 'amplitude':
        output = 20 * np.log10(val)
    else:
        raise ValueError(f"dB conversion type ({conversion_type}) not supported")

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
