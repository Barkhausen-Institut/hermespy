# -*- coding: utf-8 -*-
"""
=====================
Communication Symbols
=====================
"""

from __future__ import annotations
from copy import deepcopy
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "0.2.6"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Symbols(object):
    """A time-series of communication symbols located somewhere on the complex plane."""

    __symbols: np.ndarray       # Internal symbol storage

    def __init__(self,
                 symbols: Optional[np.ndarray] = None) -> None:
        """
        Args:

            symbols (np.ndarray, optional):
                Raw symbol array. The first dimension denotes the number of streams,
                the second dimension the number of symbols per stream.
        """

        symbols = np.empty((1, 0), dtype=complex) if symbols is None else symbols

        # Make sure the initialization is a valid symbol sequence
        if symbols.ndim > 2:
            raise ValueError("Symbols initialization array may have a maximum of two dimensions")

        # Expand to two dimensions if required
        if symbols.ndim == 1:
            symbols = symbols[np.newaxis, ::]

        self.__symbols = symbols

    @property
    def num_streams(self) -> int:
        """Number of streams within this symbol series.

        Returns:
            int: Number of streams.
        """

        return self.__symbols.shape[0]

    @property
    def num_symbols(self) -> int:
        """Number of symbols per stream within this symbol series.

        Returns:
            int: Number of symbols
        """

        return self.__symbols.shape[1]

    @property
    def raw(self) -> np.ndarray:
        """Access the raw symbol array.

        Return:
            np.ndarray: The raw symbol array
        """

        return self.__symbols

    def copy(self) -> Symbols:
        """Create a deep copy of this symbol sequence.

        Returns:
            Symbols: Copied sequence.
        """

        return deepcopy(self)

    def __getitem__(self, section: slice) -> Symbols:
        """Slice this symbol series.

        Args:
            section (slice):
                Slice symbol selection.

        Returns:
            Symbols:
                New Symbols object representing the selected `section`.
        """

        return Symbols(self.__symbols[section])

    def __setitem__(self, section: slice, value: Union[Symbols, np.ndarray]) -> None:
        """Set symbols within this series.

        Args:
            section (slice):
                Slice pointing to the symbol positions to be updated.

            value (Union[Symbols, np.ndarray]):
                The symbols to be set.
        """

        if isinstance(value, Symbols):

            self.__symbols[slice] = value.__symbols

        else:

            self.__symbols[slice] = value

    def plot_constellation(self) -> plt.Figure:
        """Plot the symbol constellation.

        Essentially projects the time-series of symbols onto a single complex plane.

        Returns:

            Figure:
                Handle to the created matplotlib.pyplot figure object.
        """

        symbols = self.__symbols.flatten()

        figure, axes = plt.subplots()
        figure.suptitle("Symbol Constellation")

        axes.scatter(symbols.real, symbols.imag)
        axes.set(ylabel="Imag")
        axes.set(xlabel="Real")
        axes.grid(True, which='both')
        axes.axhline(y=0, color='k')
        axes.axvline(x=0, color='k')

        return figure
