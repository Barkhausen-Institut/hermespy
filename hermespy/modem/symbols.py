# -*- coding: utf-8 -*-
"""
=====================
Communication Symbols
=====================
"""

from __future__ import annotations
from copy import deepcopy
from enum import Enum
from typing import Optional, Union, Iterable, Type

import matplotlib.pyplot as plt
import numpy as np
from h5py import Group

from hermespy.core import Executable, HDFSerializable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class SymbolType(Enum):
    """Communication symbol type flag."""

    DATA = 0
    """Data symbol transmitting information."""

    REFERENCE = 1
    """Reference symbol for channel estimation."""

    PILOT = 2
    """Pilot symbol for frame detection."""


class Symbol(object):
    """A single communication symbol located somewhere on the complex plane."""

    value: complex
    """Value of the symbol."""

    flag: SymbolType
    """Type of the symbol."""

    def __init__(self, value: complex, flag: SymbolType = SymbolType.DATA) -> None:
        """
        Args:

            value (complex):
                Symbol value.

            flag (SymbolType, optional):
                Assumed symbol type.
                Data is assumed by default.
        """

        self.value = value
        self.flag = flag


class Symbols(HDFSerializable):
    """A time-series of communication symbols located somewhere on the complex plane."""

    __symbols: np.ndarray  # Internal symbol storage

    def __init__(self, symbols: Optional[Union[Iterable, np.ndarray]] = None) -> None:
        """
        Args:

            symbols (Union[Iterable, numpy.ndarray], optional):
                A three-dimensional array of complex-valued communication symbols.
                The first dimension denotes the number of streams,
                the second dimension the number of symbol blocks per stream,
                the the dimension the number of symbols per block.
        """

        symbols = np.empty((0, 0, 0), dtype=complex) if symbols is None else symbols
        symbols = np.array(symbols) if not isinstance(symbols, np.ndarray) else symbols

        # Make sure the initialization is a valid symbol sequence
        if symbols.ndim > 3:
            raise ValueError("Symbols initialization array may have a maximum of three dimensions")

        # Exand the dimensions if required
        if symbols.ndim == 1:
            symbols = symbols[np.newaxis, :, np.newaxis]

        elif symbols.ndim == 2:
            symbols = symbols[:, :, np.newaxis]

        self.__symbols = symbols

    @property
    def num_streams(self) -> int:
        """Number of streams within this symbol series.

        Returns:
            int: Number of streams.
        """

        return self.__symbols.shape[0]

    @property
    def num_blocks(self) -> int:
        """Number of symbol blocks within this symbol series.

        Returns:
            int: Number of symbols
        """

        return self.__symbols.shape[1]

    @property
    def num_symbols(self) -> int:
        """Number of symbols per stream within this symbol series.

        Returns:
            int: Number of symbols
        """

        return self.__symbols.shape[2]

    def append_stream(self, symbols: Union[Symbols, np.ndarray]) -> None:
        """Append a new symbol stream to this symbol seris.

        Represents a matrix concatenation in the first dimensions.

        Args:

            symbols (Union[Symbols, np.ndarray]):
                Symbol stream to be appended to this symbol series.

        Raises:

            ValueError: If the number of symbols in time-domain do not match.
        """

        if isinstance(symbols, Symbols):
            symbols = symbols.raw

        if symbols.ndim == 1:
            symbols = symbols[np.newaxis, :, np.newaxis]

        elif symbols.ndim == 2:
            symbols = symbols[:, :, np.newaxis]

        if symbols.ndim != 3:
            raise ValueError("Symbols must be matrix (an array of dimension two)")

        if self.num_symbols < 1 and self.num_streams <= 1:

            self.__symbols = symbols

        else:

            if self.num_symbols != symbols.shape[2]:
                raise ValueError("Symbol models to be concatenated do not match in time-domain")

            if self.num_blocks != symbols.shape[1]:
                raise ValueError("Symbol models to be concatenated do not match in block-domain")

            self.__symbols = np.append(self.__symbols, symbols, axis=0)

    def append_symbols(self, symbols: Union[Symbols, np.ndarray]) -> None:
        """Append a new symbol sequence to this symbol seris.

        Represents a matrix concatenation in the second dimensions.

        Args:

            symbols (Union[Symbols, np.ndarray]):
                Symbol sequence to be appended to this symbol series.

        Raises:

            ValueError: If the number of symbol streams do not match.
        """

        if isinstance(symbols, Symbols):
            symbols = symbols.raw

        if symbols.ndim == 1:
            symbols = symbols[np.newaxis, :, np.newaxis]

        elif symbols.ndim == 2:
            symbols = symbols[:, :, np.newaxis]

        if symbols.ndim != 3:
            raise ValueError("Symbols must contain three dimensions")

        if self.num_symbols < 1 and self.num_streams <= 1:

            self.__symbols = symbols

        else:

            if self.num_streams != symbols.shape[0]:
                raise ValueError("Symbol models to be concatenated do not match in stream-domain")

            self.__symbols = np.append(self.__symbols, symbols, axis=1)

    @property
    def raw(self) -> np.ndarray:
        """Access the raw symbol array.

        Return:
            np.ndarray: The raw symbol array
        """

        return self.__symbols

    @raw.setter
    def raw(self, value: np.ndarray) -> None:

        if value.ndim != 3:
            raise ValueError("Raw symbols must be a three-dimensionall array")

        self.__symbols = value

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

    def plot_constellation(self, axes: Optional[plt.axes.Axes] = None, title: str = "Symbol Constellation") -> Optional[plt.Figure]:
        """Plot the symbol constellation.

        Essentially projects the time-series of symbols onto a single complex plane.

        Args:

            axes (Optional[plt.axes.Axes], optional):
                The axes to plot the graph to.
                By default, a new matplotlib figure is created.

            title (str, optional):
                Plot title.
                Only relevant if no axes were provided.

        Returns:

            Optional[plt.Figure]:
                Handle to the created matplotlib.pyplot figure object.
                None if the axes were provided.
        """

        symbols = self.__symbols.flatten()
        figure: Optional[plt.figure.Figure] = None

        # Create a new figure and the respective axes if none were provided
        if axes is None:

            with Executable.style_context():

                figure, axes = plt.subplots()
                figure.suptitle(title)

        axes.scatter(symbols.real, symbols.imag)
        axes.set(ylabel="Imag")
        axes.set(xlabel="Real")
        axes.grid(True, which="both")
        axes.axhline(y=0, color="k")
        axes.axvline(x=0, color="k")

        return figure

    @classmethod
    def from_HDF(cls: Type[Symbols], group: Group) -> Symbols:

        # Recall datasets
        symbols = np.array(group["symbols"], dtype=complex)

        # Initialize object from recalled state
        return cls(symbols=symbols)

    def to_HDF(self, group: Group) -> None:

        # Serialize datasets
        group.create_dataset("symbols", data=self.__symbols)

        # Serialize attributes
        group.attrs["num_streams"] = self.num_streams
        group.attrs["num_blocks"] = self.num_blocks
        group.attrs["num_symbols"] = self.num_symbols


class StatedSymbols(Symbols):
    """A time-series of communication symbols and channel states located somewhere on the complex plane."""

    __states: np.ndarray  # Symbol states, four-dimensional array

    def __init__(self, symbols: Optional[Union[Iterable, np.ndarray]], states: Optional[np.ndarray]) -> None:
        """
        Args:

            symbols (Union[Iterable, numpy.ndarray]):
                A three-dimensional array of complex-valued communication symbols.
                The first dimension denotes the number of streams,
                the second dimension the number of symbol blocks per stream,
                the the dimension the number of symbols per block.

            states (np.ndarray):
                Four-dimensional numpy array with the first two dimensions indicating the
                MIMO receive and transmit streams, respectively and the last two dimensions
                indicating the number of symbol blocks and symbols per block.
        """

        Symbols.__init__(self, symbols)
        self.states = states

    @property
    def states(self) -> np.ndarray:
        """Symbol state information.

        Four-dimensional numpy array with the first two dimensions indicating the
        MIMO receive and transmit streams, respectively and the last two dimensions
        indicating the number of symbol blocks and symbols per block.

        Raises:

            ValueError: If the state array is not four-dimensional.
            ValueError: If the state dimensions don't match the symbol dimensions.
        """

        return self.__states

    @states.setter
    def states(self, value: np.ndarray) -> None:

        if value.ndim != 4:
            raise ValueError("State must be a four-dimensional numpy array")

        if value.shape[0] != self.num_streams:
            raise ValueError(f"Number of received streams don't match, expected {self.num_streams} instead of {value.shape[0]}")

        if value.shape[2] != self.num_blocks:
            raise ValueError(f"Number of received blocks don't match, expected {self.num_blocks} instead of {value.shape[2]}")

        if value.shape[3] != self.num_symbols:
            raise ValueError(f"Symbol block sizes don't match, expected {self.num_symbols} instead of {value.shape[3]}")

        self.__states = value.copy()

    @property
    def num_transmit_streams(self) -> int:
        """Number of impinging transmit streams.

        Returns: Number of streams.
        """

        return self.__states.shape[1]

    def copy(self) -> StatedSymbols:

        return StatedSymbols(self.raw.copy(), self.states.copy())

    @classmethod
    def from_HDF(cls: Type[StatedSymbols], group: Group) -> StatedSymbols:

        # Recall datasets
        symbols = np.array(group["symbols"], dtype=complex)
        states = np.array(group["states"], dtype=complex)

        # Initialize object from recalled state
        return cls(symbols=symbols, states=states)

    def to_HDF(self, group: Group) -> None:

        # Serialize base class
        Symbols.to_HDF(self, group)

        # Serialize datasets
        group.create_dataset("states", data=self.__states)
