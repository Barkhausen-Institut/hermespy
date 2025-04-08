# -*- coding: utf-8 -*-

from __future__ import annotations
from enum import Enum
from typing import Iterable, Type
from typing_extensions import override

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from sparse import SparseArray  # type: ignore

from hermespy.core import (
    DeserializationProcess,
    Serializable,
    SerializationProcess,
    VisualizableAttribute,
    ScatterVisualization,
    VAT,
)

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Tobias Kronauer"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
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


class _ConstellationPlot(VisualizableAttribute[ScatterVisualization]):
    """Plot the symbol constellation.

    Essentially projects the time-series of symbols onto a single complex plane.

    Args:

        axes:
            The axes to plot the graph to.
            By default, a new matplotlib figure is created.

        title:
            Plot title.
            Only relevant if no axes were provided.

    Returns:
        Handle to the created matplotlib.pyplot figure object.
        None if the axes were provided.
    """

    __symbols: Symbols

    def __init__(self, symbols: Symbols) -> None:
        """
        Args:

            symbols: The symbols to be plotted.
        """

        # Initialize the base class
        super().__init__()

        # Initialize attributes
        self.__symbols = symbols

    @property
    def title(self) -> str:
        return "Symbol Constellation"

    def _prepare_visualization(
        self, figure: plt.Figure | None, axes: VAT, **kwargs
    ) -> ScatterVisualization:
        ax: plt.Axes = axes.flat[0]
        ax.set(ylabel="Imag")
        ax.set(xlabel="Real")
        ax.grid(True, which="both")
        ax.axhline(y=0, color=rcParams["grid.color"])
        ax.axvline(x=0, color=rcParams["grid.color"])
        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)

        num_symbols = (
            self.__symbols.num_symbols * self.__symbols.num_blocks * self.__symbols.num_streams
        )
        zeros = np.zeros(num_symbols, dtype=np.float64)
        path_collection = np.empty((1, 1), dtype=np.object_)
        path_collection[0, 0] = ax.scatter(zeros, zeros)

        return ScatterVisualization(figure, axes, path_collection)

    def _update_visualization(self, visualization: ScatterVisualization, **kwargs) -> None:
        symbols = self.__symbols.raw.flatten()
        path: plt.PathCollection = visualization.paths[0, 0]
        path.set_offsets(np.array([symbols.real, symbols.imag]).T)


class Symbol(object):
    """A single communication symbol located somewhere on the complex plane."""

    value: complex
    """Value of the symbol."""

    flag: SymbolType
    """Type of the symbol."""

    def __init__(self, value: complex, flag: SymbolType = SymbolType.DATA) -> None:
        """
        Args:

            value:
                Symbol value.

            flag:
                Assumed symbol type.
                Data is assumed by default.
        """

        self.value = value
        self.flag = flag


class Symbols(Serializable):
    """A time-series of communication symbols located somewhere on the complex plane."""

    __symbols: np.ndarray  # Internal symbol storage
    __constellation_plot: _ConstellationPlot  # Symbol constellation plot

    def __init__(self, symbols: Iterable | np.ndarray | None = None) -> None:
        """
        Args:

            symbols:
                A three-dimensional array of complex-valued communication symbols.
                The first dimension denotes the number of streams,
                the second dimension the number of symbol blocks per stream,
                the third dimension the number of symbols per block.
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
        self.__constellation_plot = _ConstellationPlot(self)

    @property
    def num_streams(self) -> int:
        """Number of streams within this symbol series."""

        return self.__symbols.shape[0]

    @property
    def num_blocks(self) -> int:
        """Number of symbol blocks within this symbol series."""

        return self.__symbols.shape[1]

    @property
    def num_symbols(self) -> int:
        """Number of symbols per stream within this symbol series."""

        return self.__symbols.shape[2]

    def append_stream(self, symbols: Symbols | np.ndarray) -> None:
        """Append a new symbol stream to this symbol seris.

        Represents a matrix concatenation in the first dimensions.

        Args:
            symbols: Symbol stream to be appended to this symbol series.

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

    def append_symbols(self, symbols: Symbols | np.ndarray) -> None:
        """Append a new symbol sequence to this symbol seris.

        Represents a matrix concatenation in the second dimensions.

        Args:
            symbols: Symbol sequence to be appended to this symbol series.

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
        """Access the raw symbol array."""

        return self.__symbols

    @raw.setter
    def raw(self, value: np.ndarray) -> None:
        if value.ndim != 3:
            raise ValueError("Raw symbols must be a three-dimensionall array")

        self.__symbols = value

    def copy(self) -> Symbols:
        """Create a deep copy of this symbol sequence.

        Returns: Copied sequence.
        """

        return Symbols(self.__symbols.copy())

    def __getitem__(self, section: slice) -> Symbols:
        """Slice this symbol series.

        Args:
            section: Slice symbol selection.

        Returns: New Symbols object representing the selected `section`.
        """

        return Symbols(self.__symbols[section])

    def __setitem__(self, section: slice, value: Symbols | np.ndarray) -> None:
        """Set symbols within this series.

        Args:
            section: Slice pointing to the symbol positions to be updated.
            value: The symbols to be set.
        """

        if isinstance(value, Symbols):
            self.__symbols[section] = value.__symbols

        else:
            self.__symbols[section] = value

    @property
    def plot_constellation(self) -> _ConstellationPlot:
        """Plot the symbol constellation."""

        return self.__constellation_plot

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_array(self.__symbols, "symbols")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> Symbols:
        return cls(process.deserialize_array("symbols", np.complex128))


class StatedSymbols(Symbols):
    """A time-series of communication symbols and channel states located somewhere on the complex plane."""

    __states: np.ndarray  # Symbol states, four-dimensional array

    def __init__(self, symbols: Iterable | np.ndarray, states: np.ndarray | SparseArray) -> None:
        """
        Args:

            symbols:
                A three-dimensional array of complex-valued communication symbols.
                The first dimension denotes the number of streams,
                the second dimension the number of symbol blocks per stream,
                the the dimension the number of symbols per block.

            states:
                Four-dimensional numpy array with the first two dimensions indicating the
                MIMO receive and transmit streams, respectively and the last two dimensions
                indicating the number of symbol blocks and symbols per block.
        """

        Symbols.__init__(self, symbols)
        self.states = states

    @property
    def states(self) -> np.ndarray | SparseArray:
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
    def states(self, value: np.ndarray | SparseArray) -> None:
        if value.ndim != 4:
            raise ValueError("State must be a four-dimensional numpy array")

        if value.shape[0] != self.num_streams:
            raise ValueError(
                f"Number of received streams don't match, expected {self.num_streams} instead of {value.shape[0]}"
            )

        if value.shape[2] != self.num_blocks:
            raise ValueError(
                f"Number of received blocks don't match, expected {self.num_blocks} instead of {value.shape[2]}"
            )

        if value.shape[3] != self.num_symbols:
            raise ValueError(
                f"Symbol block sizes don't match, expected {self.num_symbols} instead of {value.shape[3]}"
            )

        self.__states = value.copy()

    def dense_states(self) -> np.ndarray:
        """Return the channel state in dense format.

        Note that this method will convert the channel state to dense format if it is currently in sparse format.
        This operation may be computationally expensive and should be avoided if possible.

        Returns: The channel state tensor in dense format.
        """

        return self.__states.todense() if isinstance(self.__states, SparseArray) else self.__states

    @property
    def num_transmit_streams(self) -> int:
        """Number of impinging transmit streams."""

        return self.__states.shape[1]

    def copy(self) -> StatedSymbols:
        return StatedSymbols(self.raw.copy(), self.states.copy())

    def serialize(self, process: SerializationProcess) -> None:
        Symbols.serialize(self, process)
        process.serialize_array(self.dense_states(), "states")

    @classmethod
    def Deserialize(cls: Type[StatedSymbols], process: DeserializationProcess) -> StatedSymbols:
        symbols = process.deserialize_array("symbols", np.complex128)
        states = process.deserialize_array("states", np.complex128)
        return cls(symbols, states)
