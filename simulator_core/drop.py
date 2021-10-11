# -*- coding: utf-8 -*-
"""HermesPy data drop."""

from __future__ import annotations
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from enum import IntEnum


__author__ = "Jan Adler"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ComplexVisualization(IntEnum):
    """How to visualize complex numbers in a 2D plot."""

    ABSOLUTE = 0        # Plot absolute value
    REAL = 1            # Plot real part
    IMAGINARY = 2       # Plot imaginary part
    DUAL = 3            # Plot both real and imaginary part in a dual plot

    def plot(self, axis, data: np.array, timestamps: np.array = None) -> None:
        """Plot time series data.

        Args:
            axis: The matplotlib axis to plot to.
            data (np.array): The data to be plotted.
            timestamps (np.array, optional): The time signature.
        """

        if self == ComplexVisualization.ABSOLUTE:
            plot_data = np.abs(data)

        elif self == ComplexVisualization.REAL:
            plot_data = np.real(data)

        elif self == ComplexVisualization.IMAGINARY:
            plot_data = np.imag(data)

        elif self == ComplexVisualization.DUAL:

            plot_data = [np.real(data), np.imag(data)]

        else:
            raise RuntimeError("Unsupported plot mode requested")

        if timestamps is None:
            axis.plot(plot_data)

        else:
            axis.plot(timestamps, plot_data)


class Drop:
    """Data generated within a single execution drop."""

    __transmitted_bits: List[np.ndarray]
    __transmitted_signals: List[np.ndarray]
    __received_signals: List[np.ndarray]
    __received_bits: List[np.ndarray]

    def __init__(self,
                 transmitted_bits: List[np.ndarray],
                 transmitted_signals: List[np.ndarray],
                 received_signals: List[np.ndarray],
                 received_bits: List[np.ndarray]) -> None:
        """Object initialization.

        Args:
            transmitted_bits (List[np.ndarray]): Bits fed into the transmitting modems.
            transmitted_signals (List[np.ndarray]): Modulated signals emitted by transmitting modems.
            received_signals (List[np.ndarray]): Modulated signals impinging onto receiving modems.
            received_bits (List[np.ndarray]): Bits output by receiving modems.
        """

        self.__transmitted_bits = transmitted_bits
        self.__transmitted_signals = transmitted_signals
        self.__received_signals = received_signals
        self.__received_bits = received_bits

    @property
    def transmitted_bits(self) -> List[np.ndarray]:
        """Access transmitted bits."""

        return self.__transmitted_bits

    @property
    def transmitted_signals(self) -> List[np.ndarray]:
        """Access transmitted signals."""

        return self.__transmitted_signals

    @property
    def received_signals(self) -> List[np.ndarray]:
        """Access received signals."""

        return self.__received_bits

    @property
    def received_bits(self) -> List[np.ndarray]:
        """Access received bits."""

        return self.__received_bits

    def plot_transmitted_signals(self,
                                 visualization: ComplexVisualization = ComplexVisualization.REAL) -> None:
        """Plot the transmit signals into a grid.

        Args:
            visualization (ComplexVisualization, optional): How to plot the transmitted signals.
        """

        # Infer grid dimensions
        grid_height = len(self.__transmitted_signals)
        grid_width = 0

        # Abort without error if no transmission is available
        if grid_height < 1:
            return

        # The grid width is the maximum number of transmitting antennas within this drop
        for transmission in self.__transmitted_signals:
            grid_width = max(grid_width, transmission.shape[0])

        figure, axes = plt.subplots(grid_height, grid_width, squeeze=False)
        for transmission_index, transmission in enumerate(self.__transmitted_signals):

            for antenna_index, antenna_emission in enumerate(transmission):
                visualization.plot(axes[transmission_index, antenna_index], antenna_emission)

            # Add y-label
            axes[transmission_index, transmission_index].set(ylabel="Tx {}".format(transmission_index))

        # Add labels
        figure.suptitle("Transmitted Signals")

        for antenna_index in range(grid_width):
            axes[-1, antenna_index].set(xlabel="N {}".format(antenna_index))

    def plot_received_signals(self,
                              visualization: ComplexVisualization = ComplexVisualization.REAL) -> None:
        """Plot the receive signals into a grid.

        Args:
            visualization (ComplexVisualization, optional): How to plot the received signals.
        """

        # Infer grid dimensions
        grid_height = len(self.__received_signals)
        grid_width = 0

        # Abort without error if no transmission is available
        if grid_height < 1:
            return

        # The grid width is the maximum number of transmitting antennas within this drop
        for transmission in self.__received_signals:
            grid_width = max(grid_width, transmission.shape[0])

        figure, axes = plt.subplots(grid_height, grid_width, squeeze=False)
        for transmission_index, transmission in enumerate(self.__received_signals):

            for antenna_index, antenna_emission in enumerate(transmission):
                visualization.plot(axes[transmission_index, antenna_index], antenna_emission)

            # Add y-label
            axes[transmission_index, transmission_index].set(ylabel="Tx {}".format(transmission_index))

        # Add labels
        figure.suptitle("Received Signals")

        for antenna_index in range(grid_width):
            axes[-1, antenna_index].set(xlabel="N {}".format(antenna_index))
