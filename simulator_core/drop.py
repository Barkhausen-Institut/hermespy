# -*- coding: utf-8 -*-
"""HermesPy data drop."""

from __future__ import annotations
from typing import List, Optional
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
    """Data generated within a single execution drop.

    Attributes:

        __transmitted_bits (List[np.ndarray]):
            Bits fed into the transmitting modems.

        __transmitted_signals (List[np.ndarray]):
            Modulated signals emitted by transmitting modems.

        __received_signals (List[np.ndarray]):
            Modulated signals impinging onto receiving modems.

        __received_bits (List[np.ndarray]):
            Bits output by receiving modems.

        __bit_errors (Optional[List[List[Optional[np.ndarray]]]]):
            Bit errors which occurred during transmission.
            The attribute is None if the bit errors have not been calculated yet.
            Contains a matrix of dimensions `num_transmitters`x`num_receivers`x`bit_deltas`.
            The length `bit_deltas` is the maximum of all transmitted and received bits.
            `bit_deltas` are optional, if a computation is not feasible the matrix field is set to None.

        __pad_bit_errors (bool):
            Controls behaviour during bit error computation.
            If enabled, bit streams of different length between transmitter and receiver will be extended by zeros,
            in order to match the maximum dimension and subsequently compared.
            If disabled, bit streams of different length will be ignored during bit error computation.
    """

    __transmitted_bits: List[np.ndarray]
    __transmitted_signals: List[np.ndarray]
    __received_signals: List[np.ndarray]
    __received_bits: List[np.ndarray]
    __bit_errors: Optional[List[List[Optional[np.ndarray]]]]
    __pad_bit_errors: bool

    def __init__(self,
                 transmitted_bits: List[np.ndarray],
                 transmitted_signals: List[np.ndarray],
                 received_signals: List[np.ndarray],
                 received_bits: List[np.ndarray],
                 pad_bit_errors: bool = True) -> None:
        """Object initialization.

        Args:
            transmitted_bits (List[np.ndarray]): Bits fed into the transmitting modems.
            transmitted_signals (List[np.ndarray]): Modulated signals emitted by transmitting modems.
            received_signals (List[np.ndarray]): Modulated signals impinging onto receiving modems.
            received_bits (List[np.ndarray]): Bits output by receiving modems.
            pad_bit_errors (bool, optional): Pad bit streams during error computation.
        """

        self.__transmitted_bits = transmitted_bits
        self.__transmitted_signals = transmitted_signals
        self.__received_signals = received_signals
        self.__received_bits = received_bits
        self.__bit_errors = None
        self.__pad_bit_errors = pad_bit_errors

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

        return self.__received_signals

    @property
    def received_bits(self) -> List[np.ndarray]:
        """Access received bits."""

        return self.__received_bits

    @property
    def bit_errors(self) -> List[List[Optional[np.ndarray]]]:
        """Detect the bit transmission deltas between all transmitting and receiving modems.

        The computation result gets so buffered, meaning only the first call is computationally expensive.

        Returns:
            List[List[np.ndarray]]:
                A numpy of dimensions `num_transmitters`x`num_receivers`x`bit_deltas`.
                The length `bit_deltas` is the maximum of all transmitted and received bits.
        """

        # Return the cached result if already computed
        if self.__bit_errors is not None:
            return self.__bit_errors

        self.__bit_errors: List[List[np.ndarray]] = []
        for transmitter_bits in self.transmitted_bits:

            transmit_errors: List[Optional[np.ndarray]] = []
            for receiver_bits in self.received_bits:

                # Skip error computation if padding is disabled and the stream lengths don't match
                if len(receiver_bits) != len(transmitter_bits) and not self.__pad_bit_errors:
                    transmit_errors.append(None)

                else:

                    # Pad bit sequences (if required)
                    num_bits = max(len(receiver_bits), len(transmitter_bits))
                    padded_transmission = np.append(transmitter_bits, np.zeros(num_bits - len(transmitter_bits)))
                    padded_reception = np.append(receiver_bits, np.zeros(num_bits - len(receiver_bits)))

                    # Compute bit errors as the positions where both sequences differ.
                    # Note that this requires the sequences to be in 0/1 format!
                    bit_errors = np.abs(padded_transmission - padded_reception)
                    transmit_errors.append(bit_errors)

            self.__bit_errors.append(transmit_errors)

        return self.__bit_errors

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
            axes[transmission_index, 0].set(ylabel="Tx {}".format(transmission_index))

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

        # Abort without error if no received signal is available
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
            axes[transmission_index, 0].set(ylabel="Tx {}".format(transmission_index))

        # Add labels
        figure.suptitle("Received Signals")

        for antenna_index in range(grid_width):
            axes[-1, antenna_index].set(xlabel="N {}".format(antenna_index))

    def plot_transmitted_bits(self) -> None:
        """Plot transmitted bits into a grid."""

        grid_height = len(self.__transmitted_bits)

        # Abort without error if no transmission is available
        if grid_height < 1:
            return

        figure, axes = plt.subplots(grid_height, 1, squeeze=False)
        figure.suptitle("Transmitted Bits")

        for transmission_index, bits in enumerate(self.__transmitted_bits):

            axes[transmission_index, 0].stem(bits)
            axes[transmission_index, 0].set(ylabel="Tx {}".format(transmission_index))

    def plot_received_bits(self) -> None:
        """Plot transmitted bits into a grid."""

        grid_height = len(self.__received_bits)

        # Abort without error if no transmission is available
        if grid_height < 1:
            return

        figure, axes = plt.subplots(grid_height, 1, squeeze=False)
        figure.suptitle("Received Bits")

        for reception_index, bits in enumerate(self.__received_bits):

            if len(bits) < 1:
                axes[reception_index, 0].text(0, 0, "No bits received")

            else:
                axes[reception_index, 0].stem(bits)
                axes[reception_index, 0].set(ylabel="Rx {}".format(reception_index))

    def plot_bit_errors(self) -> None:
        """Plot bit errors into a grid."""

        grid_width = len(self.__received_bits)
        grid_height = len(self.__transmitted_bits)

        # Abort without error if no transmission is available
        if grid_height < 1:
            return

        # Fetch bit errors
        bit_errors = self.bit_errors

        # Plot bit errors
        figure, axes = plt.subplots(grid_height, grid_width, squeeze=False)
        figure.suptitle("Bit Errors")

        for transmission_index in range(len(self.__transmitted_bits)):
            for reception_index in range(len(self.__received_bits)):

                link_errors: Optional[np.ndarray] = bit_errors[transmission_index][reception_index]

                if link_errors is None:
                    axes[transmission_index, reception_index].text(0, 0, "No bits exchanged")

                else:
                    axes[transmission_index, reception_index].stem(link_errors)

            axes[transmission_index, 0].set(ylabel="Tx {}".format(transmission_index))

        for reception_index in range(len(self.received_bits)):
            axes[-1, reception_index].set(xlabel="Rx {}".format(reception_index))
