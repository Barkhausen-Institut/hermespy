# -*- coding: utf-8 -*-
"""HermesPy data drop."""

from __future__ import annotations
from typing import List, Optional, Tuple
from scipy.signal import stft
from scipy.fft import fftshift
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

        __transmit_block_sizes (List[int]):
            Bit block sizes for each transmitter.

        __received_signals (List[np.ndarray]):
            Modulated signals impinging onto receiving modems.

        __received_bits (List[np.ndarray]):
            Bits output by receiving modems.

        __receive_block_sizes (List[int]):
            Bit block sizes for each receiver.

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

        __block_errors (Optional[List[List[Optional[np.ndarray]]]]):
            Bit block errors which occurred during transmission.
            The attribute is None if the block errors have not been calculated yet.
            Contains a matrix of dimensions `num_transmitters`x`num_receivers`x`block_deltas`.
            `block_deltas` are optional, if a computation is not feasible the matrix field is set to None.

        __spectrum_fft_size (Optional[int]):
            Number of discrete frequency bins within time-frequency transformations.

        __transmit_stft (Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]):
            Short time fourier transform of the transmitted antenna signals.
            By default only the first antenna within an array is considered.
            For each transmitter, a tuple of frequency bins, timestamps and respective stft matrix is cached.
            If no result has been computed yet, the attribute is None.

        __receive_stft (Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]):
            Short time fourier transform of the transmitted antenna signals.
            By default only the first antenna within an array is considered.
            For each transmitter, a tuple of frequency bins, timestamps and respective stft matrix is cached.
            If no result has been computed yet, the attribute is None.
    """

    __transmitted_bits: List[np.ndarray]
    __transmitted_signals: List[np.ndarray]
    __transmit_block_sizes: List[int]
    __received_signals: List[np.ndarray]
    __received_bits: List[np.ndarray]
    __receive_block_sizes: List[int]
    __bit_errors: Optional[List[List[Optional[np.ndarray]]]]
    __pad_bit_errors: bool
    __block_errors: Optional[List[List[Optional[np.ndarray]]]]
    __spectrum_fft_size: Optional[int]
    __transmit_stft = Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]
    __receive_stft = Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]


    def __init__(self,
                 transmitted_bits: List[np.ndarray],
                 transmitted_signals: List[np.ndarray],
                 transmit_block_sizes: List[int],
                 received_signals: List[np.ndarray],
                 received_bits: List[np.ndarray],
                 receive_block_sizes: List[int],
                 pad_bit_errors: bool = True,
                 spectrum_fft_size: Optional[int] = None) -> None:
        """Object initialization.

        Args:
            transmitted_bits (List[np.ndarray]): Bits fed into the transmitting modems.
            transmitted_signals (List[np.ndarray]): Modulated signals emitted by transmitting modems.
            transmit_block_sizes (List[int]): Bit block sizes for each transmitter.
            received_signals (List[np.ndarray]): Modulated signals impinging onto receiving modems.
            received_bits (List[np.ndarray]): Bits output by receiving modems.
            receive_block_sizes (List[int]): Bit block sizes for each receiver.
            pad_bit_errors (bool, optional): Pad bit streams during error computation.
            spectrum_fft_size (int, optional): Number of discrete frequency bins within time-frequency transformations.
        Raises:
            ValueError: If argument List dimensions do not match.
        """

        if len(transmitted_bits) != len(transmitted_signals) or len(transmitted_signals) != len(transmit_block_sizes):
            raise ValueError("Transmit argument lists must be of identical length")

        if len(received_bits) != len(received_signals) or len(received_signals) != len(receive_block_sizes):
            raise ValueError("Receive argument lists must be of identical length")

        self.__transmitted_bits = transmitted_bits
        self.__transmitted_signals = transmitted_signals
        self.__transmit_block_sizes = transmit_block_sizes
        self.__received_signals = received_signals
        self.__received_bits = received_bits
        self.__receive_block_sizes = receive_block_sizes
        self.__bit_errors = None
        self.__pad_bit_errors = pad_bit_errors
        self.__block_errors = None
        self.__spectrum_fft_size = spectrum_fft_size
        self.__transmit_stft = None
        self.__receive_stft = None

        # Infer parameters
        self.__num_transmissions = len(self.transmitted_bits)
        self.__num_receptions = len(self.received_bits)

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

        The computation result gets buffered, meaning only the first call is computationally expensive.

        Returns:
            List[List[Optional[np.ndarray]]]:
                A matrix of dimensions `num_transmitters`x`num_receivers`x`bit_deltas`.
                The length `bit_deltas` is the maximum of all transmitted and received bits.
                Matrix fields representing invalid links will be set to None.
        """

        # Return the cached result if already computed
        if self.__bit_errors is not None:
            return self.__bit_errors

        self.__bit_errors: List[List[Optional[np.ndarray]]] = []
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

    @property
    def block_errors(self) -> List[List[Optional[np.ndarray]]]:
        """Detect the block transmission deltas between all transmitting and receiving modems.

        The computation result gets buffered, meaning only the first call is computationally expensive.

        Returns:
            List[List[Optional[np.ndarray]]]:
                A matrix of dimensions `num_transmitters`x`num_receivers`x`bit_deltas`.
                The length `bit_deltas` is the maximum of all transmitted and received bits.
                Matrix fields representing invalid links will be set to None.
        """

        # Return the cached result if already computed
        if self.__block_errors is not None:
            return self.__block_errors

        # Fetch bit errors
        bit_errors = self.bit_errors

        # Calculate block errors from bit errors

        self.__block_errors: List[List[Optional[np.ndarray]]] = []
        for transmitter_index, transmitter_block_size in enumerate(self.__transmit_block_sizes):

            block_error_row: List[Optional[np.ndarray]] = []
            for receiver_index, receiver_block_size in enumerate(self.__receive_block_sizes):

                # Fetch link bit errors
                link_bit_errors = bit_errors[transmitter_index][receiver_index]

                block_errors = None

                # Ignore links with block sizes which are not identical
                # Ignore links which have already been ignored by the bit error algorithm
                if transmitter_block_size == receiver_block_size and link_bit_errors is not None:

                    bit_error_blocks = link_bit_errors.reshape((-1, transmitter_block_size))
                    block_errors = (bit_error_blocks.sum(axis=1, keepdims=False) > 0).astype(int)

                block_error_row.append(block_errors)

            self.__block_errors.append(block_error_row)

        return self.__block_errors

    @property
    def transmit_stft(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Short time fourier transform of the transmitted antenna signals.

        By default only the first antenna within an array is considered.
        The computation result gets cached, meaning only the first call is computationally expensive.

        Returns:
            List of
                - np.ndarray: Fourier transform frequency bins.
                - np.ndarray: Timestamps
                - np.ndarray: Complex fourier transform matrix.
            for each transmitting modem.
        """

        # Return the cached result if already computed
        if self.__transmit_stft is not None:
            return self.__transmit_stft

        self.__transmit_stft: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for antenna_signals in self.__transmitted_signals:

            # Consider only the first antenna signal
            signal: np.ndarray = antenna_signals[0]
            window_size = len(signal)

            # Make sure that signal is at least as long as FFT and pad it
            # with zeros is needed
            if self.__spectrum_fft_size and len(signal) < self.__spectrum_fft_size:

                signal = np.append(signal, np.zeros(self.__spectrum_fft_size - len(signal), dtype=complex))
                window_size = self.__spectrum_fft_size

            # Compute stft TODO: Add sampling rate
            frequency, time, transform = stft(signal, nperseg=window_size,
                                              noverlap=int(.5 * window_size),
                                              return_onesided=False)

            # Append result tuple
            self.__transmit_stft.append((fftshift(frequency), time, fftshift(transform, 0)))

        return self.__transmit_stft
    
    @property
    def receive_stft(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Short time fourier transform of the received antenna signals.

        By default only the first antenna within an array is considered.
        The computation result gets cached, meaning only the first call is computationally expensive.

        Returns:
            List of
                - np.ndarray: Fourier transform frequency bins.
                - np.ndarray: Timestamps
                - np.ndarray: Complex fourier transform matrix.
            for each receiveing modem.
        """

        # Return the cached result if already computed
        if self.__receive_stft is not None:
            return self.__receive_stft

        self.__receive_stft: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for antenna_signals in self.__received_signals:

            # Consider only the first antenna signal
            signal: np.ndarray = antenna_signals[0]
            window_size = len(signal)

            # Make sure that signal is at least as long as FFT and pad it
            # with zeros is needed
            if self.__spectrum_fft_size and len(signal) < self.__spectrum_fft_size:

                signal = np.append(signal, np.zeros(self.__spectrum_fft_size - len(signal), dtype=complex))
                window_size = self.__spectrum_fft_size

            # Compute stft TODO: Add sampling rate
            frequency, time, transform = stft(signal, nperseg=window_size,
                                              noverlap=int(.5 * window_size),
                                              return_onesided=False)

            # Append result tuple
            self.__receive_stft.append((fftshift(frequency), time, fftshift(transform, 0)))

        return self.__receive_stft

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

        grid_width = self.__num_receptions
        grid_height = self.__num_transmissions

        # Abort without error if no transmission is available
        if grid_height < 1:
            return

        # Fetch bit errors
        bit_errors = self.bit_errors

        # Plot bit errors
        figure, axes = plt.subplots(grid_height, grid_width, squeeze=False)
        figure.suptitle("Bit Errors")

        for transmission_index in range(self.__num_transmissions):
            for reception_index in range(self.__num_receptions):

                link_errors: Optional[np.ndarray] = bit_errors[transmission_index][reception_index]

                if link_errors is None:
                    axes[transmission_index, reception_index].text(0, 0, "No bits exchanged")

                else:
                    axes[transmission_index, reception_index].stem(link_errors)

            axes[transmission_index, 0].set(ylabel="Tx {}".format(transmission_index))

        for reception_index in range(len(self.received_bits)):
            axes[-1, reception_index].set(xlabel="Rx {}".format(reception_index))

    def plot_block_errors(self) -> None:
        """Plot block errors into a grid."""

        # Fetch bit errors
        block_errors = self.block_errors

        # Plot bit errors
        figure, axes = plt.subplots(self.__num_transmissions, self.__num_receptions, squeeze=False)
        figure.suptitle("Block Errors")

        for transmission_index in range(self.__num_transmissions):
            for reception_index in range(self.__num_receptions):

                link_errors: Optional[np.ndarray] = block_errors[transmission_index][reception_index]

                if link_errors is None:
                    axes[transmission_index, reception_index].text(0, 0, "No blocks exchanged")

                else:
                    axes[transmission_index, reception_index].stem(link_errors)

            axes[transmission_index, 0].set(ylabel="Tx {}".format(transmission_index))

        for reception_index in range(len(self.received_bits)):
            axes[-1, reception_index].set(xlabel="Rx {}".format(reception_index))

    def plot_transmit_stft(self) -> None:
        """Plot the short-time Fourier transform of transmitted waveforms."""

        # Fetch stft
        stft = self.transmit_stft

        figure, axes = plt.subplots(self.__num_transmissions, 1, squeeze=False)
        figure.suptitle("Transmit Short-Time Fourier Transform")

        for transmission_index in range(self.__num_transmissions):

            time, frequency, transform = stft[transmission_index]

            axes[transmission_index, 0].pcolormesh(frequency, time, abs(transform))
            axes[transmission_index, 0].set(ylabel="Frequency [Hz]")
            axes[transmission_index, 0].set(xlabel="Time [sec]")
            
    def plot_receive_stft(self) -> None:
        """Plot the short-time Fourier transform of received waveforms."""

        # Fetch stft
        stft = self.receive_stft

        figure, axes = plt.subplots(self.__num_receptions, 1, squeeze=False)
        figure.suptitle("Receive Short-Time Fourier Transform")

        for reception_index in range(self.__num_receptions):

            time, frequency, transform = stft[reception_index]

            axes[reception_index, 0].pcolormesh(frequency, time, abs(transform))
            axes[reception_index, 0].set(ylabel="Frequency [Hz]")
            axes[reception_index, 0].set(xlabel="Time [sec]")
