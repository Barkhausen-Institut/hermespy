# -*- coding: utf-8 -*-
"""
===============
Signal Modeling
===============
"""

from __future__ import annotations
from copy import deepcopy
from typing import Literal, List, Sequence, Tuple, Type, Any
from abc import ABC

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from h5py import Group
from numba import jit, complex128
from scipy.constants import pi
from scipy.fft import fft, fftshift, fftfreq
from scipy.ndimage import convolve1d
from scipy.signal import butter, sosfilt, firwin

from .factory import HDFSerializable
from .visualize import PlotVisualization, VAT, VisualizableAttribute

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class _SignalVisualization(VisualizableAttribute[PlotVisualization]):
    """Visualization of signal samples."""

    __signal: Signal  # The signal model to be visualized

    def __init__(self, signal: Signal) -> None:
        """
        Args:

            signal (Signal): The signal model to be visualized.
        """

        # Initialize base class
        super().__init__()

        # Initialize class attributes
        self.__signal = signal

    @property
    def signal(self) -> Signal:
        """The signal model to be visualized."""

        return self.__signal


class _SamplesVisualization(_SignalVisualization):
    """Visualize the samples of a signal model."""

    @property
    def title(self) -> str:
        return self.signal.title

    def create_figure(
        self, space: Literal["time", "frequency", "both"] = "both", **kwargs
    ) -> Tuple[plt.FigureBase, VAT]:
        return plt.subplots(self.signal.num_streams, 2 if space == "both" else 1, squeeze=False)

    def _prepare_visualization(
        self,
        figure: plt.Figure | None,
        axes: VAT,
        angle: bool = False,
        space: Literal["time", "frequency", "both"] = "both",
        legend: bool = True,
        **kwargs,
    ) -> PlotVisualization:
        # Prepare axes and lines
        lines = np.empty_like(axes, dtype=np.object_)
        zeros = np.zeros(self.signal.num_samples, dtype=np.float_)
        timestamps = self.signal.timestamps
        for stream_idx in range(self.signal.num_streams):
            ax_y_idx = 0
            if space in {"both", "time"}:
                ax: plt.Axes = axes[stream_idx, ax_y_idx]
                ax.set_ylabel("Amplitude")
                if stream_idx == self.signal.num_streams - 1:
                    ax.set_xlabel("Time-Domain [s]")
                time_lines = ax.plot(timestamps, zeros) + ax.plot(timestamps, zeros)
                if legend and stream_idx == 0:
                    time_lines[0].set_label("Real")
                    time_lines[1].set_label("Imag")
                    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=2, frameon=False)

                lines[stream_idx, ax_y_idx] = time_lines
                ax.add_line(time_lines[0])
                ax.add_line(time_lines[1])
                ax_y_idx += 1

            if space in {"both", "frequency"}:
                freq_ax: plt.Axes = axes[stream_idx, ax_y_idx]
                freq_ax.set_ylabel("Abs")
                if stream_idx == self.signal.num_streams - 1:
                    freq_ax.set_xlabel("Frequency-Domain [Hz]")
                frequency_lines = freq_ax.plot(timestamps, zeros)
                freq_ax.add_line(frequency_lines[0])

                if angle:
                    phase_axis = freq_ax.twinx()
                    frequency_lines.extend(freq_ax.plot(timestamps, zeros))
                    phase_axis.set_ylabel("Angle [Rad]")
                lines[stream_idx, ax_y_idx] = frequency_lines

        return PlotVisualization(figure, axes, lines)

    def _update_visualization(
        self,
        visualization: PlotVisualization,
        space: Literal["time", "frequency", "both"] = "both",
        angle: bool = False,
        **kwargs,
    ) -> None:
        # Generate timestamps for time-domain plotting
        timestamps = self.signal.timestamps

        # Infer the axis indices to account for different plot
        time_axis_idx = 0
        frequency_axis_idx = 1 if space == "both" else 0

        # Generate / collect the data to be plotted
        for stream_samples, axes, lines in zip(
            self.signal.getitem(), visualization.axes, visualization.lines
        ):
            # Plot time space
            if space in {"both", "time"}:
                if self.signal.num_samples < 1:
                    axes[time_axis_idx].text(0.5, 0.5, "NO SAMPLES", horizontalalignment="center")

                else:
                    lines[time_axis_idx][0].set_data(timestamps, stream_samples.real)
                    lines[time_axis_idx][1].set_data(timestamps, stream_samples.imag)

                # Rescale axes
                axes[time_axis_idx].relim()
                axes[time_axis_idx].autoscale_view(True, True, True)

            # Plot frequency space
            if space in {"both", "frequency"}:
                if self.signal.num_samples < 1:
                    axes[frequency_axis_idx].text(
                        0.5, 0.5, "NO SAMPLES", horizontalalignment="center"
                    )

                else:
                    frequencies = fftshift(
                        fftfreq(self.signal.num_samples, 1 / self.signal.sampling_rate)
                    )
                    bins = fftshift(fft(stream_samples))
                    lines[frequency_axis_idx][0].set_data(frequencies, np.abs(bins))

                if angle:
                    lines[frequency_axis_idx][1].set_data(frequencies, np.angle(bins))

                # Rescale axes
                axes[frequency_axis_idx].relim()
                axes[frequency_axis_idx].autoscale_view(True, True, True)


class _EyeVisualization(_SignalVisualization):
    """Visualize the eye diagram of a signal model.

    Depending on the `domain` flag the eye diagram will either be rendered with the time-domain on the plot's x-axis

    .. plot::

        import matplotlib.pyplot as plt

        from hermespy.modem import TransmittingModem, RaisedCosineWaveform

        transmitter = TransmittingModem()
        waveform = RaisedCosineWaveform(modulation_order=16, oversampling_factor=16, num_preamble_symbols=0, symbol_rate=1e8, num_data_symbols=1000, roll_off=.9)
        transmitter.waveform = waveform

        transmitter.transmit().signal.plot_eye(1 / waveform.symbol_rate, domain='complex')
        plt.show()


    or on the complex plane

    .. plot::

        import matplotlib.pyplot as plt

        from hermespy.modem import TransmittingModem, RaisedCosineWaveform

        transmitter = TransmittingModem()
        waveform = RaisedCosineWaveform(modulation_order=16, oversampling_factor=16, num_preamble_symbols=0, symbol_rate=1e8, num_data_symbols=1000, roll_off=.9)
        transmitter.waveform = waveform

        transmitter.transmit().signal.plot_eye(1 / waveform.symbol_rate, domain='time')
        plt.show()


    Args:

        symbol_duration (float):
            Assumed symbol repetition interval in seconds.
            Will be rounded to match the signal model's sampling rate.

        line_width (float, optional):
            Line width of a single plot line.

        title (str, optional):
            Title of the plotted figure.
            `Eye Diagram` by default.

        domain (Literal["time", "complex"]):
            Plotting behaviour of the eye diagram.
            `time` by default.
            See above examples for rendered results.

        legend (bool, optional):
            Display a plot legend.
            Enabled by default.
            Only considered when in `time` domain plotting mode.

        linewidth (float, optional):
            Line width of the eye plot.
            :math:`.75` by default.

        symbol_cutoff (float, optional):
            Relative amount of symbols ignored during plotting.
            :math:`0.1` by default.
            This is required to properly visualize intersymbol interferences within the communication frame,
            since special effects may occur at the start and end.
    """

    @property
    def title(self) -> str:
        return "Eye Diagram"

    def _prepare_visualization(
        self,
        figure: plt.Figure | None,
        axes: VAT,
        *,
        symbol_duration: float | None = None,
        domain: Literal["time", "complex"] = "time",
        legend: bool = True,
        linewidth: float = 0.75,
        symbol_cutoff: float = 0.1,
        **kwargs,
    ) -> PlotVisualization:
        if linewidth <= 0.0:
            raise ValueError(f"Plot line width must be greater than zero (not {linewidth})")

        if symbol_cutoff < 0.0 or symbol_cutoff > 1.0:
            raise ValueError(f"Symbol cutoff must be in the interval [0, 1] (not {symbol_cutoff})")

        _ax: plt.Axes = axes.flat[0]
        colors = self._get_color_cycle()

        _symbol_duration = symbol_duration if symbol_duration else 1 / self.signal.sampling_rate
        symbol_num_samples = int(_symbol_duration * self.signal.sampling_rate)
        num_symbols = self.signal.num_samples // symbol_num_samples
        num_cutoff_symbols = int(num_symbols * symbol_cutoff)
        if num_cutoff_symbols < 2:
            num_cutoff_symbols = 2
        values_per_symbol = 2 * symbol_num_samples + 2
        num_visualized_symbols = num_symbols - 2 * num_cutoff_symbols

        lines: List[Line2D] = []

        if domain == "time":
            _ax.set_xlabel("Time-Domain [s]")
            _ax.set_ylabel("Normalized Amplitude")
            _ax.set_ylim((-1.1, 1.1))
            _ax.set_xlim((-1.0, 1.0))

            timestamps = (
                np.arange(-symbol_num_samples - 1, 1 + symbol_num_samples, 1) / symbol_num_samples
            )
            times = np.hstack([timestamps] * num_visualized_symbols)
            values = np.zeros_like(times, dtype=np.float_)
            values[::values_per_symbol] = float("nan")

            lines.extend(_ax.plot(times, values, color=colors[0], linewidth=linewidth))
            lines.extend(_ax.plot(times, values, color=colors[1], linewidth=linewidth))

            if legend:
                legend_elements = [
                    Line2D([0], [0], color=colors[0], label="Real"),
                    Line2D([0], [0], color=colors[1], label="Imag"),
                ]
                _ax.legend(handles=legend_elements, loc="upper left", fancybox=True, shadow=True)

        elif domain == "complex":
            _ax.set_xlabel("Real")
            _ax.set_ylabel("Imag")

            num_cutoff_samples = num_cutoff_symbols * symbol_num_samples
            stream_slice = self.signal.getitem((0, slice(num_cutoff_samples, -num_cutoff_samples)))
            lines.extend(
                _ax.plot(stream_slice.real, stream_slice.imag, color=colors[0], linewidth=linewidth)
            )

        else:
            raise ValueError(f"Unsupported plotting domain '{domain}'")

        lines_array = np.empty_like(axes, dtype=np.object_)
        lines_array.flat[0] = lines
        return PlotVisualization(figure, axes, lines_array)

    def _update_visualization(
        self,
        visualization: PlotVisualization,
        *,
        symbol_duration: float | None = None,
        domain: Literal["time", "complex"] = "time",
        symbol_cutoff: float = 0.1,
        **kwargs,
    ) -> None:
        _symbol_duration = symbol_duration if symbol_duration else 1 / self.signal.sampling_rate
        symbol_num_samples = int(_symbol_duration * self.signal.sampling_rate)
        num_symbols = self.signal.num_samples // symbol_num_samples
        num_cutoff_symbols = int(num_symbols * symbol_cutoff)
        if num_cutoff_symbols < 2:
            num_cutoff_symbols = 2
        values_per_symbol = 2 * symbol_num_samples + 2
        num_visualized_symbols = num_symbols - 2 * num_cutoff_symbols

        samples = self.signal.getitem()
        if domain == "time":
            values = np.empty(num_visualized_symbols * values_per_symbol, dtype=np.complex_)
            for n in range(num_symbols - 2 * num_cutoff_symbols):
                sample_offset = (n + num_cutoff_symbols) * symbol_num_samples
                values[n * values_per_symbol : (n + 1) * values_per_symbol - 1] = samples[
                    0, sample_offset : sample_offset + values_per_symbol - 1
                ]
            values[values_per_symbol - 1 :: values_per_symbol] = float("nan") + 1j * float("nan")

            # Normalize values
            abs_value = np.max(np.abs(samples))
            if abs_value > 0.0:
                values /= np.max(np.abs(values))

            # Update plot lines
            visualization.lines.flat[0][0].set_ydata(values.real)
            visualization.lines.flat[0][1].set_ydata(values.imag)

        elif domain == "complex":
            num_cutoff_samples = num_cutoff_symbols * symbol_num_samples
            stream_slice = samples[0, num_cutoff_samples:-num_cutoff_samples]

            visualization.lines.flat[0][0].set_data(stream_slice.real, stream_slice.imag)


class SignalBlock(np.ndarray):
    """A TxN matrix of complex entries representing signal samples over T streams and N time samples.
    Used in Signal to store pieces of signal samples.
    Use \"offset\" property to set the offset of this block relative to a signal start sample."""

    _offset: int

    def __new__(cls, samples: Any, offset: int) -> SignalBlock:
        """Create a new SignalBlock instance.

        Args:
            samples (array_like):
                The object to be converted to a 2D matrix. Arrays of higher dim count are not allowed. Uses np.asarray for conversion.

            offset (int):
                Integer offset of this block in a signal model

        Raises:
            ValueError:
                If ndim of given samples is bigger then 2.
        """

        # Create a ndarray
        obj = np.asarray(samples, dtype=np.complex_)

        # Add streams dim if only one stream was passed
        if obj.ndim == 1:
            obj = obj.reshape((1, obj.size))
        # Validate
        elif obj.ndim > 2:
            raise ValueError(f"Samples must have ndim <= 2 ({obj.ndim} was given)")

        # Cast it to SignalBlock
        obj = obj.view(cls)

        # Assign attributes
        obj.offset = offset  # type: ignore

        return obj  # type: ignore

    def __array_finalize__(self, obj: np.ndarray | SignalBlock | None) -> None:
        if obj is None:
            return  # pragma: no cover
        self.offset = getattr(obj, "offset", 0)

    def __getitem__(self, key: Any) -> SignalBlock:
        res = super().__getitem__(key)
        return SignalBlock(res, self.offset)

    @property
    def num_streams(self) -> int:
        """The number of streams within this signal block.

        Returns:
            int: The number of streams.
        """

        return self.shape[0]

    @property
    def num_samples(self) -> int:
        """The number of samples within this signal block.

        Returns:
            int: The number of samples.
        """

        return self.shape[1]

    @property
    def offset(self) -> int:
        """Get block's offset"""

        return self._offset

    @offset.setter
    def offset(self, value: int) -> None:
        """Set block's offset"""

        if value < 0:
            raise ValueError("Offset must be non-zero")
        self._offset = value

    @property
    def end(self) -> int:
        """Get block stop sample.
        b.num_samples = b.end - b.off"""

        return self._offset + self.shape[1]

    @property
    def power(self) -> np.ndarray:
        """Compute power of the modeled signal.

        Returns: The power of each modeled stream as a numpy vector.
        """

        return self.energy / self.num_samples

    @property
    def energy(self) -> np.ndarray:
        """Compute the energy of the modeled signal.

        Returns: The energy of each modeled stream as a numpy vector.
        """

        return np.sum(self.real**2 + self.imag**2, axis=1).view(np.ndarray)

    def copy(self, order: Literal["K", "A", "C", "F"] | None = None) -> SignalBlock:
        """Copy this signal block to a new object.

        Args:
            order: Ignored.

        Returns:
            Signal: A copy of this signal model.
        """

        if order is not None:
            raise NotImplementedError("order argument is not supported")
        return deepcopy(self)

    def resample(
        self, sampling_rate_old: float, sampling_rate_new: float, aliasing_filter: bool = True
    ) -> SignalBlock:
        """Resample a signal block to a different sampling rate.

        Args:

            sampling_rate_old (float):
                Old sampling rate of the signal block in Hz.

            sampling_rate_new (float):
                New sampling rate of the signal block in Hz.

            aliasing_filter (bool, optional):
                Apply an anti-aliasing filter during downsampling.
                Enabled by default.

        Returns:
            SignalBlock:
                The resampled signal block.

        Raises:
            ValueError: If `sampling_rate_old` or `sampling_rate_new` are smaller or equal to zero.
        """

        # Validate given sampling rate
        if sampling_rate_old <= 0.0 or sampling_rate_new <= 0.0:
            raise ValueError("Sampling rate must be greater than zero.")

        # Skip resampling if the same sampling rate was given
        if sampling_rate_old == sampling_rate_new:
            return self.copy()

        # Avoid resampling if this block does not contain samples
        if self.num_samples < 1:
            return self.copy()

        # Init
        samples_new = self.view(np.ndarray).copy()

        # Apply anti-aliasing if requested and downsampled
        if aliasing_filter and sampling_rate_new < sampling_rate_old:
            samples_new = self._resample_antialiasing(
                samples_new, 8, sampling_rate_new / sampling_rate_old
            )

        # Resample
        samples_new = self._resample(samples_new, sampling_rate_old, sampling_rate_new)

        # Apply anti-aliasing if requested and upsampled
        if aliasing_filter and sampling_rate_new > sampling_rate_old:
            samples_new = self._resample_antialiasing(
                samples_new, 8, sampling_rate_old / sampling_rate_new
            )

        # Create a new SignalBlock object from the resampled samples and return it as result
        off_new = round(self.offset * sampling_rate_new / sampling_rate_old)
        return SignalBlock(samples_new, off_new)

    def append_samples(self, value: np.ndarray) -> SignalBlock:
        """Append samples to this signal block. Creates a new instance."""

        # Validate value
        if value.shape[0] != self.num_streams:
            raise ValueError("Shape mismatch")

        return SignalBlock(np.append(self, value, 1), self.offset)

    @staticmethod
    @jit(nopython=True)
    def _mix(
        target_samples: np.ndarray,
        stream_indices: np.ndarray,
        added_samples: np.ndarray,
        sampling_rate: float,
        frequency_distance: float,
    ) -> None:  # pragma: no cover
        """Internal subroutine to mix two sets of signal model samples.

        Args:
            target_samples (np.ndarray):
                Target samples onto which `added_samples` will be mixed.

            added_samples (np.ndarray):
                Samples to be mixed onto `target_samples`.

            sampling_rate (float):
                Sampling rate in Hz, which both `target_samples` and `added_samples` must share.

            frequency_distance (float):
                Distance between the carrier frequencies of `target_samples` and `added_samples` in Hz.
        """

        num_added_samples = added_samples.shape[1]

        # ToDo: Reminder, mixing like this currently does not account for possible delays.
        mix_sinusoid = np.exp(
            2j * pi * np.arange(num_added_samples) * frequency_distance / sampling_rate
        )

        target_samples[stream_indices, :num_added_samples] += added_samples * mix_sinusoid[None, :]

    @staticmethod
    def _resample_antialiasing(samples: np.ndarray, N: Any, Wn: Any) -> np.ndarray:
        """Internal subroutine for resampled method. Applies Butterworth filter to the given samples.

        Args:
            samples (np.ndarray):
                Samples to apply AA to.

            N (Any) and Wn (Any):
                The first two parameters of scipy.signal.butter."""

        aliasing_filter = butter(N, Wn, btype="low", output="sos")
        return sosfilt(aliasing_filter, samples, axis=1)

    @staticmethod
    @jit(nopython=True)
    def _resample(
        signal: np.ndarray, input_sampling_rate: float, output_sampling_rate: float
    ) -> np.ndarray:  # pragma: no cover
        """Internal subroutine to resample a given set of samples to a new sampling rate.

        Uses sinc-interpolation, therefore `signal` is assumed to be band-limited.

        Arguments:

            signal (np.ndarray):
                TxM matrix of T signal-streams to be resampled, each containing M time-discrete samples.

            input_sampling_rate (float):
                Rate at which `signal` is sampled in Hz.

            output_sampling_rate (float):
                Rate at which the resampled signal will be sampled in Hz.

        Returns:
            np.ndarray:
                MxT' matrix of M resampled signal streams.
                The number of samples T' depends on the sampling rate relations, e.i.
                T' = T * output_sampling_rate / input_sampling_rate .
        """

        num_streams = signal.shape[0]

        num_input_samples = signal.shape[1]
        num_output_samples = round(num_input_samples * output_sampling_rate / input_sampling_rate)

        input_timestamps = np.arange(num_input_samples) / input_sampling_rate
        output_timestamps = np.arange(num_output_samples) / output_sampling_rate

        output = np.empty((num_streams, num_output_samples), dtype=complex128)
        for output_idx in np.arange(num_output_samples):
            # Sinc interpolation weights for single output sample
            interpolation_weights = (
                np.sinc((input_timestamps - output_timestamps[output_idx]) * input_sampling_rate)
                + 0j
            )

            # Resample all streams simultaneously
            output[:, output_idx] = signal @ interpolation_weights

        return output


class Signal(ABC, HDFSerializable):
    """Abstract base class for all signal models in HermesPy."""

    filter_order: int = 10  # Order of the filters applied during superimposition
    _blocks: List[SignalBlock]
    _sampling_rate: float
    _carrier_frequency: float
    _noise_power: float
    _samples_visualization: _SamplesVisualization
    _eye_visualization: _EyeVisualization
    delay: float

    _num_streams: int

    def from_ndarray(self, samples: np.ndarray):
        """Create a new signal using parameters from this signal.
        Equivalent to `create(samples, **self.kwargs)`."""

        return self.Create(samples, **self.kwargs)

    @staticmethod
    def Create(
        samples: np.ndarray | Sequence[np.ndarray],
        sampling_rate: float = 1.0,
        carrier_frequency: float = 0.0,
        noise_power: float = 0.0,
        delay: float = 0.0,
        offsets: List[int] = None,
    ) -> Signal:
        """Creates a signal model instance given signal samples.
        Subclasses of Signal should reroute the given arguments to init and return the result.

        Args:
            samples (np.ndarray | Sequence[np.ndarray]):
                Single or a sequence of 2D matricies with shapes MxT_i, where
                    M - number of streams,
                    T_i - number of samples in the matrix i.
                Note that M for each entry of the sequence must be the same.
                SignalBlock and Sequence[SignalBlock] can also be passed here.

            sampling_rate (float):
                Sampling rate of the signal in Hz.

            offsets (List[int]):
                Integer offsets of the samples if given in a sequence.
                Ignored if samples is not a Sequence of np.ndarray.
                len(offsets) must be equal to len(samples).
                Offset number i must be greater then offset i-1 + samples[i-1].shape[1].

        Returns
            signal (Signal):
                SparseSignal if samples argument is a list of np.ndarrays. DenseSignal otherwise.
        """

        if isinstance(samples, list) and len(samples) != 1:
            return SparseSignal(
                samples, sampling_rate, carrier_frequency, noise_power, delay, offsets
            )
        return DenseSignal(samples, sampling_rate, carrier_frequency, noise_power, delay, offsets)

    def __iter__(self):
        yield from self._blocks

    def __len__(self):
        """Get number of blocks in the signal model."""

        return len(self._blocks)

    @staticmethod
    def _get_blocks_union(
        bs1: Sequence[SignalBlock], bs2: Sequence[SignalBlock]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate new blocks offsets and ends for a union of 2 different signals.

        Returns:
            b_offs (np.ndarray): a list of new blocks offsets
            b_ends (np.ndarray): a list of new blocks ends"""

        # Get nonzero column indices for both signals
        nonzero_bs1 = (
            np.concatenate([np.arange(b.offset, b.offset + b.num_samples) for b in bs1])
            if len(bs1) != 0
            else np.array([], int)
        )
        nonzero_bs2 = (
            np.concatenate([np.arange(b.offset, b.offset + b.num_samples) for b in bs2])
            if len(bs2) != 0
            else np.array([], int)
        )
        # Join them (calculate a union)
        nz_cols_idx = np.union1d(nonzero_bs1, nonzero_bs2)

        # Transform column indices to new block starts and stops
        # Find block starts in the nz_cols_idx
        b_offs_idx = (np.ediff1d(nz_cols_idx) - 1).nonzero()[0]
        # b_offs_idx is not finished yet, but can be used to calculate block stops here
        b_stops = nz_cols_idx[np.append(b_offs_idx, -1)] + 1
        # Now finish off the blocks offset indices array
        b_offs_idx += 1
        b_offs_idx = np.concatenate([[0], b_offs_idx])
        # Find blocks offsets
        b_offs = nz_cols_idx[b_offs_idx]

        return b_offs, b_stops

    @staticmethod
    def __parse_slice(s: slice, dim_size: int) -> Tuple[int, int, int]:
        """Helper function for _parse_validate_itemkey.
        Given a slice and a size of the sliced dimension,
        returns non-negative values for start, stop and step of the slice."""

        # Resolve None
        s0 = 0 if s.start is None else s.start
        s1 = dim_size if s.stop is None else s.stop
        s2 = 1 if s.step is None else s.step
        # Resolve negative
        s0 = s0 if s0 >= 0 else s0 % dim_size
        s1 = s1 if s1 >= 0 else s1 % dim_size
        return s0, s1, s2

    def _parse_validate_itemkey(self, key: Any) -> Tuple[int, int, int, int, int, int, bool]:
        """Parse and validate key in __getitem__ and __setitem__.

        Raises:
            TypeError if key is not
                an int,
                a slice,
                a tuple of (int, int), (slice, slice), (int, slice) or (slice, int),
                a boolean mask.
            IndexError if the key is out of bounds of the signal model.

        Returns:
            s00 (int): streams start
            s01 (int): streams stop
            s02 (int): streams step
            s10 (int): samples start
            s11 (int): samples stop
            s12 (int): samples step
            isboolmask (bool): True if key is a boolean mask, False otherwise.

            Note that if isboolmask is True, then all s?? take the following values:
            (0, self.num_streams, 1, 0, self.num_samples, 1).

            Note that if the key references any dimansion with an integer index,
            then the corresponding result start will be the index, and stop is start+1.
            For example, if key is 1, then only stream 1 is need. Then s00 is 1 and s01 is 2.
            Numpy getitem of [1] and [1:2] differ in dimensions. Flattening of the second variant should be considered.
        """

        self_num_streams = self.num_streams
        self_num_samples = self.num_samples
        isboolmask = False

        # Key is a tuple of two
        # ======================================================================
        if isinstance(key, tuple) and len(key) == 2:
            # Validate streams key
            streams_key_is_slice = isinstance(key[0], slice)
            samples_key_is_slice = isinstance(key[1], slice)
            if streams_key_is_slice:
                s00, s01, s02 = self.__parse_slice(key[0], self_num_streams)
                if s00 >= s01 and self_num_streams != 0:
                    raise IndexError(
                        f"Streams slice start must be lower then stop ({s00} >= {s01})"
                    )
            elif np.issubdtype(type(key[0]), np.integer):
                if key[0] >= self_num_streams:
                    raise IndexError(
                        f"Streams index is out of bounds (number of streams is {self_num_streams}, but stream {key[0]} was requested)"
                    )
                s00 = key[0] % self_num_streams
                s01 = s00 + 1
                s02 = 1
            else:
                raise TypeError(
                    f"Expected to get streams index as an integer or a slice, but got {type(key[0])}"
                )

            # Parse samples key (key[1])
            # Samples are indexed with an int
            if samples_key_is_slice:
                s10, s11, s12 = self.__parse_slice(key[1], self_num_samples)
            elif np.issubdtype(type(key[1]), np.integer):
                if key[1] >= self_num_samples:
                    raise IndexError(f"Samples index is out of bounds ({key[1]} > {len(self)})")
                s10 = key[1] % self_num_samples
                s11 = s10 + 1
                s12 = 1
            else:
                raise TypeError(f"Samples key is of an unsupported type ({type(key[1])})")

            return s00, s01, s02, s10, s11, s12, False
        # ======================================================================
        # done Key is a tuple of two

        # Samples key is considered to be not given from now on
        s10 = 0
        s11 = self_num_samples
        s12 = 1

        # Parse streams key
        # Key is a slice over streams
        if isinstance(key, slice):
            s00, s01, s02 = self.__parse_slice(key, self_num_streams)
            if s00 >= s01:
                raise IndexError(f"Streams slice start must be lower then stop ({s00} >= {s01})")
        # Key is an integer index over streams
        elif np.issubdtype(type(key), np.integer):
            if key >= self_num_streams:
                raise IndexError(
                    f"Streams index is out of bounds (number of streams is {self_num_streams}, but stream {key} was requested)"
                )
            s00 = key % self_num_streams
            s01 = s00 + 1
            s02 = 1
        # Key is a boolean mask or something unsupported
        else:
            try:
                if np.asarray(key, dtype=bool).shape == (self_num_streams, self_num_samples):
                    isboolmask = True
                    s00 = 0
                    s01 = self_num_streams
                    s02 = 1
                else:
                    raise ValueError
            except ValueError:
                raise TypeError(f"Unsupported key type {type(key)}")

        return s00, s01, s02, s10, s11, s12, isboolmask

    def _find_affected_blocks(self, s10: int, s11: int) -> Tuple[int, int]:
        """Find indices of blocks that are affected by the given samples slice.

        Arguments:
            s10 (int): Start of the samples slice.
            s11 (int): Stop of the samples slice.

        Return:
            b_start, b_stop (int, int):
                Indices of the first and the last affected blocks."""

        # Done with a binary search algotrithm.

        # Starting block
        L = 0
        R = len(self) - 1
        b_start = L
        while L < R:
            b_start = -((R + L) // -2)  # upside-down floor division
            if self._blocks[b_start].offset > s10:
                R = b_start - 1
            else:
                L = b_start
        if self._blocks[b_start].offset > s10 and b_start != 0:
            b_start -= 1

        # Stopping block
        L = 0
        R = len(self) - 1
        b_stop = R
        while L < R:
            b_stop = (R + L) // 2
            if self._blocks[b_stop].end < s11:
                L = b_stop + 1
            else:
                R = b_stop
        if self._blocks[b_stop].end < s11 and (len(self._blocks) - 1) > b_stop:
            b_stop += 1

        return b_start, b_stop

    def getitem(self, key: Any = slice(None, None)) -> np.ndarray:
        """Get specified samples.
        Works like np.ndarray.__getitem__, but de-sparsifies the signal.

        Arguments:
            key (Any):
                an int, a slice,
                a tuple (int, int), (int, slice), (slice, int), (slice, slice)
                or a boolean mask.
                Defaults to slice(None, None) (same as [:, :])

        Examples:
            getitem(slice(None, None)):
                Select all samples from the signal.
                Warning: can cause memory overflow if used with a sparse signal.
            getitem(0):
                Select and de-sparsify the first stream.
            getitem((slice(None, 2), slice(50, 100))):
                Select streams 0, 1 and samples 50-99.
                Same as samples_matrix[:2, 50:100]

        Returns: np.ndarray with ndim 2 and dtype np.complex_"""

        s00, s01, s02, s10, s11, s12, isboolmask = self._parse_validate_itemkey(key)
        num_streams = -((s01 - s00) // -s02)
        num_samples = -((s11 - s10) // -s12)
        if self.num_samples == 0 or self.num_streams == 0:  # if this signal is empty
            return np.zeros((num_streams, num_samples), np.complex_)

        # If key is a boolean mask
        if isboolmask:
            mask = np.array(key, dtype=bool)
            return self.to_dense()._blocks[0].view(np.ndarray)[mask]

        # Find blocks hit by the samples slice
        b_start, b_stop = self._find_affected_blocks(s10, s11)

        # Slicing is done inside one or no blocks
        if b_start == b_stop:
            b = self._blocks[b_stop]
            res = np.zeros((num_streams, num_samples), dtype=np.complex_)
            # if slicing is done inside only one block entirely
            if s10 >= b.offset:
                b_sliced = b[s00:s01:s02, s10 - b.offset : s11 - b.offset : s12]
                res[:, : b_sliced.shape[1]] = b_sliced
            # else if the slice starts in a zero gap and ends somewhere inside the block
            #               v  slice  v
            # xxxxxxxxxxxx00000xxxxxxxxxxxxxxxx
            # ^b previous^^gap^^b_start/b_stop^
            elif s11 > b.offset:
                res[:, b.offset - s10 :] = b[s00:s01:s02, :]
            return res

        # assemble the result
        res = np.zeros((self.num_streams, s11 - s10), dtype=np.complex_)
        # the first block
        b = self._blocks[b_start]
        if s10 < b.end:
            w_size = s10 - b.offset
            if w_size <= 0:  # slice starts before the first window
                res[:, -w_size : b.shape[1] - w_size] = b[:, :]
            else:
                b_sliced = b[:, -w_size:]
                res[:, : b_sliced.shape[1]] = b_sliced
        # blocks between the first and the last ones
        for i in range(b_start + 1, b_stop):
            b = self._blocks[i]
            w_start = b.offset - s10
            w_end = w_start + b.shape[1]
            res[:, w_start:w_end] = b[:, :]
        # the last block
        b = self._blocks[b_stop]
        w_start = res.shape[1] - s11 + b.offset
        w_stop = min(w_start + b.shape[1], res.shape[1])
        res[:, w_start:w_stop] = b[:, : w_stop - w_start]

        # Apply stream slicing and the samples step.

        # The following check is required for proper handling
        # of the streams key that looks like slice(None, None, -*something*).
        # In this case, (s00, s01, s02) will be (0, num_streams, -*something*).
        # Slicing by [0:num_streams:-*something*] and [::-*something*]
        # yield different results. We would like to get the second one.

        is_streams_step_reversing = (
            isinstance(key, slice) and key.start is None and key.stop is None
        )
        is_streams_step_reversing |= (
            isinstance(key, tuple)
            and isinstance(key[0], slice)
            and key[0].start is None
            and key[0].stop is None
        )
        if is_streams_step_reversing:
            return res[::s02, ::s12]
        return res[s00:s01:s02, ::s12]

    def getstreams(self, streams_key: int | slice | Sequence[int]) -> Signal:
        """Create a new signal like this, but with only the selected streams.

        Args:
            streams_key (int, slice, Sequence[int]):
                Stream indices to select.

        Return:
            signal (Signal):
                Signal of the same implementation as the caller, containing only the selected streams
        """
        blocks = [b[streams_key] for b in self]
        return self.Create(blocks, **self.kwargs)

    def __setitem__(self, key: Any, value: Any) -> None:
        """Set an np.ndarray of samples into the model."""

        ...  # pragma: no cover

    @property
    def kwargs(self) -> dict:
        """Returns:
        {"sampling_rate": self.sampling_rate,
        "carrier_frequency": self.carrier_frequency,
        "delay": self.delay,
        "noise_power": self.noise_power}"""

        return {
            "sampling_rate": self.sampling_rate,
            "carrier_frequency": self.carrier_frequency,
            "delay": self.delay,
            "noise_power": self.noise_power,
        }

    @staticmethod
    def _validate_samples_offsets(
        samples: np.ndarray | Sequence[np.ndarray], offsets: List[int] | None = None
    ) -> Tuple[Sequence[np.ndarray], List[int]]:
        """Raises ValueError if any of the following fails:
            1. samples is a 2D matrix or a Sequence of 2D matrices with similar number of streams
            (e.g. samples[i].shape[0] == samples[i+1].shape[0]).
            2. offsets (if given) has the same length as samples argument.
            3. samples windows with given offsets do not overlap.

        Returns:
            samples (Sequence[np.ndarray]):
                A python list if 2D matrices.
            offsets (List[int]):
                A list of valid offsets."""

        # np.ndarray was given
        if isinstance(samples, np.ndarray):
            if samples.ndim == 1:  # vector or np.ndarray or objects
                if samples.dtype != object:  # vector
                    samples = [samples.reshape((1, samples.size)).astype(np.complex_)]
                else:
                    samples = samples.tolist()  # pragma: no cover
            elif samples.ndim == 2:  # 2D matrix
                samples = [samples.astype(np.complex_)]
            elif samples.ndim == 3:  # Tensor of 2D matrices
                samples = [np.asarray(s, np.complex_) for s in samples]
            else:  # Higher dim tensor
                raise ValueError(
                    f"ndarrays of more then 3 dims are not acceptable ({samples.ndim} were given)"
                )

        # samples is Sequence[np.ndarray] from now on

        # Check if samples list is empty
        if len(samples) == 0:
            return ([np.ndarray((0, 0), dtype=np.complex_)], [0])

        # Validate and adjust samples
        num_streams = samples[0].shape[0] if samples[0].ndim == 2 else 1
        for i in range(len(samples)):
            # dtype
            if not np.iscomplexobj(samples[i]):
                samples[i] = samples[i].astype(np.complex_)  # type: ignore
            # ndim
            if samples[i].ndim == 1:
                samples[i] = samples[i].reshape((1, samples[i].size))  # type: ignore
            elif samples[i].ndim != 2:
                raise ValueError(
                    f"ndarrays of more then 3 dims are not acceptable (One of samples entries has {samples[i].ndim} ndim)"
                )
            # num_streams
            if samples[i].shape[0] != num_streams:
                raise ValueError(
                    f"At least one if the samples blocks contains different number of streams (samples[0].shape[0] is {num_streams}, but samples[{i}].shape[0] is {samples[i].shape[0]})"
                )

        # Validate offsets length
        if offsets is not None and len(offsets) != len(samples):
            raise ValueError(
                f"samples and offsets legthes do not match ({len(samples)} and {len(offsets)})"
            )

        # Resolve offsets from samples, if offsets argument is None
        offsets_: List[int]
        if offsets is None:
            offsets_ = np.empty((len(samples),), int).tolist()
            offsets_[0] = getattr(samples[0], "_offset", 0)
            for i in range(1, len(samples)):
                offsets_[i] = getattr(
                    samples[i], "_offset", offsets_[i - 1] + samples[i - 1].shape[1]
                )
        else:
            offsets_ = offsets

        # Validate the offsets
        for i in range(1, len(samples)):
            gap_start = offsets_[i - 1] + samples[i - 1].shape[1]
            if offsets_[i] < gap_start:
                raise ValueError(
                    f"One of the blocks contains incorrect offset in this sequence (previous block ends on {gap_start}, this block's offset is {offsets_[i]})"
                )

        return samples, offsets_  # type: ignore

    def set_samples(
        self,
        samples: np.ndarray | Sequence[np.ndarray] | Sequence[SignalBlock],
        offsets: List[int] | None = None,
    ) -> None:
        """Sets given samples into this dense signal model.
        Validates given samples and optional offsets, writes them into _blocks attribute and resambles, if needed.
        """

        samples, offsets = self._validate_samples_offsets(samples, offsets)

        # _blocks
        self._blocks = []
        for b, o in zip(samples, offsets):
            self._blocks.append(SignalBlock(b, o))

        # num_streams
        self._num_streams = 0 if len(samples) == 0 else samples[0].shape[0]

    @staticmethod
    def Empty(sampling_rate: float, num_streams: int = 0, num_samples: int = 0, **kwargs) -> Signal:
        """Creates an empty signal model instance. Initializes it with the given arguments.
        If both num_streams and num_samples are not 0, then initilizes samples with np.empty."""

        return Signal.Create(np.empty((num_streams, num_samples)), sampling_rate, **kwargs)

    @property
    def num_streams(self) -> int:
        """The number of streams within this signal model.

        Returns:
            int: The number of streams.
        """

        return self._num_streams

    @property
    def num_samples(self) -> int:
        """The number of samples within this signal model.

        Returns:
            int: The number of samples.
        """

        if len(self._blocks) == 0:
            return 0
        return self._blocks[-1].end

    @property
    def shape(self) -> tuple:
        """Returns: (num_streams, num_samples)"""

        return (self.num_streams, self.num_samples)

    @property
    def sampling_rate(self) -> float:
        """The rate at which the modeled signal was sampled.

        Returns:
            float: The sampling rate in Hz.
        """

        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        """Modify the rate at which the modeled signal was sampled.

        Args:
            value (float): The sampling rate in Hz.

        Raises:
            ValueError: If `value` is smaller or equal to zero.
        """

        if value <= 0.0:
            raise ValueError("The sampling rate of modeled signals must be greater than zero")

        self._sampling_rate = value
        for b in self:
            b._sampling_rate = value

    @property
    def carrier_frequency(self) -> float:
        """The center frequency of the modeled signal in the radio-frequency
        transmit band.

        Returns:
            float: The carrier frequency in Hz.
        """

        return self._carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:
        """Modify the center frequency of the modeled signal in the radio-frequency transmit band.

        Args:
            value (float): he carrier frequency in Hz.

        Raises:
            ValueError: If `value` is smaller than zero.
        """

        if value < 0.0:
            raise ValueError("The carrier frequency of modeled signals must be non-negative")

        self._carrier_frequency = value
        for b in self:
            b._carrier_frequency = value

    @property
    def noise_power(self) -> float:
        """Noise power of the superimposed noise signal.

        Returns:

            Noise power.

        Raises:

            ValueError: If the noise power is smaller than zero.
        """

        return self._noise_power

    @noise_power.setter
    def noise_power(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Noise power must be greater or equal to zero")

        self._noise_power = value
        for b in self:
            b._noise_power = value

    @property
    def power(self) -> np.ndarray:
        """Compute the power of the modeled signal.

        Returns: The power of each modeled stream within a numpy vector.
        """

        if self.num_samples < 1:
            return np.zeros(self.num_streams)
        return self.energy / self.num_samples

    @property
    def energy(self) -> np.ndarray:
        """Compute the energy of the modeled signal.

        Returns: The energy of each modeled stream within a numpy vector.
        """

        return np.sum([b.energy for b in self], 0)

    @property
    def timestamps(self) -> np.ndarray:
        """The sample-points of the signal block.

        Returns:
            np.ndarray: Vector of length T containing sample-timestamps in seconds.
        """

        return np.arange(self.num_samples) / self.sampling_rate

    @property
    def frequencies(self) -> np.ndarray:
        """The signal model's discrete sample points in frequcy domain.

        Returns: Numpy vector of frequency bins.
        """

        return fftfreq(self.num_samples, 1 / self.sampling_rate)

    @property
    def duration(self) -> float:
        """Signal model duration in time-domain.

        Returns:
            float: Duration in seconds.
        """

        return self.num_samples / self.sampling_rate

    @property
    def title(self) -> str:
        return "Dense Signal Model"  # pragma: no cover

    @property
    def plot(self) -> _SamplesVisualization:
        """Visualize the samples of the signal model."""

        return self._samples_visualization

    @property
    def eye(self) -> _EyeVisualization:
        """Visualize the eye diagram of the signal model."""

        return self._eye_visualization

    def copy(self) -> Signal:
        """Copy this signal model to a new object.

        Returns:
            Signal: A copy of this signal model.
        """

        return deepcopy(self)

    def resample(self, sampling_rate: float, aliasing_filter: bool = True) -> Signal:
        """Resample the modeled signal to a different sampling rate.

        Args:

            sampling_rate (float):
                Sampling rate of the new signal model in Hz.

            aliasing_filter (bool, optional):
                Apply an anti-aliasing filter during downsampling.
                Enabled by default.

        Returns:
            Signal:
                The resampled signal model.

        Raises:
            ValueError: If `sampling_rate` is smaller or equal to zero.
        """

        signal_new = self.Empty(num_streams=self.num_streams, **self.kwargs)
        signal_new._blocks = [
            b.resample(self.sampling_rate, sampling_rate, aliasing_filter) for b in self
        ]
        signal_new._sampling_rate = sampling_rate
        return signal_new

    def superimpose(
        self,
        added_signal: Signal,
        resample: bool = True,
        aliasing_filter: bool = True,
        stream_indices: Sequence[int] | None = None,
    ) -> None:
        """Superimpose an additive signal model to this model.

        Internally re-samples `added_signal` to this model's sampling rate, if required.
        Mixes `added_signal` according to the carrier-frequency distance.

        Args:

            added_signal (Signal): The signal to be superimposed onto this one.
            resample (bool): Allow for dynamic resampling during superposition.
            aliasing_filter (bool, optional): Apply an anti-aliasing filter during mixing.
            stream_indices (Sequence[int], optional): Indices of the streams to be mixed.

        Raises:

            ValueError: If `added_signal` contains a different number of streams than this signal model.
            RuntimeError: If resampling is required but not allowd.
            NotImplementedError: If the delays if this signal and `added_signal` differ.
        """

        # Do nothing if the added signal is empty
        if added_signal.num_samples == 0:
            return

        # Parse stream indices
        num_streams = added_signal.num_streams if stream_indices is None else len(stream_indices)
        _stream_indices = (
            np.arange(num_streams)
            if stream_indices is None
            else np.array(stream_indices, dtype=np.int_)
        )

        # Abort if zero streams are selected
        # This case occurs frequently when devices transmit to devices without receiving antennas
        if len(_stream_indices) < 1:
            return

        # Validate
        if _stream_indices.max() >= self.num_streams:
            raise ValueError(f"Stream indices must be in the interval [0, {self.num_streams - 1}]")
        if self.delay != added_signal.delay:
            raise NotImplementedError(
                "Superimposing signal models of differing delay is not yet supported"
            )
        if added_signal.sampling_rate != self.sampling_rate and not resample:
            raise RuntimeError("Resampling required but not allowed")

        # Apply an aliasing filter to the added signal
        frequency_distance = added_signal.carrier_frequency - self.carrier_frequency
        filter_center_frequency = 0.5 * frequency_distance
        filter_bandwidth = 0.5 * (added_signal.sampling_rate + self.sampling_rate) - abs(
            frequency_distance
        )
        if filter_bandwidth <= 0.0:
            return
        added_signal_copy = added_signal.copy()
        if aliasing_filter and filter_bandwidth < added_signal.sampling_rate:
            filter_coefficients = firwin(
                1 + self.filter_order,
                0.5 * filter_bandwidth,
                width=0.5 * filter_bandwidth,
                fs=added_signal_copy.sampling_rate,
            ).astype(complex)
            filter_coefficients *= np.exp(
                2j
                * np.pi
                * (filter_center_frequency - added_signal.carrier_frequency)
                / added_signal_copy.sampling_rate
                * np.arange(1 + self.filter_order)
            )
            for i in range(len(added_signal_copy)):
                added_signal_copy._blocks[i][:, :] = convolve1d(
                    added_signal_copy._blocks[i], filter_coefficients, axis=1
                )

        # Resample the added signal if the respective sampling rates don't match
        added_signal_resampled = added_signal_copy.resample(self.sampling_rate, False)

        # Append zeros to self if num_samples is to small
        if added_signal_resampled.num_samples > self.num_samples:
            self.append_samples(
                np.zeros(
                    (self.num_streams, added_signal_resampled.num_samples - self.num_samples),
                    np.complex_,
                )
            )

        # Create new blocks for the signals' intersections
        # and insert the mix results there
        b_offs, b_stops = self._get_blocks_union(self._blocks, added_signal_resampled._blocks)
        res_ws = np.empty((b_offs.size,), dtype=object)
        fd_sr_ratio = frequency_distance / self.sampling_rate
        for i in range(b_offs.size):
            w_off = b_offs[i]
            w_stop = b_stops[i]
            w_self = self.getitem((slice(None, None), slice(w_off, w_stop)))
            if w_self.shape[1] < w_stop - w_off:
                w_self = np.append(
                    w_self,
                    np.zeros((num_streams, w_stop - w_off - w_self.shape[1]), np.complex_),
                    axis=1,
                )  # pragma: no cover
            w_them = added_signal_resampled.getitem((slice(None, None), slice(w_off, w_stop)))
            if w_them.shape[1] < w_stop - w_off:
                w_them = np.append(
                    w_them,
                    np.zeros((num_streams, w_stop - w_off - w_them.shape[1]), np.complex_),
                    axis=1,
                )  # pragma: no cover
            mix_sin = np.exp(2.0j * pi * np.arange(w_off, w_stop) * fd_sr_ratio)
            res_ws[i] = (w_self + mix_sin * w_them)[_stream_indices]

        # Construct new signal blocks
        self._blocks = [SignalBlock(res_ws[i], b_offs[i]) for i in range(len(b_offs))]

    def append_samples(self, signal: Signal | np.ndarray) -> None:
        """Append samples in time-domain to the signal model.

        Args:
            signal (Signal): The signal to be appended.

        Raise:
            ValueError: If the number of streams don't align.
        """
        ...  # pragma: no cover

    def append_streams(self, signal: Signal | np.ndarray) -> None:
        """Append streams to the signal model.

        Args:
            signal (Signal): The signal to be appended.

        Raise:
            ValueError: If the number of samples don't align.
        """
        ...  # pragma: no cover

    def to_interleaved(self, data_type: Type = np.int16, scale: bool = True) -> np.ndarray:
        """Convert the complex-valued floating-point model samples to interleaved integers.

        Args:

            data_type (optional):
                Numpy resulting data type.

            scale (bool, optional):
                Scale the floating point values to stretch over the whole range of integers.

        Returns:
            samples (np.ndarray):
                Numpy array of interleaved samples.
                Will contain double the samples in time-domain.
        """

        # Get samples
        samples = self.getitem()

        # Scale samples if required
        if scale and samples.shape[1] > 0 and (samples.max() > 1.0 or samples.min() < 1.0):
            samples /= np.max(abs(samples))  # pragma: no cover

        samples *= np.iinfo(data_type).max
        return samples.view(np.float64).astype(data_type)

    @classmethod
    def from_interleaved(
        cls, interleaved_samples: np.ndarray, scale: bool = True, **kwargs
    ) -> Signal:
        """Initialize a signal model from interleaved samples.

        Args:

            interleaved_samples (np.ndarray):
                Numpy array of interleaved samples.

            scale (bool, optional):
                Scale the samples after interleaving

            **kwargs:
                Additional class initialization arguments.
        """

        complex_samples = (
            interleaved_samples.astype(np.float64).view(np.complex128).view(np.complex_)
        )

        if scale:
            complex_samples /= np.iinfo(interleaved_samples.dtype).max

        return cls.Create(samples=complex_samples, **kwargs)

    def to_dense(self) -> DenseSignal:
        """Concatenate all the blocks in the signal into one block.
        Accounts for offsets. Warning - if offset values are to big, memory overflow is possible.

        Returns:
            signal (DenseSignal): Dense form for this signal"""

        res_b = np.zeros(self.shape, dtype=np.complex_)
        for b in self:
            res_b[:, b.offset : b.offset + b.num_samples] = b
        return DenseSignal(res_b, **self.kwargs)

    @classmethod
    def from_HDF(cls, group: Group) -> Signal:
        # De-serialize attributes
        sampling_rate = group.attrs.get("sampling_rate", 1.0)
        carrier_frequency = group.attrs.get("carrier_frequency", 0.0)
        delay = group.attrs.get("delay", 0.0)
        noise_power = group.attrs.get("noise_power", 0.0)
        num_streams = group.attrs.get("num_streams", 0)

        # De-serialize blocks
        num_blocks = group.attrs.get("num_blocks", 0)
        offsets = np.array(group["offsets"], int)
        blocks = [
            SignalBlock(np.array(group[f"block{i}"], np.complex_), offsets[i])
            for i in range(num_blocks)
        ]

        res = cls.Create(
            samples=blocks,
            sampling_rate=sampling_rate,
            carrier_frequency=carrier_frequency,
            delay=delay,
            noise_power=noise_power,
        )
        res._num_streams = num_streams
        return res

    def to_HDF(self, group: Group) -> None:
        # Serialize attributes
        group.attrs["carrier_frequency"] = self.carrier_frequency
        group.attrs["sampling_rate"] = self.sampling_rate
        group.attrs["num_streams"] = self.num_streams
        group.attrs["num_samples"] = self.num_samples
        group.attrs["power"] = self.power
        group.attrs["delay"] = self.delay
        group.attrs["noise_power"] = self.noise_power

        # Serialize samples
        group.attrs["num_blocks"] = len(self._blocks)
        for i in range(len(self._blocks)):
            self._write_dataset(group, f"block{i}", self._blocks[i])
        self._write_dataset(group, "offsets", [b.offset for b in self])
        self._write_dataset(group, "timestamps", self.timestamps)


class DenseSignal(Signal):
    """Dense signal model class."""

    def __init__(
        self,
        samples: np.ndarray | Sequence[np.ndarray] | Sequence[SignalBlock],
        sampling_rate: float = 1.0,
        carrier_frequency: float = 0.0,
        noise_power: float = 0.0,
        delay: float = 0.0,
        offsets: List[int] = None,
    ) -> None:
        """Signal model initialization.

        Args:
            samples (np.ndarray | Sequence[np.ndarray] | Sequence[SignalBlock]):
                A MxT matrix containing uniformly sampled base-band samples of the modeled signal.
                M is the number of individual streams, T the number of available samples.
                Note that you can pass here a 2D ndarray, a SignalBlock, a Sequence[np.ndarray] or a Sequence[SignalBlock].
                Given Sequence[SignalBlock], concatenates all signal blocks into one, accounting for their offsets.
                Given Sequence[np.ndarray], concatenates them all into one matrix sequentially.
                Warning: do not pass sparse sequences of SignalBlocks here as it can lead to memory bloat. Consider SparseSignal instead.

            sampling_rate (float):
                Sampling rate of the modeled signal in Hz (in the base-band).

            carrier_frequency (float, optional):
                Carrier-frequency of the modeled signal in the radio-frequency band in Hz.
                Zero by default.

            noise_power (float, optional):
                Power of the noise superimposed to this signal model.
                Zero by default.

            delay (float, optional):
                Delay of the signal in seconds.
                Zero by default.

            offsets (List[int], optional):
                If provided, must be of the same length as the samples argument.
                If samples argument is not an Sequence[SignalBlock],
                then offsets will be used to dislocate the blocks in the signal.
        """

        # Initialize base classes
        HDFSerializable.__init__(self)

        self._blocks = []
        # Initialize attributes
        self.sampling_rate = sampling_rate
        self.carrier_frequency = carrier_frequency
        self.delay = delay
        self.noise_power = noise_power
        self._samples_visualization = _SamplesVisualization(self)
        self._eye_visualization = _EyeVisualization(self)

        # Initialize blocks
        self.set_samples(samples, offsets)

    @staticmethod
    def Create(
        samples: np.ndarray | Sequence[np.ndarray],
        sampling_rate: float = 1.0,
        carrier_frequency: float = 0.0,
        noise_power: float = 0.0,
        delay: float = 0.0,
        offsets: List[int] = None,
    ) -> DenseSignal:
        return DenseSignal(samples, sampling_rate, carrier_frequency, noise_power, delay, offsets)

    def getitem(self, key: Any = slice(None, None)) -> np.ndarray:
        """Reroutes the argument to the single block of this model.
        Refer the numpy.ndarray.__getitem__ documentation.
        The result is always a 2D ndarray."""

        res = self._blocks[0].view(np.ndarray)[key]
        # de-flatten
        if res.ndim == 1:
            return res.reshape((1, res.size))
        return res

    def __setitem__(self, key: Any, value: Any) -> None:
        self._blocks[0][key] = value

    def set_samples(
        self,
        samples: np.ndarray | Sequence[np.ndarray] | Sequence[SignalBlock],
        offsets: List[int] | None = None,
    ) -> None:
        """Sets given samples into this dense signal model.
        SignalBlock or Sequence[SignalBlock] can be provided in samples.
        In this case they all will be resampled and the offsets argument will be ignored.
        A single 2D ndarray can be provided to construct a SignalBlock. Offsets will be ignored.
        A Sequence[ndarray] can be provided. If no offsets are given, then they all will be concatenated.
        If offsets are provided, the samples will be concatenated, including zero gaps accounting for offsets.
        If samples is Sequence[SignalBlock] and offsets are provided, blocks' offsets will be ignored
        Warning: avoid using sparse offsets as this can cause a memory bloat. Consider SparseSignal instead.
        """

        super().set_samples(samples, offsets)

        # De-sparsify the samples
        if len(self) < 2:
            return
        res = np.zeros(self.shape, np.complex_)
        for b in self:
            res[:, b.offset : b.end] = b
        self._blocks = [SignalBlock(res, 0)]

    @property
    def title(self) -> str:
        return "Dense Signal Model"

    @property
    def power(self) -> np.ndarray:
        if self.num_samples < 1:
            return np.zeros(self.num_streams)
        return self._blocks[0].power

    @property
    def energy(self) -> np.ndarray:
        return self._blocks[0].energy

    @staticmethod
    def Empty(
        sampling_rate: float, num_streams: int = 0, num_samples: int = 0, **kwargs
    ) -> DenseSignal:
        return DenseSignal(
            np.empty((num_streams, num_samples), dtype=complex), sampling_rate, **kwargs
        )

    def append_samples(self, signal: Signal | np.ndarray) -> None:
        # Resample the signal
        added_blocks_resampled: Sequence[np.ndarray]
        if isinstance(signal, Signal):
            added_blocks_resampled = signal.resample(self.sampling_rate)._blocks
        else:
            added_blocks_resampled = [signal]

        # Check if all the signal attributes match
        for prop in self.kwargs.items():
            if getattr(signal, prop[0], prop[1]) != prop[1]:
                raise ValueError(f"Signal attribute {prop[0]} does not match")

        # Adapt number of streams to fit the appended signal if this signal is empty
        if self.num_streams < 1:
            self._blocks = [
                SignalBlock(np.empty((signal.shape[0], 0), dtype=complex), 0)  # pragma: no cover
            ]

        # Append all blocks from the signal to self block
        off = self._blocks[0].offset
        for b in added_blocks_resampled:
            samples_new = np.append(self._blocks[0], b, axis=1)
            self._blocks[0] = SignalBlock(samples_new, off)

    def append_streams(self, signal: Signal | np.ndarray) -> None:
        # Adapt the number of samples if this signal is empty to match the signal to be appended

        if self.num_samples == 0:
            self._blocks = [
                SignalBlock(np.zeros((self.num_streams, signal.shape[1]), np.complex_), 0)
            ]

        # Check if all the signal attributes match
        for prop in self.kwargs.items():
            if getattr(signal, prop[0], prop[1]) != prop[1]:
                raise ValueError(f"Signal attribute {prop[0]} does not match")

        # Resample the signal
        added_blocks_resampled: Sequence[np.ndarray]
        if isinstance(signal, Signal):
            added_blocks_resampled = signal.resample(self.sampling_rate)._blocks
        else:
            added_blocks_resampled = [signal]

        # Append all blocks from the signal to self block
        for b in added_blocks_resampled:
            self._blocks[0] = SignalBlock(
                np.append(self._blocks[0], b, axis=0), self._blocks[0].offset
            )

        # Update _num_streams
        if len(self._blocks) > 0:
            self._num_streams = self._blocks[0].shape[0]

    def to_dense(self) -> DenseSignal:
        return self


class SparseSignal(Signal):
    """Sparse signal model class.
    HermesPy signal sparsification can be described as follows.
    Given M signal streams, N samples are recorded for each stream with some constant temporal sampling rate.
    Thus, a MxN complex matrix of samples can be constructed.
    If at some point streams do not contain any recorded/transmitted signal, then fully zeroed columns appear in the matrix.
    This signal model contains a list of SignalBlocks, each representing non-zero regions of the original samples matrix.
    Thus, regions with only zeros are avoided.
    Note, that SignalBlocks are sorted by their offset time and don't overlap."""

    def __init__(
        self,
        samples: np.ndarray | Sequence[np.ndarray] | Sequence[SignalBlock],
        sampling_rate: float = 1.0,
        carrier_frequency: float = 0.0,
        noise_power: float = 0.0,
        delay: float = 0.0,
        offsets: List[int] = None,
    ) -> None:
        """Signal model initialization.

        Args:
            samples (np.ndarray | Sequence[SignalBlock]):
                A MxT matrix containing uniformly sampled base-band samples of the modeled signal.
                M is the number of individual streams, T the number of available samples.
                Note that you can pass here a 2D ndarray, a SignalBlock, a Sequence[np.ndarray] or a Sequence[SignalBlock].
                Given Sequence[SignalBlock], concatenates all signal blocks into one, accounting for their offsets.
                Given Sequence[np.ndarray], concatenates them all into one matrix sequentially.

            sampling_rate (float):
                Sampling rate of the modeled signal in Hz (in the base-band).

            carrier_frequency (float, optional):
                Carrier-frequency of the modeled signal in the radio-frequency band in Hz.
                Zero by default.

            noise_power (float, optional):
                Power of the noise superimposed to this signal model.
                Zero by default.

            delay (float, optional):
                Delay of the signal in seconds.
                Zero by default.

            offsets (List[int], optional):
                If provided, must be of the same length as the samples argument.
                If samples argument is not an Sequence[SignalBlock],
                then offsets will be used to dislocate the blocks in the signal.
        """

        # Initialize base classes
        HDFSerializable.__init__(self)

        self._blocks = []
        # Initialize attributes
        self.sampling_rate = sampling_rate
        self.carrier_frequency = carrier_frequency
        self.delay = delay
        self.noise_power = noise_power
        self._samples_visualization = _SamplesVisualization(self)
        self._eye_visualization = _EyeVisualization(self)

        # Initialize blocks
        self.set_samples(samples, offsets)

    @staticmethod
    def Create(
        samples: np.ndarray | Sequence[np.ndarray],
        sampling_rate: float = 1.0,
        carrier_frequency: float = 0.0,
        noise_power: float = 0.0,
        delay: float = 0.0,
        offsets: List[int] = None,
    ) -> SparseSignal:
        return SparseSignal(samples, sampling_rate, carrier_frequency, noise_power, delay, offsets)

    @staticmethod
    def __from_dense(block: np.ndarray) -> List[SignalBlock]:
        """Sparsify given signal samples matrix."""

        # Get block attributes
        offset = getattr(block, "offset", 0)

        # Boundary dimensional cases
        if block.ndim == 1:  # a vector of a stream
            block = block.reshape((1, block.size))  # pragma: no cover
        if block.shape[1] == 0 or len(block.nonzero()) == 0:  # an empty signal
            return [SignalBlock(block, offset)]

        # Find nonzero columns indices
        nonzero_cols_idx = np.sum(block, axis=0).nonzero()[
            0
        ]  # [3, 4, 5, 6, 12, 13, 14, 16, 17, ...]
        if len(nonzero_cols_idx) == 0:
            return []
        # Calculate nonzero windows offsets and stops
        nz_wins_idx = np.ediff1d(nonzero_cols_idx) - 1  # [0, 0, 0, 5, 0,  0,  1,  0,   ...]
        nz_wins_idx = np.array(nz_wins_idx.nonzero()) + 1  # [            4,          7,   ...]
        res_offsets = nonzero_cols_idx[nz_wins_idx]  # [            12,         16,  ...]
        res_offsets = np.insert(
            res_offsets, 0, nonzero_cols_idx[0]
        )  # Add offset of the first window (3)
        # Calculate windows stops
        win_stops = nonzero_cols_idx[nz_wins_idx - 1]  # [         6,         14,      ...]
        win_stops = np.append(win_stops, nonzero_cols_idx[-1])  # Add a stop for the last window
        win_stops += 1  # [         7,         15,      ...]
        # Cut the windows
        assert win_stops.shape == res_offsets.shape
        num_windows = win_stops.shape[0]
        res = []
        for i in range(num_windows):
            res_samples = block[:, res_offsets[i] : win_stops[i]].astype(np.complex_)
            res_offset = offset + res_offsets[i]
            res.append(SignalBlock(res_samples, res_offset))

        return res

    def __setitem__(self, key: Any, value: Any) -> None:
        # parse and validate key
        s00, s01, s02, s10, s11, s12, isboolmask = self._parse_validate_itemkey(key)
        if s02 <= 0 or s12 <= 0:
            raise NotImplementedError("Only positive steps are implemented")
        if s12 != 1:
            raise NotImplementedError(
                "SparseSignal __setitem__ does not support sample stepping yet"
            )
        if isboolmask:
            raise NotImplementedError("SparseSignal __setitem__ does not support boolean masks yet")
        num_streams = -((s01 - s00) // -s02)
        num_samples = -((s11 - s10) // -s12)

        # clamp the slices to the actual dimensions
        num_streams = min(self.num_streams, num_streams)
        num_samples = min(self.num_samples, num_samples)
        s01 = min(self.num_streams, s01)
        s11 = min(self.num_samples, s11)

        # If insertion happens entirely before the first block
        if s11 < self._blocks[0].offset:
            # then create new blocks and insert them into the model
            if not isinstance(value, np.ndarray):
                value = np.complex_(value)
                if value == 0.0 + 0.0j:
                    return
                bs_new = [SignalBlock(np.full((num_streams, num_samples), value, np.complex_), s10)]
            else:
                value = value.reshape((num_streams, num_samples))
                bs_new = self.__from_dense(SignalBlock(value, s10))
            self._blocks = bs_new + self._blocks
            return

        # Get sliced streames indices
        stream_idx = np.arange(s00, s01, s02)
        # Do nothing if no streams are affected
        if stream_idx.size == 0:
            return

        # parse and validate value samples
        if not isinstance(value, np.ndarray):  # If value is a scalar
            value = np.complex_(value)
            bs_new = (
                []
                if value == 0.0 + 0.0j
                else [SignalBlock(np.full((num_streams, num_samples), value, np.complex_), s10)]
            )
        else:  # else value is a samples matrix
            if value.ndim == 1:
                value = value.reshape((num_streams, num_samples))
            bs_new = self.__from_dense(SignalBlock(value, s10))
            if len(bs_new) == 0:
                return  # pragma: no cover
            if bs_new[-1].end - bs_new[0].offset > num_samples or bs_new[0].shape[0] != num_streams:
                raise ValueError("Shape mismatch")

        # Find blocks that will be overwritten or modified.
        b_start, b_stop = self._find_affected_blocks(s10, s11)

        # Example:
        #     s10                   s11
        #      v                     v
        # xxxxxxxx0000000xxxx000xxxxxxxxx -- self signal
        #      yyyyy00yy0000yyyyy0000     -- value
        # xxxxxyyyyy00yy0000yyyyy0000xxxx -- result
        # ^   ^                           -- remaining samples from the first affected block
        #      ^   value samples    ^
        #                            ^  ^ -- remaining samples from the last affected block

        # Init the resulting blocks
        # Here the resulting blocks list is divided
        # onto 3 parts as on the ascii example.
        # b_start and b_stop can be sliced partially and will be added to the lists right after.
        # value argument will be applied only to the middle blocks (_mid).
        blocks_new_beg = self._blocks[:b_start]
        blocks_new_mid = [b for b in self._blocks[b_start + 1 : b_stop]]
        blocks_new_end = self._blocks[b_stop + 1 :]

        # Cut the first affected block
        # Example:
        #      s10                  s10
        #       v                    v
        # xxxxxxxxxxxx00000 -> xxxxxxxxxxxx00000
        # [ b_start  ]         [1st ][2nd ]
        b = self._blocks[b_start]
        cut_col = max(s10 - b.offset, 0)  # max is for s10 < self._blocks[0].offset
        b_l = b[:, :cut_col]
        b_r = b[:, cut_col:]
        b_r.offset += cut_col
        if b_l.size != 0:
            blocks_new_beg.append(b_l)
        if b_r.size != 0:
            blocks_new_mid = [b_r] + blocks_new_mid

        # Cut the last affected block
        if b_start != b_stop:
            b = self._blocks[b_stop]
            cut_col = s11 - b.offset
            b_l = b[:, :cut_col]
            b_r = b[:, cut_col:]
            b_r.offset += cut_col
            if b_l.size != 0:
                blocks_new_mid.append(b_l)
            if b_r.size != 0:
                blocks_new_end = [b_r] + blocks_new_end

        # If only some of the streams are being replaced,
        # then block merging must be performed.
        if stream_idx.size != self.num_streams:
            b_offs, b_stops = self._get_blocks_union(blocks_new_mid, bs_new)

            # Asseble the new blocks
            incoming_signal = SparseSignal(bs_new, **self.kwargs)
            blocks_new_mid_new = []
            for i in range(b_offs.size):
                b_off = b_offs[i]
                b_stop = b_stops[i]
                b_new = self.getitem((slice(None, None), slice(b_off, b_stop)))
                b_new[stream_idx, :] = incoming_signal.getitem(
                    (slice(None, None), slice(b_off, min(b_stop, s11)))
                )
                blocks_new_mid_new.append(SignalBlock(b_new, b_off))
            blocks_new_mid = blocks_new_mid_new

        # Otherwise, the middle blocks are being simply replaced with the new ones
        else:
            blocks_new_mid = bs_new

        # Insert the result into this sparse signal model
        self._blocks = blocks_new_beg + blocks_new_mid + blocks_new_end
        # merge new beginning blocks if needed
        if len(blocks_new_beg) != 0:
            b2_idx = len(blocks_new_beg)
            b1 = self._blocks[b2_idx - 1]
            b2 = self._blocks[b2_idx]
            if b1.end == b2.offset:
                self._blocks[b2_idx - 1] = b1.append_samples(b2)
                self._blocks.pop(b2_idx)
        # merge new ending blocks if needed
        if len(blocks_new_end) != 0:
            b2_idx = -len(blocks_new_end)
            b1 = self._blocks[b2_idx - 1]
            b2 = self._blocks[b2_idx]
            if b1.end == b2.offset:
                self._blocks[b2_idx - 1] = b1.append_samples(b2)
                self._blocks.pop(b2_idx)

    def set_samples(
        self,
        samples: np.ndarray | Sequence[np.ndarray] | Sequence[SignalBlock],
        offsets: List[int] | None = None,
    ) -> None:
        """Sets given samples into this sparse signal model.

        Usage:
            samples (np.ndarray):
                In this case samples array (a 2D complex matrix) will be divided onto several non-zero blocks.

            samples (Sequence[np.ndarray]):
                In this case all entries (2D complex matrices) will be concatenated among the samples axis and sparsified.

            samples (Sequence[np.ndarray]), offsets(Sequence[integer]):
                In this case SignalBlocks will be constructed directly out of the samples entries, avoiding sparsification.
                Note that number of offsets entries must equal to the number of samples entries.
                Offsets must be sorted in an increasing order and samples entries must not overlap
                (e.g. `offsets[i] + samples[i].shape[1] < offsets[i+1]`).

            samples (SignalBlock):
                In this case given SignalBlock will be resampled and sparcified.
                The block's offset property will be preserved. Consider setting it to zero beforehand (`samples.offset= 0`).

            samples (Sequence[SignalBlock]):
                In this case each entry will be resampled and stored in the model, avoiding sparsification.
                Note that the entries must be sorted by an offset in an increasing order and must not overlap
                (e.g. `samples[i].offset+ samples[i].num_samples < samples[i+1].off`).

            samples (List[SignalBlock]), offsets(Sequence[integer]):
                In this case the same actions will be taken as in the previous case,
                but given offsets will be set into the samples entries before resampling.
                Note that number of offsets entries must equal to the number of samples entries.
        """

        super().set_samples(samples, offsets)

        # Sparsify the blocks
        new_blocks = []
        for b in self._blocks:
            for b_new in self.__from_dense(b):
                if b_new.size != 0:
                    new_blocks.append(b_new)
        self._blocks = new_blocks

    @property
    def title(self) -> str:
        return "Sparse Signal Model"

    @staticmethod
    def Empty(
        sampling_rate: float, num_streams: int = 0, num_samples: int = 0, **kwargs
    ) -> SparseSignal:
        res = SparseSignal(
            np.empty((num_streams, num_samples), np.complex_), sampling_rate, **kwargs
        )
        res._num_streams = num_streams
        return res

    def append_samples(self, signal: Signal | np.ndarray) -> None:
        # Check if number of streams match
        if self.num_streams != signal.shape[0]:
            raise ValueError("Number of streams do not match")

        # Resample the incoming signal
        _signal: Signal | list[np.ndarray]
        if isinstance(signal, Signal):
            _signal = signal.resample(self.sampling_rate)
        else:
            _signal = [signal]

        # Check if all the signal attributes match
        for prop in self.kwargs.items():
            if getattr(_signal, prop[0], prop[1]) != prop[1]:
                raise ValueError(f"Signal attribute {prop[0]} does not match")

        # Sparsify all blocks in the incoming signal
        # and add them to this model
        num_blocks_old = len(self._blocks)
        self_samples_stop = self._blocks[-1].end if len(self._blocks) != 0 else 0
        for b in _signal:
            for b_ in self.__from_dense(b):
                self._blocks.append(b_)
                b_.offset += self_samples_stop

        # Merge the last old and the first new blocks if no gap exists between them
        if len(self._blocks) > num_blocks_old:
            b1 = self._blocks[num_blocks_old - 1]
            b2 = self._blocks[num_blocks_old]
            if b2.offset == b1.end:
                self._blocks[num_blocks_old - 1] = b1.append_samples(b2)
                self._blocks.pop(num_blocks_old)

    def append_streams(self, signal: Signal | np.ndarray) -> None:
        # Resample the incoming signal
        _signal: Signal | list[np.ndarray]
        if isinstance(signal, Signal):
            _signal = signal.resample(self.sampling_rate)
        else:
            _signal = [signal]

        # Check if all the signal attributes match
        for prop in self.kwargs.items():
            if getattr(_signal, prop[0], prop[1]) != prop[1]:
                raise ValueError(f"Signal attribute {prop[0]} does not match")

        # Sparsify all blocks in the incoming signal
        # and add them to this model
        blocks_incoming = []
        for b in _signal:
            for b_ in self.__from_dense(b):
                blocks_incoming.append(b_)

        # Create new resulting blocks
        num_streams = self.num_streams + signal.shape[0]
        b_offs, b_stops = self._get_blocks_union(self._blocks, blocks_incoming)
        is_signal_oftype_signal = issubclass(signal.__class__, Signal)
        blocks_new = []
        for i in range(b_offs.size):
            b_new = np.zeros((num_streams, b_stops[i] - b_offs[i]), np.complex_)
            b1 = self.getitem((slice(None, None), slice(b_offs[i], b_stops[i])))
            if is_signal_oftype_signal:
                b2 = signal.getitem((slice(None, None), slice(b_offs[i], b_stops[i])))  # type: ignore
            else:
                b2 = signal[:, b_offs[i] : b_stops[i]]  # type: ignore
            b_new = np.concatenate((b1, b2), 0)
            blocks_new.append(SignalBlock(b_new, b_offs[i]))

        # Set the result into the model
        self._blocks = blocks_new
        self._num_streams = num_streams
