# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Buffer
from typing import Iterable, Iterator, Literal, Sequence, SupportsIndex, Type, Any, TypeVar
from typing_extensions import override

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from numba import jit
from scipy.constants import pi
from scipy.fft import fft, fftshift, fftfreq
from scipy.ndimage import convolve1d
from scipy.signal import butter, decimate, sosfiltfilt, firwin

from .definitions import InterpolationMode
from .factory import Serializable, SerializationProcess, DeserializationProcess
from .visualize import PlotVisualization, VAT, VisualizableAttribute

__author__ = "Jan Adler"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler", "Egor Achkasov"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


SBT = TypeVar("SBT", bound="SignalBlock")
"""Type variable for SignalBlock and its subclasses."""

ST = TypeVar("ST", bound="Signal")
"""Type variable for Signal and its subclasses."""

AT = TypeVar("AT", bound=np.ndarray)
"""Type variable for numpy ndarray and its subclasses."""


class _SignalVisualization(VisualizableAttribute[PlotVisualization]):
    """Visualization of signal samples."""

    __signal: Signal  # The signal model to be visualized

    def __init__(self, signal: Signal) -> None:
        """
        Args:

            signal: The signal model to be visualized.
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
    @override
    def title(self) -> str:
        return self.signal.title

    @override
    def create_figure(
        self, space: Literal["time", "frequency", "both"] = "both", **kwargs
    ) -> tuple[Figure, VAT]:
        return plt.subplots(self.signal.num_streams, 2 if space == "both" else 1, squeeze=False)

    @override
    def _prepare_visualization(
        self,
        figure: Figure | None,
        axes: VAT,
        angle: bool = False,
        space: Literal["time", "frequency", "both"] = "both",
        legend: bool = True,
        **kwargs,
    ) -> PlotVisualization:
        # Prepare axes and lines
        lines = np.empty((axes.shape[0], axes.shape[1]), dtype=np.object_)
        zeros = np.zeros(self.signal.num_samples, dtype=np.float64)
        timestamps = self.signal.timestamps
        for stream_idx in range(self.signal.num_streams):
            ax_y_idx = 0
            if space in {"both", "time"}:
                ax: Axes = axes[stream_idx, ax_y_idx]
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
                freq_ax: Axes = axes[stream_idx, ax_y_idx]
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

    @override
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
            self.signal.view(np.ndarray), visualization.axes, visualization.lines
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

        from hermespy.simulation import SimulatedDevice
        from hermespy.modem import TransmittingModem, RaisedCosineWaveform

        device = SimulatedDevice()
        transmitter = TransmittingModem()
        waveform = RaisedCosineWaveform(modulation_order=16, oversampling_factor=16, num_preamble_symbols=0, symbol_rate=1e8, num_data_symbols=1000, roll_off=.9)
        transmitter.waveform = waveform
        device.add_dsp(transmitter)

        device.transmit().mixed_signal.eye(symbol_duration=1/waveform.symbol_rate, domain='time')
        plt.show()


    or on the complex plane

    .. plot::

        import matplotlib.pyplot as plt

        from hermespy.simulation import SimulatedDevice
        from hermespy.modem import TransmittingModem, RaisedCosineWaveform

        device = SimulatedDevice()
        transmitter = TransmittingModem()
        waveform = RaisedCosineWaveform(modulation_order=16, oversampling_factor=16, num_preamble_symbols=0, symbol_rate=1e8, num_data_symbols=1000, roll_off=.9)
        transmitter.waveform = waveform
        device.add_dsp(transmitter)

        device.transmit().mixed_signal.eye(symbol_duration=1/waveform.symbol_rate, domain='complex')
        plt.show()

    Args:

        symbol_duration:
            Assumed symbol repetition interval in seconds.
            Will be rounded to match the signal model's sampling rate.

        line_width:
            Line width of a single plot line.

        title:
            Title of the plotted figure.
            `Eye Diagram` by default.

        domain:
            Plotting behaviour of the eye diagram.
            `time` by default.
            See above examples for rendered results.

        legend:
            Display a plot legend.
            Enabled by default.
            Only considered when in `time` domain plotting mode.

        linewidth:
            Line width of the eye plot.
            :math:`.75` by default.

        symbol_cutoff:
            Relative amount of symbols ignored during plotting.
            :math:`0.1` by default.
            This is required to properly visualize intersymbol interferences within the communication frame,
            since special effects may occur at the start and end.
    """

    @property
    @override
    def title(self) -> str:
        return "Eye Diagram"

    @override
    def _prepare_visualization(
        self,
        figure: Figure | None,
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

        _ax: Axes = axes.flat[0]
        colors = self._get_color_cycle()

        _symbol_duration = symbol_duration if symbol_duration else 1 / self.signal.sampling_rate
        symbol_num_samples = int(_symbol_duration * self.signal.sampling_rate)
        num_symbols = self.signal.num_samples // symbol_num_samples
        num_cutoff_symbols = int(num_symbols * symbol_cutoff)
        if num_cutoff_symbols < 2:
            num_cutoff_symbols = 2
        values_per_symbol = 2 * symbol_num_samples + 2
        num_visualized_symbols = num_symbols - 2 * num_cutoff_symbols

        lines: list[Line2D] = []

        if domain == "time":
            _ax.set_xlabel("Time-Domain [s]")
            _ax.set_ylabel("Normalized Amplitude")
            _ax.set_ylim((-1.1, 1.1))
            _ax.set_xlim((-1.0, 1.0))

            timestamps = (
                np.arange(-symbol_num_samples - 1, 1 + symbol_num_samples, 1) / symbol_num_samples
            )
            times = np.hstack([timestamps] * num_visualized_symbols)
            values = np.zeros_like(times, dtype=np.float64)
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

            plot_data: list[np.ndarray] = []
            for block in self.signal.blocks:
                cut_block = (
                    block[:, num_cutoff_samples:-num_cutoff_samples].view(np.ndarray).flatten()
                )
                plot_data.append(cut_block.real)
                plot_data.append(cut_block.imag)
            lines.extend(_ax.plot(*plot_data, color=colors[0], linewidth=linewidth))

        else:
            raise ValueError(f"Unsupported plotting domain '{domain}'")

        lines_array = np.empty((axes.shape[0], axes.shape[1]), dtype=np.object_)
        lines_array.flat[0] = lines
        return PlotVisualization(figure, axes, lines_array)

    @override
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

        samples = self.signal.view(np.ndarray)
        if domain == "time":
            values = np.empty(num_visualized_symbols * values_per_symbol, dtype=np.complex128)
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


class SignalBlock(np.ndarray[tuple[int, int], np.dtype[np.complex128]], Serializable):
    """A TxN matrix of complex entries representing signal samples over T streams and N time samples.

    Used in Signal to store pieces of signal samples.
    Use `offset` property to set the offset of this block relative to a signal start sample.
    """

    _DEFAULT_OFFSET: int = 0
    _offset: int

    def __new__(
        cls,
        num_streams: int,
        num_samples: int,
        offset: int = _DEFAULT_OFFSET,
        buffer: Buffer | None = None,
    ):
        """Create a new SignalBlock instance.yx

        Args:
            num_streams: Number of dedicated streams within this signal block.
            num_samples: Number of samples within this signal block.
            offset: Integer sample offset of this block relative to a signal start sample.
            buffer: Optional buffer to initialize the samples from.
        Raises:
            ValueError: If ndim of given samples is bigger then 2.
        """

        # Create a ndarray
        obj = np.ndarray.__new__(cls, (num_streams, num_samples), np.complex128, buffer=buffer)

        # Assign additional offset attribute
        obj.offset = offset
        return obj

    @override
    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return  # pragma: no cover

        # Assert the block's shape is two-dimensional
        if self.ndim != 2:
            raise ValueError(
                "HermesPy signal models must be two-dimensional, use .view(np.ndarray) before slicing or correct the slices to be two-dimensional"
            )

        self.offset = getattr(obj, "offset", self._DEFAULT_OFFSET)

    @property
    def num_streams(self) -> int:
        """The number of streams within this signal block.

        Returns: The number of streams.
        """

        return self.shape[0] if self.ndim > 0 else 0

    @property
    def num_samples(self) -> int:
        """The number of samples within this signal block.

        Returns: The number of samples.
        """

        return self.shape[1] if self.ndim > 1 else 0

    @property
    def offset(self) -> int:
        """The dense signal block's offset relative to the assumed signal start.

        Raises:
            ValueError: If the offset is smaller than zero.
        """

        return self._offset

    @offset.setter
    def offset(self, value: int) -> None:

        if value < 0:
            raise ValueError("Offset must be non-zero")
        self._offset = value

    @property
    def end(self) -> int:
        """Get block stop sample.
        b.num_samples = b.end - b.off"""

        return self._offset + self.num_samples

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

        return np.linalg.norm(self, axis=1) ** 2

    @override
    def max(self, /, *args, **kwargs):
        # Recasting ensures the 2D constraint is bypassed
        return self.view(np.ndarray).max(*args, **kwargs)

    @override
    def min(self, /, *args, **kwargs):
        # Recasting ensures the 2D constraint is bypassed
        return self.view(np.ndarray).min(*args, **kwargs)

    def resample_block(
        self, sampling_rate_old: float, sampling_rate_new: float, aliasing_filter: bool = True
    ) -> SignalBlock:
        """Resample a signal block to a different sampling rate.

        Args:

            sampling_rate_old:
                Old sampling rate of the signal block in Hz.

            sampling_rate_new:
                New sampling rate of the signal block in Hz.

            aliasing_filter:
                Apply an anti-aliasing filter during downsampling.
                Enabled by default.

        Returns: The resampled signal block.

        Raises:
            ValueError: If `sampling_rate_old` or `sampling_rate_new` are smaller or equal to zero.
        """

        # Validate given sampling rate
        if sampling_rate_old <= 0.0 or sampling_rate_new <= 0.0:
            raise ValueError("Sampling rate must be greater than zero.")

        # Skip resampling if the same sampling rate was given
        if sampling_rate_old == sampling_rate_new:
            return self.copy()

        off_new = round(self.offset * sampling_rate_new / sampling_rate_old)

        # Avoid resampling if this block does not contain samples
        if self.num_samples < 1:
            return self.copy()

        # Quickly change the number of samples if there are no streams
        if self.num_streams < 1:
            return SignalBlock(
                0, int(self.num_samples * sampling_rate_new / sampling_rate_old), off_new
            )

        # Init
        samples_new = self.view(np.ndarray)  # .copy()

        # If decimation is possible, simply decimate the samples
        if sampling_rate_new < sampling_rate_old:
            decimation = sampling_rate_old / sampling_rate_new
            if decimation % 1.0 == 0.0:
                samples_new = decimate(samples_new, int(decimation), axis=1)
                off_new = round(self.offset * sampling_rate_new / sampling_rate_old)
                return SignalBlock(
                    samples_new.shape[0], samples_new.shape[1], off_new, samples_new.tobytes()
                )

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
        return SignalBlock(self.num_streams, samples_new.shape[1], off_new, samples_new.tobytes())

    def append_samples(self, value: np.ndarray) -> SignalBlock:
        """Append samples to this signal block. Creates a new instance."""

        # Validate value
        if value.shape[0] != self.num_streams:
            raise ValueError("Shape mismatch")

        return SignalBlock(
            self.num_streams,
            self.num_samples + value.shape[1],
            self.offset,
            np.append(self, value, 1).tobytes(),
        )

    def superimpose_to(
        self,
        target_sampling_rate: float,
        target_carrier_frequency: float,
        sampling_rate: float,
        carrier_frequency: float,
    ) -> SignalBlock:
        # Ensure sampling rates are valid
        if target_sampling_rate <= 0.0 or sampling_rate <= 0.0:
            raise ValueError("Sampling rates must be greater than zero.")

        # Abort if parmeters match
        if target_sampling_rate == sampling_rate and target_carrier_frequency == carrier_frequency:
            return self

        # Frequency distance between the two signals' carrier frequencies
        frequency_distance = target_carrier_frequency - carrier_frequency

        # Bandwidth of the overlapping frequency ranges of the two signals
        overlap_bandwidth = 0.5 * (target_sampling_rate + sampling_rate) - abs(frequency_distance)

        # Abort if the signals do not overlap in frequency
        if overlap_bandwidth <= 0.0:
            return np.zeros_like(self, dtype=np.complex128).view(SignalBlock)

        # Design a complex bandpass filter to extract only the overlapping frequency range
        # from this signal block
        # Note: This approach should ideally be replaced with a more efficient digital filter
        filter_order = 256
        filter_coefficients = firwin(
            1 + filter_order,
            0.5 * overlap_bandwidth,
            width=0.5 * overlap_bandwidth,
            fs=sampling_rate,
        ).astype(complex)
        filter_coefficients *= np.exp(
            2j * np.pi * 0.5 * frequency_distance / sampling_rate * np.arange(1 + filter_order)
        )

        # Apply the bandpass filter to the samples of this signal block
        filtered_samples = convolve1d(
            self.view(np.ndarray), filter_coefficients, axis=1, mode="constant"
        )

        # Move the filtered samples to the target carrier frequency
        # ToDo: Reminder, mixing like this currently does not account for possible delays.
        if frequency_distance != 0.0:
            mix_sinusoid = np.exp(
                -2j * pi * np.arange(self.shape[1]) * frequency_distance / sampling_rate
            )
            mixed_samples = filtered_samples * mix_sinusoid[None, :]
        else:
            mixed_samples = filtered_samples

        # Superimpose the mixed samples onto the target samples
        return SignalBlock(
            self.num_streams, self.num_samples, self.offset, bytearray(mixed_samples)
        )

    @staticmethod
    def _resample_antialiasing(samples: np.ndarray, N: Any, Wn: Any) -> np.ndarray:
        """Internal subroutine for resampled method. Applies Butterworth filter to the given samples.

        Args:
            samples: Samples to apply AA to.
            N: The first two parameters of scipy.signal.butter.
        """

        aliasing_filter = butter(N, Wn, btype="low", output="sos")
        return sosfiltfilt(aliasing_filter, samples, axis=1)

    @staticmethod
    @jit(nopython=True)
    def _resample(
        signal: np.ndarray, input_sampling_rate: float, output_sampling_rate: float
    ) -> np.ndarray:  # pragma: no cover
        """Internal subroutine to resample a given set of samples to a new sampling rate.

        Uses sinc-interpolation, therefore `signal` is assumed to be band-limited.

        Arguments:
            signal:
                TxM matrix of T signal-streams to be resampled, each containing M time-discrete samples.

            input_sampling_rate:
                Rate at which `signal` is sampled in Hz.

            output_sampling_rate:
                Rate at which the resampled signal will be sampled in Hz.

        Returns:
            MxT' matrix of M resampled signal streams.
            The number of samples T' depends on the sampling rate relations, e.i.
            T' = T * output_sampling_rate / input_sampling_rate .
        """

        num_streams = signal.shape[0]

        num_input_samples = signal.shape[1]
        num_output_samples = round(num_input_samples * output_sampling_rate / input_sampling_rate)

        input_timestamps = np.arange(num_input_samples) / input_sampling_rate
        output_timestamps = np.arange(num_output_samples) / output_sampling_rate

        output = np.empty((num_streams, num_output_samples), dtype=np.complex128)
        for output_idx in np.arange(num_output_samples):
            # Sinc interpolation weights for single output sample
            interpolation_weights = (
                np.sinc((input_timestamps - output_timestamps[output_idx]) * input_sampling_rate)
                + 0j
            )

            # Resample all streams simultaneously
            output[:, output_idx] = signal @ interpolation_weights

        return output

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_array(self, "samples")
        process.serialize_integer(self.offset, "offset")

    @classmethod
    @override
    def Deserialize(cls: Type[SignalBlock], process: DeserializationProcess) -> SignalBlock:
        samples = process.deserialize_array("samples", np.complex128)
        offset = process.deserialize_integer("offset", 0)
        return cls(samples.shape[0], samples.shape[1], offset, buffer=samples)

    @override
    def __reduce__(self) -> tuple:
        return SignalBlock.__new__, (
            SignalBlock,
            self.shape[0],
            self.shape[1],
            self.offset,
            self.view(np.ndarray),
        )

    @override
    def __getitem__(self: SBT, key: Any) -> SBT:
        # Note: This override ensures that mypy correctly recognizes the return type as SignalBlock
        return super().__getitem__(key)  # type: ignore


class Signal(ABC, Iterable, Serializable):
    """Abstract base class for all signal models in HermesPy."""

    _DEFAULT_CARRIER_FREQUENCY = 0.0
    _DEFAULT_NOISE_POWER = 0.0
    _DEFAULT_DELAY = 0.0
    filter_order: int = 10

    _sampling_rate: float
    _carrier_frequency: float
    _noise_power: float
    _samples_visualization: _SamplesVisualization
    _eye_visualization: _EyeVisualization
    _delay: float

    @abstractmethod
    def __mul__(self: ST, value: complex) -> ST:
        """Multiply all samples in the signal model by a complex scalar."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def blocks(self) -> Sequence[SignalBlock]:
        """Individual dense blocks this signal model consists of."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def block_offsets(self) -> Sequence[int]:
        """List of integer offsets of each block relative to the signal start sample."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        """The number of dense blocks within this signal model."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def num_samples(self) -> int:
        """The number of discrete time-domain samples within this signal model."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def num_streams(self) -> int:
        """The number of parallel signal streams within this signal model."""
        ...  # pragma: no cover

    @abstractmethod
    def append_samples(self: ST, value: Signal | np.ndarray) -> ST:
        """Append samples in time-domain to the signal model.

        Args:
            value: The signal samples to be appended.

        Returns:
            A new signal model instance with the appended samples.

        Raise:
            ValueError: If the number of streams don't align.
        """
        ...  # pragma: no cover

    @abstractmethod
    def append_streams(self: ST, value: Signal | np.ndarray) -> ST:
        """Append streams to the signal model.

        Args:
            value: The signal samples to be appended.

        Returns:
            A new signal model instance with the appended streams.

        Raise:
            ValueError: If the number of samples don't align.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def energy(self) -> np.ndarray:
        """Sum of squared voltages of all streams in the signal model."""
        ...  # pragma: no cover

    @property
    @abstractmethod
    def power(self) -> np.ndarray:
        """Mean squared voltage of each modeled stream.

        Note that, in case of sparse signals, the power is only computed over the non-zero regions.
        """
        ...  # pragma: no cover

    @abstractmethod
    def view(self, dtype: type[AT]) -> AT:
        """View the samples of the signal model as a numpy ndarray of the given dtype.

        Args:
            dtype: The desired numpy dtype to view the samples as.

        Returns:
            The samples of the signal model as a numpy ndarray of the given dtype.
        """
        ...  # pragma: no cover

    @abstractmethod
    def __getitem__(
        self, key: tuple[slice | SupportsIndex | Sequence[SupportsIndex], ...]
    ) -> Signal:
        """Get a slice of samples from the signal model."""
        ...  # pragma: no cover

    @abstractmethod
    def __setitem__(self, key: Any, value: Any) -> None:
        """Set an np.ndarray of samples into the model."""
        ...  # pragma: no cover

    @staticmethod
    def Create(
        samples: np.ndarray[tuple[int, int], np.dtype[np.complex128]] | Sequence[SignalBlock],
        sampling_rate: float = 1.0,
        carrier_frequency: float = 0.0,
        noise_power: float = 0.0,
        delay: float = 0.0,
        offsets: Sequence[int] | None = None,
    ) -> Signal:
        """Creates a signal model instance given signal samples.

        Subclasses of Signal should reroute the given arguments to init and return the result.

        Args:

            samples:
                Single or a sequence of 2D matricies with shapes MxT_i, where M - number of streams, T_i - number of samples in the matrix i.
                Note that M for each entry of the sequence must be the same.
                SignalBlock and Sequence[SignalBlock] can also be passed here.

            sampling_rate:
                Sampling rate of the signal in Hz.

            offsets:
                Integer offsets of the samples if given in a sequence.
                Ignored if samples is not a Sequence of np.ndarray.
                len(offsets) must be equal to len(samples).
                Offset number i must be greater then offset i-1 + samples[i-1].shape[1].

        Returns:
            SparseSignal if samples argument is a list of ndarrays. DenseSignal otherwise.
        """

        if isinstance(samples, Sequence) and len(samples) != 1:
            return SparseSignal(samples, sampling_rate, carrier_frequency, noise_power, delay)

        dense_delay = delay
        dense_delay += offsets[0] / sampling_rate if offsets and len(offsets) > 0 else 0.0
        return DenseSignal.FromNDArray(
            # For now, samples are cast to complex128 to avoid issues with deserialization
            samples[0] if isinstance(samples, Sequence) else samples,
            sampling_rate,
            carrier_frequency,
            noise_power,
            dense_delay,
        )

    @staticmethod
    def Empty(sampling_rate: float, num_streams: int = 0, num_samples: int = 0, **kwargs) -> Signal:
        """Creates an empty signal model instance. Initializes it with the given arguments.
        If both num_streams and num_samples are not 0, then initilizes samples with np.empty.
        """

        return Signal.Create(
            np.empty((num_streams, num_samples), dtype=np.complex128), sampling_rate, **kwargs
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Two dimensional tuple of shape :math:`M \\times N`.

        :math:`M` is the number of streams and :math:`N` is the number of samples.
        """

        return (self.num_streams, self.num_samples)

    @property
    def sampling_rate(self) -> float:
        """The rate at which the modeled signal was sampled in Hz."""

        return self._sampling_rate

    @sampling_rate.setter
    def sampling_rate(self, value: float) -> None:
        """Modify the rate at which the modeled signal was sampled.

        Args:
            value: The sampling rate in Hz.

        Raises:
            ValueError: If `value` is smaller or equal to zero.
        """

        if value <= 0.0:
            raise ValueError("The sampling rate of modeled signals must be greater than zero")

        self._sampling_rate = value

    @property
    def carrier_frequency(self) -> float:
        """The center frequency of the modeled signal in the radio-frequency transmit band in Hz."""

        return self._carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:
        """Modify the center frequency of the modeled signal in the radio-frequency transmit band.

        Args:
            value: The carrier frequency in Hz.

        Raises:
            ValueError: If `value` is smaller than zero.
        """

        if value < 0.0:
            raise ValueError("The carrier frequency of modeled signals must be non-negative")

        self._carrier_frequency = value

    @property
    def noise_power(self) -> float:
        """Noise power of the superimposed noise signal.

        Raises:
            ValueError: If the noise power is smaller than zero.
        """

        return self._noise_power

    @noise_power.setter
    def noise_power(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Noise power must be greater or equal to zero")

        self._noise_power = value

    @property
    def delay(self) -> float:
        """Delay of the modeled signal in seconds.

        The delay is relative to a common time-reference, e.i. the start of a simulation.

        Raises:
            ValueError: If the delay is smaller than zero.
        """

        return self._delay

    @delay.setter
    def delay(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Delay must be greater or equal to zero")

        self._delay = value

    @property
    def timestamps(self) -> np.ndarray:
        """The sample-points of the signal block.

        Vector of length T containing sample-timestamps in seconds.
        """

        return np.arange(self.num_samples) / self.sampling_rate

    @property
    def frequencies(self) -> np.ndarray:
        """The signal model's discrete sample points in frequcy domain.

        Numpy vector of frequency bins.
        """

        return fftfreq(self.num_samples, 1 / self.sampling_rate)

    @property
    def duration(self) -> float:
        """Signal model duration in time-domain.

        Duration in seconds.
        """

        return self.num_samples / self.sampling_rate

    @property
    def title(self) -> str:
        return "Signal"  # pragma: no cover

    @property
    def plot(self) -> _SamplesVisualization:
        """Visualize the samples of the signal model."""

        return self._samples_visualization

    @property
    def eye(self) -> _EyeVisualization:
        """Visualize the eye diagram of the signal model."""

        return self._eye_visualization

    @abstractmethod
    def resample(self: ST, sampling_rate: float, aliasing_filter: bool = True) -> ST:
        """Resample the modeled signal to a different sampling rate.

        Args:

            sampling_rate:
                Sampling rate of the new signal model in Hz.

            aliasing_filter:
                Apply an anti-aliasing filter during downsampling.
                Enabled by default.

        Returns: The resampled signal model.

        Raises:
            ValueError: If `sampling_rate` is smaller or equal to zero.
        """
        ...  # pragma: no cover

    @abstractmethod
    def superimpose(
        self,
        added_signal: Signal,
        resample: bool = True,
        aliasing_filter: bool = True,
        stream_indices: Sequence[int] | None = None,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
    ) -> Signal:
        """Superimpose an additive signal model to this model.

        Internally re-samples `added_signal` to this model's sampling rate, if required.
        Mixes `added_signal` according to the carrier-frequency distance.

        If the delays of this signal and `added_signal` differ, the signals will be aligned according to their delays.
        `added_signals` with delays smaller than this signal's delay will be cut accordingly (as opposed to changing the start time of the resulting signal).

        Args:
            added_signal: The signal to be superimposed onto this one.
            resample: Allow for dynamic resampling during superposition.
            aliasing_filter: Apply an anti-aliasing filter during mixing.
            stream_indices: Indices of the streams to be mixed.
            interpolation:
                Interpolation strategy to be used for superimposed signals of different delays.
                Required if the delays of this signal and `added_signal` differ and are not integer multiples of the sampling period.

        Raises:

            ValueError: If `added_signal` contains a different number of streams than this signal model.
            RuntimeError: If resampling is required but not allowd.
            NotImplementedError: If the delays if this signal and `added_signal` differ.
        """
        ...  # pragma: no cover

    @abstractmethod
    def to_dense(self) -> DenseSignal:
        """Concatenate all the blocks in the signal into one block.
        Accounts for offsets by inserting zeros where necessary.

        .. note::
           If offset values are too big, memory overflow is possible.
           Instead, iterating over the signal's blocks is recommended.

        Returns:
            Dense form for this signal
        """
        ...  # pragma: no cover

    def to_interleaved(self, data_type: Type = np.int16, scale: bool = True) -> np.ndarray:
        """Convert the complex-valued floating-point model samples to interleaved integers.

        Args:

            data_type:
                Numpy resulting data type.

            scale:
                Scale the floating point values to stretch over the whole range of integers.

        Returns:
            Numpy array of interleaved samples.
            Will contain double the samples in time-domain.
        """

        # Get samples
        samples = self.to_dense().copy().view(np.ndarray)

        # Scale samples if required
        if (
            scale and samples.shape[1] > 0 and (samples.max() > 1.0 or samples.min() < 1.0)
        ):  # pragma: no cover
            samples /= np.max(abs(samples))
            samples *= np.iinfo(data_type).max

        return samples.view(np.float64).astype(data_type)

    @classmethod
    def FromInterleaved(
        cls,
        interleaved_samples: np.ndarray[tuple[int, int], np.dtype[np.int16]],
        scale: bool = True,
        **kwargs,
    ) -> Signal:
        """Initialize a signal model from interleaved samples.

        Args:

            interleaved_samples:
                Numpy array of interleaved samples.

            scale:
                Scale the samples after interleaving

            kwargs:
                Additional class initialization arguments.
        """

        complex128samples: np.ndarray[tuple[int, int], np.dtype[np.complex128]] = interleaved_samples.astype(np.float64).view(np.complex128)  # type: ignore

        if scale:  # pragma: no cover
            complex128samples /= np.iinfo(interleaved_samples.dtype).max

        return cls.Create(complex128samples, **kwargs)

    @abstractmethod
    def copy(self: ST) -> ST:
        """Create a deep copy of this signal model.

        Returns: A deep copy of this signal model.
        """
        ...  # pragma: no cover


class DenseSignal(SignalBlock, Signal):  # type: ignore[misc]
    """Dense signal model class."""

    def __new__(
        cls,
        num_streams: int,
        num_samples: int,
        sampling_rate: float,
        carrier_frequency: float = Signal._DEFAULT_CARRIER_FREQUENCY,
        noise_power: float = Signal._DEFAULT_NOISE_POWER,
        delay: float = Signal._DEFAULT_DELAY,
        buffer: Buffer | None = None,
    ):
        """Create a new DenseSignal instance.

        Args:
            num_streams: Number of dedicated streams within this signal.
            num_samples: Number of samples within this signal.
            sampling_rate: Sampling rate of the signal in Hz.
            carrier_frequency: Carrier frequency of the signal in Hz.
            no_delayower: Noise power of the signal.
            delay: Delay of the signal in seconds.
            buffer: Optional buffer to initialize the samples from.
        """

        # Init base class
        dense = SignalBlock.__new__(cls, num_streams, num_samples, 0, buffer)

        # Add attributes
        dense.sampling_rate = sampling_rate
        dense.carrier_frequency = carrier_frequency
        dense.noise_power = noise_power
        dense.delay = delay

        # Debug: Add plot visualizations
        dense._samples_visualization = _SamplesVisualization(dense)
        dense._eye_visualization = _EyeVisualization(dense)

        # Return initalized instance
        return dense

    @staticmethod
    def Zeros(
        num_streams: int,
        num_samples: int,
        sampling_rate: float,
        carrier_frequency: float = Signal._DEFAULT_CARRIER_FREQUENCY,
        noise_power: float = Signal._DEFAULT_NOISE_POWER,
        delay: float = Signal._DEFAULT_DELAY,
    ) -> DenseSignal:
        """Create a new zero-initialized dense signal.

        Args:
            num_streams: Number of streams in the signal.
            num_samples: Number of samples in the signal.
            sampling_rate: Sampling rate of the signal in Hz.
            carrier_frequency: Carrier frequency of the signal in Hz.
            noise_power: Noise power of the signal.
            delay: Delay of the signal in seconds.

        Returns:
            The created signal model.
        """

        return DenseSignal(
            num_streams,
            num_samples,
            sampling_rate,
            carrier_frequency,
            noise_power,
            delay,
            buffer=np.zeros((num_streams, num_samples), dtype=np.complex128),
        )

    @classmethod
    def FromNDArray(
        cls,
        array: np.ndarray[tuple[int, int], np.dtype[np.complex128]],
        sampling_rate: float,
        carrier_frequency: float = Signal._DEFAULT_CARRIER_FREQUENCY,
        noise_power: float = Signal._DEFAULT_NOISE_POWER,
        delay: float = Signal._DEFAULT_DELAY,
    ) -> DenseSignal:
        """Create a new dense signal from a given numpy ndarray.

        Args:
            array:
                Array containing the signal samples.
                Must be of shape (num_streams, num_samples).
            sampling_rate: Sampling rate of the signal in Hz.
            carrier_frequency: Carrier frequency of the signal in Hz.
            noise_power: Noise power of the signal.
            delay: Delay of the signal in seconds.

        Returns:
            The created dense signal model.
        """

        # Assert array is 2D
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D ndarray, but got {array.ndim}D")

        # Cast ndarray to dense signal
        dense_signal = array.astype(np.complex128).view(DenseSignal)

        # Update additional attributes
        dense_signal.sampling_rate = sampling_rate
        dense_signal.carrier_frequency = carrier_frequency
        dense_signal.noise_power = noise_power
        dense_signal.delay = delay

        return dense_signal

    @property
    @override
    def blocks(self) -> list[SignalBlock]:
        # Since, per definition, dense signals represent only a single block
        # its blocks property only contains a reference to itself
        return [self.view(SignalBlock)]

    @property
    @override
    def block_offsets(self) -> list[int]:
        # Since, per definition, dense signals represent only a single block
        # its block_offsets property only contains a reference to its own offset
        return [0]

    @property
    @override
    def num_blocks(self) -> int:
        # Since, per definition, dense signals represent only a single block
        # its num_blocks property is always 1
        return 1

    @override
    def __array_finalize__(self, obj: np.ndarray | DenseSignal | None) -> None:
        if obj is None:
            return  # pragma: no cover

        # Finalize signal block
        SignalBlock.__array_finalize__(self, obj)

        # Finalize dense signal
        self.carrier_frequency = getattr(obj, "carrier_frequency", self._DEFAULT_CARRIER_FREQUENCY)
        self.sampling_rate = getattr(obj, "sampling_rate", 1.0)
        self.noise_power = getattr(obj, "noise_power", self._DEFAULT_NOISE_POWER)
        self.delay = getattr(obj, "delay", self._DEFAULT_DELAY)
        self._samples_visualization = _SamplesVisualization(self)
        self._eye_visualization = _EyeVisualization(self)

    def _correct_key_slicing(self, key) -> Any:
        """Correct the key slicing to ensure two-dimensional output.

        Args:
            key: The key to be corrected.

        Returns: The corrected key.
        """

        if isinstance(key, int):
            key = (key, slice(None))
        if isinstance(key, Sequence) and isinstance(key[0], int):
            key = ([key[0]], *key[1:])

        return key

    @override
    def __getitem__(self, key) -> DenseSignal:
        key = self._correct_key_slicing(key)
        # Numpy will automatically return a view of the same class
        # by calling __array_finalize. This is not represented by the type-hinting
        return SignalBlock.__getitem__(self, key)  # type: ignore

    @override
    def __setitem__(self, key, value) -> None:
        key = self._correct_key_slicing(key)
        SignalBlock.__setitem__(self, key, value)

    @property
    def title(self) -> str:
        return "Dense Signal Model"

    @override
    def append_samples(self, value: Signal | np.ndarray) -> DenseSignal:
        # Resample the signal
        addedblocks_resampled: Sequence[SignalBlock]
        if isinstance(value, Signal):
            addedblocks_resampled = value.resample(self.sampling_rate).blocks

        else:
            addedblocks_resampled = [
                SignalBlock(value.shape[0], value.shape[1], 0, value.tobytes())
            ]

        # Allocate new memory
        new_num_samples = self.num_samples + max(b.end for b in addedblocks_resampled)
        new_signal = DenseSignal.Zeros(
            self.num_streams,
            new_num_samples,
            self.sampling_rate,
            self.carrier_frequency,
            self.noise_power,  # Noise power is technically undefined
            self.delay,
        )

        # Copy old samples
        new_signal[:, : self.num_samples] = self

        # Append all blocks from the signal to self block
        for b in addedblocks_resampled:
            new_signal[:, self.num_samples + b.offset : self.num_samples + b.end] = b

        return new_signal

    @override
    def append_streams(self, value: Signal | np.ndarray) -> DenseSignal:
        # Resample the signal
        addedblocks_resampled: Sequence[SignalBlock]
        if isinstance(value, Signal):
            addedblocks_resampled = value.resample(self.sampling_rate).blocks

        else:
            addedblocks_resampled = [
                SignalBlock(value.shape[0], value.shape[1], 0, value.tobytes())
            ]

        # Allocate new memory
        new_num_streams = self.num_streams + max(b.num_streams for b in addedblocks_resampled)
        new_signal = DenseSignal.Zeros(
            new_num_streams,
            self.num_samples,
            self.sampling_rate,
            self.carrier_frequency,
            self.noise_power,  # Noise power is technically undefined
            self.delay,
        )

        # Copy old samples
        new_signal[: self.num_streams, :] = self

        # Append all blocks from the signal to self block
        for b in addedblocks_resampled:
            new_signal[self.num_streams : self.num_streams + b.num_streams, b.offset : b.end] = b

        return new_signal

    @override
    def resample(self, sampling_rate: float, aliasing_filter: bool = True) -> DenseSignal:
        return DenseSignal.FromNDArray(
            self.resample_block(self.sampling_rate, sampling_rate, aliasing_filter).view(
                np.ndarray
            ),
            sampling_rate,
            self.carrier_frequency,
            self.noise_power,
            self.delay,
        )

    @override
    def superimpose(
        self,
        added_signal: Signal,
        resample: bool = True,
        aliasing_filter: bool = True,
        stream_indices: Sequence[int] | slice | None = None,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
    ) -> Signal:
        if stream_indices is None:
            stream_indices = slice(None)

        # If this signal is empty, simply return the added signal
        if self.size < 1:
            return added_signal

        # Resample the added signal to the target sampling rate
        if added_signal.sampling_rate != self.sampling_rate:
            if not resample:
                raise RuntimeError(
                    "Resampling is required, but not allowed. "
                    "Set resample=True to allow for resampling."
                )
            _added_signal = added_signal.resample(
                self.sampling_rate, aliasing_filter=aliasing_filter
            )

        # If no resampling is required, simply use the added signal
        else:
            _added_signal = added_signal

        # Account for possible delay differences
        if self.delay != _added_signal.delay:
            if interpolation == InterpolationMode.NEAREST:
                if self.delay > _added_signal.delay:
                    _added_signal = _added_signal[
                        :,
                        int(
                            round((self.delay - _added_signal.delay) * _added_signal.sampling_rate)
                        ) :,
                    ]
                else:
                    _added_signal = _added_signal[
                        :,
                        : _added_signal.num_samples
                        - int(
                            round((_added_signal.delay - self.delay) * _added_signal.sampling_rate)
                        ),
                    ]

            else:
                raise NotImplementedError(
                    f"Interpolation mode {interpolation} is not implemented yet."
                )

        if (
            _added_signal.num_streams <= self.num_streams
            and _added_signal.num_samples <= self.num_samples
        ):
            return_signal = self

        else:
            # Generate a new dense signal with enough memory to hold both signals
            new_num_streams = max(self.num_streams, _added_signal.num_streams)
            new_num_samples = max(self.num_samples, _added_signal.num_samples)
            dense_signal = DenseSignal.Zeros(
                new_num_streams,
                new_num_samples,
                self.sampling_rate,
                self.carrier_frequency,
                self.noise_power,
                self.delay,
            )

            # Add the current signal to the new dense signal
            dense_signal[: self.num_streams, : self.num_samples] += self  # type: ignore
            return_signal = dense_signal

        # Add the added_signal to the new dense signal
        _stream_indices = slice(None) if stream_indices is None else stream_indices
        for b in _added_signal.blocks:
            return_signal[_stream_indices, b.offset : b.end] += b[_stream_indices, :].superimpose_to(  # type: ignore
                return_signal.sampling_rate,
                return_signal.carrier_frequency,
                _added_signal.sampling_rate,
                _added_signal.carrier_frequency,
            )

        # Tadaaa
        return return_signal

    @override
    def to_dense(self) -> DenseSignal:
        return self

    @override
    def __repr__(self) -> str:
        return self.__str__()

    @override
    def __str__(self) -> str:
        return f"{type(self).__name__}(num_streams={self.num_streams}, num_samples={self.num_samples}, sampling_rate={self.sampling_rate}, carrier_frequency={self.carrier_frequency})"

    @override
    def copy(self, order: Literal["K", "A", "C", "F"] | None = None) -> DenseSignal:
        """Copy the dense signal samples into a new signal model with identical parameters."""

        return DenseSignal(
            self.num_streams,
            self.num_samples,
            self.sampling_rate,
            self.carrier_frequency,
            self.noise_power,
            self.delay,
            bytearray(self),
        )

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_array(self.view(np.ndarray).astype(np.complex128), "samples")
        process.serialize_floating(self.sampling_rate, "sampling_rate")
        process.serialize_floating(self.carrier_frequency, "carrier_frequency")
        process.serialize_floating(self.noise_power, "noise_power")
        process.serialize_floating(self.delay, "delay")

    @classmethod
    @override
    def Deserialize(cls: Type[DenseSignal], process: DeserializationProcess) -> DenseSignal:
        samples = process.deserialize_array("samples", np.complex128)
        return cls(
            samples.shape[0],
            samples.shape[1],
            process.deserialize_floating("sampling_rate"),
            process.deserialize_floating("carrier_frequency", 0.0),
            process.deserialize_floating("noise_power", 0.0),
            process.deserialize_floating("delay", 0.0),
            samples.tobytes(),
        )

    @override
    def __reduce__(self) -> tuple:
        return DenseSignal.__new__, (
            DenseSignal,
            self.num_streams,
            self.num_samples,
            self.sampling_rate,
            self.carrier_frequency,
            self.noise_power,
            self.delay,
            self.tobytes(),
        )


class SparseSignal(Signal):
    """Sparse signal model class.

    HermesPy signal sparsification can be described as follows.
    Given M signal streams, N samples are recorded for each stream with some constant temporal sampling rate.
    Thus, a MxN complex matrix of samples can be constructed.
    If at some point streams do not contain any recorded/transmitted signal, then fully zeroed columns appear in the matrix.
    This signal model contains a list of SignalBlocks, each representing non-zero regions of the original samples matrix.
    Thus, regions with only zeros are avoided.
    Note, that SignalBlocks are sorted by their offset time and don't overlap.
    """

    __blocks: list[SignalBlock]

    def __init__(
        self,
        samples: np.ndarray | Sequence[SignalBlock],
        sampling_rate: float = 1.0,
        carrier_frequency: float = 0.0,
        noise_power: float = 0.0,
        delay: float = 0.0,
    ) -> None:
        """Signal model initialization.

        Args:
            samples:
                A MxT matrix containing uniformly sampled base-band samples of the modeled signal.
                M is the number of individual streams, T the number of available samples.
                Note that you can pass here a 2D ndarray, a SignalBlock, a Sequence[np.ndarray] or a Sequence[SignalBlock].
                Given Sequence[SignalBlock], concatenates all signal blocks into one, accounting for their offsets.
                Given Sequence[np.ndarray], concatenates them all into one matrix sequentially.

            sampling_rate:
                Sampling rate of the modeled signal in Hz (in the base-band).

            carrier_frequency:
                Carrier-frequency of the modeled signal in the radio-frequency band in Hz.
                Zero by default.

            noise_power:
                Power of the noise superimposed to this signal model.
                Zero by default.

            delay:
                Delay of the signal in seconds.
                Zero by default.
        """

        # Initialize base classes
        Signal.__init__(self)

        # Initialize attributes
        self.__blocks = []
        self.sampling_rate = sampling_rate
        self.carrier_frequency = carrier_frequency
        self.delay = delay
        self.noise_power = noise_power
        self._samples_visualization = _SamplesVisualization(self)
        self._eye_visualization = _EyeVisualization(self)

        # Initialize blocks
        # Note that this assumes that sequences of blocks or arrays are already sparsified
        if isinstance(samples, np.ndarray):
            self.__blocks = self.__from_dense(samples, 0)
        else:
            self.__blocks = list(samples)

    @override
    def __mul__(self, value: complex) -> SparseSignal:
        return SparseSignal(
            [(b * value).view(SignalBlock) for b in self.blocks],
            self.sampling_rate,
            self.carrier_frequency,
            self.noise_power,
            self.delay,
        )

    @property
    @override
    def blocks(self) -> list[SignalBlock]:
        return self.__blocks

    @property
    @override
    def block_offsets(self) -> list[int]:
        return [b.offset for b in self.blocks]

    @property
    @override
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    @override
    def num_samples(self) -> int:
        if self.num_blocks < 1:
            return 0
        return self.blocks[-1].end

    @property
    @override
    def num_streams(self) -> int:
        if len(self.blocks) == 0:
            return 0
        return self.blocks[0].shape[0]

    @override
    def copy(self) -> SparseSignal:
        """Copy the sparse signal samples into a new signal model with identical parameters."""
        return SparseSignal(
            [b.copy() for b in self.blocks],
            self.sampling_rate,
            self.carrier_frequency,
            self.noise_power,
            self.delay,
        )

    @property
    @override
    def energy(self) -> np.ndarray:
        return np.sum([b.energy for b in self.blocks])

    @property
    @override
    def power(self) -> np.ndarray:
        return np.mean([b.power for b in self.blocks], axis=0)

    @override
    def view(self, dtype: type[AT]) -> AT:
        # If the requested dtype is already the current class, return self
        if dtype is type(self):
            return self  # type: ignore

        return self.to_dense().view(dtype)

    @override
    def to_dense(self) -> DenseSignal:
        dense_samples = np.zeros(self.shape, dtype=np.complex128)
        for b in self.blocks:
            dense_samples[:, b.offset : b.offset + b.num_samples] = b
        return DenseSignal.FromNDArray(
            dense_samples, self.sampling_rate, self.carrier_frequency, self.noise_power, self.delay
        )

    @staticmethod
    def __from_dense(block: np.ndarray, offset: int = 0) -> list[SignalBlock]:
        """Sparsify given signal samples matrix.

        Args:
            block:
                An MxT matrix containing uniformly sampled base-band samples of the modeled signal.
                M is the number of individual streams, T the number of available samples.
            offset:
                Offset of the given block relative to the start of the signal in samples.

        Returns:
            A list of SignalBlocks representing non-zero regions of the original samples matrix.
        """

        # Get block attributes
        offset = getattr(block, "offset", offset)

        # Boundary dimensional cases
        if block.ndim == 1:  # a vector of a stream
            block = block.reshape((1, block.size))  # pragma: no cover
        if block.shape[1] == 0 or len(block.nonzero()) == 0:  # an empty signal
            return [SignalBlock(block.shape[0], block.shape[1], offset, block.tobytes())]

        # Find nonzero columns indices
        nonzero_cols_idx = np.sum(block.view(np.ndarray), axis=0).nonzero()[
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
            res_samples = block[:, res_offsets[i] : win_stops[i]].astype(np.complex128)
            res_offset = offset + res_offsets[i]
            res.append(
                SignalBlock(
                    res_samples.shape[0], res_samples.shape[1], res_offset, res_samples.tobytes()
                )
            )

        return res

    @staticmethod
    def _getblocks_union(
        bs1: Sequence[SignalBlock], bs2: Sequence[SignalBlock]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate new blocks offsets and ends for a union of 2 different signals.

        Returns:
            A tuple of two numpy arrays:
            1. A list of new blocks offsets
            2. a list of new blocks ends
        """

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
    def __parse_slice(s: slice, dim_size: int) -> tuple[int, int, int]:
        """Helper function for _parse_validate_itemkey.
        Given a slice and a size of the sliced dimension,
        returns non-negative values for start, stop and step of the slice.
        """

        # Resolve None
        s0 = 0 if s.start is None else s.start
        s1 = dim_size if s.stop is None else s.stop
        s2 = 1 if s.step is None else s.step
        # Resolve negative
        s0 = s0 if s0 >= 0 else s0 % dim_size
        s1 = s1 if s1 >= 0 else s1 % dim_size
        return s0, s1, s2

    def _parse_validate_itemkey(
        self, key: Any
    ) -> tuple[int, int, int, int, int, int, bool, bool, bool]:
        """Parse and validate key in __getitem__ and __setitem__.

        Raises:
            TypeError if key is not
                an int,
                a slice,
                a tuple of (int, int), (slice, slice), (int, slice) or (slice, int),
                a boolean mask.
            IndexError if the key is out of bounds of the signal model.

        Returns:
            s00: streams start
            s01: streams stop
            s02: streams step
            s10: samples start
            s11: samples stop
            s12: samples step
            isboolmask: True if key is a boolean mask, False otherwise.
            should_flatten_streams: True if numpy's getitem would flatten the streams (1) dimension with this key
            should_flatten_samples: True if numpy's getitem would flatten the samples (2) dimension with this key

            Note that if isboolmask is True, then all s?? take the following values:
            (0, self.num_streams, 1, 0, self.num_samples, 1).

            Note that if the key references any dimension with an integer index,
            then the corresponding result start will be the index, and stop is start+1.
            For example, if key is 1, then only stream 1 is need. Then s00 is 1 and s01 is 2.
            Numpy getitem of [1] and [1:2] differ in dimensions. Flattening of the second variant should be considered.
        """

        self_num_streams = self.num_streams
        self_num_samples = self.num_samples
        isboolmask = False
        should_flatten_streams = False
        should_flatten_samples = False

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
                should_flatten_streams = True
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
                    raise IndexError(
                        f"Samples index is out of bounds ({key[1]} > {self_num_samples})"
                    )
                s10 = key[1] % self_num_samples
                s11 = s10 + 1
                s12 = 1
                should_flatten_samples = True
            else:
                raise TypeError(f"Samples key is ofan unsupported type ({type(key[1])})")

            return (
                s00,
                s01,
                s02,
                s10,
                s11,
                s12,
                False,
                should_flatten_streams,
                should_flatten_samples,
            )
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
            should_flatten_streams = True
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

        return (
            s00,
            s01,
            s02,
            s10,
            s11,
            s12,
            isboolmask,
            should_flatten_streams,
            should_flatten_samples,
        )

    def _find_affectedblocks(self, s10: int, s11: int) -> tuple[int, int]:
        """Find indices of blocks that are affected by the given samples slice.

        Arguments:
            s10: Start of the samples slice.
            s11: Stop of the samples slice.

        Return:
            b_start, b_stop:
                Indices of the first and the last affected blocks.
        """

        # Done with a binary search algotrithm.

        # Starting block
        L = 0
        R = len(self.blocks) - 1
        b_start = L
        while L < R:
            b_start = -((R + L) // -2)  # upside-down floor division
            if self.blocks[b_start].offset > s10:
                R = b_start - 1
            else:
                L = b_start
        if self.blocks[b_start].offset > s10 and b_start != 0:
            b_start -= 1

        # Stopping block
        L = 0
        R = len(self.blocks) - 1
        b_stop = R
        while L < R:
            b_stop = (R + L) // 2
            if self.blocks[b_stop].end < s11:
                L = b_stop + 1
            else:
                R = b_stop
        if self.blocks[b_stop].end < s11 and (len(self.blocks) - 1) > b_stop:
            b_stop += 1

        return b_start, b_stop

    @override
    def __getitem__(
        self, key: tuple[slice | SupportsIndex | Sequence[SupportsIndex], ...]
    ) -> SparseSignal | DenseSignal:
        """Select specific streams and samples from the sparse signal model.

        Arguments:
            key:
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
                Result shape is (1, num_samples)
            getitem(0, False):
                Same, but allow the numpy flattening.
                Result shape is (num_samples,)
            getitem((slice(None, 2), slice(50, 100))):
                Select streams 0, 1 and samples 50-99.
                Same as samples_matrix[:2, 50:100]

        Returns: Array with ndim 2 or less and  dtype dtype np.complex128
        """

        if not isinstance(key, tuple) or len(key) > 2:
            raise TypeError("SparseSignal __getitem__ key must be a tuple of at most 2 elements")

        stream_selector: slice | SupportsIndex | Sequence[SupportsIndex] = key[0]
        sample_selector: slice | SupportsIndex | Sequence[SupportsIndex] = (
            key[1] if isinstance(key, tuple) else slice(0, self.num_samples, 1)
        )
        selected_blocks: list[SignalBlock] = []

        # Handle the case in which samples are selcted by a slice, i.e. [..., a:b:c]
        if isinstance(sample_selector, slice):
            samples_start: int = sample_selector.start if sample_selector.start is not None else 0
            samples_end: int = (
                sample_selector.stop if sample_selector.stop is not None else self.num_samples
            )
            samples_step: int = sample_selector.step if sample_selector.step is not None else 1

            for block in self.blocks:

                # Ignore blocks that are entirely outside the selected slice
                if block.offset >= samples_end or block.end <= samples_start:
                    continue

                # Calculate the first index of the new block in the result
                block_start_index = max(samples_start - block.offset, 0)
                block_start_index += (
                    block_start_index + block.offset - samples_start
                ) % samples_step

                # Calculate the last index of the new block in the result
                block_end_index = min(samples_end - block.offset, block.num_samples)

                # Correctly slice the block
                sliced_block_samples = block.view(np.ndarray)[
                    stream_selector, block_start_index:block_end_index:samples_step  # type: ignore
                ]
                sliced_block = SignalBlock(
                    sliced_block_samples.shape[0],
                    sliced_block_samples.shape[1],
                    (block_start_index + block.offset - samples_start) // samples_step,
                    sliced_block_samples.tobytes(),
                )

                selected_blocks.append(sliced_block)

        elif isinstance(sample_selector, Sequence):
            raise NotImplementedError(
                "SparseSignal __getitem__ does not support sample indexing yet"
            )

        else:
            raise ValueError("Unsupported sparse signal key")

        # If key is a boolean mask
        # if isboolmask:
        #     mask = np.array(key, dtype=bool)
        #     return self.to_dense().blocks[0].view(np.ndarray)[mask]

        # The following check is required for proper handling
        # of the streams key that looks like slice(None, None, -*something*).
        # In this case, (s00, s01, s02) will be (0, num_streams, -*something*).
        # Slicing by [0:num_streams:-*something*] and [::-*something*]
        # yield different results. We would like to get the second one.

        # is_streams_step_reversing = (
        #     isinstance(key, slice) and key.start is None and key.stop is None
        # )
        # is_streams_step_reversing |= (
        #     isinstance(key, tuple)
        #     and isinstance(key[0], slice)
        #     and key[0].start is None
        #     and key[0].stop is None
        # )
        # if is_streams_step_reversing:
        #     return res[::s02, ::s12]
        # res = res[s00:s01:s02, ::s12]

        # If only a single block is selected, return it as a dense signal
        if len(selected_blocks) == 1 and selected_blocks[0].offset == 0:
            return DenseSignal.FromNDArray(
                selected_blocks[0].view(np.ndarray),
                self.sampling_rate,
                self.carrier_frequency,
                self.noise_power,
                self.delay,
            )

        # Allocate the result array
        return SparseSignal(
            selected_blocks,
            self.sampling_rate,
            self.carrier_frequency,
            self.noise_power,
            self.delay,
        )

    @override
    def __setitem__(self, key: tuple[slice | Sequence[SupportsIndex], ...], value: Any) -> None:
        # parse and validate key
        s00, s01, s02, s10, s11, s12, isboolmask, _, _ = self._parse_validate_itemkey(key)
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
        if s11 < self.blocks[0].offset:
            # then create new blocks and insert them into the model
            if not isinstance(value, np.ndarray):
                value = np.complex128(value)
                if value == 0.0 + 0.0j:
                    return
                bs_new = [
                    SignalBlock(
                        num_streams,
                        num_samples,
                        s10,
                        bytearray(np.full((num_streams, num_samples), value, np.complex128)),
                    )
                ]
            else:
                bs_new = self.__from_dense(value.reshape((num_streams, num_samples)))
            self.__blocks = bs_new + self.blocks
            return

        # Get sliced streames indices
        stream_idx = np.arange(s00, s01, s02)
        # Do nothing if no streams are affected
        if stream_idx.size == 0:
            return

        # Parse and validate value samples
        # Parse scalar values
        if not isinstance(value, np.ndarray):
            value = np.complex128(value)

            # Special case: Do nothing if the value is zero
            if value == 0.0 + 0.0j:
                bs_new = []

            # Otherwise the added blocks are a single block of the given value
            else:
                bs_new = [
                    SignalBlock(
                        num_streams,
                        num_samples,
                        s10,
                        bytearray(np.full((num_streams, num_samples), value, np.complex128)),
                    )
                ]

        # Parse matrix values
        else:  # else value is a samples matrix
            if value.ndim == 1:
                value = value.reshape((num_streams, num_samples))
            bs_new = self.__from_dense(value, s10)
            if len(bs_new) == 0:
                return  # pragma: no cover
            if bs_new[-1].end - bs_new[0].offset > num_samples or bs_new[0].shape[0] != num_streams:
                raise ValueError("Shape mismatch")

        # Find blocks that will be overwritten or modified.
        b_start, b_stop = self._find_affectedblocks(s10, s11)

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
        blocks_new_beg = self.__blocks[:b_start]
        blocks_new_mid = [b for b in self.__blocks[b_start + 1 : b_stop]]
        blocks_new_end = self.__blocks[b_stop + 1 :]

        # Cut the first affected block
        # Example:
        #      s10                  s10
        #       v                    v
        # xxxxxxxxxxxx00000 -> xxxxxxxxxxxx00000
        # [ b_start  ]         [1st ][2nd ]
        b = self.blocks[b_start]
        cut_col = max(s10 - b.offset, 0)  # max is for s10 < self.blocks[0].offset
        b_l = b[:, :cut_col]
        b_r = b[:, cut_col:]
        b_r.offset += cut_col
        if b_l.size != 0:
            blocks_new_beg.append(b_l)
        if b_r.size != 0:
            blocks_new_mid = [b_r] + blocks_new_mid

        # Cut the last affected block
        if b_start != b_stop:
            b = self.blocks[b_stop]
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
            b_offs, b_stops = self._getblocks_union(blocks_new_mid, bs_new)

            # Asseble the new blocks
            incoming_signal = SparseSignal(
                bs_new, self.sampling_rate, self.carrier_frequency, self.noise_power, self.delay
            )
            blocks_new_mid_new = []
            for i in range(b_offs.size):
                b_off = b_offs[i]
                b_stop = b_stops[i]
                b_new = self[(slice(None, None), slice(b_off, b_stop))].view(np.ndarray)
                b_new[stream_idx, :] = incoming_signal[:, b_off : min(b_stop, s11)]

                blocks_new_mid_new.append(
                    SignalBlock(b_new.shape[0], b_new.shape[1], b_off, bytearray(b_new))
                )
            blocks_new_mid = blocks_new_mid_new

        # Otherwise, the middle blocks are being simply replaced with the new ones
        else:
            blocks_new_mid = bs_new

        # Insert the result into this sparse signal model
        self.__blocks = blocks_new_beg + blocks_new_mid + blocks_new_end
        # merge new beginning blocks if needed
        if len(blocks_new_beg) != 0:
            b2_idx = len(blocks_new_beg)
            b1 = self.__blocks[b2_idx - 1]
            b2 = self.__blocks[b2_idx]
            if b1.end == b2.offset:
                self.__blocks[b2_idx - 1] = b1.append_samples(b2)
                self.__blocks.pop(b2_idx)
        # merge new ending blocks if needed
        if len(blocks_new_end) != 0:
            b2_idx = -len(blocks_new_end)
            b1 = self.__blocks[b2_idx - 1]
            b2 = self.__blocks[b2_idx]
            if b1.end == b2.offset:
                self.__blocks[b2_idx - 1] = b1.append_samples(b2)
                self.__blocks.pop(b2_idx)

    @override
    def resample(self, sampling_rate: float, aliasing_filter: bool = True) -> SparseSignal:
        return SparseSignal(
            [
                b.resample_block(self.sampling_rate, sampling_rate, aliasing_filter)
                for b in self.blocks
            ],
            sampling_rate,
            self.carrier_frequency,
            self.noise_power,
            self.delay,
        )

    @override
    def superimpose(
        self,
        added_signal: Signal,
        resample: bool = True,
        aliasing_filter: bool = True,
        stream_indices: Sequence[int] | None = None,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
    ) -> SparseSignal:
        # Account for the delay difference (if any)

        # If the superimposed signal's delay is smaller,
        # cut its leading samples accordingly
        if self.delay > added_signal.delay:
            if interpolation is InterpolationMode.NEAREST:
                added_signal_shift = int(
                    round((self.delay - added_signal.delay) * added_signal.sampling_rate)
                )
                added_signal = added_signal[:, added_signal_shift:]
            else:
                raise NotImplementedError("Only InterpolationMode.NEAREST is implemented")

        # If the superimposed signal's delay is larger,
        # shift its blocks accordingly
        elif self.delay < added_signal.delay:
            num_delay_samples = int(
                round((added_signal.delay - self.delay) * added_signal.sampling_rate)
            )
            added_signal = added_signal.copy()
            added_signal.delay = 0.0
            if interpolation is InterpolationMode.NEAREST:
                for b in added_signal.blocks:
                    b.offset += num_delay_samples

            else:
                raise NotImplementedError("Only InterpolationMode.NEAREST is implemented")

        # Do nothing if the added signal is empty
        if added_signal.num_samples == 0:
            return self

        # Parse stream indices
        num_streams = added_signal.num_streams if stream_indices is None else len(stream_indices)
        _stream_indices: np.ndarray = (
            np.arange(num_streams)
            if stream_indices is None
            else np.asarray(stream_indices, dtype=np.int64).flatten()
        )

        # Abort if zero streams are selected
        # This case occurs frequently when devices transmit to devices without receiving antennas
        if len(_stream_indices) < 1:
            return self

        # Validate
        if _stream_indices.max() >= self.num_streams:
            raise ValueError(f"Stream indices must be in the interval [0, {self.num_streams - 1}]")

        if added_signal.sampling_rate != self.sampling_rate and not resample:
            raise RuntimeError("Resampling required but not allowed")

        # Generate superimposed signal blocks
        added_resampled_blocks = [
            b.resample_block(added_signal.sampling_rate, self.sampling_rate, False).superimpose_to(
                self.sampling_rate,
                self.carrier_frequency,
                self.sampling_rate,
                added_signal.carrier_frequency,
            )
            for b in added_signal.blocks
        ]

        # Create new blocks for the signals' intersections
        # and insert the mix results there
        b_offs, b_stops = self._getblocks_union(self.blocks, added_resampled_blocks)
        unified_blocks: list[SignalBlock] = []
        for w_off, w_stop in zip(b_offs, b_stops):
            w_size = w_stop - w_off

            # Initialize a new block with zeros
            unified_block = np.zeros((num_streams, w_size), np.complex128)

            # Sum the samples from self
            for blocks in [self.blocks, added_resampled_blocks]:
                for b in blocks:

                    # If the block ends before this window's start, skip it
                    if b.end < w_off:
                        continue

                    # If the block starts after this window's end, stop processing blocks
                    # CAUTION: This assumes that blocks are sorted by their offsets
                    if b.offset > w_stop:
                        break

                    # Find the section within the current window that overlaps with the current block
                    w_start_index = max(0, b.offset - w_off)
                    w_stop_index = min(w_size, b.offset - w_off + b.num_samples)

                    # Find the section within the current block that overlaps with the current window
                    b_start_index = max(0, w_off - b.offset)
                    b_stop_index = min(b.num_samples, w_off - b.offset + w_size)

                    # Add the overlapping sections
                    unified_block[:, w_start_index:w_stop_index] += b[:, b_start_index:b_stop_index]

            unified_blocks.append(
                SignalBlock(self.num_streams, w_size, w_off, unified_block.tobytes())
            )

        return SparseSignal(
            unified_blocks, self.sampling_rate, self.carrier_frequency, self.noise_power, self.delay
        )

    @override
    def __iter__(self) -> Iterator[SparseSignal | DenseSignal]:
        for s in range(self.num_streams):
            yield self[s : s + 1, :]

    @property
    def title(self) -> str:
        return "Sparse Signal Model"

    @staticmethod
    def Empty(
        sampling_rate: float, num_streams: int = 0, num_samples: int = 0, **kwargs
    ) -> SparseSignal:
        res = SparseSignal(
            np.empty((num_streams, num_samples), np.complex128), sampling_rate, **kwargs
        )
        return res

    @staticmethod
    def _validate_samples_offsets(
        samples: np.ndarray | Sequence[np.ndarray], offsets: list[int] | None = None
    ) -> tuple[Sequence[np.ndarray], list[int]]:
        """Raises ValueError if any of the following fails:
            1. samples is a 2D matrix or a Sequence of 2D matrices with similar number of streams
            (e.g. samples[i].shape[0] == samples[i+1].shape[0]).
            2. offsets (if given) has the same length as samples argument.
            3. samples windows with given offsets do not overlap.

        Returns:
            samples:
                A python list if 2D matrices.
            offsets:
                A list of valid offsets.
        """

        # np.ndarray was given
        if isinstance(samples, np.ndarray):
            if samples.ndim == 1:  # vector or np.ndarray or objects
                if samples.dtype != object:  # vector
                    samples = [samples.reshape((1, samples.size)).astype(np.complex128)]
                else:
                    samples = samples.tolist()  # pragma: no cover
            elif samples.ndim == 2:  # 2D matrix
                samples = [samples.astype(np.complex128)]
            elif samples.ndim == 3:  # Tensor of 2D matrices
                samples = [np.asarray(s, np.complex128) for s in samples]
            else:  # Higher dim tensor
                raise ValueError(
                    f"ndarrays of more then 3 dims are not acceptable ({samples.ndim} were given)"
                )

        # samples is Sequence[np.ndarray] from now on

        # Check if samples list is empty
        if len(samples) == 0:
            return ([np.ndarray((0, 0), dtype=np.complex128)], [0])

        # Validate and adjust samples
        num_streams = samples[0].shape[0] if samples[0].ndim == 2 else 1
        for i in range(len(samples)):
            # dtype
            if not np.iscomplexobj(samples[i]):
                samples[i] = samples[i].astype(np.complex128)  # type: ignore
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
        offsets_: list[int]
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

    @override
    def append_samples(self, value: Signal | np.ndarray) -> SparseSignal:
        # Check if number of streams match
        if self.num_streams != value.shape[0]:
            raise ValueError("Number of streams do not match")

        # Resample the incoming signal
        added_blocks: list[SignalBlock]
        if isinstance(value, Signal):
            added_blocks = []
            for b in value.blocks:
                resampled_block = b.resample_block(value.sampling_rate, self.sampling_rate)
                resampled_block.offset += self.num_samples
                added_blocks.append(resampled_block)
        else:
            added_blocks = [
                SignalBlock(value.shape[0], value.shape[1], self.num_samples, value.tobytes())
            ]

        # Attach new blocks to existing ones
        new_blocks = self.blocks + added_blocks
        return SparseSignal(
            new_blocks, self.sampling_rate, self.carrier_frequency, self.noise_power, self.delay
        )

    @override
    def append_streams(self, value: Signal | np.ndarray) -> SparseSignal:
        # Resample the incoming signal
        signal_blocks: Signal | list[np.ndarray]
        if isinstance(value, Signal):
            signal_blocks = [
                b.resample_block(value.sampling_rate, self.sampling_rate) for b in value.blocks
            ]
        else:
            signal_blocks = [value]

        # Sparsify all blocks in the incoming signal
        # and add them to this model
        blocks_incoming = []
        for b in signal_blocks:
            for b_ in self.__from_dense(b):
                blocks_incoming.append(b_)

        # Create new resulting blocks
        new_num_streams = self.num_streams + value.shape[0]
        b_offs, b_stops = self._getblocks_union(self.blocks, blocks_incoming)

        blocks_new = []
        for block_start, block_stop in zip(b_offs, b_stops):

            # Select the respective blocks to be mergerd from both signal models
            initial_block = self[:, block_start:block_stop].view(np.ndarray)
            appended_block = value[:, block_start:block_stop].view(np.ndarray)

            # Merge the blocks into a new block with increased number of streams
            new_block = np.concatenate((initial_block, appended_block), axis=0)
            blocks_new.append(
                SignalBlock(
                    new_num_streams, block_stop - block_start, block_start, new_block.tobytes()
                )
            )

        return SparseSignal(
            blocks_new, self.sampling_rate, self.carrier_frequency, self.noise_power, self.delay
        )

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.sampling_rate, "sampling_rate")
        process.serialize_floating(self.carrier_frequency, "carrier_frequency")
        process.serialize_floating(self.delay, "delay")
        process.serialize_floating(self.noise_power, "noise_power")
        process.serialize_integer(self.num_streams, "num_streams")
        process.serialize_object_sequence(self.blocks, "blocks")

    @classmethod
    @override
    def Deserialize(cls, process: DeserializationProcess) -> SparseSignal:
        return SparseSignal(
            process.deserialize_object_sequence("blocks", SignalBlock),
            process.deserialize_floating("sampling_rate"),
            process.deserialize_floating("carrier_frequency", cls._DEFAULT_CARRIER_FREQUENCY),
            process.deserialize_floating("noise_power", cls._DEFAULT_NOISE_POWER),
            process.deserialize_floating("delay", cls._DEFAULT_DELAY),
        )
