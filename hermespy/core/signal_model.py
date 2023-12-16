# -*- coding: utf-8 -*-
"""
===============
Signal Modeling
===============
"""

from __future__ import annotations
from copy import deepcopy
from math import floor
from typing import Literal, Sequence, Tuple, Type

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from h5py import Group
from numba import jit, complex128
from scipy.constants import pi
from scipy.fft import fft, fftshift, fftfreq
from scipy.ndimage import convolve1d
from scipy.signal import butter, sosfilt, firwin

from .executable import Executable
from .factory import HDFSerializable
from .visualize import VAT, Visualizable

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.2.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class Signal(HDFSerializable, Visualizable):
    """Base class of signal models in HermesPy.

    Attributes:

        __samples (np.ndarray):
            An MxT matrix containing uniformly sampled base-band samples of the modeled signal.
            M is the number of individual streams, T the number of available samples.

        __sampling_rate (float):
            Sampling rate of the modeled signal in Hz (in the base-band).

        __carrier_frequency (float):
            Carrier-frequency of the modeled signal in the radio-frequency band,
            i.e. the central frequency in Hz.
    """

    filter_order: int = 10
    """Order of the filters applied during resampling and supoperosition mixing."""

    __samples: np.ndarray
    __sampling_rate: float
    __carrier_frequency: float
    __noise_power: float
    delay: float

    def __init__(
        self,
        samples: np.ndarray,
        sampling_rate: float,
        carrier_frequency: float = 0.0,
        delay: float = 0.0,
        noise_power: float = 0.0,
    ) -> None:
        """Signal model initialization.

        Args:
            samples (np.ndarray):
                An MxT matrix containing uniformly sampled base-band samples of the modeled signal.
                M is the number of individual streams, T the number of available samples.

            sampling_rate (float):
                Sampling rate of the modeled signal in Hz (in the base-band).

            carrier_frequency (float, optional):
                Carrier-frequency of the modeled signal in the radio-frequency band,
                i.e. the central frequency in Hz.
                Zero by default.

            delay (float, optional):
                Delay of the signal in seconds.
                Zero by default.

            noise_power (float, optional):
                Power of the noise superimposed to this signal model.
                Zero by default.
        """

        # Initialize base classes
        HDFSerializable.__init__(self)
        Visualizable.__init__(self)

        # Initialize class attributes
        self.samples = samples.astype(complex).copy()
        self.sampling_rate = sampling_rate
        self.carrier_frequency = carrier_frequency
        self.delay = delay
        self.noise_power = noise_power

    @property
    def title(self) -> str:
        return "Signal Model"

    @classmethod
    def empty(
        cls, sampling_rate: float, num_streams: int = 0, num_samples: int = 0, **kwargs
    ) -> Signal:
        """Create a new empty signal model instance.

        Args:
            sampling_rate (float):
                Sampling rate of the modeled signal in Hz (in the base-band).

            num_streams (int, optional):
                Number of signal streams within this empty model.

            num_samples (int, optional):
                Number of signal samples within this empty model.

            kwargs:
                Additional initialization arguments, piped through to the class init.
        """

        return Signal(np.empty((num_streams, num_samples), dtype=complex), sampling_rate, **kwargs)

    @property
    def samples(self) -> np.ndarray:
        """Uniformly sampled complex signal model in time-domain.

        Returns:
            np.ndarray:
                An MxT matrix of samples,
                where M is the number of individual streams and T the number of samples.
        """

        return self.__samples

    @samples.setter
    def samples(self, value: np.ndarray) -> None:
        """Modify the base-band samples of the modeled signal.

        Args:
            value (np.ndarray):
                An MxT matrix of samples,
                where M is the number of individual streams and T the number of samples.

        Raises:
            ValueError: If `value` can't be interpreted as a matrix.
        """

        # Expand value vectors to matrices, indicating a signal model with one stream
        if value.ndim == 1:
            value = value[np.newaxis, :]

        if value.ndim != 2:
            raise ValueError("Signal model samples must be a matrix (a two-dimensional array)")

        self.__samples = value

    @property
    def num_streams(self) -> int:
        """The number of streams within this signal model.

        Returns:
            int: The number of streams.
        """

        return self.__samples.shape[0]

    @property
    def num_samples(self) -> int:
        """The number of samples within this signal model.

        Returns:
            int: The number of samples.
        """

        return self.__samples.shape[1]

    @property
    def sampling_rate(self) -> float:
        """The rate at which the modeled signal was sampled.

        Returns:
            float: The sampling rate in Hz.
        """

        return self.__sampling_rate

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

        self.__sampling_rate = value

    @property
    def carrier_frequency(self) -> float:
        """The center frequency of the modeled signal in the radio-frequency
        transmit band.

        Returns:
            float: The carrier frequency in Hz.
        """

        return self.__carrier_frequency

    @carrier_frequency.setter
    def carrier_frequency(self, value: float) -> None:
        """Modify the center frequency of the modeled signal in the radio-frequency transmit band.

        Args:
            value (float): he carrier frequency in Hz.

        Raises:
            ValueError: If `value` is smaller than zero.
        """

        if value < 0.0:
            raise ValueError(
                "The carrier frequency of modeled signals must be greater or equal to zero"
            )

        self.__carrier_frequency = value

    @property
    def noise_power(self) -> float:
        """Noise power of the superimposed noise signal.

        Returns:

            Noise power.

        Raises:

            ValueError: If the noise power is smaller than zero.
        """

        return self.__noise_power

    @noise_power.setter
    def noise_power(self, value: float) -> None:
        if value < 0.0:
            raise ValueError("Noise power must be greater or equal to zero")

        self.__noise_power = value

    @property
    def power(self) -> np.ndarray:
        """Compute the power of the modeled signal.

        Returns: The power of each modeled stream within a numpy vector.
        """

        if self.num_samples < 1:
            return np.zeros(self.num_streams)

        stream_power = (
            np.sum(self.__samples.real**2 + self.__samples.imag**2, axis=1) / self.num_samples
        )
        return stream_power

    @property
    def energy(self) -> np.ndarray:
        """Compute the energy of the modeled signal.

        Returns: The energy of each modeled stream within a numpy vector.
        """

        stream_energy = np.sum(self.__samples.real**2 + self.__samples.imag**2, axis=1)
        return stream_energy

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

        if sampling_rate <= 0.0:
            raise ValueError("Sampling rate for resampling must be greater than zero.")

        if self.num_samples < 1:
            return Signal(
                np.empty((self.num_streams, 0), dtype=np.complex_),
                sampling_rate,
                carrier_frequency=self.__carrier_frequency,
                delay=self.delay,
                noise_power=self.noise_power,
            )

        # Resample the internal samples
        if self.__sampling_rate != sampling_rate:
            # Apply an anti-aliasing filter if the respective flag is enabled
            if aliasing_filter:
                # Apply an anti-aliasing filter after resampling if the signal is upsampled
                if sampling_rate > self.sampling_rate:
                    samples = Signal.__resample(self.__samples, self.__sampling_rate, sampling_rate)

                    aliasing_filter = butter(
                        8, self.sampling_rate / sampling_rate, btype="low", output="sos"
                    )
                    samples = sosfilt(aliasing_filter, samples, axis=1)

                # Apply an anti-aliasing filter before resampling if the signal is downsampled
                elif sampling_rate < self.sampling_rate:
                    aliasing_filter = butter(
                        8, sampling_rate / self.sampling_rate, btype="low", output="sos"
                    )
                    samples = sosfilt(aliasing_filter, self.__samples, axis=1)

                    samples = Signal.__resample(samples, self.__sampling_rate, sampling_rate)

            else:
                samples = Signal.__resample(self.__samples, self.__sampling_rate, sampling_rate)

        # Skip resampling if both sampling rates are identical
        else:
            samples = self.__samples.copy()

        # Create a new signal object from the resampled samples and return it as result
        return Signal(
            samples,
            sampling_rate,
            carrier_frequency=self.__carrier_frequency,
            delay=self.delay,
            noise_power=self.noise_power,
        )

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

        num_streams = added_signal.num_streams if stream_indices is None else len(stream_indices)
        _stream_indices = (
            np.arange(num_streams)
            if stream_indices is None
            else np.array(stream_indices, dtype=np.int_)
        )

        # Abort if not streams are selected
        # This case occurs frequently when devices transmit to devices without receiving antennas
        if len(_stream_indices) < 1:
            return

        if _stream_indices.max() >= self.num_streams:
            raise ValueError(f"Stream indices must be in the interval [0, {self.num_streams - 1}]")

        if self.delay != added_signal.delay:
            raise NotImplementedError(
                "Superimposing signal models of differing delay is not yet supported"
            )

        # Apply an aliasing filter to the added signal
        frequency_distance = added_signal.carrier_frequency - self.carrier_frequency
        filter_center_frequency = 0.5 * frequency_distance
        filter_bandwidth = 0.5 * (added_signal.sampling_rate + self.sampling_rate) - abs(
            frequency_distance
        )

        if filter_bandwidth <= 0.0:
            return

        if aliasing_filter and filter_bandwidth < added_signal.sampling_rate:
            filter_coefficients = firwin(
                1 + self.filter_order,
                0.5 * filter_bandwidth,
                width=0.5 * filter_bandwidth,
                fs=added_signal.sampling_rate,
            ).astype(complex)
            filter_coefficients *= np.exp(
                2j
                * np.pi
                * (filter_center_frequency - added_signal.carrier_frequency)
                / added_signal.sampling_rate
                * np.arange(1 + self.filter_order)
            )

            added_samples = convolve1d(added_signal.samples, filter_coefficients, axis=1)

        else:
            added_samples = added_signal.samples

        # Resample the added signal if the respective sampling rates don't match
        if self.sampling_rate != added_signal.sampling_rate:
            if not resample:
                raise RuntimeError("Resampling required but not allowed")

            resampled_added_samples = self.__resample(
                added_samples, added_signal.sampling_rate, self.sampling_rate
            )

        else:
            resampled_added_samples = added_samples

        if self.num_samples < resampled_added_samples.shape[1]:
            self.__samples = np.append(
                self.__samples,
                np.zeros(
                    (self.num_streams, resampled_added_samples.shape[1] - self.num_samples),
                    dtype=complex,
                ),
                axis=1,
            )

        # Mix the added signal onto this signal's samples according to the carrier frequency distance
        self.__mix(
            self.__samples,
            _stream_indices,
            resampled_added_samples,
            self.__sampling_rate,
            frequency_distance,
        )

    @staticmethod
    @jit(nopython=True)
    def __mix(
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
                Samples to be mixed ont `target_samples`.

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

    @property
    def timestamps(self) -> np.ndarray:
        """The sample-points of the signal model.

        Returns:
            np.ndarray: Vector of length T containing sample-timestamps in seconds.
        """

        return np.arange(self.num_samples) / self.__sampling_rate - self.delay

    @property
    def frequencies(self) -> np.ndarray:
        """The signal model's discrete sample points in frequcy domain.

        Returns: Numpy vector of frequency bins.
        """

        return fftfreq(self.num_samples, 1 / self.sampling_rate)

    def _new_axes(
        self,
        num_streams: int | None = None,
        space: Literal["time", "frequency", "both"] = "both",
        **kwargs,
    ) -> Tuple[plt.Figure, np.ndarray[plt.Axes, np.dtype[np.object_]]]:
        """Generate a new figure and axes to plot into.

        Can be overriden by subclasses to configure custom axes flags.

        Args:

            space (Literal["time", "frequency", "both"], optional):
                Signal space to be plotted.
                Both by default.

            **kwargs:
                Additional keyword arguments to be used by subclasses.

        Returns: Tuple of matplotlib figure and axes.
        """

        num_streams = self.num_streams if num_streams is None else num_streams
        num_axes = 2 if space == "both" else 1

        figure, axes = plt.subplots(num_streams, num_axes, squeeze=False)
        return figure, axes

    def plot(
        self,
        axes: VAT | None = None,
        *,
        title: str | None = None,
        angle: bool = False,
        space: Literal["time", "frequency", "both"] = "both",
        legend: bool = True,
    ) -> plt.FigureBase:
        """Plot the current signal in time- and frequency-domain.

        .. plot::

            import matplotlib.pyplot as plt
            import numpy as np

            from hermespy.core import Signal

            sampling_rate = 100
            timestamps = np.arange(100) / sampling_rate
            signal = Signal(np.exp(-.25j * np.pi * timestamps * sampling_rate), sampling_rate)

            signal.plot()
            plt.show()

        Args:

            axes (plt.Axes, optional):
                The axis object into which the information should be plotted.
                If not specified, the routine will generate and return a new figure.

            title (str, optional):
                Title of the generated plot.

            angle (bool, optional):
                Plot the angle of complex frequency bins.

            space (Literal["time", "frequency", "both"], optional):
                Signal space to be plotted.
                By default, both spaces are visualized.

            legend (bool, optional):
                Add a legend to the plots.
                Enabled by default.

        Returns:
            The created matplotlib figure.
            `None`, if axes were provided.
        """

        title = "Signal Model" if title is None else title
        figure, _axes = self._prepare_axes(axes, title, num_streams=self.num_streams, space=space)

        # Generate timestamps for time-domain plotting
        timestamps = self.timestamps

        # Infer the axis indices to account for different plot
        time_axis_idx = 0
        frequency_axis_idx = 1 if space == "both" else 0

        for stream_idx, stream_samples in enumerate(self.__samples):
            # Plot time space
            if space in {"both", "time"}:
                if self.num_samples < 1:
                    _axes[stream_idx, time_axis_idx].text(
                        0.5, 0.5, "NO SAMPLES", horizontalalignment="center"
                    )

                else:
                    _axes[stream_idx, time_axis_idx].plot(
                        timestamps, stream_samples.real, label="Real"
                    )
                    _axes[stream_idx, time_axis_idx].plot(
                        timestamps, stream_samples.imag, label="Imag"
                    )

                _axes[stream_idx, time_axis_idx].set_xlabel("Time-Domain [s]")

                if legend:
                    with Executable.style_context():
                        _axes[stream_idx, time_axis_idx].legend(
                            loc="upper left", fancybox=True, shadow=True
                        )

            # Plot frequency space
            if space in {"both", "frequency"}:
                if self.num_samples < 1:
                    _axes[stream_idx, time_axis_idx].text(
                        0.5, 0.5, "NO SAMPLES", horizontalalignment="center"
                    )

                else:
                    frequencies = fftshift(fftfreq(self.num_samples, 1 / self.sampling_rate))
                    bins = fftshift(fft(stream_samples))

                    _axes[stream_idx, frequency_axis_idx].plot(frequencies, np.abs(bins))

                _axes[stream_idx, frequency_axis_idx].set_ylabel("Abs")
                _axes[stream_idx, frequency_axis_idx].set_xlabel("Frequency-Domain [Hz]")

                if angle:
                    phase = _axes[stream_idx, frequency_axis_idx].twinx()
                    phase.plot(frequencies, np.angle(bins))
                    phase.set_ylabel("Angle [Rad]")

        return figure

    def plot_eye(
        self,
        symbol_duration: float,
        axes: VAT | None = None,
        *,
        title: str | None = None,
        domain: Literal["time", "complex"] = "time",
        legend: bool = True,
        linewidth: float = 0.75,
        symbol_cutoff: float = 0.1,
    ) -> plt.FigureBase:
        """Plot the signal model's eye diagram.

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

        if symbol_duration <= 0.0:
            raise ValueError("Symbol duration must be greater than zero")

        if linewidth <= 0.0:
            raise ValueError(f"Plot line width must be greater than zero (not {linewidth})")

        if symbol_cutoff < 0.0 or symbol_cutoff > 1.0:
            raise ValueError(f"Symbol cutoff must be in the interval [0, 1] (not {symbol_cutoff})")

        figure, axes = self._prepare_axes(axes, "Eye Diagram" if title is None else title)

        symbol_num_samples = int(symbol_duration * self.sampling_rate)
        num_symbols = int(self.num_samples / symbol_num_samples)
        num_cutoff_symbols = int(num_symbols * symbol_cutoff)

        colors = self._get_color_cycle()

        if domain == "time":
            timestamps = (
                np.arange(-symbol_num_samples, 1 + symbol_num_samples, 1) / symbol_num_samples
            )

            for n in range(num_cutoff_symbols, num_symbols - num_cutoff_symbols):
                stream_slice = self.__samples[
                    0, symbol_num_samples * n : symbol_num_samples * (2 + n) + 1
                ]
                axes.flat[0].plot(
                    timestamps[: len(stream_slice)],
                    stream_slice.real,
                    color=colors[0],
                    linewidth=linewidth,
                )
                axes.flat[0].plot(
                    timestamps[: len(stream_slice)],
                    stream_slice.imag,
                    color=colors[1],
                    linewidth=linewidth,
                )

            axes.flat[0].set_xlabel("Time-Domain [s]")
            axes.flat[0].set_ylabel("Amplitude")

            if legend:
                legend_elements = [
                    Line2D([0], [0], color=colors[0], label="Real"),
                    Line2D([0], [0], color=colors[1], label="Imag"),
                ]
                with Executable.style_context():
                    axes.flat[0].legend(
                        handles=legend_elements, loc="upper left", fancybox=True, shadow=True
                    )

        elif domain == "complex":
            symbol_num_samples = floor(symbol_duration * self.sampling_rate)
            num_cutoff_samples = num_cutoff_symbols * symbol_num_samples

            stream_slice = self.__samples[0, num_cutoff_samples:-num_cutoff_samples]
            axes.flat[0].plot(
                stream_slice.real, stream_slice.imag, color=colors[0], linewidth=linewidth
            )

            axes.flat[0].set_xlabel("Real")
            axes.flat[0].set_ylabel("Imag")

        else:
            raise ValueError(f"Unsupported plotting domain '{domain}'")

        # Return resulting figure
        return figure

    @staticmethod
    @jit(nopython=True)
    def __resample(
        signal: np.ndarray, input_sampling_rate: float, output_sampling_rate: float
    ) -> np.ndarray:  # pragma: no cover
        """Internal subroutine to resample a given set of samples to a new sampling rate.

        Uses sinc-interpolation, therefore `signal` is assumed to be band-limited.

        Arguments:

            signal (np.ndarray):
                MxT matrix of M signal-streams to be resampled, each containing T time-discrete samples.

            input_sampling_rate (float):
                Rate at which `signal` is sampled in Hz.

            output_sampling_rate (float):
                Rate at which the resampled signal will be sampled in Hz.

        Returns:
            np.ndarray:
                MxT' matrix of M resampled signal streams.
                The number of samples T' depends on the sampling rate relations, i.e.
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

    def append_samples(self, signal: Signal) -> None:
        """Append samples in time-domain to the signal model.

        Args:
            signal (Signal): The signal to be appended.

        Raise:
            ValueError: If the number of streams don't align.
        """

        # Adapt number of streams to fit the appended signal if this signal is empty
        if self.num_streams < 1:
            self.__samples = np.empty((signal.num_streams, 0), dtype=complex)

        if signal.num_streams != self.num_streams:
            raise ValueError(
                "Appending signal models with differing amount of streams is not defined"
            )

        if signal.carrier_frequency != self.__carrier_frequency:
            raise ValueError("Appending signals of differing carrier frequencies is not defined")

        # Resample the signal if required
        if signal.sampling_rate != self.sampling_rate:
            raise NotImplementedError("Resampling not yet implemented")

        self.__samples = np.append(self.__samples, signal.samples, axis=1)

    def append_streams(self, signal: Signal) -> None:
        """Append streams to the signal model.

        Args:
            signal (Signal): The signal to be appended.

        Raise:
            ValueError: If the number of samples don't align.
        """

        # Adapt the number of samples if this signal is empty to match the signal to be appended
        if self.num_samples < 1:
            self.__samples = np.empty((0, signal.num_samples), dtype=complex)

        if signal.num_samples != self.num_samples:
            raise ValueError(
                "Appending signal models streams with differing amount of samples is not supported"
            )

        if signal.carrier_frequency != self.__carrier_frequency:
            raise ValueError("Appending signals of differing carrier frequencies is not defined")

        # Resample the signal if required
        if signal.sampling_rate != self.sampling_rate:
            raise NotImplementedError("Resampling not yet implemented")

        self.__samples = np.append(self.__samples, signal.samples, axis=0)

    @property
    def duration(self) -> float:
        """Signal model duration in time-domain.

        Returns:
            float: Duration in seconds.
        """

        return self.num_samples / self.sampling_rate

    def to_interleaved(self, data_type: Type = np.int16, scale: bool = False) -> np.ndarray:
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

        samples = self.__samples.copy()

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

        complex_samples = interleaved_samples.astype(np.float64).view(np.complex128)

        if scale:
            complex_samples /= np.iinfo(interleaved_samples.dtype).max

        return cls(samples=complex_samples, **kwargs)

    @classmethod
    def from_HDF(cls, group: Group) -> Signal:
        # De-serialize attributes
        sampling_rate = group.attrs.get("sampling_rate", 1.0)
        carrier_frequency = group.attrs.get("carrier_frequency", 0.0)
        delay = group.attrs.get("delay", 0.0)
        noise_power = group.attrs.get("noise_power", 0.0)

        # De-serialize samples
        samples = np.array(group["samples"], dtype=np.complex_)

        return Signal(
            samples=samples,
            sampling_rate=sampling_rate,
            carrier_frequency=carrier_frequency,
            delay=delay,
            noise_power=noise_power,
        )

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
        self._write_dataset(group, "samples", self.samples)
        self._write_dataset(group, "timestamps", self.timestamps)
