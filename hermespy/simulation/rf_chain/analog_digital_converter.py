# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
from matplotlib import pyplot as plt

from hermespy.core import Serializable, SerializableEnum, Signal

from hermespy.tools.math import rms_value

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class GainControlType(SerializableEnum):
    """Type of automatig gain control"""

    NONE = 0
    MAX_AMPLITUDE = 1
    RMS_AMPLITUDE = 2


GainType = TypeVar("GainType", bound="Gain")
"""Type of gain."""


class GainControlBase(ABC):
    """Base class for all ADC gain control models."""

    __rescale_quantization: bool

    def __init__(self, rescale_quantization: bool = False) -> None:
        """
        Args:

            rescale_quantization (bool, optional):
                If enabled, the quantized signal is rescaled to the original signal range before gain adjustment.
                Disabled by default.
        """

        self.rescale_quantization = rescale_quantization

    @property
    def rescale_quantization(self) -> bool:
        """Rescale the quantized signal to the original signal range before gain adjustment."""

        return self.__rescale_quantization

    @rescale_quantization.setter
    def rescale_quantization(self, value: bool) -> None:
        self.__rescale_quantization = value

    @abstractmethod
    def estimate_gain(self, input_signal: Signal) -> float:
        """Estimate the gain required to adjust the signal to the ADC input range.

        Args:

            input_signal (Signal):
                Input signal to be adjusted.

        Returns: Linear gain to be applied to the `input_signal`'s Voltage samples.
        """
        ...  # pragma: no cover

    def adjust_signal(self, input_signal: Signal, gain: float) -> Signal:
        """Adjust the signal to the ADC input range.

        Args:

            input_signal (Signal):
                Input signal to be adjusted.

            gain (float):
                Linear gain to be applied to the `input_signal`'s Voltage samples.

        Returns: The adjusted signal.
        """

        adjusted_signal = input_signal.copy()
        adjusted_signal.samples = adjusted_signal.samples * gain

        return adjusted_signal

    def scale_quantized_signal(self, quantized_signal: Signal, gain: float) -> Signal:
        """Scale the quantized signal back to the original signal range before gain adjustment.

        Only applied if :py:attr:`rescale_quantization` is enabled.

        Args:

            quantized_signal (Signal):
                Quantized signal to be adjusted.

            gain (float):
                Linear gain to applied to the `input_signal`'s Voltage samples before quantization.

        Returns: The scaled qzanitized signal.
        """

        if not self.rescale_quantization:
            return quantized_signal

        scaled_signal = quantized_signal.copy()
        scaled_signal.samples = scaled_signal.samples / gain

        return scaled_signal


class Gain(Serializable, GainControlBase):
    """Constant gain model."""

    yaml_tag = "Gain"
    """YAML serialization tag."""

    __gain: float

    def __init__(self, gain: float = 1.0, rescale_quantization: bool = False) -> None:
        """
        Args:

            gain (float, optional):
                Linear signal gain to be applied before ADC quantization.
                Unit by default, meaning no gain adjustment.

            rescale_quantization (bool, optional):
                If enabled, the quantized signal is rescaled to the original signal range before gain adjustment.
                Disabled by default.
        """

        # Initialize base class
        GainControlBase.__init__(self, rescale_quantization=rescale_quantization)

        # Initialize attributes
        self.gain = gain

    @property
    def gain(self) -> float:
        """Linear gain before ADC quantization.

        Quantizer operates by default between -1. and +1.
        Signal can be adjusted by to this range by appropriate gain setting.

        Returns: Gain in Volt.

        Raises:

            ValueError: If gain is smaller or equal to zero.
        """
        return self.__gain

    @gain.setter
    def gain(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Gain must be larger than zero")

        self.__gain = value

    def estimate_gain(self, input_signal: Signal) -> float:
        return self.gain


class AutomaticGainControl(Serializable, GainControlBase):
    """Analog-to-digital conversion automatic gain control modeling."""

    yaml_tag = "AutomaticGainControl"
    """YAML serialization tag."""

    __agc_type: GainControlType
    __backoff: float

    def __init__(
        self,
        agc_type: GainControlType = GainControlType.MAX_AMPLITUDE,
        backoff: float = 1.0,
        rescale_quantization: bool = False,
    ) -> None:
        """
        Args:

            agc_type (GainControlType, optional):
                Type of amplitude gain control at ADC input. Default is GainControlType.MAX_AMPLITUDE.

            backoff (float, optional):
                this is the ratio between maximum amplitude and the rms value or maximum of input signal,
                depending on AGC type. Default value is 1.0.

            rescale_quantization (bool, optional):
                If enabled, the quantized signal is rescaled to the original signal range before gain adjustment.
                Disabled by default.
        """

        # Initialize base class
        GainControlBase.__init__(self, rescale_quantization=rescale_quantization)

        # Initialize attributes
        self.agc_type = agc_type
        self.backoff = backoff

    @property
    def agc_type(self) -> GainControlType:
        """Automatic Gain Control

        The AGC may have the following types of gain control, which wil specify the quantizer range:
        - GainControlType.NONE: no gain control, range must be specified
        - GainControlType.MAX_AMPLITUDE: the range is given by the maximum amplitude of the
        - GainControlType.RMS_AMPLITUDE: the range is given by the rms value plus a given backoff
        Note the for complex numbers, amplitude is calculated for both real and imaginary parts separately, and the
        greatest value is considered.

        Returns:
            GainControlType:

        """
        return self.__agc_type

    @agc_type.setter
    def agc_type(self, value: GainControlType | str) -> None:
        self.__agc_type = value if isinstance(value, GainControlType) else GainControlType[value]

    @property
    def backoff(self) -> float:
        """Quantizer backoff in linear scale

        This quantity determines the ratio between the maximum quantization level and the signal rms value
        Returns:
            float: the backoff in linear scale

        """
        return self.__backoff

    @backoff.setter
    def backoff(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Backoff must be larger than 0")

        self.__backoff = value

    def estimate_gain(self, input_signal: Signal) -> float:
        if self.agc_type == GainControlType.MAX_AMPLITUDE:
            max_amplitude = max(
                np.abs(np.real(input_signal.samples)).max(),
                np.abs(np.imag(input_signal.samples)).max(),
            )

        elif self.agc_type == GainControlType.RMS_AMPLITUDE:
            max_amplitude = max(
                rms_value(np.real(input_signal.samples)), rms_value(np.imag(input_signal.samples))
            )

        else:
            raise RuntimeError("Unsupported gain control type")

        return 1 / (max_amplitude * self.backoff) if max_amplitude > 0.0 else 1.0


class QuantizerType(SerializableEnum):
    """Type of quantizer"""

    MID_RISER = 0
    MID_TREAD = 1


class AnalogDigitalConverter(Serializable):
    """Implements an ADC (analog-to-digital converter)

    Models the behaviour of an ADC, including:
    - Sampling Jitter (to be implemented)
    - Automatic Gain Control
    - Quantization. Currently only uniform and symmetric quantization is supported.

    This class only implements the quantization noise, the output data is still in floating point representation with
    the same amplitude as the input.
    """

    yaml_tag = "ADC"
    """YAML serialization tag"""

    __num_quantization_bits: int | None
    gain: Gain
    __quantizer_type: QuantizerType

    def __init__(
        self,
        num_quantization_bits: int | None = None,
        gain: Gain | None = None,
        quantizer_type: QuantizerType = QuantizerType.MID_RISER,
    ) -> None:
        """
        Args:

            num_quantization_bits (int, optional):
                ADC resolution in bits. Default is infinite resolution (no quantization)

            gain (Gain, optional):
                Amplitude gain control at ADC input. Default is Gain(1.0), i.e., no gain.

            quantizer_type (QuantizerType, optional):
                Determines quantizer behaviour at zero. Default is QuantizerType.MID_RISER.
        """

        self.num_quantization_bits = num_quantization_bits
        self.gain = Gain() if gain is None else gain
        self.quantizer_type = quantizer_type

    @property
    def num_quantization_bits(self) -> int | None:
        """Quantization resolution in bits

        Returns:
            Bit resolution, `None` if no quantization is applied.

        Raises:
            ValueError: If resolution is less than zero.

        """

        return self.__num_quantization_bits

    @num_quantization_bits.setter
    def num_quantization_bits(self, value: int | None) -> None:
        if value is None:
            self.__num_quantization_bits = None

        else:
            if value < 0 or not isinstance(value, (int, np.int_)):
                raise ValueError("Number of bits must be a non-negative integer")

            else:
                self.__num_quantization_bits = int(value) if value > 0 else None

    @property
    def num_quantization_levels(self) -> float:
        """Number of quantization levels

        Returns:
            int: Number of levels

        """

        if self.__num_quantization_bits is None:
            return np.inf

        return 2**self.num_quantization_bits

    @property
    def quantizer_type(self) -> QuantizerType:
        """Type of quantizer

        - QuantizationType.MID_TREAD: 0 can be the output of a quantization step. Since the number of quantization step
                                      must be even, negative values will have one step more than positive values
        - QuantizerType.MID_RISE: input values around zero are quantized as either -delta/2 or delta/2, with delta the
                                  quantization step

        Returns:
            QuantizerType: type of quantizer
        """

        return self.__quantizer_type

    @quantizer_type.setter
    def quantizer_type(self, value: QuantizerType) -> None:
        self.__quantizer_type = value

    def __convert_frame(self, frame_signal: Signal) -> Signal:
        """Converts an analog frame into a digitally quantized frame.

        Subroutine of :meth:`convert`.

        Args:
            input_signal (Signal): Signal to be converted.

        Returns: Gain adjusted and quantized signal.
        """

        # Initially, estimate the required gain to avoid clipping
        gain = self.gain.estimate_gain(frame_signal)

        # Scale the input signal according to the estimated gain
        adjusted_signal = self.gain.adjust_signal(frame_signal, gain)

        # Quantize adjusted signal
        adjusted_signal.samples = self._quantize(adjusted_signal.samples)

        # Rescale adjusted signal to the original amplitude range
        output_signal = self.gain.scale_quantized_signal(adjusted_signal, gain)

        return output_signal

    def convert(self, input_signal: Signal, frame_duration: float = 0.0) -> Signal:
        """Converts an analog signal into a digitally quantized signal.

        Args:

            input_signal (Signal):
                Signal to be converted.

            frame_duration (float, optional):
                Duration of a signal frame frame in seconds.
                Each frame will get converted indepentedly.
                By default the whole signal is converted at once.

        Returns: Gain adjusted and quantized signal.
        """

        num_frame_samples = (
            int(round(frame_duration * input_signal.sampling_rate))
            if frame_duration > 0
            else input_signal.num_samples
        )
        num_frames = (
            int(np.ceil(input_signal.num_samples / num_frame_samples))
            if num_frame_samples > 0
            else 0
        )
        converted_signal = Signal.empty(
            input_signal.sampling_rate,
            input_signal.num_streams,
            0,
            carrier_frequency=input_signal.carrier_frequency,
        )

        # Iterate over each frame independtenly
        for f in range(num_frames):
            frame_samples = input_signal.samples[
                :, f * num_frame_samples : (f + 1) * num_frame_samples
            ]
            frame_signal = Signal(
                frame_samples, input_signal.sampling_rate, input_signal.carrier_frequency
            )

            converted_frame_signal = self.__convert_frame(frame_signal)
            converted_signal.append_samples(converted_frame_signal)

        return converted_signal

    def _quantize(self, input_signal: np.ndarray) -> np.ndarray:
        """Quantizes the input signal

        Args:
            input_signal(np.ndarray):
                Sample vector of the signal feeding into the quantizer.

        Returns:
            np.ndarray:
                Distorted signal after quantization.
                Note that the original signal amplitude will be preserved.
        """

        quantized_signal = np.zeros(input_signal.shape, dtype=complex)

        if self.num_quantization_bits is None:
            quantized_signal = input_signal

        else:
            max_amplitude = 1.0
            step = 2 * max_amplitude / self.num_quantization_levels

            if self.quantizer_type == QuantizerType.MID_RISER:
                bins = np.arange(-max_amplitude + step, max_amplitude, step)
                offset = 0.0

            elif self.quantizer_type == QuantizerType.MID_TREAD:
                bins = np.arange(-max_amplitude + step / 2, max_amplitude - step / 2, step)
                offset = -0.5 * step

            quant_idx = np.digitize(np.real(input_signal), bins)
            quantized_signal += quant_idx * step - (max_amplitude - step / 2) + offset
            quant_idx = np.digitize(np.imag(input_signal), bins)
            quantized_signal += 1j * (quant_idx * step - (max_amplitude - step / 2) + offset)

        return quantized_signal

    def plot_quantizer(
        self,
        input_samples: np.ndarray | None = None,
        label: str = "",
        fig_axes: plt.Axes | None = None,
    ) -> None:
        """Plot the quantizer characteristics.

        Generates a matplotlib plot depicting the staircase amplitude response.
        Note that only the real part is plotted, as the quantizer is applied separately in the real and imaginary parts.

        Args:

            input_samples (np.ndarray, optional):
                Sample points at which to evaluate the characteristics, i.e., the x-axis of the resulting
                characteristics plot. It should be a sorted number sequence.

            label(str, optional):
                A label describing the desired plot.

            fig_axes (Optional[plt.axes], optional):
                Axes to which to plot the charateristics.
                By default, a new figure is created.
        """

        _input_samples = (
            np.arange(-1, 1, 0.01) + 1j * np.arange(1, -1, -0.01)
            if input_samples is None
            else input_samples.flatten()
        )

        figure: plt.Figure | None = None
        if fig_axes is None:
            figure, quant_axes = plt.subplots()

            quant_axes.set_xlabel("Input Amplitude")
            quant_axes.set_ylabel("Output Amplitude")
        else:
            quant_axes = fig_axes

        output_samples = self.convert(Signal(_input_samples, 1.0)).samples.flatten()
        quant_axes.plot(np.real(_input_samples), np.real(output_samples))

        quant_axes.axhline(0)
        quant_axes.axvline(0)

        quant_axes.set_title(self.__class__.__name__ + " - " + label)
