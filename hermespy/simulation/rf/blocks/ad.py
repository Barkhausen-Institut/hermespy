# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from typing import TypeVar
from typing_extensions import override

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from hermespy.core import (
    SerializableEnum,
    Serializable,
    SerializationProcess,
    DeserializationProcess,
)
from hermespy.tools.math import rms_value
from ..block import (
    RFBlock,
    RFBlockPort,
    RFBlockPortType,
    DSPInputBlock,
    DSPOutputBlock,
    RFBlockRealization,
)
from ..signal import RFSignal
from ..noise import NoiseModel, NoiseLevel

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2025, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class ConverterBase(RFBlock):
    """Base class for analaog-digital and digital-analog converters."""

    __num_ports: int

    def __init__(
        self, num_quantization_bits: int | None = None, num_ports: int = 1, seed: int | None = None
    ) -> None:
        """
        Args:
            num_quantization_bits: ADC resolution in bits. Default is infinite resolution (no quantization)
            num_ports: Number of digital/analog ports of the converter.
            seed: Seed with which to initialize the block's random state.
        """

        # Init base class
        RFBlock.__init__(self, seed=seed)

        # Initialize attributes
        self.num_quantization_bits = num_quantization_bits
        self.__num_ports = num_ports

    @property
    @override
    def num_input_ports(self) -> int:
        return self.__num_ports

    @property
    @override
    def num_output_ports(self) -> int:
        return self.__num_ports

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

        Returns: Number of levels.
        """

        if self.__num_quantization_bits is None:
            return np.inf

        return 2**self.num_quantization_bits


class GainControlType(SerializableEnum):
    """Type of automatig gain control"""

    NONE = 0
    MAX_AMPLITUDE = 1
    RMS_AMPLITUDE = 2


GainType = TypeVar("GainType", bound="Gain")
"""Type of gain."""


class GainControlBase(Serializable):
    """Base class for all ADC gain control models."""

    __rescale_quantization: bool

    def __init__(self, rescale_quantization: bool = False) -> None:
        """
        Args:

            rescale_quantization:
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
    def estimate_gain(self, input_signal: RFSignal) -> float:
        """Estimate the gain required to adjust the signal to the ADC input range.

        Args:

            input_signal:
                Input signal to be adjusted.

        Returns: Linear gain to be applied to the `input_signal`'s Voltage samples.
        """
        ...  # pragma: no cover

    def adjust_signal(self, input_signal: RFSignal, gain: float) -> RFSignal:
        """Adjust the signal to the ADC input range.

        Args:

            input_signal:
                Input signal to be adjusted.

            gain:
                Linear gain to be applied to the `input_signal`'s Voltage samples.

        Returns: The adjusted signal.
        """

        return (input_signal * gain).view(RFSignal)

    def scale_quantized_signal(self, quantized_signal: RFSignal, gain: float) -> RFSignal:
        """Scale the quantized signal back to the original signal range before gain adjustment.

        Only applied if :py:attr:`rescale_quantization` is enabled.

        Args:

            quantized_signal:
                Quantized signal to be adjusted.

            gain:
                Linear gain to applied to the `input_signal`'s Voltage samples before quantization.

        Returns: The scaled qzanitized signal.
        """

        if not self.rescale_quantization:
            return quantized_signal

        scaled_signal = quantized_signal / gain
        return scaled_signal.view(RFSignal)


class Gain(GainControlBase):
    """Constant gain model."""

    __gain: float

    def __init__(self, gain: float = 1.0, rescale_quantization: bool = False) -> None:
        """
        Args:

            gain:
                Linear signal gain to be applied before ADC quantization.
                Unit by default, meaning no gain adjustment.

            rescale_quantization:
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

    def estimate_gain(self, input_signal: RFSignal) -> float:
        return self.gain

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.__gain, "gain")
        process.serialize_integer(int(self.rescale_quantization), "rescale_quantization")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> Gain:
        return Gain(
            process.deserialize_floating("gain", 1.0),
            bool(process.deserialize_integer("rescale_quantization", 0)),
        )


class AutomaticGainControl(GainControlBase):
    """Analog-to-digital conversion automatic gain control modeling."""

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

            agc_type:
                Type of amplitude gain control at ADC input. Default is GainControlType.MAX_AMPLITUDE.

            backoff:
                this is the ratio between maximum amplitude and the rms value or maximum of input signal,
                depending on AGC type. Default value is 1.0.

            rescale_quantization:
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
        """
        return self.__agc_type

    @agc_type.setter
    def agc_type(self, value: GainControlType | str) -> None:
        self.__agc_type = value if isinstance(value, GainControlType) else GainControlType[value]

    @property
    def backoff(self) -> float:
        """Quantizer backoff in linear scale

        This quantity determines the ratio between the maximum quantization level and the signal rms value

        Returns: The backoff in linear scale
        """
        return self.__backoff

    @backoff.setter
    def backoff(self, value: float) -> None:
        if value <= 0:
            raise ValueError("Backoff must be larger than 0")

        self.__backoff = value

    @override
    def estimate_gain(self, input_signal: RFSignal) -> float:
        if self.agc_type == GainControlType.MAX_AMPLITUDE:
            max_amplitude = 0
            max_amplitude = max(
                np.abs(np.real(input_signal)).max(),
                np.abs(np.imag(input_signal)).max(),
                max_amplitude,
            )

        elif self.agc_type == GainControlType.RMS_AMPLITUDE:
            max_amplitude = 0
            max_amplitude = max(
                rms_value(np.real(input_signal)), rms_value(np.imag(input_signal)), max_amplitude
            )

        else:
            raise RuntimeError("Unsupported gain control type")

        return 1 / (max_amplitude * self.backoff) if max_amplitude > 0.0 else 1.0

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_object(self.agc_type, "agc_type")
        process.serialize_floating(self.backoff, "backoff")
        process.serialize_integer(int(self.rescale_quantization), "rescale_quantization")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> AutomaticGainControl:
        return AutomaticGainControl(
            process.deserialize_object("agc_type", GainControlType, GainControlType.MAX_AMPLITUDE),
            process.deserialize_floating("backoff", 1.0),
            bool(process.deserialize_integer("rescale_quantization", 0)),
        )


class QuantizerType(SerializableEnum):
    """Type of quantizer"""

    MID_RISER = 0
    MID_TREAD = 1


class ADC(ConverterBase, DSPOutputBlock):
    """Model of an analog-digital converter (ADC).

    The ADC quantizes the input signal to a given number of bits, applying a gain control model before quantization.
    The quantizer can be configured to behave as either a mid-riser or mid-tread quantizer.
    """

    NO_RESAMPlING: float = 0.0
    """Magic number of the ADC's sampling rate indicating no resampling is applied."""

    gain: GainControlBase
    __quantizer_type: QuantizerType
    __i: RFBlockPort[ADC]

    def __init__(
        self,
        num_quantization_bits: int | None = None,
        gain: GainControlBase | None = None,
        quantizer_type: QuantizerType = QuantizerType.MID_RISER,
        num_ports: int = 1,
        noise_model: NoiseModel | None = None,
        noise_level: NoiseLevel | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            num_quantization_bits: ADC resolution in bits. Default is infinite resolution (no quantization)
            gain: Amplitude gain control at ADC input. Default is Gain(1.0), i.e., no gain.
            quantizer_type: Determines quantizer behaviour at zero. Default is QuantizerType.MID_RISER.
            num_ports: Number of analog ports of the ADC.
            seed: Seed with which to initialize the block's random state.
        """

        # Init base classes
        ConverterBase.__init__(self, num_quantization_bits, num_ports, seed)
        DSPOutputBlock.__init__(self, noise_model, noise_level, seed)

        # Initialize attributes
        self.gain = Gain() if gain is None else gain
        self.quantizer_type = quantizer_type
        self.__i = RFBlockPort(self, [p for p in range(num_ports)], RFBlockPortType.IN)

    @override
    def realize(
        self, bandwidth: float, oversampling_factor: int, carrier_frequency: float
    ) -> RFBlockRealization:
        return RFBlockRealization(
            bandwidth,
            oversampling_factor,
            self.noise_model.realize(self.noise_level.get_power(bandwidth)),
        )

    @property
    def i(self) -> RFBlockPort[ADC]:
        """Input ports of the analog-digital converter."""

        return self.__i

    @property
    def quantizer_type(self) -> QuantizerType:
        """Type of quantizer

        - QuantizationType.MID_TREAD: 0 can be the output of a quantization step. Since the number of quantization step
                                      must be even, negative values will have one step more than positive values
        - QuantizerType.MID_RISE: input values around zero are quantized as either -delta/2 or delta/2, with delta the
                                  quantization step
        """

        return self.__quantizer_type

    @quantizer_type.setter
    def quantizer_type(self, value: QuantizerType) -> None:
        self.__quantizer_type = value

    def _quantize(self, input_signal: np.ndarray) -> np.ndarray:
        """Quantizes the input signal

        Args:
            input_signal:
                Sample vector of the signal feeding into the quantizer.

        Returns:
            Distorted signal after quantization.
            Note that the original signal amplitude will be preserved.
        """

        quantized_signal = np.zeros(input_signal.shape, dtype=complex)

        if self.num_quantization_bits is None:
            quantized_signal = input_signal

        else:
            max_amplitude = 1.0

            # Mid-riser quantization
            if self.quantizer_type == QuantizerType.MID_RISER:
                step = 2 * max_amplitude / self.num_quantization_levels

                quantized_signal.real = step * (np.floor(input_signal.real / step) + 0.5)
                quantized_signal.imag = step * (np.floor(input_signal.imag / step) + 0.5)

                quantized_signal.real = np.clip(
                    quantized_signal.real, -max_amplitude + step / 2, max_amplitude - step / 2
                )
                quantized_signal.imag = np.clip(
                    quantized_signal.imag, -max_amplitude + step / 2, max_amplitude - step / 2
                )

            # Mid-tread quantization
            # Note that mid-tread quantization generates an odd number of levels
            else:
                step = 2 * max_amplitude / (self.num_quantization_levels + 1)

                clipped_signal = np.empty_like(input_signal)
                clipped_signal.real = np.clip(
                    input_signal.real, -max_amplitude, max_amplitude - step
                )
                clipped_signal.imag = np.clip(
                    input_signal.imag, -max_amplitude, max_amplitude - step
                )

                quantized_signal.real = step * np.floor(clipped_signal.real / step + 0.5)
                quantized_signal.imag = step * np.floor(clipped_signal.imag / step + 0.5)

        return quantized_signal

    def __convert_frame(self, frame_signal: RFSignal) -> RFSignal:
        """Converts an analog frame into a digitally quantized frame.

        Subroutine of :meth:`convert`.

        Args:
            input_signal: Signal to be converted.

        Returns: Gain adjusted and quantized signal.
        """

        # Initially, estimate the required gain to avoid clipping
        gain = self.gain.estimate_gain(frame_signal)

        # Scale the input signal according to the estimated gain
        adjusted_signal = self.gain.adjust_signal(frame_signal, gain)

        # Quantize adjusted signal
        quantized_signal = self._quantize(adjusted_signal.view(np.ndarray))

        # Rescale adjusted signal to the original amplitude range
        output_signal = self.gain.scale_quantized_signal(quantized_signal.view(RFSignal), gain)

        return output_signal

    @override
    def _propagate(
        self, realization: RFBlockRealization, input: RFSignal, frame_duration: float = 0.0
    ) -> RFSignal:
        # Downsample the input signal to the ADC's target sampling rate
        # if input.sampling_rate != self.sampling_rate and self.sampling_rate > 0.0:
        #     decimated_samples = decimate(input, int(input.sampling_rate / self.sampling_rate), axis=1)
        #     input = RFSignal(decimated_samples, self.sampling_rate, input.carrier_frequencies, input.noise_powers, input.delay)

        num_frame_samples = (
            int(round(frame_duration * input.sampling_rate))
            if frame_duration > 0
            else input.num_samples
        )
        num_frames = (
            int(np.ceil(input.num_samples / num_frame_samples)) if num_frame_samples > 0 else 0
        )
        converted_signal = RFSignal(
            input.num_streams,
            num_frames * num_frame_samples,
            sampling_rate=input.sampling_rate,
            carrier_frequencies=input.carrier_frequencies,
            noise_powers=input.noise_powers,
            delay=input.delay,
        )

        # Iterate over each frame independtenly
        for f in range(num_frames):
            frame_signal = input[:, f * num_frame_samples : (f + 1) * num_frame_samples]
            converted_frame_signal = self.__convert_frame(frame_signal)
            converted_signal[:, f * num_frame_samples : (f + 1) * num_frame_samples] = (
                converted_frame_signal
            )

        return converted_signal

    def plot_quantizer(
        self, input_samples: np.ndarray | None = None, label: str = "", fig_axes: Axes | None = None
    ) -> None:
        """Plot the quantizer characteristics.

        Generates a matplotlib plot depicting the staircase amplitude response.
        Note that only the real part is plotted, as the quantizer is applied separately in the real and imaginary parts.

        Args:

            input_samples:
                Sample points at which to evaluate the characteristics, i.e., the x-axis of the resulting
                characteristics plot. It should be a sorted number sequence.

            label: A label describing the desired plot.

            fig_axes:
                Axes to which to plot the charateristics.
                By default, a new figure is created.
        """

        _input_samples = (
            np.arange(-1.1, 1.1, 0.01) + 1j * np.arange(1.1, -1.1, -0.01)
            if input_samples is None
            else input_samples.flatten()
        )

        figure: Figure | None = None
        if fig_axes is None:
            figure, quant_axes = plt.subplots()

            quant_axes.set_xlabel("Input Amplitude")
            quant_axes.set_ylabel("Output Amplitude")
        else:
            quant_axes = fig_axes

        input_signal = RFSignal.FromNDArray(_input_samples.reshape(1, -1), 1.0)
        output_samples = (
            self._propagate(self.realize(1.0, 1, 0.0), input_signal).view(np.ndarray).flatten()
        )
        quant_axes.plot(np.real(_input_samples), np.real(output_samples))

        quant_axes.axhline(0)
        quant_axes.axvline(0)

        quant_axes.set_title(self.__class__.__name__ + " - " + label)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        if self.num_quantization_bits is not None:
            process.serialize_integer(self.num_quantization_bits, "num_quantization_bits")
        process.serialize_object(self.gain, "gain")
        process.serialize_object(self.quantizer_type, "quantizer_type")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> ADC:
        return ADC(
            process.deserialize_integer("num_quantization_bits", None),
            process.deserialize_object("gain", GainControlBase, None),
            process.deserialize_object("quantizer_type", QuantizerType, QuantizerType.MID_RISER),
        )


class DAC(ConverterBase, DSPInputBlock):

    __o: RFBlockPort[DAC]

    def __init__(
        self,
        num_quantization_bits: int | None = None,
        num_ports: int = 1,
        noise_model: NoiseModel | None = None,
        noise_level: NoiseLevel | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:
            num_quantization_bits: DAC resolution in bits. Default is infinite resolution (no quantization)
            num_ports: Number of analog ports of the DAC.
            seed: Seed with which to initialize the block's random state.
        """

        # Init base classes
        ConverterBase.__init__(self, num_quantization_bits, num_ports, seed)
        DSPOutputBlock.__init__(self, noise_model, noise_level, seed)

        # Initialize attributes
        self.__o = RFBlockPort(self, [p for p in range(num_ports)], RFBlockPortType.OUT)

    @override
    def realize(
        self, bandwidth: float, oversampling_factor: int, carrier_frequency: float
    ) -> RFBlockRealization:
        return RFBlockRealization(
            bandwidth,
            oversampling_factor,
            self.noise_model.realize(self.noise_level.get_power(bandwidth)),
        )

    @override
    def _propagate(self, realization: RFBlockRealization, input: RFSignal) -> RFSignal:
        return input

    @property
    def o(self) -> RFBlockPort[DAC]:
        """Output ports of the digital-analog converter."""

        return self.__o

    @override
    def serialize(self, process: SerializationProcess) -> None:
        if self.num_quantization_bits is not None:
            process.serialize_integer(self.num_quantization_bits, "num_quantization_bits")
        process.serialize_integer(self.num_input_ports, "num_ports")
        process.serialize_object(self.noise_model, "noise_model")
        process.serialize_object(self.noise_level, "noise_level")
        if self.seed is not None:
            process.serialize_integer(self.seed, "seed")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> DAC:
        return DAC(
            process.deserialize_integer("num_quantization_bits", None),
            process.deserialize_integer("num_ports", 1),
            process.deserialize_object("noise_model", NoiseModel, None),
            process.deserialize_object("noise_level", NoiseLevel, None),
            process.deserialize_integer("seed", None),
        )
