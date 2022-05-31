# -*- coding: utf-8 -*-
"""
=======================================
Analog-to-Digital Converter
=======================================

Implements an analog-to digital converter.
Currently only uniform quantization is considered.


The following figure visualizes the quantizer responses.

.. plot:: scripts/plot_quantizer.py
   :align: center

"""

from __future__ import annotations
from enum import Enum
from typing import Type, Optional

import numpy as np
from matplotlib import pyplot as plt
from typing import Optional, Union

from hermespy.core import Serializable, Signal
from ruamel.yaml import ScalarNode, MappingNode, SafeRepresenter,  SafeConstructor
from ruamel.yaml.constructor import ConstructorError

from hermespy.tools.math import lin2db, DbConversionType, rms_value

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class GainControlType(Enum):
    """Type of automatig gain control """
    
    MAX_AMPLITUDE = 1
    RMS_AMPLITUDE = 2


class Gain(Serializable):
    """Base class for analog-to-digital conversion gain modeling."""
    
    yaml_tag = u'Gain'
    """YAML serialization tag."""

    __gain: float

    def __init__(self,
                 gain=1.0) -> None:
        """
        Args:
            gain (float, optional):
                signal gain
        """

        self.gain = gain

    @property
    def gain(self) -> float:
        """Gain before quantizer

        Quantizer operates by default between -1. and +1.
        Signal can be adjusted by to this range by appropriate gain setting.

        Returns:
            float: fixed gain

        """
        return self.__gain

    @gain.setter
    def gain(self, value: float) -> None:

        if value <= 0:
            raise ValueError("Gain must be larger than 0")

        self.__gain = value

    def multiply_signal(self, input_signal: Signal) -> None:
        input_signal.samples = input_signal.samples * self.gain

    def divide_signal(self, input_signal: Signal) -> None:
        input_signal.samples = input_signal.samples / self.gain

    @classmethod
    def to_yaml(cls: Type[Gain], representer: SafeRepresenter, node: AutomaticGainControl) -> MappingNode:
        """Serialize a `Gain` object to YAML.

        Args:
            representer (BaseRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (AutomaticGainControl):
                The ADC instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """
        state = {'gain': node.gain}

        return representer.represent_mapping(cls.yaml_tag, state)


class AutomaticGainControl(Gain):
    """Analog-to-digital conversion automatic gain control modeling."""
    
    yaml_tag = u'AutomaticGainControl'
    """YAML serialization tag."""

    __agc_type: GainControlType
    __backoff: float

    def __init__(self,
                 agc_type=GainControlType.MAX_AMPLITUDE,
                 backoff=1.0) -> None:
        """
        Args:
            agc_type (GainControlType, optional):
                Type of amplitude gain control at ADC input. Default is GainControlType.MAX_AMPLITUDE.

            backoff (float, optional):
                this is the ratio between maximum amplitude and the rms value or maximum of input signal,
                depending on AGC type. Default value is 1.0.

         """

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
    def agc_type(self, value: GainControlType) -> None:
        self.__agc_type = value

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

    def multiply_signal(self, input_signal: Signal) -> None:
        samples = input_signal.samples
        if self.agc_type == GainControlType.MAX_AMPLITUDE:
            max_amplitude = np.maximum(np.amax(np.real(samples)),
                                       np.amax(np.imag(samples))) * self.backoff
        elif self.agc_type == GainControlType.RMS_AMPLITUDE:
            max_amplitude = np.maximum(rms_value(np.real(samples)),
                                       rms_value(np.imag(samples))) * self.backoff

        self.gain = 1/max_amplitude

        super().multiply_signal(input_signal)

    @classmethod
    def from_yaml(cls: Type[AutomaticGainControl],
                  constructor: SafeConstructor,
                  node: Union[ScalarNode, MappingNode]) -> AutomaticGainControl:
        """Recall a new `AnalogDigitalConverter` instance from YAML.

        Args:
            constructor (RoundTripConstructor):
                A handle to the constructor extracting the YAML information.

            node (Union[ScalarNode, MappingNode]):
                YAML node representing the `AnalogDigitalConverter` serialization.

        Returns:
            AnalogDigitalConverter:
                Newly created `AnalogDigitalConverter` instance.
            """

        if isinstance(node, ScalarNode):
            return cls()

        state = SafeConstructor.construct_mapping(constructor, node, deep=False)

        return cls.InitializationWrapper(state)

    @classmethod
    def to_yaml(cls: Type[AutomaticGainControl],
                representer: SafeRepresenter, node: AutomaticGainControl) -> MappingNode:
        """Serialize a `AutomaticGainControl` object to YAML.

        Args:
            representer (BaseRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (AutomaticGainControl):
                The ADC instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """
        state = {'backoff': node.backoff, 'agc_type': node.agc_type.name}

        return representer.represent_mapping(cls.yaml_tag, state)


class QuantizerType(Enum):
    """Type of quantizer """
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

    yaml_tag = u'ADC'
    """YAML serialization tag"""

    __num_quantization_bits: Union[int, float]
    gain: Gain
    __quantizer_type: QuantizerType

    def __init__(self,
                 num_quantization_bits: int = np.inf,
                 gain: Optional[Gain] = None,
                 quantizer_type: QuantizerType = QuantizerType.MID_RISER) -> None:
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
    def num_quantization_bits(self) -> int:
        """Quantization resolution in bits

        Returns:
            int: Bit resolution (if 0 or np.inf, then no quantization is applied)

        Raises:
            ValueError: If resolution is less than zero.

        """

        return self.__num_quantization_bits

    @num_quantization_bits.setter
    def num_quantization_bits(self, value: Union[int, float]) -> None:

        if value < 0 or (isinstance(value, float) and value != np.inf and value != np.round(value)):
            raise ValueError("Number of bits must be a non-negative integer")
        elif value == 0 or value == np.inf:
            self.__num_quantization_bits = np.inf
        else:
            self.__num_quantization_bits = int(value)

    @property
    def num_quantization_levels(self) -> Union[int, float]:
        """Number of quantization levels

        Returns:
            int: Number of levels

        """
        return 2 ** self.num_quantization_bits

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

    def convert(self, input_signal: Signal) -> Signal:
        output_signal = input_signal.copy()
        self.gain.multiply_signal(output_signal)
        output_signal.samples = self._quantize(output_signal.samples)
        self.gain.divide_signal(output_signal)

        return output_signal

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
        if self.num_quantization_bits == np.inf:
            quantized_signal = input_signal
        else:
            max_amplitude = 1.0

            step = 2 * max_amplitude / self.num_quantization_levels

            if self.quantizer_type == QuantizerType.MID_RISER:
                bins = np.arange(-max_amplitude + step, max_amplitude, step)
                offset = 0
            elif self.quantizer_type == QuantizerType.MID_TREAD:
                bins = np.arange(-max_amplitude + step/2, max_amplitude - step/2, step)
                offset = -step/2

            quant_idx = np.digitize(np.real(input_signal), bins)
            quantized_signal += quant_idx * step - (max_amplitude - step / 2) + offset
            quant_idx = np.digitize(np.imag(input_signal), bins)
            quantized_signal += 1j * (quant_idx * step - (max_amplitude - step / 2) + offset)

        return quantized_signal

    def plot_quantizer(self,
                       input_samples: Optional[np.ndarray] = None,
                       label: str = "",
                       fig_axes: Optional[plt.axes] = None) -> None:
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

        if input_samples is None:
            input_samples = np.arange(-1, 1, .01) + 1j*np.arange(1, -1, -.01)

        figure: Optional[plt.figure] = None
        if fig_axes is None:

            figure = plt.figure()
            quant_axes = figure.add_axes()

            quant_axes.set_xlabel("Input Amplitude")
            quant_axes.set_ylabel("Output Amplitude")
        else:
            quant_axes = fig_axes

        output_samples = self.convert(input_samples)
        quant_axes.plot(np.real(input_samples), np.real(output_samples))

        quant_axes
        quant_axes.axhline(0)
        quant_axes.axvline(0)

        quant_axes.set_title(self.__class__.__name__ + " - " + label)

    @classmethod
    def from_yaml(cls: Type[AnalogDigitalConverter],
                  constructor: SafeConstructor,
                  node: Union[ScalarNode, MappingNode]) -> AnalogDigitalConverter:
        """Recall a new `AnalogDigitalConverter` instance from YAML.

        Args:
            constructor (RoundTripConstructor):
                A handle to the constructor extracting the YAML information.

            node (Union[ScalarNode, MappingNode]):
                YAML node representing the `AnalogDigitalConverter` serialization.

        Returns:
            AnalogDigitalConverter:
                Newly created `AnalogDigitalConverter` instance.
            """

        if isinstance(node, ScalarNode):
            return cls()

        state = SafeConstructor.construct_mapping(constructor, node)
        return cls.InitializationWrapper(state)

    @classmethod
    def to_yaml(cls: Type[AnalogDigitalConverter], representer: SafeRepresenter, node: AnalogDigitalConverter) -> MappingNode:
        """Serialize a `AnalogDigitalConverter` object to YAML.

        Args:
            representer (BaseRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (AnalogDigitalConverter):
                The ADC instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """
        state = {
            'num_quantization_bits': node.num_quantization_bits,
        }

        if not node.num_quantization_bits == np.inf:
            state['quantizer_type'] = node.quantizer_type.name
            state['gain_control'] = node.gain

        return representer.represent_mapping(cls.yaml_tag, state)

