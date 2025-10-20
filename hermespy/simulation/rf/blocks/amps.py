# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, TypeVar
from typing_extensions import override

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.constants import pi

from hermespy.core import Serializable, VAT, SerializationProcess, DeserializationProcess
from ..block import RFBlock, RFBlockPort, RFBlockPortType, RFBlockRealization
from ..signal import RFSignal
from ..noise import NoiseModel, NoiseLevel

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.5.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


_PAT = TypeVar("_PAT", bound="PowerAmplifier")
"""Type of power amplifier."""


class PowerAmplifier(RFBlock, Serializable):
    """Base class of a distorionless power-amplifier model."""

    __gain: float
    __i: RFBlockPort[PowerAmplifier]
    __o: RFBlockPort[PowerAmplifier]

    def __init__(
        self,
        gain: float = 1.0,
        saturation_amplitude: float = float("inf"),
        noise_model: NoiseModel | None = None,
        noise_level: NoiseLevel | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            gain:
                Linear voltage gain of the power amplifier.

            saturation_amplitude:
                Cut-off point for the linear behaviour of the amplification in Volt.
        """

        self.gain = gain
        self.saturation_amplitude = saturation_amplitude
        self.__i = RFBlockPort(self, 0, RFBlockPortType.IN)
        self.__o = RFBlockPort(self, 0, RFBlockPortType.OUT)

        RFBlock.__init__(self, noise_model, noise_level, seed)
        Serializable.__init__(self)

    @property
    def gain(self) -> float:
        """Linear voltage gain of the power amplifier.

        Raises:
            ValueError: If gain is smaller than zero.
        """

        return self.__gain

    @gain.setter
    def gain(self, value: float) -> None:
        """Set the linear voltage gain of the power amplifier."""

        if value < 0.0:
            raise ValueError("Power-Amplifier model gain must be greater or equal to zero")

        self.__gain = value

    @property
    def saturation_amplitude(self) -> float:
        """Cut-off point for the linear behaviour of the amplification.

        Referred to as :math:`s_\\mathrm{sat} \\ \\mathbb{R}_{+}` in equations.

        Returns: Saturation amplitude in Volt.

        Raises:
            ValueError: If amplitude is smaller than zero.
        """

        return self.__saturation_amplitude

    @saturation_amplitude.setter
    def saturation_amplitude(self, value: float) -> None:
        """Set the cut-off point for the linear behaviour of the amplification."""

        if value < 0.0:
            raise ValueError(
                "Power-Amplifier model saturation amplitude must be greater or equal to zero"
            )

        self.__saturation_amplitude = value

    @property
    @override
    def num_input_ports(self) -> int:
        return 1

    @property
    @override
    def num_output_ports(self) -> int:
        return 1

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

        # Amplify the input signal
        amplified_signal: RFSignal = input * self.gain  # type: ignore

        # Model amplification characteristics
        distorted_signal = self.model(amplified_signal)
        return distorted_signal

    @property
    def i(self) -> RFBlockPort[PowerAmplifier]:
        """Input port of the power amplifier."""

        return self.__i

    @property
    def o(self) -> RFBlockPort[PowerAmplifier]:
        """Output port of the power amplifier."""

        return self.__o

    def model(self, input_signal: RFSignal) -> RFSignal:
        """Model signal amplification characteristics.

        Args:
            input_signal: Sample vector of the signal feeding into the power amplifier.

        Returns: Distorted signal after amplification modeling.
        """

        # No modeling in the prototype, just return the non-distorted input signal
        return input_signal

    @property
    def title(self) -> str:
        return self.__class__.__name__ + " Characteristics"

    def plot_characteristics(
        self,
        axes: VAT | None = None,
        *,
        title: str | None = None,
        samples: np.ndarray | None = None,
    ) -> Figure:
        """Plot the power amplifier distortion characteristics.

        Generates a matplotlib plot depicting the phase/amplitude.

        Args:

            axes:
                The axis object into which the information should be plotted.
                If not specified, the routine will generate and return a new figure.

            title:
                Title of the generated plot.

            samples:
                Sample points at which to evaluate the characteristics.
                In other words, the x-axis of the resulting characteristics plot.

        Returns: Handle to the generated figure.
        """

        fig: Figure
        if axes:
            _axes = axes.flatten()[0]
            fig = _axes.get_figure()  # type: ignore
        else:
            fig, _axes = plt.subplots(1, 1, squeeze=True)
            fig.suptitle(title if title else self.title)

        if samples is None:
            samples = (np.arange(0, 2, 0.01, np.complex128) * self.saturation_amplitude).reshape((1, -1))

        model = self.model(
            RFSignal(samples.shape[0], samples.shape[1], 1.0, buffer=bytearray(samples))
        )
        amplitude = abs(model)
        phase = np.angle(model)

        amplitude_axes: Axes = _axes
        phase_axes: Axes = amplitude_axes.twinx()  # type: ignore

        amplitude_axes.set_xlabel("Input Amplitude")
        amplitude_axes.set_ylabel("Output Amplitude")

        phase_axes.set_ylabel("Output Phase")
        phase_axes.set_ylim((-pi, pi))

        amplitude_axes.plot(samples, amplitude, "-")
        phase_axes.plot(samples, phase, "--")

        return fig

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.gain, "gain")
        process.serialize_floating(self.saturation_amplitude, "saturation_amplitude")
        process.serialize_object(self.noise_model, "noise_model")
        process.serialize_object(self.noise_level, "noise_level")
        if self.seed is not None:
            process.serialize_integer(self.seed, "seed")

    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> PowerAmplifier:
        gain = process.deserialize_floating("gain")
        saturation_amplitude = process.deserialize_floating("saturation_amplitude")
        noise_model = process.deserialize_object("noise_model", NoiseModel)
        noise_level = process.deserialize_object("noise_level", NoiseLevel)
        seed = process.deserialize_integer("seed", None)
        return cls(gain, saturation_amplitude, noise_model, noise_level, seed)


class ClippingPowerAmplifier(PowerAmplifier):
    """Model of a clipping power amplifier."""

    def __init__(
        self,
        gain: float = 1.0,
        saturation_amplitude: float = float("inf"),
        noise_model: NoiseModel | None = None,
        noise_level: NoiseLevel | None = None,
        seed: int | None = None,
    ) -> None:
        """
        Args:

            gain:
                Linear voltage gain of the power amplifier.

            saturation_amplitude:
                Cut-off point for the linear behaviour of the amplification in Volt.
        """

        # Initialize base class
        PowerAmplifier.__init__(self, gain, saturation_amplitude, noise_model, noise_level, seed)

    def model(self, input_signal: RFSignal) -> RFSignal:
        output_signal = input_signal.copy()

        clip_idx = np.nonzero(np.abs(input_signal) > self.saturation_amplitude)
        output_signal[clip_idx] = self.saturation_amplitude * np.exp(
            1j * np.angle(input_signal.view(np.ndarray)[clip_idx])
        )

        return output_signal


class RappPowerAmplifier(PowerAmplifier):
    """Model of a power amplifier according to Rapp's model.

    See :footcite:t:`1991:rapp` for further details.
    """

    def __init__(self, smoothness_factor: float = 1.0, **kwargs: Any) -> None:
        """
        Args:

            smoothness_factor:
                Smoothness factor of the amplification saturation characteristics.

            kwargs:
                PowerAmplifier base class initialization arguments.
        """

        self.smoothness_factor = smoothness_factor

        # Initialize base class
        PowerAmplifier.__init__(self, **kwargs)

    @property
    def smoothness_factor(self) -> float:
        """Smoothness factor of the amplification saturation characteristics.

        Also referred to as Rapp-factor :math:`p_\\mathrm{Rapp}`.

        Returns: Smoothness factor.

        Raises:
            ValueError: If smoothness factor is smaller than one.
        """

        return self.__smoothness_factor

    @smoothness_factor.setter
    def smoothness_factor(self, value: float) -> None:
        """Set smoothness factor of the amplification saturation characteristics."""

        if value <= 0.0:
            raise ValueError("Smoothness factor must be greater than zero.")

        self.__smoothness_factor = value

    @override
    def model(self, input_signal: RFSignal) -> RFSignal:
        p = self.smoothness_factor
        gain = (1 + (np.abs(input_signal) / self.saturation_amplitude) ** (2 * p)) ** (-1 / (2 * p))

        return input_signal * gain  # type: ignore

    @override
    def serialize(self, process: SerializationProcess) -> None:
        PowerAmplifier.serialize(self, process)
        process.serialize_floating(self.smoothness_factor, "smoothness_factor")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> RappPowerAmplifier:
        return cls(
            process.deserialize_floating("smoothness_factor", 1.0),
            saturation_amplitude=process.deserialize_floating("saturation_amplitude", float("inf")),
        )


class SalehPowerAmplifier(PowerAmplifier):
    """Model of a power amplifier according to Saleh.

    See :footcite:t:`1981:saleh` for further details.
    """

    phase_alpha: float
    """Phase model factor :math:`\\alpha_\\Phi`."""

    phase_beta: float
    """Phase model factor :math:`\\beta_\\Phi`."""

    __amplitude_alpha: float  # Amplitude model factor alpha.
    __amplitude_beta: float  # Amplitude model factor beta.

    def __init__(
        self,
        amplitude_alpha: float,
        amplitude_beta: float,
        phase_alpha: float,
        phase_beta: float,
        **kwargs: Any,
    ) -> None:
        """
        Args:

            amplitude_alpha:
                Amplitude model factor alpha.

            amplitude_beta:
                Amplitude model factor beta.

            phase_alpha (float)
                Phase model factor alpha.

            phase_beta:
                Phase model factor beta.

            kwargs:
                PowerAmplifier base class initialization arguments.
        """

        self.amplitude_alpha = amplitude_alpha
        self.amplitude_beta = amplitude_beta
        self.phase_alpha = phase_alpha
        self.phase_beta = phase_beta

        # Initialize base class
        PowerAmplifier.__init__(self, **kwargs)

    @property
    def amplitude_alpha(self) -> float:
        """Amplitude model factor :math:`\\alpha_\\mathrm{a}`.

        Returns:
            float: Amplitude factor.

        Raises:
            ValueError: If the factor is smaller than zero.
        """

        return self.__amplitude_alpha

    @amplitude_alpha.setter
    def amplitude_alpha(self, value: float) -> None:
        """Set the amplitude model factor alpha."""

        if value < 0.0:
            raise ValueError("Amplitude model factor alpha must be greater or equal to zero")

        self.__amplitude_alpha = value

    @property
    def amplitude_beta(self) -> float:
        """Amplitude model factor :math:`\\beta_\\mathrm{a}`.

        Returns: Amplitude factor.

        Raises:
            ValueError: If the factor is smaller than zero.
        """

        return self.__amplitude_beta

    @amplitude_beta.setter
    def amplitude_beta(self, value: float) -> None:
        """Set the amplitude model factor beta."""

        if value < 0.0:
            raise ValueError("Amplitude model factor beta must be greater or equal to zero")

        self.__amplitude_beta = value

    @override
    def model(self, input_signal: RFSignal) -> RFSignal:
        amp = np.abs(input_signal) / self.saturation_amplitude
        gain = self.__amplitude_alpha / (1 + self.__amplitude_beta * amp**2)
        phase_shift = self.phase_alpha * amp**2 / (1 + self.phase_beta * amp**2)

        return input_signal * gain * np.exp(1j * phase_shift)

    @override
    def serialize(self, process: SerializationProcess) -> None:
        PowerAmplifier.serialize(self, process)
        process.serialize_floating(self.amplitude_alpha, "amplitude_alpha")
        process.serialize_floating(self.amplitude_beta, "amplitude_beta")
        process.serialize_floating(self.phase_alpha, "phase_alpha")
        process.serialize_floating(self.phase_beta, "phase_beta")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> SalehPowerAmplifier:
        return cls(
            process.deserialize_floating("amplitude_alpha"),
            process.deserialize_floating("amplitude_beta"),
            process.deserialize_floating("phase_alpha"),
            process.deserialize_floating("phase_beta"),
            saturation_amplitude=process.deserialize_floating("saturation_amplitude", float("inf")),
        )


class CustomPowerAmplifier(PowerAmplifier):
    """Model of a customized power amplifier."""

    __input: np.ndarray
    __gains: np.ndarray
    __phases: np.ndarray

    def __init__(
        self, input: np.ndarray, gains: np.ndarray, phases: np.ndarray, **kwargs: Any
    ) -> None:
        """
        Args:

            input:
            gains:
            phases:

            **kwargs:
                PowerAmplifier base class initialization arguments.

        Raises:
            ValueError: If `input`, `gain`, and `phase` are not vectors of identical length.
        """

        if input.ndim != 1:
            raise ValueError("Custom power amplifier input must be a vector")

        if gains.ndim != 1:
            raise ValueError("Custom power amplifier gain must be a vector")

        if phases.ndim != 1:
            raise ValueError("Custom power amplifier phase must be a vector")

        if len(input) != len(gains) != len(phases):
            raise ValueError(
                "Custom power amplifier input, gain and phase vectors must be of identical length"
            )

        self.__input = input
        self.__gains = gains
        self.__phases = phases

        PowerAmplifier.__init__(self, **kwargs)

    @override
    def model(self, input_signal: RFSignal) -> RFSignal:
        amp = np.abs(input_signal) / self.saturation_amplitude
        gain = np.interp(amp, self.__input, self.__gains)
        phase_shift = np.interp(amp, self.__input, self.__phases)

        return input_signal * gain * np.exp(1j * phase_shift)

    @property
    def input(self) -> np.ndarray:
        return self.__input.copy()

    @property
    def gains(self) -> np.ndarray:
        return self.__gains.copy()

    @property
    def phases(self) -> np.ndarray:
        return self.__phases.copy()

    @override
    def serialize(self, process: SerializationProcess) -> None:
        PowerAmplifier.serialize(self, process)
        process.serialize_array(self.__input, "input")
        process.serialize_array(self.__gains, "gain")
        process.serialize_array(self.__phases, "phase")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> CustomPowerAmplifier:
        return cls(
            process.deserialize_array("input", np.float64),
            process.deserialize_array("gain", np.float64),
            process.deserialize_array("phase", np.float64),
            saturation_amplitude=process.deserialize_floating("saturation_amplitude", float("inf")),
        )
