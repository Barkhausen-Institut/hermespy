# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Any, TypeVar
from typing_extensions import override

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi

from hermespy.core import Serializable, VAT, SerializationProcess, DeserializationProcess

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


class PowerAmplifier(Serializable):
    """Base class of a distorionless power-amplifier model."""

    adjust_power: bool
    """Power adjustment flag.

    If enabled, the power amplifier will normalize the distorted signal after propagation modeling.
    """

    __saturation_amplitude: float

    def __init__(
        self, saturation_amplitude: float = float("inf"), adjust_power: bool = False
    ) -> None:
        """
        Args:

            saturation_amplitude:
                Cut-off point for the linear behaviour of the amplification in Volt.

            adjust_power:
                Power adjustment flag.
        """

        self.saturation_amplitude = saturation_amplitude
        self.adjust_power = adjust_power

        Serializable.__init__(self)

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

    def send(self, input_signal: np.ndarray) -> np.ndarray:
        """Model signal amplification characteristics.

        Internally calls the model subroutine of power-amplifier models implementing this prototype-class.

        Args:
            input_signal: Sample vector of the signal feeding into the power amplifier.

        Returns: Distorted signal after amplification modeling.
        """

        distorted_signal = self.model(input_signal)

        # Adjust distorted signal if the respective flag is enabled
        if self.adjust_power:
            loss = np.linalg.norm(distorted_signal) / np.linalg.norm(input_signal)
            distorted_signal /= loss

        return distorted_signal

    def model(self, input_signal: np.ndarray) -> np.ndarray:
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
    ) -> plt.Figure:
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

        fig: plt.Figure
        if axes:
            _axes = axes.flatten()[0]
            fig = _axes.get_figure()  # type: ignore
        else:
            fig, _axes = plt.subplots(1, 1, squeeze=True)
            fig.suptitle(title if title else self.title)

        if samples is None:
            samples = np.arange(0, 2, 0.01) * self.saturation_amplitude

        model = self.model(samples.astype(complex))
        amplitude = abs(model)
        phase = np.angle(model)

        amplitude_axes: plt.Axes = _axes
        phase_axes: plt.Axes = amplitude_axes.twinx()  # type: ignore

        amplitude_axes.set_xlabel("Input Amplitude")
        amplitude_axes.set_ylabel("Output Amplitude")

        phase_axes.set_ylabel("Output Phase")
        phase_axes.set_ylim((-pi, pi))

        amplitude_axes.plot(samples, amplitude, "-")
        phase_axes.plot(samples, phase, "--")

        return fig

    @override
    def serialize(self, process: SerializationProcess) -> None:
        process.serialize_floating(self.saturation_amplitude, "saturation_amplitude")
        process.serialize_integer(self.adjust_power, "adjust_power")

    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> PowerAmplifier:
        saturation_amplitude = process.deserialize_floating("saturation_amplitude")
        adjust_power = bool(process.deserialize_integer("adjust_power", 0))

        return cls(saturation_amplitude, adjust_power)


class ClippingPowerAmplifier(PowerAmplifier):
    """Model of a clipping power amplifier."""

    def __init__(
        self, saturation_amplitude: float = float("inf"), adjust_power: bool = False
    ) -> None:
        """
        Args:

            saturation_amplitude:
                Cut-off point for the linear behaviour of the amplification in Volt.

            adjust_power:
                Power adjustment flag.
        """

        # Initialize base class
        PowerAmplifier.__init__(self, saturation_amplitude, adjust_power)

    def model(self, input_signal: np.ndarray) -> np.ndarray:
        output_signal = input_signal.copy()

        clip_idx = np.nonzero(np.abs(input_signal) > self.saturation_amplitude)
        output_signal[clip_idx] = self.saturation_amplitude * np.exp(
            1j * np.angle(input_signal[clip_idx])
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

            \*\*kwargs:
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

    def model(self, input_signal: np.ndarray) -> np.ndarray:
        p = self.smoothness_factor
        gain = (1 + (np.abs(input_signal) / self.saturation_amplitude) ** (2 * p)) ** (-1 / (2 * p))

        return input_signal * gain

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
            adjust_power=bool(process.deserialize_integer("adjust_power", 0)),
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

            \*\*kwargs:
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

    def model(self, input_signal: np.ndarray) -> np.ndarray:
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
            adjust_power=bool(process.deserialize_integer("adjust_power", 0)),
        )


class CustomPowerAmplifier(PowerAmplifier):
    """Model of a customized power amplifier."""

    __input: np.ndarray
    __gain: np.ndarray
    __phase: np.ndarray

    def __init__(
        self, input: np.ndarray, gain: np.ndarray, phase: np.ndarray, **kwargs: Any
    ) -> None:
        """
        Args:

            input:
            gain:
            phase:

            \*\*kwargs:
                PowerAmplifier base class initialization arguments.

        Raises:
            ValueError: If `input`, `gain`, and `phase` are not vectors of identical length.
        """

        if input.ndim != 1:
            raise ValueError("Custom power amplifier input must be a vector")

        if gain.ndim != 1:
            raise ValueError("Custom power amplifier gain must be a vector")

        if phase.ndim != 1:
            raise ValueError("Custom power amplifier phase must be a vector")

        if len(input) != len(gain) != len(phase):
            raise ValueError(
                "Custom power amplifier input, gain and phase vectors must be of identical length"
            )

        self.__input = input
        self.__gain = gain
        self.__phase = phase

        PowerAmplifier.__init__(self, **kwargs)

    def model(self, input_signal: np.ndarray) -> np.ndarray:
        amp = np.abs(input_signal) / self.saturation_amplitude
        gain = np.interp(amp, self.__input, self.__gain)
        phase_shift = np.interp(amp, self.__input, self.__phase)

        return input_signal * gain * np.exp(1j * phase_shift)

    @property
    def input(self) -> np.ndarray:
        return self.__input.copy()

    @property
    def gain(self) -> np.ndarray:
        return self.__gain.copy()

    @property
    def phase(self) -> np.ndarray:
        return self.__phase.copy()

    @override
    def serialize(self, process: SerializationProcess) -> None:
        PowerAmplifier.serialize(self, process)
        process.serialize_array(self.__input, "input")
        process.serialize_array(self.__gain, "gain")
        process.serialize_array(self.__phase, "phase")

    @override
    @classmethod
    def Deserialize(cls, process: DeserializationProcess) -> CustomPowerAmplifier:
        return cls(
            process.deserialize_array("input", np.float64),
            process.deserialize_array("gain", np.float64),
            process.deserialize_array("phase", np.float64),
            saturation_amplitude=process.deserialize_floating("saturation_amplitude", float("inf")),
            adjust_power=bool(process.deserialize_integer("adjust_power", 0)),
        )
