# -*- coding: utf-8 -*-
"""
=========================
Power Amplifier Modeling
=========================

Power amplification modeling represents a single processing step within the radio-frequency hardware model of
transmitting modems.

.. autoclasstree:: hermespy.simulation.rf_chain.power_amplifier
   :strict:
   :full:

Each power amplifier model instance implements an equation

.. math::

   s'(t) = f\\lbrace s(t), t \\rbrace \\in \\mathbb{C}

distorting complex-valued signal samples :math:`s(t) \\in \\mathbb{C}` feeding into the power amplifier.
For time-invariant (i.e. memoryless) models :math:`f\\lbrace s(t), t \\rbrace = f\\lbrace s(t) \\rbrace`.
The following figure visualizes the gain characteristics for the implemented amplification models for a saturation point :math:`s_\\mathrm{sat} = 1`.

.. plot:: scripts/plot_pa_characteristics.py
   :align: center


"""

from __future__ import annotations
from abc import abstractmethod
from math import sqrt
from typing import Any, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from ruamel.yaml import ScalarNode, MappingNode, SafeRepresenter,  SafeConstructor
from ruamel.yaml.constructor import ConstructorError
from scipy.constants import pi

from hermespy.core import Serializable

__author__ = "Andre Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Andre Noll Barreto", "Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class PowerAmplifier(Serializable):
    """Base class of a power-amplifier model.

    Implements a distortion-less amplification model

    .. math::

       s'(t) = s(t) \\text{,}

    which may be overwritten by classes inheriting from this base.
    """

    yaml_tag: str = u'Distortionless'
    """YAML serialization tag."""

    adjust_power: bool
    """Power adjustment flag.
    
    If enabled, the power amplifier will normalize the distorted signal after propagation modeling.
    """

    __saturation_amplitude: float

    def __init__(self,
                 saturation_amplitude: float = float('inf'),
                 adjust_power: bool = False) -> None:
        """
        Args:

            saturation_amplitude (float, optional):
                Cut-off point for the linear behaviour of the amplification in Volt.

            adjust_power (bool, optional):
                Power adjustment flag.
        """

        self.saturation_amplitude = saturation_amplitude
        self.adjust_power = adjust_power
        
        Serializable.__init__(self)

    @property
    def saturation_amplitude(self) -> float:
        """Cut-off point for the linear behaviour of the amplification.
        
        Referred to as :math:`s_\\mathrm{sat} \\ \\mathbb{R}_{+}` in equations.

        Returns:
            float: Saturation amplitude in Volt.

        Raises:
            ValueError:
                If amplitude is smaller than zero.
        """

        return self.__saturation_amplitude

    @saturation_amplitude.setter
    def saturation_amplitude(self, value: float) -> None:
        """Set the cut-off point for the linear behaviour of the amplification."""

        if value < 0.:
            raise ValueError("Power-Amplifier model saturation amplitude must be greater or equal to zero")

        self.__saturation_amplitude = value

    def send(self, input_signal: np.ndarray) -> np.ndarray:
        """Model signal amplification characteristics.

        Internally calls the model subroutine of power-amplifier models implementing this prototype-class.

        Args:
            input_signal(np.ndarray):
                Sample vector of the signal feeding into the power amplifier.

        Returns:
            np.ndarray:
                Distorted signal after amplification modeling.
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
            input_signal(np.ndarray):
                Sample vector of the signal feeding into the power amplifier.

        Returns:
            np.ndarray:
                Distorted signal after amplification modeling.
        """

        # No modeling in the prototype, just return the non-distorted input signal
        return input_signal

    def plot(self,
             samples: Optional[np.ndarray] = None,
             axes: Optional[Tuple[plt.axes, plt.axes]] = None) -> None:
        """Plot the power amplifier distortion characteristics.

        Generates a matplotlib plot depicting the phase/amplitude.

        Args:
        
            samples (np.ndarray, optional):
                Sample points at which to evaluate the characteristics.
                In other words, the x-axis of the resulting characteristics plot.
                
            axes (Optional[Tuple[plt.axes, plt.axes]], optional):
                Axes to which to plot the characteristics.
                By default, a new figure is created.
        """

        if samples is None:
            samples = np.arange(0, 2, 0.01) * self.saturation_amplitude

        model = self.model(samples.astype(complex))
        amplitude = abs(model)
        phase = np.angle(model)

        figure: Optional[plt.figure] = None
        if axes is None:
            
            figure, amplitude_axes = plt.subplots()
            figure.suptitle(self.__class__.__name__ + " Characteristics")

            amplitude_axes.set_xlabel("Input Amplitude")
            amplitude_axes.set_ylabel("Output Amplitude")
            
            phase_axes = amplitude_axes.twinx()
            phase_axes.set_ylabel("Output Phase")
            phase_axes.set_ylim([-pi, pi])
            
        else:
            
            amplitude_axes = axes[0]
            phase_axes = axes[1]

        amplitude_axes.plot(samples, amplitude)
        phase_axes.plot(samples, phase)
        
        if figure is not None:
            figure.tight_layout()

    @classmethod
    def to_yaml(cls: Type[PowerAmplifier], representer: SafeRepresenter, node: PowerAmplifier) -> MappingNode:
        """Serialize a `PowerAmplifier` object to YAML.

        Args:
            representer (BaseRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (PowerAmplifier):
                The amplifier instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            'saturation_amplitude':  node.__saturation_amplitude,
            'adjust_power': node.adjust_power,
        }

        return representer.represent_mapping(cls.yaml_tag, state)


class ClippingPowerAmplifier(PowerAmplifier):
    """Model of a clipping power amplifier.

    Complex signal samples with amplitudes above :math:`s_\\mathrm{sat} \\in \\mathbb{R}` will be clipped
    to the maximum amplitude value.
    In case of clipping, the respective sample angles will be preserved, so that

    .. math::

       s'(t) = \\frac{s(t)}{|s(t)|} \\cdot \\min{(|s(t)|, s_\\mathrm{sat})} \\text{.}
    """

    yaml_tag = u'Clipping'
    """YAML serialization tag."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Args:
            kwargs (Any):
                PowerAmplifier base class initialization arguments.
        """

        # Initialize base class
        PowerAmplifier.__init__(self, **kwargs)

    def model(self, input_signal: np.ndarray) -> np.ndarray:

        output_signal = input_signal.copy()

        clip_idx = np.nonzero(np.abs(input_signal) > self.saturation_amplitude)
        output_signal[clip_idx] = self.saturation_amplitude * np.exp(1j * np.angle(input_signal[clip_idx]))

        return output_signal


class RappPowerAmplifier(PowerAmplifier):
    """Model of a power amplifier according to Rapp's model.

    Implements a saturation characteristic according to

    .. math::

       s'(t) = s(t) \\cdot \\left( 1 + \\left( \\frac{|s(t)|}{s_\\mathrm{sat}} \\right)^{2p_\\mathrm{Rapp}} \\right)^{-\\frac{1}{2p_\\mathrm{Rapp}}}
       \\text{,}

    where :math:`p_\\mathrm{Rapp} \\in \\lbrace x \\in \\mathbb{R} | x \\geq 1 \\rbrace`
    denotes the smoothness factor of the saturation curve.

    See :footcite:t:`1991:rapp` for further details.
    """

    yaml_tag = u'Rapp'
    """YAML serialization tag."""

    def __init__(self,
                 smoothness_factor: float = 1.,
                 **kwargs: Any) -> None:
        """Clipping Power Amplifier object initialization.

        Args:

            smoothness_factor(float, optional):
                Smoothness factor of the amplification saturation characteristics.

            kwargs (Any):
                PowerAmplifier base class initialization arguments.
        """

        self.smoothness_factor = smoothness_factor

        # Initialize base class
        PowerAmplifier.__init__(self, **kwargs)

    @property
    def smoothness_factor(self) -> float:
        """Smoothness factor of the amplification saturation characteristics.

        Also referred to as Rapp-factor :math:`p_\\mathrm{Rapp}`.

        Returns:
            float: Smoothness factor.

        Raises:
            ValueError: If smoothness factor is smaller than one.
        """

        return self.__smoothness_factor

    @smoothness_factor.setter
    def smoothness_factor(self, value: float) -> None:
        """Set smoothness factor of the amplification saturation characteristics."""

        if value <= 0.:
            raise ValueError("Smoothness factor must be greater than zero.")

        self.__smoothness_factor = value

    def model(self, input_signal: np.ndarray) -> np.ndarray:

        p = self.smoothness_factor
        gain = (1 + (np.abs(input_signal) / self.saturation_amplitude) ** (2 * p)) ** (-1 / (2 * p))

        return input_signal * gain

    @classmethod
    def to_yaml(cls: Type[RappPowerAmplifier], representer: SafeRepresenter, node: RappPowerAmplifier) -> MappingNode:
        """Serialize a `RappPowerAmplifier` object to YAML.

        Args:
            representer (BaseRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (RappPowerAmplifier):
                The amplifier instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            'smoothness_factor': node.__smoothness_factor,
        }

        representation = representer.represent_mapping(cls.yaml_tag, state)
        representation.value.extend(PowerAmplifier.to_yaml(representer, node).value)
        
        return representation


class SalehPowerAmplifier(PowerAmplifier):
    """Model of a power amplifier according to Saleh.

    Implements a saturation characteristic according to

    .. math::

       s'(t) = s(t) \\cdot A\\lbrace s(t) \\rbrace e^{\\mathrm{j} \\Phi\\lbrace s(t) \\rbrace}

    where

    .. math::

       A\\lbrace s \\rbrace =
       \\frac{ \\alpha_\\mathrm{a} \\frac{|s|}{s_\\mathrm{sat}} }
             { 1 + \\beta_\\mathrm{a} \\frac{|s|^2}{s_\\mathrm{sat}^2} }

    describes the amplitude model depending on two parameters
    :math:`\\alpha_\\mathrm{a}, \\beta_\\mathrm{a} \\in \\mathbb{R}_{+}`
    and

    .. math::

       \\Phi\\lbrace s \\rbrace =
       \\frac{ \\alpha_\\Phi \\frac{|s|}{s_\\mathrm{sat}} }
             { 1 + \\beta_\\Phi \\frac{|s|^2}{s_\\mathrm{sat}^2} }

    describes the phase model depending on
    :math:`\\alpha_\\Phi, \\beta_\\Phi \\in \\mathbb{R}`, respectively.


    See :footcite:t:`1981:saleh` for further details.
    """

    yaml_tag = u'Saleh'
    """YAML serialization tag."""

    phase_alpha: float
    """Phase model factor :math:`\\alpha_\\Phi`."""

    phase_beta: float
    """Phase model factor :math:`\\beta_\\Phi`."""

    __amplitude_alpha: float            # Amplitude model factor alpha.
    __amplitude_beta: float             # Amplitude model factor beta.

    def __init__(self,
                 amplitude_alpha: float,
                 amplitude_beta: float,
                 phase_alpha: float,
                 phase_beta: float,
                 **kwargs: Any) -> None:
        """
        Args:

            amplitude_alpha (float):
                Amplitude model factor alpha.

            amplitude_beta (float):
                Amplitude model factor beta.

            phase_alpha (float)
                Phase model factor alpha.

            phase_beta (float):
                Phase model factor beta.

            kwargs (Any):
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

        if value < 0.:
            raise ValueError("Amplitude model factor alpha must be greater or equal to zero")

        self.__amplitude_alpha = value

    @property
    def amplitude_beta(self) -> float:
        """Amplitude model factor :math:`\\beta_\\mathrm{a}`.

        Returns:
            float: Amplitude factor.

        Raises:
            ValueError: If the factor is smaller than zero.
        """

        return self.__amplitude_beta

    @amplitude_beta.setter
    def amplitude_beta(self, value: float) -> None:
        """Set the amplitude model factor beta."""

        if value < 0.:
            raise ValueError("Amplitude model factor beta must be greater or equal to zero")

        self.__amplitude_beta = value

    def model(self, input_signal: np.ndarray) -> np.ndarray:

        amp = np.abs(input_signal) / self.saturation_amplitude
        gain = self.__amplitude_alpha / (1 + self.__amplitude_beta * amp ** 2)
        phase_shift = self.phase_alpha * amp ** 2 / (1 + self.phase_beta * amp ** 2)

        return input_signal * gain * np.exp(1j * phase_shift)

    @classmethod
    def to_yaml(cls: Type[SalehPowerAmplifier], representer: SafeRepresenter, node: SalehPowerAmplifier) -> MappingNode:
        """Serialize a `SalehPowerAmplifier` object to YAML.

        Args:
            representer (BaseRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (SalehPowerAmplifier):
                The amplifier instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
        """

        state = {
            'amplitude_alpha': node.__amplitude_alpha,
            'amplitude_beta': node.__amplitude_beta,
            'phase_alpha': node.phase_alpha,
            'phase_beta': node.phase_beta,
        }

        representation = representer.represent_mapping(cls.yaml_tag, state)
        representation.value.extend(PowerAmplifier.to_yaml(representer, node).value)

        return representation


class CustomPowerAmplifier(PowerAmplifier):
    """Model of a customized power amplifier."""

    yaml_tag = u'Custom'
    """YAML serialization tag."""

    __input: np.ndarray
    __gain: np.ndarray
    __phase: np.ndarray

    def __init__(self,
                 input: np.ndarray,
                 gain: np.ndarray,
                 phase: np.ndarray,
                 **kwargs: Any) -> None:
        """
        Args:

            input (np.ndarray):
            gain (np.ndarray):
            phase (np.ndarray):

            kwargs (Any):
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
            raise ValueError("Custom power amplifier input, gain and phase vectors must be of identical length")

        self.__input = input
        self.__gain = gain
        self.__phase = phase

        PowerAmplifier.__init__(self, **kwargs)

    def model(self, input_signal: np.ndarray) -> np.ndarray:

        amp = np.abs(input_signal) / self.saturation_amplitude
        gain = np.interp(amp, self.__input, self.__gain)
        phase_shift = np.interp(amp, self.__input, self.__phase)

        return input_signal * gain * np.exp(1j * phase_shift)
