# -*- coding: utf-8 -*-
"""
=======================================
Hardware Radio Frequency Chain Modeling
=======================================

Isolation model (to be implemented): :footcite:t:`2018:kiayni`
"""

from __future__ import annotations

import numpy as np

from hermespy.core.signal_model import Signal
from hermespy.core.factory import Serializable
from .analog_digital_converter import AnalogDigitalConverter
from .phase_noise import PhaseNoise, NoPhaseNoise
from .power_amplifier import PowerAmplifier

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2023, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.1.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RfChain(Serializable):
    """Implements an RF chain model.

    Only PA is modelled.
    """

    yaml_tag = "RfChain"

    __phase_offset: float
    __amplitude_imbalance: float
    __power_amplifier: PowerAmplifier | None
    __phase_noise: PhaseNoise
    __adc: AnalogDigitalConverter

    def __init__(
        self,
        phase_offset: float | None = None,
        amplitude_imbalance: float | None = None,
        adc: AnalogDigitalConverter | None = None,
    ) -> None:
        # Initialize class attributes
        self.__phase_offset = 0.0
        self.__amplitude_imbalance = 0.0
        self.__power_amplifier = None
        self.__phase_noise = NoPhaseNoise()
        self.adc = AnalogDigitalConverter() if adc is None else adc

        if phase_offset is not None:
            self.__phase_offset = phase_offset

        if amplitude_imbalance is not None:
            self.amplitude_imbalance = amplitude_imbalance

    @property
    def amplitude_imbalance(self) -> float:
        """I/Q amplitude imbalance.

        Raises:

            ValueError: If the imbalance is less than -1 or more than one.
        """

        return self.__amplitude_imbalance

    @amplitude_imbalance.setter
    def amplitude_imbalance(self, val) -> None:
        if abs(val) >= 1:
            raise ValueError("Amplitude imbalance must be within interval (-1, 1).")

        self.__amplitude_imbalance = val

    @property
    def phase_offset(self) -> float:
        """I/Q phase offset.

        Returns: Phase offset in radians.
        """

        return self.__phase_offset

    @phase_offset.setter
    def phase_offset(self, value: float) -> None:
        self.__phase_offset = value

    @property
    def adc(self) -> AnalogDigitalConverter:
        """The analog to digital converter at the end of the RF receive chain."""

        return self.__adc

    @adc.setter
    def adc(self, value: AnalogDigitalConverter) -> None:
        self.__adc = value

    def transmit(self, input_signal: Signal) -> Signal:
        """Returns the distorted version of signal in "input_signal".

        According to transmission impairments.
        """

        transmitted_signal = input_signal.copy()

        # Simulate IQ imbalance
        transmitted_signal.samples = self.add_iq_imbalance(transmitted_signal.samples)

        # Simulate phase noise
        transmitted_signal = self.phase_noise.add_noise(transmitted_signal)

        # Simulate power amplifier
        if self.power_amplifier is not None:
            transmitted_signal.samples = self.power_amplifier.send(transmitted_signal.samples)

        return transmitted_signal

    def add_iq_imbalance(self, input_signal: np.ndarray) -> np.ndarray:
        """Adds Phase offset and amplitude error to input signal.

        Notation taken from https://en.wikipedia.org/wiki/IQ_imbalance.

        Args:
            input_signal (np.ndarray):
                Signal to be deteriorated as a matrix in shape `#no_antennas x #no_samples`.
                `#no_antennas` depends if on receiver or transmitter side.

        Returns:
            np.ndarray:
                Deteriorated signal with the same shape as `input_signal`.
        """
        x = input_signal
        eps_delta = self.__phase_offset
        eps_a = self.__amplitude_imbalance

        eta_alpha = np.cos(eps_delta / 2) + 1j * eps_a * np.sin(eps_delta / 2)
        eta_beta = eps_a * np.cos(eps_delta / 2) - 1j * np.sin(eps_delta / 2)

        return eta_alpha * x + eta_beta * np.conj(x)

    def receive(self, input_signal: Signal) -> Signal:
        """Returns the distorted version of signal in "input_signal".

        According to reception impairments.
        """

        input_signal = input_signal.copy()

        # Simulate IQ imbalance
        input_signal.samples = self.add_iq_imbalance(input_signal.samples)

        # Simulate phase noise
        input_signal = self.phase_noise.add_noise(input_signal)

        return input_signal

    @property
    def power_amplifier(self) -> PowerAmplifier:
        """Access the `PowerAmplifier` of the rf chain.

        Returns:
            A handle to the `PowerAmplifier`.
        """

        return self.__power_amplifier

    @power_amplifier.setter
    def power_amplifier(self, value: PowerAmplifier) -> None:
        self.__power_amplifier = value

    @property
    def phase_noise(self) -> PhaseNoise:
        """Phase Noise model configuration.

        Returns: Handle to the pase noise model.
        """

        return self.__phase_noise

    @phase_noise.setter
    def phase_noise(self, value: PhaseNoise) -> None:
        self.__phase_noise = value
