# -*- coding: utf-8 -*-
"""
=======================================
Hardware Radio Frequency Chain Modeling
=======================================

Isolation model (to be implemented): :footcite:t:`2018:kiayni`
"""


from __future__ import annotations
from typing import Type, Optional

import numpy as np
from ruamel.yaml import SafeConstructor, SafeRepresenter, Node

from hermespy.core.signal_model import Signal
from hermespy.core.factory import Serializable
from .power_amplifier import PowerAmplifier

__author__ = "André Noll Barreto"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["André Barreto", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.7"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class RfChain(Serializable):
    """Implements an RF chain model.

    Only PA is modelled.
    """

    yaml_tag = u'RfChain'
    __tx_power: float
    __phase_offset: float
    __amplitude_imbalance: float

    __power_amplifier: Optional[PowerAmplifier]

    def __init__(self,
                 tx_power: float = None,
                 phase_offset: float = None,
                 amplitude_imbalance: float = None) -> None:

        self.__tx_power = 1.0
        self.__phase_offset = 0.0
        self.__amplitude_imbalance = 0.0

        self.__power_amplifier = None

        if tx_power is not None:
            self.__tx_power = tx_power

        if phase_offset is not None:
            self.__phase_offset = phase_offset

        if amplitude_imbalance is not None:
            self.amplitude_imbalance = amplitude_imbalance

    @property
    def amplitude_imbalance(self) -> float:
        return self.__amplitude_imbalance

    @amplitude_imbalance.setter
    def amplitude_imbalance(self, val) -> None:
        if abs(val) >= 1:
            raise ValueError("Amplitude imbalance must be within interval (-1, 1).")

        self.__amplitude_imbalance = val

    @property
    def phase_offset(self) -> float:
        return self.__phase_offset

    @classmethod
    def to_yaml(cls: Type[RfChain], representer: SafeRepresenter, node: RfChain) -> Node:
        """Serialize an RfChain object to YAML.

        Args:
            representer (SafeRepresenter):
                A handle to a representer used to generate valid YAML code.
                The representer gets passed down the serialization tree to each node.

            node (RfChain):
                The `RfChain` instance to be serialized.

        Returns:
            Node:
                The serialized YAML node.
                None if the object state is default.
        """

        state = {}

        if node.__power_amplifier is not None:
            state[node.power_amplifier.yaml_tag] = node.__power_amplifier

        if node.__amplitude_imbalance != 0.0:
            state['amplitude_imbalance'] = node.__amplitude_imbalance

        if node.__phase_offset != 0.0:
            state['phase_offset'] = node.__phase_offset

        if len(state) < 1:
            return representer.represent_none(None)

        return representer.represent_mapping(cls.yaml_tag, state)

    def transmit(self, input_signal: Signal) -> Signal:
        """Returns the distorted version of signal in "input_signal".

        According to transmission impairments.
        """

        transmitted_signal = input_signal.copy()

        # Simulate IQ imbalance
        transmitted_signal.samples = self.add_iq_imbalance(transmitted_signal.samples)

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

        eta_alpha = np.cos(eps_delta/2) + 1j * eps_a * np.sin(eps_delta/2)
        eta_beta = eps_a * np.cos(eps_delta/2) - 1j * np.sin(eps_delta/2)

        return eta_alpha * x + eta_beta * np.conj(x)

    def receive(self, input_signal: Signal) -> Signal:
        """Returns the distorted version of signal in "input_signal".

        According to reception impairments.
        """

        input_signal = input_signal.copy()

        # Simulate IQ imbalance
        input_signal.samples = self.add_iq_imbalance(input_signal.samples)

        return input_signal

    @property
    def power_amplifier(self) -> PowerAmplifier:
        """Access the `PowerAmplifier` of the rf chain.

        Returns:
            A handle to the `PowerAmplifier`.
        """

        return self.__power_amplifier

    @power_amplifier.setter
    def power_amplifier(self, power_amplifier: PowerAmplifier) -> None:
        """Reassign the power amplifier configuration.

        Args:
            power_amplifier (PowerAmplifier):
                The new power amplifier configuration.
        """

        self.__power_amplifier = power_amplifier
