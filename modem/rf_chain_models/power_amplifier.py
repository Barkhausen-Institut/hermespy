from __future__ import annotations
import numpy as np
from enum import Enum
from typing import Type
from ruamel.yaml import Node, RoundTripRepresenter, RoundTripConstructor, SafeConstructor

__author__ = "Tobias Kronauer"
__copyright__ = "Copyright 2021, Barkhausen Institut gGmbH"
__credits__ = ["Tobias Kronauer", "Jan Adler"]
__license__ = "AGPLv3"
__version__ = "0.2.0"
__maintainer__ = "Tobias Kronauer"
__email__ = "tobias.kronauer@barkhauseninstitut.org"
__status__ = "Prototype"


class PowerAmplifier:
    """This module implements a distortion model for a power amplifier (PA).

    The following PA models are currently supported:

    - Rapp's model, as described in
      C. Rapp, "Effects of HPA Nonlinearity on a 4-PI DPSK/OFDM Signal for a digital Sound Broadcasting System"
      in Proc. Eur. Conf. Satellite Comm., 1991

    - Saleh's model, as described in
      A.A.M. Saleh, "Frequency-independent and frequency dependent nonlinear models of TWT amplifiers"
      in IEEE Trans. Comm., Nov. 1981

    - any custom AM/AM and AM/PM response as a vector

    Currently only memoryless models are supported.

    TODO: Refactor models into dedicated subclasses of `PowerAmplifier`
    """

    class Model(Enum):
        """Supported power amplifier models.
        """

        NONE = 0
        CLIP = 1
        RAPP = 2
        SALEH = 3
        CUSTOM = 4

    yaml_tag = 'PowerAmplifier'
    __model: Model
    __saturation_amplitude: float
    __input_backoff_pa_db: float
    __rapp_smoothness_factor: float
    __saleh_alpha_a: float
    __saleh_alpha_phi: float
    __saleh_beta_a: float
    __saleh_beta_phi: float
    __custom_pa_input: np.array
    __custom_pa_output: np.array
    __custom_pa_gain: np.array
    __custom_pa_phase: np.array
    __adjust_power_after_pa: bool

    def __init__(self,
                 model: Model = None,
                 tx_power: float = None,
                 saturation_amplitude: float = None,
                 input_backoff_pa_db: float = None,
                 rapp_smoothness_factor: float = None,
                 saleh_alpha_a: float = None,
                 saleh_alpha_phi: float = None,
                 saleh_beta_a: float = None,
                 saleh_beta_phi: float = None,
                 custom_pa_input: np.array = None,
                 custom_pa_output: np.array = None,
                 custom_pa_gain: np.array = None,
                 custom_pa_phase: np.array = None,
                 adjust_power_after_pa: bool = None) -> None:
        """Creates a power amplifier object

        TODO: ARGUMENTS
        """

        self.__model = self.Model.NONE
        self.__tx_power = 1.0
        self.__saturation_amplitude = 0.0
        self.__input_backoff_pa_db = 0.0
        self.__rapp_smoothness_factor = 1.0
        self.__power_backoff = 1.0
        self.__saleh_alpha_a = 2.0
        self.__saleh_alpha_phi = 1.0
        self.__saleh_beta_a = 2.0
        self.__saleh_beta_phi = 1.0
        self.__custom_pa_input = np.empty(0, dtype=float)
        self.__custom_pa_output = np.empty(0, dtype=float)
        self.__custom_pa_gain = np.empty(0, dtype=float)
        self.__custom_pa_phase = np.empty(0, dtype=float)
        self.__adjust_power_after_pa = False

        if model is not None:
            self.model = model

        if tx_power is not None:
            self.__tx_power = tx_power

        if saturation_amplitude is not None:
            self.__saturation_amplitude = saturation_amplitude

        if input_backoff_pa_db is not None:
            self.__input_backoff_pa_db = input_backoff_pa_db

        if rapp_smoothness_factor is not None:
            self.rapp_smoothness_factor = rapp_smoothness_factor

        if saleh_alpha_a is not None:
            self.__saleh_alpha_a = saleh_alpha_a

        if saleh_alpha_phi is not None:
            self.__saleh_alpha_phi = saleh_alpha_phi

        if saleh_beta_a is not None:
            self.__saleh_beta_a = saleh_beta_a

        if saleh_beta_phi is not None:
            self.__saleh_beta_phi = saleh_beta_phi

        if custom_pa_input is not None:
            self.__custom_pa_input = custom_pa_input

        if custom_pa_output is not None:
            self.__custom_pa_output = custom_pa_output

        if custom_pa_gain is not None:
            self.__custom_pa_gain = custom_pa_gain

        if custom_pa_phase is not None:
            self.__custom_pa_phase = custom_pa_phase

        if adjust_power_after_pa is not None:
            self.__adjust_power_after_pa = adjust_power_after_pa

        if self.__model != self.Model.NONE:
            self.__power_backoff = 10**(self.__input_backoff_pa_db/10)

            saturation_power = tx_power * self.__power_backoff
            self.__saturation_amplitude = np.sqrt(saturation_power)

    @classmethod
    def to_yaml(cls: Type[PowerAmplifier], representer: RoundTripRepresenter, node: PowerAmplifier) -> Node:
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
            'model': node.__model.name,
            'tx_power': node.__tx_power,
            'saturation_amplitude':  node.__saturation_amplitude,
            'input_backoff_pa_db': node.__input_backoff_pa_db,
            'rapp_smoothness_factor': node.__rapp_smoothness_factor,
            'saleh_alpha_a': node.__saleh_alpha_a,
            'saleh_alpha_phi': node.__saleh_alpha_phi,
            'saleh_beta_a': node.__saleh_beta_a,
            'saleh_beta_phi': node.__saleh_beta_phi,
            #'custom_pa_input': node.__custom_pa_input,
            #'custom_pa_output': node.__custom_pa_output,
            #'custom_pa_gain': node.__custom_pa_gain,
            #'custom_pa_phase': node.__custom_pa_phase,
            'adjust_power_after_pa': node.__adjust_power_after_pa
        }

        return representer.represent_mapping(cls.yaml_tag, state)

    @classmethod
    def from_yaml(cls: Type[PowerAmplifier], constructor: SafeConstructor, node: Node) -> PowerAmplifier:
        """Recall a new `PowerAmplifier` instance from YAML.

        Args:
            constructor (RoundTripConstructor):
                A handle to the constructor extracting the YAML information.

            node (Node):
                YAML node representing the `PowerAmplifier` serialization.

        Returns:
            PowerAmplifier:
                Newly created `PowerAmplifier` instance.
            """

        state = SafeConstructor.construct_mapping(constructor, node, deep=False)

        if 'model' in state.keys():

            if isinstance(state['model'], str):
                state['model'] = PowerAmplifier.Model[state['model']]

            elif isinstance(state['model'], int):
                state['model'] = PowerAmplifier.Model(state['model'])

        return PowerAmplifier(**state)

    def send(self, input_signal: np.ndarray) -> np.ndarray:
        """Returns the distorted version of the input signal according to the desired model

        Args:
            input_signal(np.ndarray): vector with input signal

        Returns:
            numpy.ndarray: Distorted signal of same size as `input_signal`.
        """

        output_signal = np.ndarray([])
        if self.__model == PowerAmplifier.Model.NONE:
            output_signal = input_signal

        elif self.__model == self.Model.CLIP:
            # clipping
            clip_idx = np.nonzero(np.abs(input_signal) > self.__saturation_amplitude)
            output_signal = np.copy(input_signal)
            output_signal[clip_idx] = self.__saturation_amplitude * np.exp(1j * np.angle(input_signal[clip_idx]))

        elif self.__model == PowerAmplifier.Model.RAPP:
            p = self.__rapp_smoothness_factor
            gain = (1 + (np.abs(input_signal) / self.__saturation_amplitude)**(2 * p))**(-1/2/p)
            output_signal = input_signal * gain

        elif self.__model == PowerAmplifier.Model.SALEH:
            amp = np.abs(input_signal) / self.__saturation_amplitude
            gain = self.__saleh_alpha_a / (1 + self.__saleh_beta_a * amp**2)
            phase_shift = self.__saleh_alpha_phi * amp**2 / (1 + self.__saleh_beta_phi * amp**2)
            output_signal = input_signal * gain * np.exp(1j * phase_shift)

        elif self.__model == PowerAmplifier.Model.CUSTOM:
            amp = np.abs(input_signal) / self.__saturation_amplitude
            gain = np.interp(amp, self.__custom_pa_input, self.__custom_pa_gain)
            phase_shift = np.interp(amp, self.__custom_pa_input, self.__custom_pa_phase)
            output_signal = input_signal * gain * np.exp(1j * phase_shift)

        else:
            ValueError(f"Power amplifier model {self.__model.name} not supported")

        if self.__adjust_power_after_pa:
            loss = np.linalg.norm(output_signal) / np.linalg.norm(input_signal)
            output_signal = output_signal / loss

        return output_signal

    @property
    def model(self) -> PowerAmplifier.Model:
        """Access the configured model.

        Returns (PowerAmplifier.Model):
            The configured model."""

        return self.__model

    @model.setter
    def model(self, model: Model) -> None:
        """Modify the configured model.

        Args:
            model (Model):
                The new model.

        Raises:
            ValueError:
                If he model is not supported.
        """

        if model not in PowerAmplifier.Model:
            raise ValueError("Model not supported")

        self.__model = model

    @property
    def rapp_smoothness_factor(self) -> float:
        """Access the configured smoothness factor for Rapp's model.

        Returns (float):
            The smoothness factor.

        """

        return self.__rapp_smoothness_factor

    @rapp_smoothness_factor.setter
    def rapp_smoothness_factor(self, factor: float) -> None:
        """Modify the configured smoothness factor for Rapp's model.

        Args:
            factor (float):
                The new factor. Must be greater than zero.

        Raises:
            ValueError:
                Should the `factor` be smaller or equal to zero.
        """

        if factor <= 0.0:
            raise ValueError("Smoothness factor ({}) in Rapp's model must be greater than zero".format(factor))

        self.__rapp_smoothness_factor = factor
